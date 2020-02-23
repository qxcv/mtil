"""Wrappers for rlpyt algorithms that inject reward from a custom reward model
at execution time.

TODO: need to figure out exactly how I'm going to do this for algorithms other
than PG. Some notes:

- For PG algorithms (PPO + A2C) it's easy to override the reward used at
  training time by subclassing & overriding the process_returns(samples)
  method. This won't work for algos with replay buffers!
- Not sure what to do for DQN. Prioritised DQN is a pain (and probably not
  possible to do efficiently anyway, so I may as well skip it). Probably my
  best bet is to override the loss() function to use actual reward. That will
  also be annoying b/c by default the algorithm uses a "return_" thing
  calculated by the replay buffer; to avoid that, I'll have to touch most parts
  of the loss() method (so say goodbye to forward-compat with future versions
  of rlpyt…).
- Also not yet sure how to customise reward evaluation in samplers; perhaps I
  shouldn't be doing that at all, and should instead write my own eval code?
  I'll definitely need my own code if I want to display both eval_score and the
  learnt reward."""

from collections import namedtuple
import warnings

from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppo import PPO
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch

from mtil.common import RunningMeanVariance

# ################# #
# For PG algorithms #
# ################# #


class RewardEvaluator:
    """Batching reward evaluator which can optionally standardise reward
    values."""
    def __init__(self,
                 reward_model,
                 obs_dims,
                 batch_size=256,
                 target_std=0.1,
                 normalise=False):
        self.batch_size = batch_size
        self.target_std = target_std
        self.normalise = normalise
        self.obs_dims = obs_dims
        if normalise:
            self.rew_running_average = RunningMeanVariance(())
        self.dev = next(iter(reward_model.parameters())).device
        self.reward_model = reward_model

    def evaluate(self, obs_tensor, act_tensor, update_stats=True):
        # put model into eval mode if necessary
        old_training = self.reward_model.training
        if old_training:
            self.reward_model.eval()

        with torch.no_grad():
            # flatten observations & actions
            old_dev = obs_tensor.device
            lead_dim, T, B, _ = infer_leading_dims(obs_tensor, self.obs_dims)
            obs_flat = obs_tensor.view((T*B, ) + obs_tensor.shape[lead_dim:])
            act_flat = act_tensor.view((T*B, ) + act_tensor.shape[lead_dim:])

            # now evaluate one batch at a time
            reward_tensors = []
            for b_start in range(0, T * B, self.batch_size):
                obs_batch = obs_flat[b_start:b_start + self.batch_size]
                act_batch = act_flat[b_start:b_start + self.batch_size]
                dev_obs = obs_batch.to(self.dev)
                dev_acts = act_batch.to(self.dev)
                dev_reward = self.reward_model(dev_obs, dev_acts)
                reward_tensors.append(dev_reward.to(old_dev))

            # join together the batch results
            new_reward_flat = torch.cat(reward_tensors, 0)
            new_reward = restore_leading_dims(new_reward_flat, lead_dim, T, B)

        # put back into training mode if necessary
        if old_training:
            self.reward_model.train(old_training)

        # normalise if necessary
        if self.normalise:
            if update_stats:
                self.rew_running_average.update(new_reward.flatten())
            mu = self.rew_running_average.mean.item()
            std = self.rew_running_average.std.item()
            denom = max(std / self.target_std, 1e-3)
            new_reward = (new_reward - mu) / denom

        return new_reward


class CustomRewardMixinPg:
    # filled in by set_reward_model()
    _reward_model = None
    # filled in by set_reward_model()
    _dev = None
    # filled in by process_returns()
    _last_rew_ret_adv = None
    # filled in by optimize_agent()
    _RRAInfo = None
    # _custom_logging_fields is used by GAILMinibatchRl (also also be
    # optimize_agent())
    _custom_logging_fields = ('synthRew', 'synthRet', 'synthAdv')

    def __init__(self, *args, true_reward_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._true_reward_weight = true_reward_weight
        if self._true_reward_weight:
            assert 0 < self._true_reward_weight <= 1.0, \
                f"true_reward_weight must be in [0,1], but was " \
                f"{self._true_reward_weight} (this code takes a convex " \
                f"combination of true & GAIL rewards)"
            warnings.warn(
                "USING GROUND TRUTH REWARD (!) in CustomRewardMixinPg. This "
                "is only for debugging, so remember to disable it later!")

    def set_reward_evaluator(self, reward_evaluator):
        self._reward_eval = reward_evaluator

    def process_returns(self, samples):
        assert self._reward_eval is not None, \
            "must call .set_reward_eval() on algorithm before continuing"

        # evaluate new rewards
        new_reward = self._reward_eval.evaluate(
            samples.env.observation, samples.agent.action)

        # sanity-check reward shapes
        assert new_reward.shape == samples.env.reward.shape, \
            (new_reward.shape, samples.env.reward.shape)

        if self._true_reward_weight:
            # debuging branch: use ground truth rewards
            alpha = self._true_reward_weight
            warnings.warn(f"USING GROUND TRUTH REWARD (!) at alpha={alpha}")
            env_reward = samples.env.reward
            new_reward = (1 - alpha) * new_reward + alpha * env_reward

        # replace rewards
        new_samples = samples._replace(env=samples.env._replace(
            reward=new_reward))

        # actually do return/advantage calculations
        return_, advantage, valid = super().process_returns(new_samples)

        # record old reward, return, and advantage so that we can log stats
        # later
        self._last_rew_ret_adv = (new_reward, return_, advantage)

        return return_, advantage, valid

    @staticmethod
    def _to_cpu_list(t):
        # rlpyt logging code only pays attention to lists of numbers, so if I
        # want my synthetic reward etc. to be logged then I need to put it here
        return list(t.detach().cpu().flatten().numpy())

    def optimize_agent(self, itr, samples):
        # slightly hacky, but whatever
        opt_info = super().optimize_agent(itr, samples)

        # log extra data
        if self._RRAInfo is None:
            old_fields = opt_info._fields
            all_fields = [*old_fields, *self._custom_logging_fields]
            self._RRAInfo = namedtuple('_RRAInfo', all_fields)
        rew, ret, adv = self._last_rew_ret_adv
        rra_info = self._RRAInfo(**opt_info._asdict(),
                                 synthRew=self._to_cpu_list(rew),
                                 synthRet=self._to_cpu_list(ret),
                                 synthAdv=self._to_cpu_list(adv))

        # being defensive: I want to see exception if we try to unpack the same
        # thing twice
        self._last_rew_ret_adv = None

        return rra_info


class CustomRewardPPO(CustomRewardMixinPg, PPO):
    pass


class CustomRewardA2C(CustomRewardMixinPg, A2C):
    pass


# ################ #
# For DQN variants #
# ################ #

# (TODO: going to try policy gradient first & then move to DQN if it seems more
# efficient)

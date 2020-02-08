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

import numpy as np
from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppo import PPO
import torch

# ################# #
# For PG algorithms #
# ################# #


class _RunningMeanVariance:
    """Exponentially-weighted running mean and variance, for reward
    normalisation."""
    def __init__(self, shape, discount=0.98):
        assert isinstance(shape, tuple)
        self._shape = shape
        self._fo = np.zeros(shape)
        self._so = np.zeros(shape)
        self.discount = discount
        self._n_updates = 0

    @property
    def mean(self):
        # bias correction like Adam
        ub_fo = self._fo / (1 - self.discount ** self._n_updates)
        return ub_fo

    @property
    def std(self):
        # same bias correction
        ub_fo = self.mean
        ub_so = self._so / (1 - self.discount ** self._n_updates)
        return np.sqrt(ub_so - ub_fo ** 2)

    def update(self, new_values):
        new_values = np.asarray(new_values)
        assert len(new_values) >= 1
        assert new_values[0].shape == self._shape
        nv_mean = np.mean(new_values, axis=0)
        nv_sq_mean = np.mean(new_values ** 2, axis=0)
        self._fo = self.discount * self._fo + (1 - self.discount) * nv_mean
        self._so = self.discount * self._so + (1 - self.discount) * nv_sq_mean
        self._n_updates += 1


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rew_running_average = _RunningMeanVariance((), discount=0.98)

    def set_reward_model(self, reward_model):
        self._reward_model = reward_model
        self._dev = next(iter(reward_model.parameters())).device

    def process_returns(self, samples):
        # inject custom reward into samples
        assert self._reward_model is not None, \
            "must call .set_reward_model() on algorithm before continuing"

        old_training = self._reward_model.training
        if old_training:
            self._reward_model.eval()
        with torch.no_grad():
            old_dev = samples.env.observation.device
            dev_obs = samples.env.observation.to(self._dev)
            dev_acts = samples.agent.action.to(self._dev)
            dev_reward = self._reward_model(dev_obs, dev_acts)
            new_reward = dev_reward.to(old_dev)
        if old_training:
            self._reward_model.train(old_training)

        # normalise
        # TODO: this logic should be put in self._reward_model, not here
        self._rew_running_average.update(new_reward.flatten())
        mu = self._rew_running_average.mean.item()
        std = self._rew_running_average.std.item()
        new_reward = (new_reward - mu) / max(std, 1e-3)

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

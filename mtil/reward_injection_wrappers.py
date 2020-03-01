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

from rlpyt.agents.base import AgentInputs
from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.base import OptInfo
from rlpyt.algos.pg.ppo import PPO, LossInputs
from rlpyt.utils.buffer import buffer_method, buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.tensor import (infer_leading_dims, restore_leading_dims,
                                valid_mean)
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
            obs_flat = obs_tensor.view((T * B, ) + obs_tensor.shape[lead_dim:])
            act_flat = act_tensor.view((T * B, ) + act_tensor.shape[lead_dim:])

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
        new_reward = self._reward_eval.evaluate(samples.env.observation,
                                                samples.agent.action)

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


BCLossInputs = namedarraytuple(
    "BCLossInputs", ["bc_agent_inputs", "bc_expert_action", "bc_valid"])


class BehaviouralCloningPPOMixin:
    def __init__(self, bc_loss_coeff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_loss_coeff = bc_loss_coeff

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        return_, advantage, valid = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle
        # all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be
                # OK.
                loss, entropy, perplexity = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state=rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) \
                / self.n_itr

        return opt_info

    def loss(self,
             agent_inputs,
             action,
             return_,
             advantage,
             valid,
             old_dist_info,
             bc_agent_inputs,
             bc_expert_actions,
             bc_valid,
             init_rnn_state=None):
        """
        Compute the BC-augmented training loss:
            policy_loss + value_loss + entropy_loss + bc_loss
        Policy loss: min(likelhood-ratio * advantage,
                         clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        BC loss: xent(policy(demo_states), action_labels)
        Calls the agent to compute forward pass on training data, and uses the
        ``agent.distribution`` to compute likelihoods and entropies. Valid for
        feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs,
                                                      init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action,
                                      old_dist_info=old_dist_info,
                                      new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
                                    1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = -valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_)**2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = -self.entropy_loss_coeff * entropy

        # BC loss (this is the only new part)
        if init_rnn_state is not None:
            bc_dist_info, _, _ = self.agent(*bc_agent_inputs, init_rnn_state)
        else:
            bc_dist_info, _ = self.agent(*bc_agent_inputs)
        expert_ll = dist.log_likelihood(bc_expert_actions, bc_dist_info)
        bc_loss = -self.bc_loss_coeff * valid_mean(expert_ll, bc_valid)

        loss = pi_loss + value_loss + entropy_loss + bc_loss

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity


class BCCustomRewardPPO(BehaviouralCloningPPOMixin, CustomRewardMixinPg, PPO):
    pass


# ################ #
# For DQN variants #
# ################ #

# (TODO: going to try policy gradient first & then move to DQN if it seems more
# efficient)

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

from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.utils import (discount_return,
                               generalized_advantage_estimation,
                               valid_from_done)
import torch

# ################# #
# For PG algorithms #
# ################# #


class CustomRewardMixinPg:
    reward_model = None

    def set_reward_model(self, reward_model):
        self.reward_model = reward_model

    def process_returns(self, samples):
        # TODO: shorten this by just manipulating samples directly & then
        # passing it to super().process_returns() (want to make sure I don't
        # modify the samples in-place, which will be annoying)
        assert self._reward_model is not None, \
            "must call .set_reward_model() on algorithm before continuing"
        # rest copied from rlpyt/algos/pg/base.py (but with reward modified)
        _, done, value, bv = (samples.env.reward, samples.env.done,
                              samples.agent.agent_info.value,
                              samples.agent.bootstrap_value)

        old_training = self._reward_model.training
        if old_training:
            self._reward_model.eval()
        with torch.no_grad():
            # TODO: is this reward even the right shape?
            reward = self._reward_model(samples.env.obs, samples.env.action)
        if old_training:
            self._reward_model.train(old_training)

        done = done.type(reward.dtype)

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(
                reward, value, done, bv, self.discount, self.gae_lambda)

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(
                done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        return return_, advantage, valid


class CustomRewardPPO(PPO, CustomRewardMixinPg):
    pass


class CustomRewardA2C(A2C, CustomRewardMixinPg):
    pass


# ################ #
# For DQN variants #
# ################ #

# (TODO: going to try policy gradient first & then move to DQN if it seems more
# efficient)

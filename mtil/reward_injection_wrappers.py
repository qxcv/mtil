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

from rlpyt.algos.pg.a2c import A2C
from rlpyt.algos.pg.ppo import PPO
import torch

# ################# #
# For PG algorithms #
# ################# #


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

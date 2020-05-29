from collections import namedtuple
import multiprocessing
import os

import gym
from milbench import register_envs
from rlpyt.envs.gym_schema import GymEnvWrapper
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.logging import context as log_ctx
from rlpyt.utils.logging import logger

from mtil.utils.misc import make_unique_run_name


class MILBenchGymEnv(GymEnvWrapper):
    """Useful for constructing rlpyt environments from Gym environment names
    (as needed to, e.g., create agents/samplers/etc.). Will automatically
    register MILBench envs first."""
    def __init__(self, env_name, **kwargs):
        register_envs()
        env = gym.make(env_name)
        super().__init__(env, **kwargs)


def make_logger_ctx(out_dir,
                    algo,
                    orig_env_name,
                    custom_run_name=None,
                    snapshot_gap=10,
                    **kwargs):
    # for logging & model-saving
    if custom_run_name is None:
        run_name = make_unique_run_name(algo, orig_env_name)
    else:
        run_name = custom_run_name
    logger.set_snapshot_gap(snapshot_gap)
    log_dir = os.path.abspath(out_dir)
    # this is irrelevant so long as it's a prefix of log_dir
    # FIXME: update rlpyt so that I can remove this LOG_DIR kludge.
    log_ctx.LOG_DIR = log_dir
    os.makedirs(out_dir, exist_ok=True)
    return log_ctx.logger_context(out_dir,
                                  run_ID=run_name,
                                  name="mtil",
                                  snapshot_mode="gap",
                                  **kwargs)


class MILBenchTrajInfo(AttrDict):
    """TrajInfo class that returns includes a score for the agent. Also
    includes trajectory length and 'base' reward to ensure that they are both
    zero."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Score = 0
        self.Length = 0
        self.BaseReward = 0
        self.Task = 0
        self.Variant = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Score += env_info.eval_score
        self.Length += 1
        self.BaseReward += reward

    def terminate(self, observation):
        self.Task = int(observation.task_id)
        self.Variant = int(observation.variant_id)
        return self


def get_policy_spec_milbench(env_metas):
    """Get `MultiHeadPolicyNet`'s `in_chans` and `n_actions` kwargs
    automatically from env metadata from a MILBench environment. Does sanity
    check to ensure that input & output shapes are the same for all envs."""
    obs_space = env_metas[0].observation_space
    act_space = env_metas[0].action_space
    assert all(em.observation_space == obs_space for em in env_metas)
    assert all(em.action_space == act_space for em in env_metas)
    assert len(obs_space.shape) == 3, obs_space.shape
    in_chans = obs_space.shape[-1]
    n_actions = act_space.n  # categorical action space
    model_kwargs = dict(in_chans=in_chans, n_actions=n_actions)
    return model_kwargs


# Spec data:
#
# - spec.id
# - spec.reward_threshold
# - spec.nondeterministic
# - spec.max_episode_steps
# - spec.entry_point
# - spec._kwargs
#
# Everything except entry_point and spec._kwargs is probably pickle-safe.
EnvMeta = namedtuple('EnvMeta', ['observation_space', 'action_space', 'spec'])
FilteredSpec = namedtuple(
    'FilteredSpec',
    ['id', 'reward_threshold', 'nondeterministic', 'max_episode_steps'])


def _get_env_meta_target(env_names, rv_dict):
    register_envs()  # in case this proc was spawned
    metas = []
    for env_name in env_names:
        # construct a bunch of envs in turn to get info about their observation
        # spaces, action spaces, etc.
        env = gym.make(env_name)
        spec = FilteredSpec(*(getattr(env.spec, field)
                              for field in FilteredSpec._fields))
        meta = EnvMeta(observation_space=env.observation_space,
                       action_space=env.action_space,
                       spec=spec)
        metas.append(meta)
        env.close()
    rv_dict['result'] = tuple(metas)


def get_env_metas(*env_names, ctx=multiprocessing):
    """Spawn a subprocess and use that to get metadata about an environment
    (env_spec, observation_space, action_space, etc.). Can optionally be passed
    a custom multiprocessing context to spawn subprocess with (e.g. so you can
    use 'spawn' method rather than the default 'fork').

    This is useful for environments that pollute some global state of the
    process which constructs them. For instance, the MILBench environments
    create some X resources that cannot be forked gracefully. If you do fork
    and then try to create a new env in the child, then you will end up with
    inscrutable resource errors."""
    mgr = ctx.Manager()
    rv_dict = mgr.dict()
    proc = ctx.Process(target=_get_env_meta_target, args=(env_names, rv_dict))
    try:
        proc.start()
        proc.join(30)
    finally:
        proc.terminate()
    if proc.exitcode != 0:
        raise multiprocessing.ProcessError(
            f"nonzero exit code {proc.exitcode} when collecting metadata "
            f"for '{env_names}'")
    result = rv_dict['result']
    return result

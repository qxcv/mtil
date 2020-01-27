"""Common tools for all of mtil package."""

from collections import namedtuple
import datetime
import multiprocessing
import os
import random
import sys
import uuid

import click
import gym
from milbench import register_envs
import numpy as np
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.logging import context as log_ctx
from rlpyt.utils.logging import logger
import torch
from torch import nn
from torch.utils import data


class MILBenchPreprocLayer(nn.Module):
    """Takes a uint8 image in format [N,T,H,W,C] (a batch of several time steps
    of H,W,C images) or [N,H,W,C] (i.e. T=1 time steps) and returns float image
    (elements in [-1,1]) in format [N,C*T,H,W] for Torch."""
    def forward(self, x):
        assert x.dtype == torch.uint8, \
            f"expected uint8 tensor but got {x.dtype} tensor"

        assert len(x.shape) == 4, x.shape
        N, H, W, C = x.shape
        # just transpose channels axis to front, do nothing else
        x = x.permute((0, 3, 1, 2))

        assert (H, W) == (128, 128), \
            f"(height,width)=({H},{W}), but should be (128,128) (try " \
            f"resizing)"

        # convert format and scale to [0,1]
        x = x.to(torch.float32) / 127.5 - 1.0

        return x


class MILBenchFeatureNetwork(nn.Module):
    """Convolutional feature extractor to process 128x128 images down into
    1024-dimensional feature vectors."""
    def __init__(self, in_chans=3, ActivationCls=torch.nn.ReLU):
        super().__init__()
        self.feature_generator = nn.Sequential(
            # TODO: consider adding batch norm, skip layers, etc. to this
            # at input: (128, 128)
            nn.Conv2d(in_chans, 64, kernel_size=5, stride=1),
            ActivationCls(),
            # now: (124, 124)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            ActivationCls(),
            # now: (64, 64)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            ActivationCls(),
            # now: (32, 32)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            ActivationCls(),
            # now: (16, 16)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            ActivationCls(),
            # now (8, 8)
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            ActivationCls(),
            # now (4, 4), 64 channels, for 16*64=1024 elements total
            # nn.Flatten(),
            # finally: flat 1024-elem vector
            # EDIT 2020-01-13: removing Flatten() layer & doing flattening
            # manually for Torch 1.2 compat.
        )

    def forward(self, x):
        nonflat_feats = self.feature_generator(x)
        flat_feats = nonflat_feats.view(nonflat_feats.shape[:1] + (-1, ))
        return flat_feats


class MILBenchGymEnv(GymEnvWrapper):
    """Useful for constructing rlpyt environments from Gym environment names
    (as needed to, e.g., create agents/samplers/etc.). Will automatically
    register MILBench envs first."""
    def __init__(self, env_name, **kwargs):
        register_envs()
        env = gym.make(env_name)
        super().__init__(env, **kwargs)


def set_seeds(seed):
    """Set all relevant PRNG seeds."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_unique_run_name(algo, orig_env_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    unique_suff = uuid.uuid4().hex[-6:]
    return f"{algo}-{orig_env_name}-{timestamp}-{unique_suff}"


def make_logger_ctx(out_dir,
                    algo,
                    orig_env_name,
                    custom_run_name=None,
                    **kwargs):
    # for logging & model-saving
    if custom_run_name is None:
        run_name = make_unique_run_name(algo, orig_env_name)
    else:
        run_name = custom_run_name
    logger.set_snapshot_gap(10)
    log_dir = os.path.abspath(out_dir)
    # this is irrelevant so long as it's a prefix of log_dir
    log_ctx.LOG_DIR = log_dir
    os.makedirs(out_dir, exist_ok=True)
    return log_ctx.logger_context(out_dir,
                                  run_ID=run_name,
                                  name="mtil",
                                  snapshot_mode="gap",
                                  **kwargs)


def trajectories_to_loader(demo_trajs, batch_size):
    """Re-format demonstration trajectories as a Torch DataLoader."""
    # convert dataset to Torch (always stored on CPU; we'll move one batch at a
    # time to the GPU)
    cpu_dev = torch.device("cpu")
    all_obs = torch.cat([
        torch.as_tensor(traj.obs[:-1], device=cpu_dev) for traj in demo_trajs
    ])
    all_acts = torch.cat(
        [torch.as_tensor(traj.acts, device=cpu_dev) for traj in demo_trajs])
    dataset = data.TensorDataset(all_obs, all_acts)
    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             pin_memory=False,
                             shuffle=True,
                             drop_last=True)
    return loader


def trajectories_to_loader_mt(demo_trajs_by_env, batch_size):
    """Like trajectories_to_loader, but for multi-task data, so it also yields
    a vector of task IDs with every sample."""

    # TODO: replace all uses of trajectories_to_loader with this. No sense
    # having two things that do the same thing.

    cpu_dev = torch.device("cpu")

    # create some numeric task IDs
    env_names = sorted(demo_trajs_by_env.keys())
    env_ids = torch.arange(len(env_names), device=cpu_dev)
    np_env_ids = env_ids.numpy()
    env_name_to_id = dict(zip(env_names, np_env_ids))
    env_id_to_name = dict(zip(np_env_ids, env_names))

    # make big list of trajectories in a deterministic order
    all_obs = []
    all_acts = []
    all_ids = []
    for env_name, env_id in sorted(env_name_to_id.items()):
        demo_trajs = demo_trajs_by_env[env_name]
        for traj in demo_trajs:
            all_obs.append(torch.as_tensor(traj.obs[:-1], device=cpu_dev))
            all_acts.append(torch.as_tensor(traj.acts, device=cpu_dev))
            id_tensor = torch.full((len(traj.acts), ),
                                   env_id,
                                   dtype=torch.long)
            all_ids.append(id_tensor)

    # join together trajectories into Torch dataset
    all_obs = torch.cat(all_obs)
    all_acts = torch.cat(all_acts)
    all_ids = torch.cat(all_ids)
    dataset = data.TensorDataset(all_ids, all_obs, all_acts)

    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             pin_memory=False,
                             shuffle=True,
                             drop_last=True)

    return loader, env_name_to_id, env_id_to_name


class MILBenchTrajInfo(AttrDict):
    """TrajInfo class that returns includes a score for the agent. Also
    includes trajectory length and 'base' reward to ensure that they are both
    zero."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Score = 0
        self.Length = 0
        self.BaseReward = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Score += env_info.eval_score
        self.Length += 1
        self.BaseReward += reward

    def terminate(self, observation):
        return self


def sane_click_init(cli):
    """Initialise Click in a sensible way that prevents it from catching
    KeyboardInterrupt exceptions, but still allows it to show usage messages.
    `cli` should be a function decorated with `@click.group` that you want to
    execute, or a function decorated with `@cli.command` for some group
    `cli`."""
    try:
        with cli.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            cli.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        if e.exit_code == 0:
            sys.exit(e.exit_code)
        raise


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


def _get_env_meta_target(env_name, rv_dict):
    register_envs()  # in case this proc was spawned
    env = gym.make(env_name)
    spec = FilteredSpec(*(getattr(env.spec, field)
                          for field in FilteredSpec._fields))
    meta = EnvMeta(observation_space=env.observation_space,
                   action_space=env.action_space,
                   spec=spec)
    rv_dict['result'] = meta
    env.close()


def get_env_meta(env_name, ctx=multiprocessing):
    """Spawn a subprocess and use that to get metadata about an environment
    (env_spec, observation_space, action_space, etc.). Can optionally be passed
    a custom multiprocessing context to spawn subprocess with (e.g. so you can
    use 'spawn' method rather than the default 'fork')."""
    mgr = ctx.Manager()
    rv_dict = mgr.dict()
    proc = ctx.Process(target=_get_env_meta_target, args=(env_name, rv_dict))
    try:
        proc.start()
        proc.join(30)
    finally:
        proc.terminate()
    if proc.exitcode != 0:
        raise multiprocessing.ProcessError(
            f"nonzero exit code {proc.exitcode} when collecting metadata "
            f"for '{env_name}'")
    result = rv_dict['result']
    return result

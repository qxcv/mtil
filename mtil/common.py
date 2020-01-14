"""Common tools for all of mtil package."""

import datetime
import os
import random
import uuid

import gym
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
        flat_feats = nonflat_feats.view(nonflat_feats.shape[:1] + (-1,))
        return flat_feats


class VanillaGymEnv(GymEnvWrapper):
    """Useful for constructing rlpyt environments from Gym environment names
    (as needed to, e.g., create agents/samplers/etc.)."""
    def __init__(self, env_name, **kwargs):
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


def make_logger_ctx(out_dir, algo, orig_env_name, custom_run_name=None,
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
                             pin_memory=True,
                             shuffle=True,
                             drop_last=True)
    return loader


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

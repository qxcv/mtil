"""Common tools for all of mtil package."""

import random

import gym
import numpy as np
from rlpyt.envs.gym import GymEnvWrapper
import torch
from torch import nn


class MILBenchPreprocLayer(nn.Module):
    """Takes a uint8 image in format [N,T,H,W,C] (a batch of several time steps
    of H,W,C images) or [N,H,W,C] (i.e. T=1 time steps) and returns float image
    (elements in [-1,1]) in format [N,C*T,H,W] for Torch."""
    def forward(self, x):
        assert x.dtype == torch.uint8, \
            f"expected uint8 tensor but got {x.dtype} tensor"

        if len(x.shape) == 5:
            N, T, H, W, C = x.shape
            # move channels to the beginning so it's [N,T,C,H,W]
            x = x.permute((0, 1, 4, 2, 3))
            # flatten along channels axis
            x = x.reshape((N, T * C, H, W))
        else:
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


class MILBenchPolicyNet(nn.Module):
    """Convolutional policy network that yields some action logits. Note this
    network is specific to MILBench (b/c of preproc layer and action count)."""
    def __init__(self,
                 in_chans=3,
                 n_actions=3 * 3 * 2,
                 ActivationCls=torch.nn.ReLU):
        super().__init__()
        # this asserts that input is 128x128, and ensures that it is in
        # [N,C,H,W] format with float32 values in [-1,1].
        self.preproc = MILBenchPreprocLayer()
        self.logit_generator = nn.Sequential(
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
            nn.Flatten(),
            # now: flat 1024-elem vector
            nn.Linear(1024, 256),
            ActivationCls(),
            # now: flat 256-elem vector
            nn.Linear(256, 256),
            ActivationCls(),
            # now: flat 256-elem vector
            nn.Linear(256, n_actions),
            # now: we get n_actions logits to use with softmax etc.
        )

    def forward(self, x):
        preproc = self.preproc(x)
        logits = self.logit_generator(preproc)
        return logits


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

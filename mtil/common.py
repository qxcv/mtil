"""Common tools for all of mtil package."""

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
    """Convolutional policy network that yields some actino logits. Note this
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


def _preproc_n_actions(n_actions_per_dim, other_act_tensor):
    n_actions_per_dim = torch.as_tensor(n_actions_per_dim,
                                        dtype=other_act_tensor.dtype,
                                        device=other_act_tensor.device)
    # assert n_actions_per_dim.ndim == 1, n_actions_per_dim.shape
    # This is ridiculous. There has to be a better way of doing it.
    single_one = torch.ones((1, ),
                            device=n_actions_per_dim.device,
                            dtype=n_actions_per_dim.dtype)
    cat_action_nums = torch.cat((single_one, n_actions_per_dim[:-1]))
    return n_actions_per_dim, torch.cumprod(cat_action_nums, 0)


def md_to_flat_action(md_actions, n_actions_per_dim):
    """Convert a tensor of MultiDiscrete actions to "flat" discrete actions."""
    # should be int
    assert not md_actions.is_floating_point(), md_actions.dtype
    assert len(n_actions_per_dim) == md_actions.shape[-1], \
        (n_actions_per_dim, md_actions.shape)

    n_actions_per_dim, action_prods \
        = _preproc_n_actions(n_actions_per_dim, md_actions)
    action_prods_bcast = torch.reshape(
        action_prods, (1, ) * (len(md_actions.shape) - 1) + action_prods.shape)

    flat_acts = torch.sum(md_actions * action_prods_bcast, -1)

    return flat_acts


def flat_to_md_action(flat_actions, n_actions_per_dim):
    """Convert a tensor of flat discrete actions to "MultiDiscrete" actions.
    `flat_actions` can be of arbitrary shape [A, B, ...]; `n_actions_per_dim`
    must be of dim 1 (call its length `N`). Will return tensor of shape [A, B,
    ..., N]. Inverse of `md_to_flat_action`."""
    assert not flat_actions.is_floating_point(), flat_actions.dtype
    n_actions_per_dim, action_prods \
        = _preproc_n_actions(n_actions_per_dim, flat_actions)

    # decompose each "flat" action into a weighted sum of elements of
    # action_prods
    flat_copy = flat_actions.clone()
    md_actions = flat_actions.new(flat_actions.shape +
                                  (n_actions_per_dim.shape[0], ))
    for i in range(n_actions_per_dim.shape[0])[::-1]:
        md_actions[..., i] = flat_copy // action_prods[i]
        flat_copy = flat_copy % action_prods[i]

    return md_actions

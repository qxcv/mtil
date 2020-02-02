"""Multi-Task Behavioural Cloning (MTBC). Train on several environments, with
one "head" per environment. For now this only works with MILBench environments,
so it assumes that all environments have the same input & output spaces."""

import collections
import re

import numpy as np
from rlpyt.utils.prog_bar import ProgBarCounter
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import nn
import torch.nn.functional as F

from mtil.algos.bc import AgentModelWrapper
from mtil.common import MILBenchFeatureNetwork, MILBenchPreprocLayer


class MultiHeadPolicyNet(nn.Module):
    """Like MILBenchPolicyNet in bc.py, but for multitask policies with one
    head per environment."""
    def __init__(self,
                 env_ids_and_names,
                 in_chans=3,
                 n_actions=3 * 3 * 2,
                 fc_dim=256,
                 ActivationCls=torch.nn.ReLU):
        super().__init__()
        self.preproc = MILBenchPreprocLayer()
        self.feature_extractor = MILBenchFeatureNetwork(
            in_chans=in_chans, ActivationCls=ActivationCls)
        self.fc_postproc = nn.Sequential(
            nn.Linear(1024, 256),
            ActivationCls(),
            # now: flat 256-elem vector
            nn.Linear(256, fc_dim),
            ActivationCls(),
            # now: flat 256-elem vector
        )
        # these "embeddings" are actually affine transformation parameters,
        # with weight matrix at the beginning and bias at the end
        embedding_size = fc_dim * n_actions + n_actions
        self.task_embeddings = nn.Embedding(len(env_ids_and_names),
                                            embedding_size)
        # now: n_actions-dimensional logit vector

        # save env IDs and names so that we know how to reconstruct, as well as
        # fc_dim and n_actionsso that we can do reshaping
        self.env_ids_and_names = sorted(env_ids_and_names)
        self.fc_dim = fc_dim
        self.n_actions = n_actions

    def forward(self, obs, task_ids):
        preproc = self.preproc(obs)
        features = self.feature_extractor(preproc)
        fc_features = self.fc_postproc(features)

        # slightly messy: have to reshape "embeddings" into weight matrices &
        # affine parameters, then apply them like a batch of different affine
        # layers
        embeddings = self.task_embeddings(task_ids)
        bias_begin = self.fc_dim * self.n_actions
        matrices_flat = embeddings[..., :bias_begin]
        mats_shape = matrices_flat.shape[:-1] + (self.n_actions, self.fc_dim)
        matrices = matrices_flat.view(mats_shape)
        biases = embeddings[..., bias_begin:]
        bc_fc_features = fc_features[..., None]
        mm_result = torch.squeeze(matrices @ bc_fc_features, dim=-1)
        assert mm_result.shape == biases.shape, (mm_result.shape, biases.shape)
        logits = mm_result + biases

        expected_shape = task_ids.shape + (self.n_actions, )
        assert logits.shape == expected_shape, \
            f"expected logits to be shape {expected_shape}, but got " \
            f"shape {logits.shape} (might be reshaping bug)"

        return logits


class FixedTaskModelWrapper(AgentModelWrapper):
    """Like AgentModelWrapper, but for multi-head policies that expect task IDs
    as input. Assumes that it is only ever getting applied to one task,
    identified by a given integer `task_id`. Good when you have one sampler per
    task."""
    def __init__(self, task_id, **kwargs):
        # This should be given 'model_ctor', 'model_kwargs', and optionally
        # 'model' kwargs.
        super().__init__(**kwargs)
        self.task_id = task_id

    def forward(self, obs, prev_act, prev_rew):
        # similar to AgentModelWrapper.forward(), but also constructs task IDs
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        task_ids = torch.full((T * B, ), self.task_id, dtype=torch.long) \
            .to(obs.device)
        logits = self.model(obs.view(T * B, *img_shape), task_ids)
        pi = F.softmax(logits, dim=-1)
        v = torch.zeros((T * B, ), device=pi.device, dtype=pi.dtype)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v


def copy_model_into_sampler(model, sampler, prefix='model'):
    """Update the `.agent` inside `sampler` so that it contains weights from
    `model`. Should call this before doing evaluation rollouts between epochs
    of training."""
    state_dict = model.state_dict()
    assert hasattr(sampler.agent, prefix)
    updated_state_dict = {
        prefix + '.' + key: value
        for key, value in state_dict.items()
    }
    sampler.agent.load_state_dict(updated_state_dict)


def do_epoch_training_mt(loader, model, opt, dev, passes_per_eval):
    # @torch.jit.script
    def do_loss_forward_back(task_ids_batch, obs_batch, acts_batch):
        logits_flat = model(obs_batch, task_ids_batch)
        losses = F.cross_entropy(logits_flat, acts_batch.long(),
                                 reduction='none')
        loss = losses.mean()
        loss.backward()
        return losses.detach().cpu().numpy()

    # make sure we're in train mode
    model.train()

    # for logging
    loss_ewma = None
    losses = []
    per_task_losses = collections.defaultdict(lambda: [])
    progress = ProgBarCounter(len(loader) * passes_per_eval)
    for pass_num in range(passes_per_eval):
        for batches_done, (task_ids_batch, obs_batch, acts_batch) \
                in enumerate(loader, start=1):
            # copy to GPU
            obs_batch = obs_batch.to(dev)
            acts_batch = acts_batch.to(dev)
            task_ids_batch = task_ids_batch.to(dev)

            # compute loss & take opt step
            opt.zero_grad()
            batch_losses = do_loss_forward_back(
                task_ids_batch, obs_batch, acts_batch)
            opt.step()

            # for logging

            progress.update(batches_done + len(loader) * pass_num)
            f_loss = np.mean(batch_losses)
            loss_ewma = f_loss if loss_ewma is None \
                else 0.9 * loss_ewma + 0.1 * f_loss
            losses.append(f_loss)

            # also track separately for each task
            np_task_ids = task_ids_batch.cpu().numpy()
            for task_id in np.unique(np_task_ids):
                rel_losses = batch_losses[np_task_ids == task_id]
                if len(rel_losses) > 0:
                    per_task_losses[task_id].append(np.mean(rel_losses))

    progress.stop()

    return loss_ewma, losses, per_task_losses


_no_version_re = re.compile(r'^(?P<env_name>.*?)(-v\d+)?$')
_alnum_re = re.compile(r'[a-zA-Z0-9]+')


def make_env_tag(env_name):
    """Take a Gym env name like 'fooBar-BazQux-v3' and return more concise string
    of the form 'FooBarBazQux' (no version string, no non-alphanumeric
    characters, letters that formerly separated words are always
    capitalised)."""
    no_version = _no_version_re.match(env_name).groupdict()['env_name']
    alnum_parts = _alnum_re.findall(no_version)
    final_name = ''.join(part[0].upper() + part[1:] for part in alnum_parts)
    return final_name

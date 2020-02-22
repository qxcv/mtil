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
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
import numpy as np
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.logging import context as log_ctx
from rlpyt.utils.logging import logger
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import nn
import torch.nn.functional as F
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


class MultiTaskAffineLayer(nn.Module):
    """A multi-task version of torch.nn.Linear. It keeps a separate set of
    weights for each of `n_tasks` tasks. On the forward pass, it takes a batch
    of task IDs in addition to a batch of feature inputs, and uses those to
    look up the appropriate affine transform to apply to each batch element."""
    def __init__(self, in_chans, out_chans, n_tasks):
        super().__init__()

        # these "embeddings" are actually affine transformation parameters,
        # with weight matrix at the beginning and bias at the end
        self._embedding_shapes = [(out_chans, in_chans), (out_chans, )]
        self._embedding_sizes = [
            np.prod(shape) for shape in self._embedding_shapes
        ]
        full_embed_size = sum(self._embedding_sizes)
        self.task_embeddings = nn.Embedding(n_tasks, full_embed_size)

    def _retrieve_embeddings(self, task_ids):
        embeddings = self.task_embeddings(task_ids)
        stops = list(np.cumsum(self._embedding_sizes))
        starts = [0] + stops[:-1]
        reshaped = []
        for start, stop, shape in zip(starts, stops, self._embedding_shapes):
            full_shape = embeddings.shape[:-1] + shape
            part = embeddings[..., start:stop].view(full_shape)
            reshaped.append(part)
        return reshaped

    def forward(self, inputs, task_ids):
        # messy: must reshape "embeddings" into weight matrices & affine
        # parameters, then apply them like a batch of different affine layers
        matrices, biases \
            = self._retrieve_embeddings(task_ids)
        bc_fc_features = inputs[..., None]
        mm_result = torch.squeeze(matrices @ bc_fc_features, dim=-1)
        assert mm_result.shape == biases.shape, \
            (mm_result.shape, biases.shape)
        result = mm_result + biases

        return result


class MultiHeadPolicyNet(nn.Module):
    """Like MILBenchPolicyNet in bc.py, but for multitask policies with one
    head per environment. Returns both logits and values, for use in algorithms
    other than BC."""
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
            nn.Linear(1024, fc_dim),
            ActivationCls(),
            # now: flat <fc_dim>-elem vector
            nn.Linear(fc_dim, fc_dim),
            ActivationCls(),
            # now: flat <fc_dim>-elem vector
        )
        # this produces both a single value output and a vector of policy
        # logits for each sample
        self.mt_fc_layer = MultiTaskAffineLayer(fc_dim, n_actions + 1,
                                                len(env_ids_and_names))

        # save env IDs and names so that we know how to reconstruct, as well as
        # fc_dim and n_actionsso that we can do reshaping
        self.env_ids_and_names = sorted(env_ids_and_names)
        self.fc_dim = fc_dim
        self.n_actions = n_actions

    def forward(self, obs, task_ids=None):
        if task_ids is None:
            # if the task is unambiguous, then it's fine not to pass IDs
            n_tasks = len(self.env_ids_and_names)
            assert n_tasks == 1, \
                "no task_ids given, but have {n_tasks} tasks to choose from"
            task_ids = obs.new_zeros(obs.shape[1:], dtype=torch.long)

        preproc = self.preproc(obs)
        features = self.feature_extractor(preproc)
        fc_features = self.fc_postproc(features)
        logits_and_values = self.mt_fc_layer(fc_features, task_ids)
        logits = logits_and_values[..., :-1]
        values = logits_and_values[..., -1]

        l_expected_shape = task_ids.shape + (self.n_actions, )
        assert logits.shape == l_expected_shape, \
            f"expected logits to be shape {l_expected_shape}, but got " \
            f"shape {logits.shape}"
        assert values.shape == task_ids.shape, \
            f"expected values to be shape {task_ids.shape}, but got " \
            f"shape {values.shape}"

        return logits, values


class AgentModelWrapper(nn.Module):
    """Wraps a normal (observation -> logits) feedforward network so that (1)
    it deals gracefully with the variable-dimensional inputs that the rlpyt
    sampler gives it, (2) it produces action probabilities instead of logits,
    and (3) it produces some value 'values' to keep CategoricalPgAgent
    happy."""
    def __init__(self, model_ctor, model_kwargs, model=None):
        super().__init__()
        if model is not None:
            self.model = model
        else:
            self.model = model_ctor(**model_kwargs)

    def forward(self, obs, prev_act, prev_rew):
        # copied from AtariFfModel, then modified to match own situation
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        logits = self.model(obs.view(T * B, *img_shape))
        pi = F.softmax(logits, dim=-1)
        # fake values (BC doesn't use them)
        v = torch.zeros((T * B, ), device=pi.device, dtype=pi.dtype)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v


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
        logits, v = self.model(obs.view(T * B, *img_shape), task_ids)
        pi = F.softmax(logits, dim=-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v


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
    log_ctx.LOG_DIR = log_dir
    os.makedirs(out_dir, exist_ok=True)
    return log_ctx.logger_context(out_dir,
                                  run_ID=run_name,
                                  name="mtil",
                                  snapshot_mode="gap",
                                  **kwargs)


def trajectories_to_dataset_mt(demo_trajs_by_env):
    """Re-format multi-task trajectories into a Torch dataset."""

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
        n_samples = 0
        for traj in demo_trajs:
            all_obs.append(torch.as_tensor(traj.obs[:-1], device=cpu_dev))
            all_acts.append(torch.as_tensor(traj.acts, device=cpu_dev))
            id_tensor = torch.full((len(traj.acts), ),
                                   env_id,
                                   dtype=torch.long)
            all_ids.append(id_tensor)
            n_samples += len(id_tensor)

        # weight things inversely proportional to their frequency
        assert n_samples > 0, demo_trajs

    # join together trajectories into Torch dataset
    all_obs = torch.cat(all_obs)
    all_acts = torch.cat(all_acts)
    all_ids = torch.cat(all_ids)
    dataset = data.TensorDataset(all_ids, all_obs, all_acts)

    return dataset, env_name_to_id, env_id_to_name


def make_loader_mt(dataset, batch_size):
    """Construct sampler that randomly chooses N items from N-sample dataset,
    weighted so that it's even across all tasks (so no task implicitly has
    higher priority than the others). Assumes the given dataset is a
    TensorDataset produced by trajectories_to_dataset_mt."""
    task_ids = dataset.tensors[0]
    unique_ids, frequencies = torch.unique(task_ids,
                                           return_counts=True,
                                           sorted=True)
    # all tasks must be present for this to work
    assert torch.all(unique_ids == torch.arange(len(unique_ids))), (unique_ids)
    freqs_total = torch.sum(frequencies).to(torch.float)
    unique_weights = frequencies.to(torch.float) / freqs_total
    weights = unique_weights[task_ids]

    weighted_sampler = data.WeightedRandomSampler(weights,
                                                  len(weights),
                                                  replacement=True)
    batch_sampler = data.BatchSampler(weighted_sampler,
                                      batch_size=batch_size,
                                      drop_last=True)

    loader = data.DataLoader(dataset,
                             pin_memory=False,
                             batch_sampler=batch_sampler)

    return loader


# TODO: unify this, make_loader_mt, and trajectories_to_dataset_mt into one big
# class.
def load_demos_mt(demo_paths, add_preproc=None):
    """Load multi-task demonstrations. Can apply any desired MILBench
    preprocessor as needed."""
    demo_dicts = load_demos(demo_paths)
    orig_env_names = [d['env_name'] for d in demo_dicts]
    orig_names_uniq = sorted(set(orig_env_names))
    if add_preproc:
        env_names = [
            splice_in_preproc_name(orig_env_name, add_preproc)
            for orig_env_name in orig_env_names
        ]
        print(f"Splicing preprocessor '{add_preproc}' into environments "
              f"{orig_names_uniq}. New names are {sorted(set(env_names))}")
    else:
        env_names = orig_env_names
    # pair of (original name, name with preprocessor spliced in)
    name_pairs = sorted(set(zip(orig_env_names, env_names)))
    demo_trajs_by_env = {
        env_name: [
            demo_dict['trajectory'] for demo_dict in demo_dicts
            if demo_dict['env_name'] == orig_env_name
        ]
        for orig_env_name, env_name in name_pairs
    }
    assert sum(map(len, demo_trajs_by_env.values())) == len(demo_dicts)
    if add_preproc:
        demo_trajs_by_env = {
            env_name:
            preprocess_demos_with_wrapper(demo_trajs_by_env[env_name],
                                          orig_env_name, add_preproc)
            for orig_env_name, env_name in name_pairs
        }
    dataset_mt, env_name_to_id, env_id_to_name = trajectories_to_dataset_mt(
        demo_trajs_by_env)

    return dataset_mt, env_name_to_id, env_id_to_name, name_pairs


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
    use 'spawn' method rather than the default 'fork').

    This is useful for environments that pollute some global state of the
    process which constructs them. For instance, the MILBench environments
    create some X resources that cannot be forked gracefully. If you do fork
    and then try to create a new env in the child, then you will end up with
    inscrutable resource errors."""
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


class RunningMeanVarianceEWMA:
    """Running mean and variance. Intended for reward normalisation."""
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
        ub_fo = self._fo / (1 - self.discount**self._n_updates)
        return ub_fo

    @property
    def std(self):
        # same bias correction
        ub_fo = self.mean
        ub_so = self._so / (1 - self.discount**self._n_updates)
        return np.sqrt(ub_so - ub_fo**2)

    def update(self, new_values):
        new_values = np.asarray(new_values)
        assert len(new_values) >= 1
        assert new_values[0].shape == self._shape
        nv_mean = np.mean(new_values, axis=0)
        nv_sq_mean = np.mean(new_values**2, axis=0)
        self._fo = self.discount * self._fo + (1 - self.discount) * nv_mean
        self._so = self.discount * self._so + (1 - self.discount) * nv_sq_mean
        self._n_updates += 1


class RunningMeanVariance:
    """This version computes full history instead of EWMA. Copy-pasted from
    Stable Baslines."""
    def __init__(self, shape=(), epsilon=1e-4):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param shape: (tuple) the shape of the data stream's output
        :param epsilon: (float) helps with arithmetic issues
        """
        self.mean = torch.zeros(shape, dtype=torch.double)
        self.var = torch.ones(shape, dtype=torch.double)
        self.count = epsilon

    @property
    def std(self):
        return torch.sqrt(self.var)

    def update(self, arr):
        # reshape as appropriate (assume last dimension(s) contain data
        # elements)
        shape_idx = len(arr.shape) - len(self.mean.shape)
        assert shape_idx >= 0
        tail_shape = arr.shape[shape_idx:]
        assert tail_shape == self.mean.shape, (tail_shape, self.mean.shape)
        arr = arr.view((-1, ) + tail_shape)

        batch_var, batch_mean = torch.var_mean(arr)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta ** 2) * self.count * batch_count \
            / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

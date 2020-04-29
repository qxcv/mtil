"""MTGAIL-specific tools for loading trajectories.

TODO: unify this with code in common.py. Really this should replace that code
because it's much cleaner :)"""

import collections

from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
from milbench.benchmarks import EnvName
import numpy as np
from rlpyt.agents.pg.categorical import CategoricalPgAgent
import torch
from torch.utils import data

# TODO: factor load_state_dict_or_model from mtbc out into common
from mtil.algos.mtgail.sample_mux import (EnvIDObsArray,
                                          MILBenchEnvMultiplexer,
                                          MuxCpuSampler, MuxGpuSampler,
                                          MuxTaskModelWrapper)
from mtil.common import (MILBenchTrajInfo, MultiHeadPolicyNet, get_env_metas,
                         tree_map)

# vvvvvvvvvvv STUFF TO FACTOR OUT vvvvvvvvvvvvvvv


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


def get_demos_meta(*,
                   demo_paths,
                   omit_noop=False,
                   transfer_variants=(),
                   preproc_name=None):
    dataset_mt, variant_groups = load_demos_mtgail(demo_paths,
                                                   transfer_variants,
                                                   mb_preproc=preproc_name,
                                                   omit_noop=omit_noop)

    print("Getting env metadata")
    all_env_names = sorted(variant_groups.task_variant_by_name.keys())
    env_metas = get_env_metas(*all_env_names)

    task_ids_and_demo_env_names = [
        (env_name, task_id) for env_name, (task_id, variant_id)
        in variant_groups.task_variant_by_name.items()
        # variant ID 0 is (usually) the demo env
        # TODO: make sure this is ACTUALLY the demo env
        if variant_id == 0
    ]  # yapf: disable

    rv = {
        'dataset_mt': dataset_mt,
        'variant_groups': variant_groups,
        'env_metas': env_metas,
        'task_ids_and_demo_env_names': task_ids_and_demo_env_names,
    }
    return rv


def make_mux_sampler(*, variant_groups, env_metas, use_gpu, batch_B, batch_T):
    print("Setting up environment multiplexer")
    # local copy of Gym env, w/ args to create equivalent env in the sampler
    env_mux = MILBenchEnvMultiplexer(variant_groups)
    new_batch_B, env_ctor_kwargs = env_mux.get_batch_size_and_kwargs(batch_B)
    all_env_names = sorted(variant_groups.task_variant_by_name.keys())
    if new_batch_B != batch_B:
        print(f"Increasing sampler batch size from '{batch_B}' to "
              f"'{new_batch_B}' to be divisible by number of "
              f"environments ({len(all_env_names)})")
        batch_B = new_batch_B

    # number of transitions collected during each round of sampling will be
    # batch_T * batch_B = n_steps * n_envs
    max_steps = max(
        [env_meta.spec.max_episode_steps for env_meta in env_metas])

    print("Setting up sampler")
    if use_gpu:
        sampler_ctor = MuxGpuSampler
    else:
        sampler_ctor = MuxCpuSampler
    sampler = sampler_ctor(env_mux,
                           env_ctor_kwargs,
                           max_decorrelation_steps=max_steps,
                           TrajInfoCls=MILBenchTrajInfo,
                           batch_T=batch_T,
                           batch_B=batch_B)

    return sampler, batch_B


def make_agent_policy_mt(env_metas, task_ids_and_demo_env_names):
    model_in_out_kwargs = get_policy_spec_milbench(env_metas)
    model_kwargs = {
        'env_ids_and_names': task_ids_and_demo_env_names,
        **model_in_out_kwargs,
    }
    model_ctor = MultiHeadPolicyNet
    ppo_agent = CategoricalPgAgent(
        ModelCls=MuxTaskModelWrapper,
        model_kwargs=dict(
            model_ctor=model_ctor,
            # task_id=env_id,
            model_kwargs=model_kwargs))
    return ppo_agent, model_ctor, model_kwargs


# ^^^^^^^^^^^ STUFF TO FACTOR OUT ^^^^^^^^^^^^^^^


# FIXME: is it even worth dealing with dicts? Instead should I just make
# EVERYTHING into a namedarraytuple?
class DictTensorDataset(data.Dataset):
    def __init__(self, tensor_dict):
        # need at least one tensor
        assert len(tensor_dict) > 0

        # make sure batch size is uniform
        batch_sizes = set()
        tree_map(lambda t: batch_sizes.add(t.size(0)), tensor_dict)
        assert len(batch_sizes) == 1, batch_sizes

        self.tensor_dict = tensor_dict

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensor_dict.items()}

    def __len__(self):
        first_val = next(iter(self.tensor_dict.values()))
        if isinstance(first_val, torch.Tensor):
            return first_val.size(0)
        elif isinstance(first_val, tuple):
            sizes = set()
            tree_map(lambda t: sizes.add(t.size(0)), first_val)
            assert len(sizes) == 1, sizes
            return next(iter(sizes))
        raise TypeError(f"can't handle value of type '{type(first_val)}'")


def make_tensor_dict_dataset(demo_trajs_by_env, omit_noop=False):
    """Re-format multi-task trajectories into a Torch dataset of dicts."""

    cpu_dev = torch.device("cpu")

    # make big list of trajectories in a deterministic order
    all_obs = []
    all_acts = []
    for env_name in sorted(demo_trajs_by_env.keys()):
        demo_trajs = demo_trajs_by_env[env_name]
        n_samples = 0
        for traj in demo_trajs:
            # The observation trajectories are one elem longer than the act
            # trajectories because they include terminal obs. We lop that off
            # here.
            all_obs.append(
                tree_map(lambda t: torch.as_tensor(t, device=cpu_dev)[:-1],
                         traj.obs))
            all_acts.append(torch.as_tensor(traj.acts, device=cpu_dev))
            n_samples += len(traj.acts)

        # weight things inversely proportional to their frequency
        assert n_samples > 0, demo_trajs

    # join together trajectories into Torch dataset
    all_obs = tree_map(lambda *t: torch.cat(t), *all_obs)
    all_acts = torch.cat(all_acts)

    if omit_noop:
        # omit action 0 (helps avoid the "agent does nothing initially" problem
        # for MILBench)
        valid_inds = torch.squeeze(torch.nonzero(all_acts), 1)
        all_obs = tree_map(lambda t: t[valid_inds], all_obs)
        all_acts = all_acts[valid_inds]

    dataset = DictTensorDataset({
        'obs': all_obs,
        'acts': all_acts,
    })

    return dataset


def _stem_env_name_mb(env_name):
    ename = EnvName(env_name)
    return ename.name_prefix


class GroupedVariantNames:
    """Class for storing a bunch of variant env names, grouped by task. Useful
    for, e.g., assigning consistent IDs to things."""
    def __init__(self, demo_env_names, variant_env_names):
        demo_names_uniq = sorted(set(demo_env_names))
        var_names_uniq = sorted(set(variant_env_names))
        intersect = sorted(set(demo_names_uniq) & set(var_names_uniq))
        if intersect:
            raise ValueError(
                "original names ({orig_names_uniq}) and variant names "
                "({var_names_uniq}) have overlapping entries: {intersect}")
        self.name_by_prefix = {}
        for name in demo_names_uniq:
            stem = _stem_env_name_mb(name)
            # having two envs with different stems is sign of possible bug, b/c
            # all demos with the same stem should really be coming from one
            # variant (the demo env)
            assert stem not in self.name_by_prefix, \
                f"{name} & {self.name_by_prefix[stem]} in demos have same stem"
            self.name_by_prefix[stem] = [name]
        for name in var_names_uniq:
            stem = _stem_env_name_mb(name)
            self.name_by_prefix[stem].append(name)
        # sort the dict & maintain sorted order (to preserve task identities or
        # whatever)
        self.name_by_prefix = collections.OrderedDict(
            sorted(self.name_by_prefix.items()))

        # also make some IDs
        self.task_variant_by_name = {
            env_name: (task_id, variant_id)
            for task_id, env_list in enumerate(self.name_by_prefix.values())
            for variant_id, env_name in enumerate(env_list)
        }

        # invert to map from (task id, variant id) back to env name
        self.env_name_by_task_variant = {
            value: key
            for key, value in self.task_variant_by_name.items()
        }

    @property
    def num_tasks(self):
        return len(self.name_by_prefix)

    @property
    def max_num_variants(self):
        return max(map(len, self.name_by_prefix.values()))


def add_mb_preproc(demo_trajs_by_env, variant_names, mb_preproc):
    demo_env_names = sorted(set(demo_trajs_by_env.keys()))
    variant_names = sorted(set(variant_names))
    _orig_names = demo_env_names + variant_names  # for debug prints

    # update the names of the demo envs (& keep map between new & old
    # names)
    new_demo_env_names = []
    new_env_names = {}
    for old_name in demo_env_names:
        new_name = splice_in_preproc_name(old_name, mb_preproc)
        new_demo_env_names.append(new_name)
        new_env_names[old_name] = new_name
    demo_env_names = new_demo_env_names
    del new_demo_env_names

    # update names of variant envs (don't don't need mapping for these ones,
    # but whatever)
    new_variant_names = []
    for variant_name in variant_names:
        new_variant_name = splice_in_preproc_name(variant_name, mb_preproc)
        new_variant_names.append(new_variant_name)
        new_env_names[variant_name] = new_variant_name
    variant_names = new_variant_names
    del new_variant_names

    # debug print to tell us what names changed
    _new_names = demo_env_names + variant_names
    print(f"Splicing preprocessor '{add_mb_preproc}' into environments "
          f"{_orig_names}. New names are {_new_names}")
    del _orig_names, _new_names

    # apply appropriate preprocessors
    demo_trajs_by_env = {
        new_env_names[orig_env_name]:
        preprocess_demos_with_wrapper(demo_trajs_by_env[orig_env_name],
                                      orig_env_name, mb_preproc)
        for orig_env_name, traj in demo_trajs_by_env.items()
    }

    return demo_trajs_by_env, demo_env_names, variant_names


def insert_task_ids(obs_seq, task_id, variant_id):
    nobs = len(obs_seq.obs)
    task_id_array = np.full((nobs, ), task_id, dtype='int64')
    variant_id_array = np.full((nobs, ), variant_id, dtype='int64')
    eio_arr = EnvIDObsArray(observation=obs_seq.obs,
                            task_id=task_id_array,
                            variant_id=variant_id_array)
    return obs_seq._replace(obs=eio_arr)


def load_demos_mtgail(demo_paths,
                      variant_names=(),
                      mb_preproc=None,
                      omit_noop=False):
    # load demo pickles from disk and sort into dict
    demo_dicts = load_demos(demo_paths)
    demo_trajs_by_env = {}
    for demo_dict in demo_dicts:
        demo_trajs_by_env.setdefault(demo_dict['env_name'], []) \
                         .append(demo_dict['trajectory'])

    # if necessary, update the environment names to also include a preprocessor
    # name
    if add_mb_preproc:
        demo_trajs_by_env, demo_env_names, variant_names \
            = add_mb_preproc(demo_trajs_by_env, variant_names, mb_preproc)

    # now make a grouped index of env names & add env IDs to observations
    variant_groups = GroupedVariantNames(demo_env_names, variant_names)
    for env_name in demo_trajs_by_env.keys():
        task_id, variant_id = variant_groups.task_variant_by_name[env_name]
        demo_trajs_by_env[env_name] = [
            insert_task_ids(traj, task_id, variant_id)
            for traj in demo_trajs_by_env[env_name]
        ]

    # Convert the loaded trajectories into a torch Dataset. This removes
    # temporal order.
    # TODO: add train/test split
    dataset_mt = make_tensor_dict_dataset(demo_trajs_by_env,
                                          omit_noop=omit_noop)

    return dataset_mt, variant_groups

"""Multi-task-capable tools for loading trajectories."""

import collections

# FIXME: is it even worth dealing with dicts? Instead should I just make
# EVERYTHING into a namedarraytuple?
from magical.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
from magical.benchmarks import EnvName
import numpy as np
from rlpyt.utils.collections import NamedArrayTupleSchema
import torch
from torch.utils import data

from mtil.sample_mux import EnvIDObsArray
from mtil.utils.misc import tree_map
from mtil.utils.rlpyt import get_env_metas
from mtil.utils.torch import fixed_default_collate


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
        # for MAGICAL)
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
    print(f"Splicing preprocessor '{mb_preproc}' into environments "
          f"{_orig_names}. New names are {_new_names}")
    del _orig_names, _new_names

    # apply appropriate preprocessors
    demo_trajs_by_env = {
        new_env_names[orig_env_name]:
        preprocess_demos_with_wrapper(demo_trajs_by_env[orig_env_name],
                                      orig_env_name, mb_preproc)
        for orig_env_name, traj in demo_trajs_by_env.items()
    }

    return demo_trajs_by_env, demo_env_names, variant_names, new_env_names


def insert_task_source_ids(obs_seq, task_id, variant_id, source_id):
    # figuring out the length of the sequence requires a bit of judo because
    # the observations are stored as namedarraytuples
    lens = []
    tree_map(lambda t: lens.append(len(t)), obs_seq.obs)
    nobs, = set(lens)

    task_id_array = np.full((nobs, ), task_id, dtype='int64')
    variant_id_array = np.full((nobs, ), variant_id, dtype='int64')
    source_id_array = np.full((nobs, ), source_id, dtype='int64')
    eio_arr = EnvIDObsArray(observation=obs_seq.obs,
                            task_id=task_id_array,
                            variant_id=variant_id_array,
                            source_id=source_id_array)
    return obs_seq._replace(obs=eio_arr)


def convert_obs_to_namedarraytuples(traj):
    # If using an appropriate preprocessor (e.g. LoResStack4E), MAGICAL
    # observations will be plain numpy ndarray, in which case we don't need to
    # do anything here
    if isinstance(traj.obs[0], np.ndarray):
        return traj

    # Otherwise, all MAGICAL observations are dicts with "ego" and "allo"
    # keys. We convert the observations to named(array)tuples so that it's
    # easier to work with them.
    fields = tuple(traj.obs[0].keys())
    nats = NamedArrayTupleSchema('MILBenchObs', fields)
    # for each key in the observation dict, stack values for that key along
    # time axis
    by_field = collections.defaultdict(list)
    for item in traj.obs:
        for field in fields:
            by_field[field].append(item[field])
    # now convert dict of stacked observations to a NamedArrayTuple
    return traj._replace(obs=nats._make(*(np.stack(by_field[field], axis=0)
                                          for field in fields)))


def load_demos_mt(demo_paths,
                  variant_names=(),
                  mb_preproc=None,
                  omit_noop=False):
    # load demo pickles from disk and sort into dict
    demo_dicts = load_demos(demo_paths)
    demo_trajs_by_env = {}
    source_ids_by_env = {}
    num_demo_sources = 0
    # FIXME: number the sources separately for each task
    for source_id, demo_dict in enumerate(demo_dicts, start=1):
        env_name = demo_dict['env_name']
        traj = demo_dict['trajectory']
        demo_trajs_by_env.setdefault(env_name, []).append(traj)
        source_ids_by_env.setdefault(env_name, []).append(source_id)
        num_demo_sources += 1

    # if necessary, update the environment names to also include a preprocessor
    # name (we want to apply preprocessors before breaking dict structure
    # apart)
    if mb_preproc:
        demo_trajs_by_env, demo_env_names, variant_names, new_env_names \
            = add_mb_preproc(demo_trajs_by_env, variant_names, mb_preproc)
        source_ids_by_env = {
            new_env_names[k]: v
            for k, v in source_ids_by_env.items()
        }
    else:
        demo_env_names = sorted(set(demo_trajs_by_env))

    # demo observations will be dicts containing ego and/or allo view; we want
    # to turn those into namedarraytuples
    demo_trajs_by_env = {
        key: [convert_obs_to_namedarraytuples(traj) for traj in trajs]
        for key, trajs in demo_trajs_by_env.items()
    }

    # now make a grouped index of env names & add env IDs to observations
    variant_groups = GroupedVariantNames(demo_env_names, variant_names)
    for env_name in demo_trajs_by_env.keys():
        task_id, variant_id = variant_groups.task_variant_by_name[env_name]
        trajs = demo_trajs_by_env[env_name]
        source_ids = source_ids_by_env[env_name]
        assert len(trajs) == len(source_ids)
        demo_trajs_by_env[env_name] = [
            insert_task_source_ids(traj, task_id, variant_id, source_id)
            for traj, source_id in zip(trajs, source_ids)
        ]

    # Convert the loaded trajectories into a torch Dataset. This removes
    # temporal order.
    # TODO: add train/test split
    dataset_mt = make_tensor_dict_dataset(demo_trajs_by_env,
                                          omit_noop=omit_noop)

    return dataset_mt, variant_groups, num_demo_sources


def make_loader_mt(dataset, batch_size):
    """Construct sampler that randomly chooses N items from N-sample dataset,
    weighted so that it's even across all tasks (so no task implicitly has
    higher priority than the others). Assumes the given dataset is a
    TensorDataset produced by trajectories_to_dataset_mt."""
    task_ids = dataset.tensor_dict['obs'].task_id
    assert len(task_ids) > 0 and batch_size > 0, \
        f"either {len(task_ids)}=0 task IDs or {batch_size}=0 batch size"
    unique_ids, frequencies = torch.unique(task_ids,
                                           return_counts=True,
                                           sorted=True)
    # all tasks must be present for this to work
    assert torch.all(unique_ids == torch.arange(len(unique_ids))), (unique_ids)
    freqs_total = torch.sum(frequencies).to(torch.float)
    unique_weights = freqs_total / frequencies.to(torch.float)
    unique_weights = unique_weights / unique_weights.sum()
    weights = unique_weights[task_ids]

    # even out the number of samples to be a multiple of batch size, always
    n_samples = len(weights) + (-len(weights)) % batch_size
    assert n_samples >= len(weights) and 0 == n_samples % batch_size, \
        (batch_size, n_samples)
    weighted_sampler = data.WeightedRandomSampler(weights,
                                                  n_samples,
                                                  replacement=True)
    batch_sampler = data.BatchSampler(weighted_sampler,
                                      batch_size=batch_size,
                                      drop_last=True)

    loader = data.DataLoader(dataset,
                             pin_memory=False,
                             batch_sampler=batch_sampler,
                             collate_fn=fixed_default_collate)

    return loader


def _conv_out_d(d, kernel_size=1, stride=1, padding=0, dilation=1):
    numerator = d + 2 * padding - dilation * (kernel_size - 1) - 1
    return int(np.floor(numerator / stride + 1))


def get_demos_meta(*,
                   demo_paths,
                   omit_noop=False,
                   transfer_variants=(),
                   preproc_name=None):
    dataset_mt, variant_groups, num_demo_sources = load_demos_mt(
        demo_paths,
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
        'num_demo_sources': num_demo_sources,
    }
    return rv

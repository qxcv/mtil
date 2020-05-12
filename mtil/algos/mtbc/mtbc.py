"""Multi-Task Behavioural Cloning (MTBC). Train on several environments, with
one "head" per environment. For now this only works with MILBench environments,
so it assumes that all environments have the same input & output spaces."""

import collections
import os
import re

from milbench.benchmarks import EnvName
import numpy as np
from rlpyt.utils.prog_bar import ProgBarCounter
import torch
from torch import nn
import torch.nn.functional as F

from mtil.models import FixedTaskModelWrapper
from mtil.utils.misc import load_state_dict_or_model, tree_map

LATEST_MARKER = 'LATEST'


def get_latest_path(path_template):
    abs_path = os.path.abspath(path_template)
    dir_name, base_name = os.path.split(abs_path)
    # find last occurrence
    latest_ind_rev = base_name[::-1].find(LATEST_MARKER[::-1])
    if latest_ind_rev == -1:
        raise ValueError(f"No occurrence of marker '{LATEST_MARKER}' in "
                         f"path template '{path_template}'")
    latest_start = len(base_name) - latest_ind_rev - len(LATEST_MARKER)
    latest_stop = latest_start + len(LATEST_MARKER)
    bn_prefix = base_name[:latest_start]
    bn_suffix = base_name[latest_stop:]
    best_num = None
    best_fn = None
    for entry in os.listdir(dir_name):
        if not (entry.startswith(bn_prefix) and entry.endswith(bn_suffix)):
            continue
        end_idx = len(entry) - len(bn_suffix)
        num_str = entry[latest_start:end_idx]
        try:
            num = int(num_str)
        except ValueError as ex:
            raise ValueError(
                f"Error trying to parse file name '{entry}' with template "
                f"'{path_template}': {ex.message}")
        if best_fn is None or num > best_num:
            best_fn = entry
            best_num = num
    if best_fn is None:
        raise ValueError("Couldn't find any files matching path template "
                         f"'{path_template}' in '{dir_name}'")
    return os.path.join(dir_name, best_fn)


def copy_model_into_agent_eval(model, agent, prefix='model'):
    """Update the `.agent` inside `sampler` so that it contains weights from
    `model`. Should call this before doing evaluation rollouts between epochs
    of training."""
    state_dict = model.state_dict()
    assert hasattr(agent, prefix)
    updated_state_dict = {
        prefix + '.' + key: value
        for key, value in state_dict.items()
    }
    agent.load_state_dict(updated_state_dict)
    # make sure agent is in eval mode no matter what
    agent.model.eval()


class MinBCWeightingModule(nn.Module):
    """Module for computing min-BC loss weights."""
    __constants__ = ['num_demo_sources', 'num_tasks']

    def __init__(self, num_tasks, num_demo_sources):
        super().__init__()
        self.num_demo_sources = num_demo_sources
        self.num_tasks = num_tasks
        self.weight_logits = nn.Parameter(
            # id 0 should never be used
            torch.zeros(num_tasks, num_demo_sources + 1))
        self.register_parameter('weight_logits', self.weight_logits)

    def forward(self, task_ids, variant_ids, source_ids):
        # perform a separate softmax for each task_id, then average uniformly
        # over task IDs
        # TODO: torch.jit() this once you know it works
        orig_shape = task_ids.shape
        task_ids = task_ids.flatten()
        source_ids = source_ids.flatten()
        result = torch.zeros_like(task_ids, dtype=torch.float)
        max_id = torch.max(source_ids)
        min_id = torch.min(source_ids)
        assert min_id >= 1 and max_id <= self.num_demo_sources, \
            (min_id, max_id, self.num_demo_sources)
        for task_id in task_ids.unique():
            selected_mask = task_ids == task_id
            selected_sources = source_ids[selected_mask]
            task_weights = self.weight_logits[task_id]
            source_weights = task_weights[selected_sources]
            # scaling by this target_weight ensures that each task counts
            # equally
            target_weight = selected_mask.float().mean()
            final_weights = target_weight * F.softmax(source_weights, dim=0)
            result[selected_mask] = final_weights
        return result.reshape(orig_shape)


def do_epoch_training_mt(loader, model, opt, dev, passes_per_eval, aug_model,
                         min_bc_module):
    # @torch.jit.script
    def do_loss_forward_back(obs_batch_obs, obs_batch_task, obs_batch_var,
                             obs_batch_source, acts_batch):
        # we don't use the value output
        logits_flat, _ = model(obs_batch_obs, task_ids=obs_batch_task)
        losses = F.cross_entropy(logits_flat,
                                 acts_batch.long(),
                                 reduction='none')
        if min_bc_module is not None:
            # weight using a model-dependent strategy
            mbc_weights = min_bc_module(obs_batch_task, obs_batch_var,
                                        obs_batch_source)
            assert mbc_weights.shape == losses.shape, (mbc_weights.shape,
                                                       losses.shape)
            loss = (losses * mbc_weights).sum()
        else:
            # no weighting
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
        for batches_done, loader_batch in enumerate(loader, start=1):
            # (task_ids_batch, obs_batch, acts_batch)
            # copy to GPU
            obs_batch = loader_batch['obs']
            acts_batch = loader_batch['acts']
            # reminder: attributes are .observation, .task_id, .variant_id
            obs_batch = tree_map(lambda t: t.to(dev), obs_batch)
            acts_batch = acts_batch.to(dev)

            if aug_model is not None:
                # apply augmentations
                obs_batch = obs_batch._replace(
                    observation=aug_model(obs_batch.observation))

            # compute loss & take opt step
            opt.zero_grad()
            batch_losses = do_loss_forward_back(obs_batch.observation,
                                                obs_batch.task_id,
                                                obs_batch.variant_id,
                                                obs_batch.source_id,
                                                acts_batch)
            opt.step()

            # for logging
            progress.update(batches_done + len(loader) * pass_num)
            f_loss = np.mean(batch_losses)
            loss_ewma = f_loss if loss_ewma is None \
                else 0.9 * loss_ewma + 0.1 * f_loss
            losses.append(f_loss)

            # also track separately for each task
            tv_ids = torch.stack((obs_batch.task_id, obs_batch.variant_id),
                                 axis=1)
            np_tv_ids = tv_ids.cpu().numpy()
            assert len(np_tv_ids.shape) == 2 and np_tv_ids.shape[1] == 2, \
                np_tv_ids.shape
            for tv_id in np.unique(np_tv_ids, axis=0):
                tv_mask = np.all(np_tv_ids == tv_id[None], axis=-1)
                rel_losses = batch_losses[tv_mask]
                if len(rel_losses) > 0:
                    task_id, variant_id = tv_id
                    per_task_losses[(task_id, variant_id)] \
                        .append(np.mean(rel_losses))

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


def strip_mb_preproc_name(env_name):
    """Strip any preprocessor name from a MILBench env name."""
    en = EnvName(env_name)
    return '-'.join((en.name_prefix, en.demo_test_spec, en.version_suffix))


def wrap_model_for_fixed_task(model, env_name):
    """Wrap a loaded multi-task model in a `FixedTaskModelWrapper` that _only_
    uses the weights for the given env. Useful for `test` and `testall`."""
    # contra its name, .env_ids_and_names is list of tuples of form
    # (environment name, numeric environment ID)
    env_name_to_id = dict(model.env_ids_and_names)
    if env_name not in env_name_to_id:
        env_names = ', '.join(
            [f'{name} ({eid})' for name, eid in model.env_ids_and_names])
        raise ValueError(
            f"Supplied environment name '{env_name}' is not supported by "
            f"model. Supported names (& IDs) are: {env_names}")
    env_id = env_name_to_id[env_name]
    # this returns (pi, v)
    ft_wrapper = FixedTaskModelWrapper(task_id=env_id,
                                       model_ctor=None,
                                       model_kwargs=None,
                                       model=model)
    return ft_wrapper


def eval_model(sampler_mt, itr, n_traj=10):
    # BUG: this doesn't reset the sampler envs & agent (because I don't know
    # how), so it yields somewhat inaccurate results when called repeatedly on
    # envs with different horizons.
    # BUG: if you only call this once (or if you fix the reset issue so that it
    # always resets when called) then it will be biased towards short
    # trajectories. Not an issue for fixed horizon, but will be an issue for
    # other things.
    scores_by_task_var = collections.defaultdict(lambda: [])
    while (not scores_by_task_var
           or min(map(len, scores_by_task_var.values())) < n_traj):
        samples_pyt, _ = sampler_mt.obtain_samples(itr)
        eval_scores = samples_pyt.env.env_info.eval_score
        dones = samples_pyt.env.done.flatten()
        done_scores = eval_scores.flatten()[dones]
        done_task_ids = samples_pyt.env.observation.task_id.flatten()[dones]
        done_var_ids = samples_pyt.env.observation.variant_id.flatten()[dones]
        for score, task_id, var_id in zip(done_scores, done_task_ids,
                                          done_var_ids):
            key = (task_id.item(), var_id.item())
            scores_by_task_var[key].append(score.item())
    # clip any extra rollouts
    scores_by_task_var = {k: v[:n_traj] for k, v in scores_by_task_var.items()}
    return scores_by_task_var


def eval_model_st(sampler_st, itr, n_traj=10):
    # BUGS: same as eval_model()
    scores = []
    while len(scores) < n_traj:
        samples_pyt, _ = sampler_st.obtain_samples(itr)
        eval_scores = samples_pyt.env.env_info.eval_score
        dones = samples_pyt.env.done.flatten()
        done_scores = eval_scores.flatten()[dones]
        scores.extend(s.item() for s in done_scores)
    # clip any extra rollouts
    return scores[:n_traj]


def saved_model_loader_ft(state_dict_or_model_path, env_name):
    """Loads a saved policy model and then wraps it in a
    FixedTaskModelWrapper."""
    model = load_state_dict_or_model(state_dict_or_model_path)
    ft_wrapper = wrap_model_for_fixed_task(model, env_name)
    return ft_wrapper

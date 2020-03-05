"""Multi-Task Behavioural Cloning (MTBC). Train on several environments, with
one "head" per environment. For now this only works with MILBench environments,
so it assumes that all environments have the same input & output spaces."""

import collections
import os
import re

import numpy as np
from rlpyt.utils.prog_bar import ProgBarCounter
import torch
import torch.nn.functional as F

from mtil.common import FixedTaskModelWrapper


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
    # make sure sampler is in eval mode no matter what
    sampler.agent.model.eval()


def do_epoch_training_mt(loader, model, opt, dev, passes_per_eval):
    # @torch.jit.script
    def do_loss_forward_back(task_ids_batch, obs_batch, acts_batch):
        # we don't use the value output
        logits_flat, _ = model(obs_batch, task_ids_batch)
        losses = F.cross_entropy(logits_flat,
                                 acts_batch.long(),
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
            batch_losses = do_loss_forward_back(task_ids_batch, obs_batch,
                                                acts_batch)
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


def load_state_dict_or_model(state_dict_or_model_path):
    """Load a model from a path to either a state dict or a full PyTorch model.
    If it's just a state dict path then the corresponding model path will be
    inferred (assuming the state dict was produced by `train`). This is useful
    for the `test` and `testall` commands."""
    state_dict_or_model_path = os.path.abspath(state_dict_or_model_path)
    cpu_dev = torch.device('cpu')
    state_dict_or_model = torch.load(state_dict_or_model_path,
                                     map_location=cpu_dev)
    if not isinstance(state_dict_or_model, dict):
        print(f"Treating supplied path '{state_dict_or_model_path}' as "
              f"policy (type {type(state_dict_or_model)})")
        model = state_dict_or_model
    else:
        state_dict = state_dict_or_model['model_state']
        state_dict_dir = os.path.dirname(state_dict_or_model_path)
        # we save full model once at beginning of training so that we have
        # architecture saved; not sure how to support loading arbitrary models
        fm_path = os.path.join(state_dict_dir, 'full_model.pt')
        print(f"Treating supplied path '{state_dict_or_model_path}' as "
              f"state dict to insert into model at '{fm_path}'")
        model = torch.load(fm_path, map_location=cpu_dev)
        model.load_state_dict(state_dict)

    return model


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


def eval_model(sampler, itr, n_traj=10):
    scores = []
    while len(scores) < n_traj:
        # can't see an obvious purpose to the 'itr' argument, so setting it to
        # None
        samples_pyt, _ = sampler.obtain_samples(itr)
        eval_scores = samples_pyt.env.env_info.eval_score
        dones = samples_pyt.env.done
        done_scores = eval_scores.flatten()[dones.flatten()]
        scores.extend(done_scores)
    return scores[:n_traj]

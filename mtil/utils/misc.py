"""Common tools for all of mtil package."""

from collections import OrderedDict
import datetime
import os
import random
import sys
import uuid

import click
import numpy as np
import torch


def set_seeds(seed):
    """Set all relevant PRNG seeds."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_unique_run_name(algo, orig_env_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    unique_suff = uuid.uuid4().hex[-6:]
    return f"{algo}-{orig_env_name}-{timestamp}-{unique_suff}"


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


def tree_map(f, *structures):
    """Map a function `f` over some sequence of complicated data structures.
    They could be dicts, lists, namedarratuples, etc. etc."""
    s0 = structures[0]
    if hasattr(s0, '_fields'):
        # namedtuple, NamedTuple, namedarraytuple etc.
        return s0._make(tree_map(f, *zs) for zs in zip(*structures))
    elif isinstance(s0, (list, tuple)):
        return type(s0)(tree_map(f, *zs) for zs in zip(*structures))
    elif isinstance(s0, (dict, OrderedDict)):
        return type(s0)(
            (k, tree_map(f, *(s[k] for s in structures))) for k in s0.keys())
    return f(*structures)


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

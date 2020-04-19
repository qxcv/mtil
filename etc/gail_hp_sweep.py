#!/usr/bin/env python3

import collections
import glob
import os
import subprocess
import weakref

import click
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.skopt import SkOptSearch
from skopt.optimizer import Optimizer
from skopt.space import space as opt_space

DEMO_PATTERN \
    = '~/repos/milbench/demos-simplified/match-regions-2020-03-01/*.pkl.gz'


def get_demo_paths():
    glob_pattern = os.path.expanduser(DEMO_PATTERN)
    demos = glob.glob(glob_pattern)
    assert len(demos) >= 1, glob_pattern
    return demos


def get_stats():
    pattern = os.path.join(os.getcwd(), 'scratch/run_*/progress.csv')
    prog_paths = glob.glob(pattern)
    assert len(prog_paths) == 1, (len(prog_paths), pattern)
    prog_path, = prog_paths
    data = pd.read_csv(prog_path)
    # just return all stats averaged over last 10 time steps
    last_ten = data.iloc[-10:]
    assert len(last_ten) >= 1, last_ten
    stats = last_ten.mean().to_dict()
    return stats


def run_gail(gpu_idx, **cfg_kwargs):
    auto_args = []
    for k, v in sorted(cfg_kwargs.items()):
        dashed = k.replace('_', '-')
        if v is True or v is False:
            # boolean args: set by using "--foo-bar" or "--no-foo-bar"
            if v:
                auto_args.append('--' + dashed)
            else:
                auto_args.append('--no-' + dashed)
        else:
            # other args: pass stringified argument explicitly
            auto_args.append('--' + dashed)
            auto_args.append(str(v))
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        # use --gpu-idx=0 b/c we don't know what device we're on
        assert cvd == str(gpu_idx), (gpu_idx, cvd)
        gpu_idx = 0
    cmd = [
        *'xvfb-run -a python -m mtil.algos.mtgail'.split(),
        *f'--gpu-idx {gpu_idx} --snapshot-gap 1000'.split(),
        *f'--total-n-steps {int(1e6)}'.split(),
        *auto_args,
        *get_demo_paths(),
    ]
    subprocess.run(cmd, check=True)
    return get_stats()


def ray_tune_trial(conf, reporter):
    # Ray chooses weird way of assigning GPUs
    cuda_vis_devs = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert cuda_vis_devs is not None
    dev_nums = list(map(int, cuda_vis_devs.split(',')))
    assert len(dev_nums) == 1, (dev_nums, cuda_vis_devs)
    gpu_idx, = dev_nums
    stats = run_gail(**conf, gpu_idx=gpu_idx)
    # just look for a high average (could also use ScoreStd, but I doubt it
    # will make a difference)
    hp_score = stats['ScoreAverage']

    reporter(hp_score=hp_score, **stats)


class CheckpointFIFOScheduler(FIFOScheduler):
    """Variant of FIFOScheduler that periodically saves the given search
    algorithm. Useful for, e.g., SkOptSearch, where it is helpful to be able to
    re-instantiate the search object later on."""
    # FIXME: this is a stupid hack. There should be a better way of saving
    # skopt internals as part of Ray Tune. Perhaps defining a custom trainable
    # would do the trick?
    def __init__(self, search_alg):
        self.search_alg = weakref.proxy(search_alg)

    def on_trial_complete(self, trial_runner, trial, result):
        rv = super().on_trial_complete(trial_runner, trial, result)
        # references to _local_checkpoint_dir and _session_dir are a bit hacky
        checkpoint_path = os.path.join(
            trial_runner._local_checkpoint_dir,
            f'search-alg-{trial_runner._session_str}.pkl')
        self.search_alg.save(checkpoint_path + '.tmp')
        os.rename(checkpoint_path + '.tmp', checkpoint_path)
        return rv


@click.command()
@click.option("--ray-address",
              default=None,
              help="address of Ray instance to attach to")
def run_ray_tune(ray_address):
    sk_space = collections.OrderedDict()
    sk_space['omit_noop'] = [True, False]  # ???
    sk_space['disc_up_per_iter'] = (8, 32)  # small values don't work
    sk_space['sampler_time_steps'] = (8, 64)  # small is okay?
    sk_space['sampler_batch_envs'] = (16, 64)  # bigger = better?
    # leaving this out for now; don't want to just get a finely-tuned BC
    # baseline
    # sk_space['bc_loss'] = ['0.0', str(int(1e-3)), str(1)]
    sk_space['ppo_lr'] = (5e-5, 5e-4, 'log-uniform')
    sk_space['ppo_gamma'] = (0.9, 1.0, 'log-uniform')
    sk_space['ppo_lambda'] = (0.9, 1.0, 'log-uniform')
    sk_space['ppo_ent'] = (1e-6, 1e-4, 'log-uniform')
    sk_space['ppo_adv_clip'] = (0.01, 0.2, 'uniform')
    # things that don't matter that much:
    # sk_space['disc_lr'] = (1e-5, 5e-4, 'log-uniform')  # fix to 1e-4
    # sk_space['disc_use_act'] = [True, False]  # fix to True
    # sk_space['disc_all_frames'] = [True, False]  # fix to True
    # sk_space['disc_replay_mult'] = opt_space.Integer(1, 32, 'log-uniform')  # fix to 4 # noqa: E501
    # sk_space['ppo_norm_adv'] = [True, False]  # fix to False
    known_working = {
        'omit_noop': True,
        'disc_up_per_iter': 16,
        'sampler_time_steps': 16,
        'sampler_batch_envs': 32,
        'bc_loss': 0.0,
        'ppo_lr': 2.5e-4,
        'ppo_gamma': 0.95,
        'ppo_lambda': 0.95,
        'ppo_ent': 1e-5,
        'ppo_adv_clip': 0.05,
        # things that don't matter much:
        # 'disc_lr': 1e-4,
        # 'disc_use_act': True,
        # 'disc_all_frames': True,
        # 'disc_replay_mult': 4,
        # 'ppo_norm_adv': False,
    }
    for k, v in list(sk_space.items()):
        new_v = opt_space.check_dimension(v)
        new_v.name = k
        sk_space[k] = new_v
    sk_optimiser = Optimizer(list(sk_space.values()), base_estimator='GP')
    search_alg = SkOptSearch(
        sk_optimiser,
        sk_space.keys(),
        max_concurrent=4,
        metric='hp_score',
        mode='max',
        points_to_evaluate=[[known_working[k] for k in sk_space]],
    )

    if ray_address:
        ray.init(redis_address=ray_address)
    tune.run(
        ray_tune_trial,
        search_alg=search_alg,
        local_dir='ray-tune-results',
        resources_per_trial={"gpu": 1},
        # this could be 2 days to a week of runs, depending on the env
        num_samples=200,
        scheduler=CheckpointFIFOScheduler(search_alg))


if __name__ == '__main__':
    run_ray_tune()

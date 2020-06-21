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

# FIXME: don't make this relative to /home, but rather read it from cmdline or
# (better yet) a config file
DEMO_PATTERN \
    = '~/repos/magical/demos-ea/cluster-shape-2020-05-18/demo-*.pkl.gz'


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
        *'python -m mtil.algos.mtgail'.split(),
        *f'--gpu-idx {gpu_idx} --snapshot-gap 1000'.split(),
        # we're using small number of steps so we can get shoot for MAX
        # EFFICIENCY (like RAD)
        *f'--total-n-steps {int(0.3e6)}'.split(),
        *f'--transfer-variant MoveToRegion-TestAll-v0'.split(),
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
    stat_values = []
    for stat_key, stat_value in stats.items():
        # if there are several variants, average all their scores
        if stat_key.startswith('Score') and stat_key.endswith('Average'):
            stat_values.append(stat_value)

    assert len(stat_values) > 0, stats
    hp_score = sum(stat_values) / len(stat_values)

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

    sk_space['disc_up_per_iter'] = (2, 12)  # small values don't work
    sk_space['sampler_time_steps'] = (8, 20)  # small is okay?
    sk_space['sampler_batch_envs'] = (8, 24)  # bigger = better?
    sk_space['ppo_lr'] = (1e-6, 1e-3, 'log-uniform')
    sk_space['ppo_gamma'] = (0.9, 1.0, 'log-uniform')
    sk_space['ppo_lambda'] = (0.9, 1.0, 'log-uniform')
    sk_space['ppo_ent'] = (1e-6, 1e-4, 'log-uniform')
    sk_space['ppo_adv_clip'] = (0.05, 0.2, 'uniform')
    sk_space['add_preproc'] = ['LoRes4E', 'LoRes3EA']
    # allow us to have smaller batches or run more of them
    sk_space['ppo_minibatches'] = [4, 6]
    sk_space['ppo_epochs'] = [2, 10]
    sk_space['ppo_use_bn'] = [True, False]
    sk_space['ppo_aug'] = ['none', 'all', 'crop']

    # things I'm commenting out for simplicity:
    # sk_space['bc_loss'] = ['0.0', str(int(1e-3)), str(1)]
    # sk_space['ppo_use_bn'] = [True, False]

    # things that don't matter that much:
    # sk_space['omit_noop'] = [True, False]  # ???
    # sk_space['disc_lr'] = (1e-5, 5e-4, 'log-uniform')  # fix to 1e-4
    # sk_space['disc_use_act'] = [True, False]  # fix to True
    # sk_space['disc_all_frames'] = [True, False]  # fix to True
    # sk_space['disc_replay_mult'] = opt_space.Integer(1, 32, 'log-uniform')  # fix to 4 # noqa: E501
    # sk_space['ppo_norm_adv'] = [True, False]  # fix to False

    known_working = {
        'disc_up_per_iter': [4, 2],
        'sampler_time_steps': [16, 16],
        'sampler_batch_envs': [32, 12],
        # 'bc_loss': [0.0, 0.0],
        'ppo_lr': [2.5e-4, 2e-4],
        'ppo_adv_clip': [0.05, 0.1],
        'ppo_minibatches': [4, 5],
        'ppo_epochs': [4, 6],
        'ppo_use_bn': [False, False],
        'ppo_aug': ['none', 'none'],
        'ppo_gamma': [0.95, 0.9],
        'ppo_lambda': [0.95, 0.9],
        'ppo_ent': [1e-5, 1.2e-5],
        'add_preproc': ['LoRes4E', 'LoRes4E']
        # things that I'm removing because they'll take too much time
        # 'omit_noop': [True],
        # things that don't matter much:
        # 'disc_lr': [1e-4],
        # 'disc_use_act': [True],
        # 'disc_all_frames': [True],
        # 'disc_replay_mult': [4],
        # 'ppo_norm_adv': [False],
    }
    for k, v in list(sk_space.items()):
        new_v = opt_space.check_dimension(v)
        new_v.name = k
        sk_space[k] = new_v
    sk_optimiser = Optimizer(list(sk_space.values()), base_estimator='GP')
    n_known_working, = set(map(len, known_working.values()))
    search_alg = SkOptSearch(
        sk_optimiser,
        sk_space.keys(),
        max_concurrent=8,  # XXX figure out how to make this configurable
        metric='hp_score',
        mode='max',
        points_to_evaluate=[
            [known_working[k][i] for k in sk_space]
            for i in range(n_known_working)
        ])

    if ray_address:
        ray.init(redis_address=ray_address)
    tune.run(
        ray_tune_trial,
        search_alg=search_alg,
        local_dir='ray-tune-results',
        resources_per_trial={"gpu": 0.24},
        # this could be 2 days to a week of runs, depending on the env
        num_samples=200,
        scheduler=CheckpointFIFOScheduler(search_alg))


if __name__ == '__main__':
    run_ray_tune()

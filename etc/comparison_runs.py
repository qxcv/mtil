#!/usr/bin/env python3
"""Main script for doing experiments that compare different methods."""
import collections
import datetime
import glob
import multiprocessing as mp
import os
import subprocess

import click
from milbench.benchmarks import EnvName
import numpy as np
import ray
import yaml

DEMO_PATH_PATTERNS = {
    'move-to-corner': '~/repos/milbench/demos-simplified/move-to-corner-2020-03-*/*.pkl.gz',  # noqa: E501
    'move-to-region': '~/repos/milbench/demos-simplified/move-to-region-2020-04-*/*.pkl.gz',  # noqa: E501
    'match-regions': '~/repos/milbench/demos-simplified/match-regions-2020-03-*/*.pkl.gz',  # noqa: E501
    'find-dupe': '~/repos/milbench/demos-simplified/find-dupe-2020-04-*/*.pkl.gz',  # noqa: E501
    'cluster-colour': '~/repos/milbench/demos-simplified/cluster-colour-2020-03-*/*.pkl.gz',  # noqa: E501
    'cluster-type': '~/repos/milbench/demos-simplified/cluster-type-2020-03-*/*.pkl.gz',  # noqa: E501
}  # yapf: disable
ENV_NAMES = {
    'move-to-corner': 'MoveToCorner-Demo-LoResStack-v0',
    'move-to-region': 'MoveToRegion-Demo-LoResStack-v0',
    'match-regions': 'MatchRegions-Demo-LoResStack-v0',
    'find-dupe': 'FindDupe-Demo-LoResStack-v0',
    'cluster-colour': 'ClusterColour-Demo-LoResStack-v0',
    'cluster-type': 'ClusterType-Demo-LoResStack-v0',
}
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_GPUS = 4
# for the final paper I'm just going to spend a few days running like 10 seeds
# or something to get not-insane error bars
DEFAULT_NUM_SEEDS = 3
BASE_START_SEED = 3255779925


def insert_variant(env_name, variant):
    """Insert a variant name into an environment name. For instance,
    `insert_variant("MoveToCorner-Demo-LoResStack-v0", "TestAll")` yields
    `"MoveToCorner-TestAll-LoResStack-v0"`."""
    parsed = EnvName(env_name)
    new_name = (parsed.name_prefix + '-' + variant + parsed.env_name_suffix +
                parsed.version_suffix)
    return new_name


def expand_patterns(*patterns):
    result = []
    for pattern in patterns:
        user_exp = os.path.expanduser(pattern)
        glob_results = glob.glob(user_exp)
        if len(glob_results) == 0:
            raise ValueError(
                f"supplied pattern '{pattern}' yielded 0 results! (expansion: "
                f"'{user_exp}'")
        result.extend(glob_results)
    return result


def parse_unknown_args(**unk_args):
    """Parse kwargs like `dict(foo=False, bar=42)` into parameters like
    `--no-foo --bar 42`."""
    parsed = []
    for key, value in unk_args.items():
        base_name = key.replace('_', '-')
        if isinstance(value, bool):
            # boolean values don't follow the normal "--key value" format; have
            # to handle them like "--key" or "--no-key" instead.
            if value:
                parsed.append('--' + base_name)
            else:
                parsed.append('--no-' + base_name)
        else:
            # otherwise we assume normal "--key value" format.
            parsed.extend(['--' + base_name, str(value)])
    return parsed


def generate_seeds(nseeds, start_seed=BASE_START_SEED):
    # generate non-sequential seeds via repeated calls to a PRNG
    rng = np.random.RandomState(start_seed)
    seeds = rng.randint(0, 1 << 31, size=nseeds)
    return seeds.tolist()


def sample_core_range(num_cores):
    """Sample a contiguous block of random CPU cores.

    (why contiguous? because default Linux/Intel core enumeration puts distinct
    physical cores on the same socket next to each other)"""
    ncpus = mp.cpu_count()
    assert (num_cores % ncpus) == 0, \
        f"number of CPUs ({ncpus}) is not divisible by number of cores " \
        f"({num_cores}) for this job"
    ngroups = num_cores // ncpus
    group = np.random.randint(ngroups)
    cores = list(range(group * num_cores, (group + 1) * num_cores))
    return cores


# Signature for algorithms:
# - Takes in a list of demo paths, a run name, a seed, and a bunch of kwargs
#   specific to that algorithm.
# - Returns a list of command parts for `subprocess` (or `shlex` or whatever),
#   and a path to a scratch dir that should contain a file matching the
#   "itr_LATEST.pkl" used by mtbc.py.


def gen_command_gail(*,
                     demo_paths,
                     run_name,
                     out_root,
                     seed,
                     n_steps=None,
                     log_interval_steps=1e4,
                     trans_env_names=(),
                     **kwargs):
    extras = parse_unknown_args(**kwargs)
    cmd_parts = [
        *("xvfb-run -a python -m mtil.algos.mtgail").split(),
        "--out-dir",
        out_root,
        "--run-name",
        str(run_name),
        "--seed",
        str(seed),
        "--gpu-idx",
        "0",
        "--log-interval-steps",
        str(log_interval_steps),
        *extras,
    ]
    if n_steps is not None:
        assert n_steps == int(n_steps), n_steps
        cmd_parts.extend(['--total-n-steps', str(int(n_steps))])
    for trans_variant in trans_env_names:
        cmd_parts.extend(('--transfer-variant', trans_variant))
    cmd_parts.extend(demo_paths)
    out_dir = os.path.join(out_root, f'run_{run_name}')
    return cmd_parts, out_dir


def gen_command_bc(*,
                   demo_paths,
                   run_name,
                   out_root,
                   seed,
                   eval_n_traj=5,
                   snapshot_gap=1,
                   trans_env_names=(),
                   **kwargs):
    extras = parse_unknown_args(**kwargs)
    assert len(trans_env_names) == 0, \
        f"transfer envs not yet supported for BC, but got transfer env " \
        f"names '{trans_env_names}'"
    cmd_parts = [
        "xvfb-run",
        "-a",
        "python",
        "-m",
        "mtil.algos.mtbc",
        "train",
        "--out-dir",
        out_root,
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--gpu-idx",
        "0",
        "--eval-n-traj",
        str(eval_n_traj),
        "--snapshot-gap",
        str(snapshot_gap),
        *extras,
    ]
    cmd_parts.extend(demo_paths)
    out_dir = os.path.join(out_root, f'run_{run_name}')
    return cmd_parts, out_dir


def make_eval_cmd(run_name, snap_dir, env_shorthand, env_name):
    new_cmd = [
        "xvfb-run",
        "-a",
        "python",
        "-m",
        "mtil.algos.mtbc",
        "testall",
        "--env-name",
        env_name,
        "--run-id",
        f"{run_name}-on-{env_shorthand}",
        "--write-latex",
        f"{snap_dir}/eval-{env_shorthand}.tex",
        "--write-csv",
        f"{snap_dir}/eval-{env_shorthand}.csv",
        "--n-rollouts",
        "100",
        "--gpu-idx",
        "0",
        "--load-latest",
        os.path.join(snap_dir, 'itr_LATEST.pkl'),
    ]
    return new_cmd


def parse_expts_file(file_path):
    # parse experiment spec file, validating except the method-specific
    # arguments
    with open(file_path, 'r') as fp:
        data = yaml.safe_load(fp)
    assert isinstance(data, dict)
    run_specs = []
    for run_name, run_dict in data.items():
        assert isinstance(run_dict, dict)
        algo = run_dict.pop('algo')
        generator = globals()['gen_command_' + algo]

        is_multi_task = run_dict.pop('multi', False)
        assert isinstance(is_multi_task, bool)

        # these go straight through to the algorithm and will be validated
        # later
        args = run_dict.pop('args', {})
        assert isinstance(args, dict)

        nseeds = run_dict.pop('nseeds', DEFAULT_NUM_SEEDS)
        assert isinstance(nseeds, int) and nseeds >= 1

        trans_variants = run_dict.pop('transfer-variants', [])
        assert isinstance(trans_variants, list)

        if len(run_dict) > 0:
            raise ValueError(
                f"unrecognised options in expt spec '{run_name}': "
                f"{sorted(run_dict.keys())}")

        run_specs.append({
            'run_name': run_name,
            'algo': algo,
            'generator': generator,
            'is_multi_task': is_multi_task,
            'trans_variants': trans_variants,
            'args': args,
            'nseeds': nseeds,
        })

    return run_specs


Run = collections.namedtuple('Run', ('train_cmd', 'test_cmds'))


def generate_runs(*, run_name, algo, generator, is_multi_task, args, nseeds,
                  suffix, out_dir, trans_variants):
    runs = []
    seeds = generate_seeds(nseeds)
    for seed in seeds:
        if is_multi_task:
            # one train command corresponding to N test commands
            mt_run_name = f"{run_name}-{suffix}-s{seed}"
            demo_paths = sum([
                glob.glob(os.path.expanduser(DEMO_PATH_PATTERNS[expt]))
                for expt in sorted(DEMO_PATH_PATTERNS.keys())
            ], [])

            mt_trans_variants = [
                insert_variant(ENV_NAMES[expt], variant)
                for variant in trans_variants
                for expt in sorted(DEMO_PATH_PATTERNS.keys())
            ]
            mt_cmd, mt_dir = generator(demo_paths=demo_paths,
                                       run_name=mt_run_name,
                                       seed=seed,
                                       out_root=out_dir,
                                       trans_env_names=mt_trans_variants,
                                       **args)

            eval_cmds = []
            for task_shorthand, env_name in sorted(ENV_NAMES.items()):
                mt_eval_cmd = make_eval_cmd(run_name, mt_dir, task_shorthand,
                                            env_name)
                eval_cmds.append(mt_eval_cmd)
            runs.append(Run(mt_cmd, eval_cmds))

        else:
            # one test command for each of N train commands
            for task in sorted(DEMO_PATH_PATTERNS.keys()):
                st_run_name = f"{run_name}-{task}-{suffix}-s{seed}"
                demo_paths = glob.glob(
                    os.path.expanduser(DEMO_PATH_PATTERNS[task]))
                st_trans_variants = [
                    insert_variant(ENV_NAMES[task], variant)
                    for variant in trans_variants
                ]

                st_cmd, st_dir = generator(demo_paths=demo_paths,
                                           run_name=st_run_name,
                                           seed=seed,
                                           out_root=out_dir,
                                           trans_env_names=st_trans_variants,
                                           **args)

                st_eval_cmd = make_eval_cmd(run_name, st_dir, task,
                                            ENV_NAMES[task])
                runs.append(Run(st_cmd, [st_eval_cmd]))

    return runs


def run_check(*args):
    """Like subprocess.run, but passes check=True. Need a separate function for
    this because ray.remote doesn't allow remote calls to take keyword
    arguments :("""
    return subprocess.run(*args, check=True)


class RunDummy:
    def remote(self, *args, **kwargs):
        """Simply prints command."""
        sargs = ", ".join(map(repr, args))
        if kwargs:
            values = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            if args:
                skwargs = ", " + values
            else:
                skwargs = values
        else:
            skwargs = ""
        print(f"subprocess.run({sargs}{skwargs})")
        # object() can be hashed and compared for equality, so it acts like a
        # Ray remote call handle when we put it in a dict
        return object()


@click.command()
@click.option("--suffix",
              default=None,
              help="extra suffix to add to runs (defaults to timestamp)")
@click.option("--out-dir", default="scratch")
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="only print experiment commands, without starting experiments")
@click.option(
    '--ray-connect',
    default=None,
    help='connect Ray to this Redis DB instead of starting new cluster')
@click.option('--ray-ncpus',
              default=None,
              help='number of CPUs to use if starting new Ray instance')
@click.option('--job-ngpus',
              default=0.3,
              help='number of GPUs per train job (can be fractional)')
@click.option('--job-ngpus-eval',
              default=0.45,
              help='number of GPUs per eval job (can be fractional)')
# @click.option('--job-ncpus',
#               default=8,
#               help='number of CPU cores to use for sampler in each job')
@click.argument("spec")
def main(spec, suffix, out_dir, ray_connect, ray_ncpus, job_ngpus,
         job_ngpus_eval, dry_run):
    """Execute some experiments with Ray."""
    # spin up Ray cluster
    new_cluster = ray_connect is None
    ray_kwargs = {}
    if not new_cluster:
        ray_kwargs["redis_address"] = ray_connect
        assert ray_ncpus is None, \
            "can't provide --ray-ncpus and --ray-connect"
    else:
        if ray_ncpus is not None:
            ray_kwargs["num_cpus"] = ray_ncpus
    if not dry_run:
        ray.init(**ray_kwargs)

    if suffix is None:
        # add 'T%H:%M' for hours and minutes
        suffix = datetime.datetime.now().strftime('%Y-%m-%d')

    expts_specs = parse_expts_file(spec)

    all_runs = []
    for expt_spec in expts_specs:
        # we have to run train_cmds first, then test_cmds second
        new_runs = generate_runs(**expt_spec, suffix=suffix, out_dir=out_dir)
        all_runs.extend(new_runs)
    if dry_run:
        call_remote = RunDummy()
        call_remote_eval = RunDummy()
    else:
        call_remote = ray.remote(num_gpus=job_ngpus)(run_check)
        call_remote_eval = ray.remote(num_gpus=job_ngpus_eval)(run_check)

    # first launch all train CMDs
    running_train_cmds = collections.OrderedDict()
    for run in all_runs:
        wait_handle = call_remote.remote(run.train_cmd)
        running_train_cmds[wait_handle] = run

    running_test_cmds = collections.OrderedDict()
    while running_train_cmds or running_test_cmds:
        if dry_run:
            finished = list(running_train_cmds.keys())
        else:
            finished, _ = ray.wait(list(running_train_cmds.keys()),
                                   timeout=1.0)

        for f_handle in finished:
            run = running_train_cmds[f_handle]
            del running_train_cmds[f_handle]
            try:
                if not dry_run:
                    ret_value = ray.get(f_handle)
                    print(f"Run {run} returned value {ret_value.returncode}")
            except Exception as ex:
                print(f"Got exception while popping run {run}: {ex}")
                continue

            for test_cmd in run.test_cmds:
                test_handle = call_remote_eval.remote(test_cmd)
                running_test_cmds[test_handle] = run

        if dry_run:
            done_test_cmds = list(running_test_cmds.keys())
        else:
            done_test_cmds, _ = ray.wait(list(running_test_cmds.keys()),
                                         timeout=1.0)
        for d_handle in done_test_cmds:
            run = running_test_cmds[d_handle]
            del running_test_cmds[d_handle]
            try:
                if not dry_run:
                    ret_value = ray.get(d_handle)
                    print(f"Test command on run {run} returned value "
                          f"{ret_value.returncode}")
            except Exception as ex:
                print(f"Got exception from test cmd for run {run}: {ex}")
                continue

    print("Everything finished!")

    # Q: what is the end result meant to be? What output do we want this to
    # produce? It would be ideal if it produced a *huge* CSV with all the
    # results it could possibly gather. After that I can break this code up
    # into several scripts that can be run in sequence, so that I can still
    # re-run later stages (in case of failure of modifications) while caching
    # early-stage results.


if __name__ == '__main__':
    main()

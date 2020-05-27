#!/usr/bin/env python3
"""Main script for doing experiments that compare different methods."""
import collections
import datetime
import glob
import multiprocessing as mp
import os
import re
import subprocess

import click
import numpy as np
import ray
import yaml

DEMO_PATH_PATTERNS = {
    'move-to-corner': '~/repos/milbench/demos-simplified/move-to-corner-2020-03-*/*.pkl.gz',  # noqa: E501
    'move-to-region': '~/repos/milbench/demos-simplified/move-to-region-2020-04-*/*.pkl.gz',  # noqa: E501
    'match-regions': '~/repos/milbench/demos-simplified/match-regions-2020-05-*/*.pkl.gz',  # noqa: E501
    'find-dupe': '~/repos/milbench/demos-simplified/find-dupe-2020-05-*/*.pkl.gz',  # noqa: E501
    'cluster-colour': '~/repos/milbench/demos-simplified/cluster-colour-2020-05-*/*.pkl.gz',  # noqa: E501
    'cluster-shape': '~/repos/milbench/demos-simplified/cluster-shape-2020-05-*/*.pkl.gz',  # noqa: E501
    'fix-colour': '~/repos/milbench/demos-simplified/fix-colour-2020-05-*/*.pkl.gz',  # noqa: E501
    'make-line': '~/repos/milbench/demos-simplified/make-line-2020-05-*/*.pkl.gz',  # noqa: E501
}  # yapf: disable
ENV_NAMES = {
    'move-to-corner': 'MoveToCorner-Demo-LoResStack-v0',
    'move-to-region': 'MoveToRegion-Demo-LoResStack-v0',
    'match-regions': 'MatchRegions-Demo-LoResStack-v0',
    'find-dupe': 'FindDupe-Demo-LoResStack-v0',
    'cluster-colour': 'ClusterColour-Demo-LoResStack-v0',
    'cluster-shape': 'ClusterShape-Demo-LoResStack-v0',
    'fix-colour': 'FixColour-Demo-LoResStack-v0',
    'make-line': 'MakeLine-Demo-LoResStack-v0',
}
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_GPUS = 4
DEFAULT_NUM_SEEDS = 5
BASE_START_SEED = 3255779925
# default number of trajectories for training; can go up to 25
DEFAULT_NUM_TRAJ = 10

# FIXME: I had to vendor EnvName and _ENV_NAME_RE from MILBench because
# directly importing from MILBench loads Pyglet, and Pyglet tries to make an
# XOrg window on import (?!), which breaks on the server without xvfb-run :(
_ENV_NAME_RE = re.compile(
    r'^(?P<name_prefix>[^-]+)(?P<demo_test_spec>-(Demo|Test[^-]*))'
    r'(?P<env_name_suffix>(-[^-]+)*)(?P<version_suffix>-v\d+)$')


class EnvName:
    """Vendored version of milbench.benchmarks.EnvName."""
    def __init__(self, env_name):
        match = _ENV_NAME_RE.match(env_name)
        if match is None:
            raise ValueError(
                "env name '{env_name}' does not match _ENV_NAME_RE spec")
        groups = match.groupdict()
        self.env_name = env_name
        self.name_prefix = groups['name_prefix']
        self.demo_test_spec = groups['demo_test_spec']
        self.env_name_suffix = groups['env_name_suffix']
        self.version_suffix = groups['version_suffix']
        self.demo_env_name = self.name_prefix + '-Demo' \
            + self.env_name_suffix + self.version_suffix
        self.is_test = self.demo_test_spec.startswith('-Test')
        if not self.is_test:
            assert self.demo_env_name == self.env_name, \
                (self.demo_env_name, self.env_name)


def insert_variant(env_name, variant):
    """Insert a variant name into an environment name, omitting suffix. For
    instance, `insert_variant("MoveToCorner-Demo-LoResStack-v0", "TestAll")`
    yields `"MoveToCorner-TestAll-v0"`."""
    parsed = EnvName(env_name)
    new_name = (parsed.name_prefix + '-' + variant + parsed.version_suffix)
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


def generate_core_list(block_size):
    """Sample a contiguous block of random CPU cores, and return core numbers
    as a comma-separated string (for --cpu-list).

    (why contiguous? because default Linux/Intel core enumeration puts distinct
    physical cores on the same socket next to each other)"""
    ncpus = mp.cpu_count()
    assert (ncpus % block_size) == 0, \
        f"number of CPUs ({ncpus}) is not divisible by number of cores per " \
        f"job ({block_size})"
    ngroups = ncpus // block_size
    group = np.random.randint(ngroups)
    core_iter = range(group * block_size, (group + 1) * block_size)
    core_str = ",".join(map(str, core_iter))
    return core_str


# def make_tee_magic(out_root, run_name):
#     full_out_dir = os.path.join(out_root, run_name)
#     os.makedirs(full_out_dir)
#     stdout_path = os.path.join(full_out_dir, 'stdout.log')
#     stderr_path = os.path.join(full_out_dir, 'stderr.log')
#     # bash witchcraft from
#     # https://stackoverflow.com/questions/692000/how-do-i-write-stderr-to-a-file-while-using-tee-with-a-pipe
#     magic = [
#         '>',
#         '>(',
#         'tee',
#         '-a',
#         stdout_path,
#         ')',
#         '2>',
#         '>(tee',
#         '-a',
#         stderr_path,
#         '>&2)',
#     ]
#     return magic


def rand_assign_cores(n_cores):
    """Randomly assign a block of CPU cores."""
    n_blocks = os.cpu_count() // n_cores
    block_num = os.random.randint(n_blocks)
    return np.arange(block_num * n_cores, (block_num + 1) * n_cores).tolist()


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
                     cpu_list,
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
        "--cpu-list",
        cpu_list,
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
                   cpu_list,
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
        "--cpu-list",
        cpu_list,
        *extras,
    ]
    cmd_parts.extend(demo_paths)
    out_dir = os.path.join(out_root, f'run_{run_name}')
    return cmd_parts, out_dir


def make_eval_cmd(run_name, snap_dir, env_shorthand, env_name, *, cpu_list):
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
        "--cpu-list",
        cpu_list,
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

        ntraj = run_dict.pop('ntraj', DEFAULT_NUM_TRAJ)
        assert isinstance(ntraj, int) and ntraj >= 1

        trans_variants = run_dict.pop('transfer-variants', [])
        assert isinstance(trans_variants, list)

        env_subset = run_dict.pop('env-subset', list(ENV_NAMES.keys()))
        assert isinstance(env_subset, list) \
            and len(set(env_subset)) == len(env_subset)\
            and set(env_subset) <= ENV_NAMES.keys()
        env_subset = sorted(env_subset)

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
            'ntraj': ntraj,
            'env_subset': env_subset,
            'nseeds': nseeds,
        })

    return run_specs


def select_subset(collection, n, rng):
    """Use the given RNG to randomly select a subset of `n` items from
    collection."""
    nitems = len(collection)
    assert nitems >= n, \
        f"cannot select {n} items from collection of size {nitems}"
    indices = rng.permutation(nitems)[:n]
    return [collection[i] for i in indices]


Run = collections.namedtuple('Run', ('train_cmd', 'test_cmds'))


def generate_runs(*, run_name, algo, generator, is_multi_task, args, nseeds,
                  suffix, out_dir, trans_variants, ntraj, env_subset,
                  nworkers):
    runs = []
    seeds = generate_seeds(nseeds)
    for seed in seeds:
        rng = np.random.RandomState(seed)
        if is_multi_task:
            # one train command corresponding to N test commands
            mt_run_name = f"{run_name}-{suffix}-s{seed}"
            demo_paths = []
            for expt in env_subset:
                globbed = glob.glob(
                    os.path.expanduser(DEMO_PATH_PATTERNS[expt]))
                chosen = select_subset(globbed, ntraj, rng)
                demo_paths.extend(chosen)

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
                                       cpu_list=generate_core_list(nworkers),
                                       **args)

            eval_cmds = []
            for task_shorthand in env_subset:
                env_name = ENV_NAMES[task_shorthand]
                mt_eval_cmd = make_eval_cmd(
                    run_name,
                    mt_dir,
                    task_shorthand,
                    env_name,
                    cpu_list=generate_core_list(nworkers))
                eval_cmds.append(mt_eval_cmd)
            runs.append(Run(mt_cmd, eval_cmds))

        else:
            # one test command for each of N train commands
            for task in sorted(env_subset):
                st_run_name = f"{run_name}-{task}-{suffix}-s{seed}"
                demo_paths_all = glob.glob(
                    os.path.expanduser(DEMO_PATH_PATTERNS[task]))
                demo_paths = select_subset(demo_paths_all, ntraj, rng)
                st_trans_variants = [
                    insert_variant(ENV_NAMES[task], variant)
                    for variant in trans_variants
                ]

                st_cmd, st_dir = generator(
                    demo_paths=demo_paths,
                    run_name=st_run_name,
                    seed=seed,
                    out_root=out_dir,
                    trans_env_names=st_trans_variants,
                    cpu_list=generate_core_list(nworkers),
                    **args)

                st_eval_cmd = make_eval_cmd(
                    run_name,
                    st_dir,
                    task,
                    ENV_NAMES[task],
                    cpu_list=generate_core_list(nworkers))
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
@click.option('--job-nworkers',
              default=None,
              help='number of CPU workers per job')
@click.argument("spec")
def main(spec, suffix, out_dir, ray_connect, ray_ncpus, job_ngpus,
         job_ngpus_eval, job_nworkers, dry_run):
    """Execute some experiments with Ray."""
    # spin up Ray cluster
    if job_nworkers is None:
        job_nworkers = min(8, max(1, os.cpu_count() // 2))
    assert os.cpu_count() % job_nworkers == 0, \
        (os.cpu_count(), job_nworkers)
    new_cluster = ray_connect is None
    ray_kwargs = {}
    if not new_cluster:
        assert job_nworkers is None, \
            "the --job-nworkers option is only supported when spinning up a " \
            "new Ray instance"
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
        new_runs = generate_runs(**expt_spec,
                                 suffix=suffix,
                                 out_dir=out_dir,
                                 nworkers=job_nworkers)
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


if __name__ == '__main__':
    main()

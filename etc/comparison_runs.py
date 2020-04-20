#!/usr/bin/env python3
"""Main script for doing experiments that compare different methods."""
import datetime
import glob
import itertools
import os
import stat
import subprocess

import click
import yaml

DEMO_PATH_PATTERNS = {
    'move-to-corner': '~/repos/milbench/demos-simplified/move-to-corner-2020-03-*/demo-MoveToCorner-Demo-v0-2020-03-*T*.pkl.gz',  # noqa: E501
    'match-regions': '~/repos/milbench/demos-simplified/match-regions-2020-03-*/demo-MatchRegions-Demo-v0-2020-03-*T*.pkl.gz',  # noqa: E501
    'cluster-colour': '~/repos/milbench/demos-simplified/cluster-colour-2020-03-*/demo-ClusterColour-Demo-v0-2020-03-*T*.pkl.gz',  # noqa: E501
    'cluster-type': '~/repos/milbench/demos-simplified/cluster-type-2020-03-*/demo-ClusterType-Demo-v0-2020-03-*T*.pkl.gz',  # noqa: E501
}  # yapf: disable
ENV_NAMES = {
    'move-to-corner': 'MoveToCorner-Demo-LoResStack-v0',
    'match-regions': 'MatchRegions-Demo-LoResStack-v0',
    'cluster-colour': 'ClusterColour-Demo-LoResStack-v0',
    'cluster-type': 'ClusterType-Demo-LoResStack-v0',
}
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_GPUS = 4


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


# Signature for algorithms:
# - Takes in a list of demo paths, a run name, a seed, and a bunch of kwargs
#   specific to that algorithm.
# - Returns a list of command parts for `subprocess` (or `shlex` or whatever),
#   and a path to a scratch dir that should contain a file matching the
#   "itr_LATEST.pkl" used by mtbc.py.


def gen_command_gail(demo_paths, run_name, seed, n_steps=None):
    cmd_parts = [
        *("xvfb-run -a python -m mtil.algos.mtgail").split(),
        "--run-name",
        str(run_name),
        "--seed",
        str(seed),
    ]
    if n_steps is not None:
        assert n_steps == int(n_steps), n_steps
        cmd_parts.extend(['--total-n-steps', str(int(n_steps))])
    cmd_parts.extend(demo_paths)
    out_dir = f"./scratch/run_mtgail-{run_name}/"
    return cmd_parts, out_dir


def gen_command_bc(demo_paths, run_name, seed):
    cmd_parts = [
        "xvfb-run",
        "-a",
        "python",
        "-m",
        "mtil.algos.mtbc",
        "train",
        "--run-name",
        run_name,
        "--eval-n-traj",
        "10",
        "--passes-per-eval",
        "20",
        # snapshot gap of 1 is going to generate a lot of noise!
        "--snapshot-gap",
        "1",
        "--seed",
        str(seed),
    ]
    cmd_parts.extend(demo_paths)
    out_dir = f'./scratch/run_{run_name}'
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
        os.path.join(snap_dir, 'itr_LATEST.pkl'),
        "--run-id",
        f"{run_name}-on-{env_shorthand}",
        "--write-latex",
        f"{snap_dir}/eval-{env_shorthand}.tex",
        "--write-csv",
        f"{snap_dir}/eval-{env_shorthand}.csv",
        "--n-rollouts",
        "100",
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

        if len(run_dict) > 0:
            raise ValueError(
                f"unrecognised options in expt spec '{run_name}': "
                f"{sorted(run_dict.keys())}")

        run_specs.append({
            'run_name': run_name,
            'algo': algo,
            'generator': generator,
            'is_multi_task': is_multi_task,
            'args': args,
        })

    return run_specs


def generate_cmds(*, run_name, algo, generator, is_multi_task, args, suffix):
    train_cmds = []
    eval_cmds = []

    seed = 42  # TODO: allow for iteration over seed

    if is_multi_task:
        # one train command, N test commands
        mt_run_name = f"{run_name}-{suffix}"
        demo_paths = [
            DEMO_PATH_PATTERNS[expt]
            for expt in sorted(DEMO_PATH_PATTERNS.keys())
        ]

        mt_cmd, mt_dir = generator(demo_paths=demo_paths,
                                   run_name=mt_run_name,
                                   seed=seed,
                                   **args)
        train_cmds.append(mt_cmd)

        for task_shorthand, env_name in sorted(ENV_NAMES.items()):
            mt_eval_cmd = make_eval_cmd(run_name, mt_dir, task_shorthand,
                                        env_name)
            eval_cmds.append(mt_eval_cmd)

    else:
        # one test command, N train commands
        for task in sorted(DEMO_PATH_PATTERNS.keys()):
            st_run_name = f"{run_name}-{task}-{suffix}"
            demo_paths = [DEMO_PATH_PATTERNS[task]]

            st_cmd, st_dir = generator(demo_paths=demo_paths,
                                       run_name=st_run_name,
                                       seed=seed,
                                       **args)
            train_cmds.append(st_cmd)

            st_eval_cmd = make_eval_cmd(run_name, st_dir, task,
                                        ENV_NAMES[task])
            eval_cmds.append(st_eval_cmd)

    return train_cmds, eval_cmds


def chmod_plusx(file_path):
    """Make file at existing path world-executable."""
    file_mode = os.stat(file_path).st_mode
    file_mode |= stat.S_IXUSR | stat.S_IXOTH | stat.S_IXGRP
    os.chmod(file_path, file_mode)


@click.command()
@click.option("--dest", default="commands.sh", help="file to write to")
@click.option("--suffix",
              default=None,
              help="extra suffix to add to runs (defaults to timestamp)")
@click.argument("spec")
def main(spec, dest, suffix):
    """Generate shell script of commands from an input YAML spec"""
    if suffix is None:
        # add 'T%H:%M' for hours and minutes
        suffix = datetime.datetime.now().strftime('%Y-%m-%d')

    expts_specs = parse_expts_file(spec)

    all_train = []
    all_test = []
    for expt_spec in expts_specs:
        train_cmds, test_cmds = generate_cmds(**expt_spec, suffix=suffix)
        all_train.extend(train_cmds)
        all_test.extend(test_cmds)

    print(f"Writing commands to '{dest}'")
    out_dir = os.path.dirname(dest)
    if out_dir:
        # create dir if needed
        os.makedirs(out_dir, exist_ok=True)
    with open(dest, "w") as fp:
        print("#!/usr/bin/env bash\n", file=fp)

        # write the commands & the GPU indices at the same time
        gpu_itr = itertools.cycle(range(NUM_GPUS))
        for cmd in all_train:
            cmd = [*cmd, "--gpu-idx", str(next(gpu_itr))]
            print(subprocess.list2cmdline(cmd) + " &\n", file=fp)

        # assume that test cmds start *after* all train CMDs have finished, so
        # it's fine to reset counter
        gpu_itr = itertools.cycle(range(NUM_GPUS))
        for cmd in all_test:
            cmd = [*cmd, "--gpu-idx", str(next(gpu_itr))]
            print(subprocess.list2cmdline(cmd) + " &\n", file=fp)

    # change file mode to make executable
    chmod_plusx(dest)


if __name__ == '__main__':
    main()

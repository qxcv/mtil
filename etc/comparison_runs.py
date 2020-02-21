#!/usr/bin/env python3
import datetime
import glob
import os
import subprocess

DEMO_PATH_PATTERNS = {
    'move-to-corner': '~/repos/milbench/demos/move-to-corner-2019-12-09/demo-MoveToCorner-Demo-v0-2019-12-09T*.pkl.gz',  # noqa: E501
    'match-regions': '~/repos/milbench/demos/match-regions-2019-12-09/demo-MatchRegions-Demo-v0-2019-12-09T*.pkl.gz',  # noqa: E501
    'cluster-colour': '~/repos/milbench/demos/cluster-colour-2019-12-09/demo-ClusterColour-Demo-v0-2019-12-09T*.pkl.gz',  # noqa: E501
    'cluster-type': '~/repos/milbench/demos/cluster-type-2019-12-09/demo-ClusterType-Demo-v0-2019-12-09T*.pkl.gz',  # noqa: E501
}  # yapf: disable
ENV_NAMES = {
    'move-to-corner': 'MoveToCorner-Demo-LoResStack-v0',
    'match-regions': 'MatchRegions-Demo-LoResStack-v0',
    'cluster-colour': 'ClusterColour-Demo-LoResStack-v0',
    'cluster-type': 'ClusterType-Demo-LoResStack-v0',
}
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# IDK why I thought this was a better idea than "python -m mtil.algos.mtbc"
MAIN_FILE = os.path.join(THIS_DIR, 'mtil/algos/mtbc/__main__.py')


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


def gen_command(expts, run_name, n_epochs, gpu_idx=0):
    demo_paths = expand_patterns(*[DEMO_PATH_PATTERNS[expt] for expt in expts])
    cmd_parts = [
        'xvfb-run',
        '-a',
        'python',
        MAIN_FILE,
        'train',
        '--run-name',
        run_name,
        '--use-gpu',
        '--epochs',
        str(n_epochs),
        '--eval-n-traj',
        '10',
        '--passes-per-eval',
        '20',
        # snapshot gap of 1 is going to generate a lot of noise!
        '--snapshot-gap',
        '1',
        *demo_paths,
        # put it at the end so it's easy to change
        '--gpu-idx',
        str(gpu_idx),
    ]
    return cmd_parts


def make_eval_cmd(run_name, env_shorthand, env_name, itr):
    run_dir = f'./scratch/run_{run_name}'
    new_cmd = [
        "xvfb-run",
        "-a",
        "python",
        "-m",
        "mtil.algos.mtbc",
        "testall",
        "--env-name",
        env_name,
        f"{run_dir}/itr_{itr}.pkl",
        "--run-id",
        f"{run_name}-on-{env_shorthand}-after-{itr+1}-epochs",
        "--write-latex",
        f"{run_dir}/eval_{itr}.tex",
        "--write-csv",
        f"{run_dir}/eval_{itr}.csv",
        "--n-rollouts",
        "30",
    ]
    return new_cmd


def gen_all_expts():
    # these will all have GPU 0, so I need to change things once I have the
    # shell script. (another reminder: there are 20 passes through the dataset
    # for each 'eval')
    date = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')
    mt_run_name = f"multi-task-bc-{date}"
    mt_cmd = gen_command(sorted(DEMO_PATH_PATTERNS.keys()), mt_run_name, 100)
    st_run_names = {
        task: f"single-task-bc-{task}-{date}"
        for task in sorted(DEMO_PATH_PATTERNS.keys())
    }
    st_cmds = sorted([
        gen_command([task], run_name, 100)
        for task, run_name in st_run_names.items()
    ])
    train_cmds = [mt_cmd, *st_cmds]

    # test CMDs will write problem data to a problem-specific dataframe, then
    # combine the frames into a plot later on
    target_itrs = [0, 9, 99]
    eval_cmds = []
    for itr in target_itrs:
        for env_shorthand, run_name in st_run_names.items():
            env_name = ENV_NAMES[env_shorthand]
            new_cmd = make_eval_cmd(run_name, env_shorthand, env_name, itr)
            eval_cmds.append(new_cmd)
    for itr in target_itrs:
        for env_shorthand, env_name in ENV_NAMES.items():
            new_cmd = make_eval_cmd(mt_run_name, env_shorthand, env_name, itr)
            eval_cmds.append(new_cmd)

    return [*train_cmds, *eval_cmds]


# Runs I want to do:
#  - One multi-task with 1/10/100 rounds of opt. (one run, three checkpoints)
#  - Four single-tasks (one per env) with 1/10/100 rounds of opt. (four runs,
#    twelve checkpoints).
# Will start ~5 relevant training runs now & then sort out eval later.


def main():
    expt_cmd_parts = gen_all_expts()
    expt_commands = [
        subprocess.list2cmdline(parts) for parts in expt_cmd_parts
    ]
    print("Writing commands to commands.sh")
    with open("commands.sh", "w") as fp:
        print("#!/bin/env bash\n", file=fp)
        for line in expt_commands:
            print(line, file=fp)
            print(file=fp)


if __name__ == '__main__':
    main()

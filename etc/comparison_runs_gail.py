#!/usr/bin/env python3
import datetime
import glob
import itertools
import os
import subprocess

DEMO_PATH_PATTERNS = {
    'move-to-corner': '~/repos/milbench/demos-simplified/move-to-corner-2020-03-*/demo-MoveToCorner-Demo-v0-2020-03-*T*.pkl.gz',  # noqa: E501
    # 'match-regions': '~/repos/milbench/demos-simplified/match-regions-2020-03-*/demo-MatchRegions-Demo-v0-2020-03-*T*.pkl.gz',  # noqa: E501
    # 'cluster-colour': '~/repos/milbench/demos-simplified/cluster-colour-2020-03-*/demo-ClusterColour-Demo-v0-2020-03-*T*.pkl.gz',  # noqa: E501
    # 'cluster-type': '~/repos/milbench/demos-simplified/cluster-type-2020-03-*/demo-ClusterType-Demo-v0-2020-03-*T*.pkl.gz',  # noqa: E501
}  # yapf: disable
ENV_NAMES = {
    'move-to-corner': 'MoveToCorner-Demo-LoResStack-v0',
    # 'match-regions': 'MatchRegions-Demo-LoResStack-v0',
    # 'cluster-colour': 'ClusterColour-Demo-LoResStack-v0',
    # 'cluster-type': 'ClusterType-Demo-LoResStack-v0',
}
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# IDK why I thought this was a better idea than "python -m mtil.algos.mtbc"
MAIN_FILE = os.path.abspath(
    os.path.join(THIS_DIR, '../mtil/algos/mtbc/__main__.py'))
NUM_GPUS = 4
N_STEPS = int(1e6)


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


def gen_command(expts, run_name, n_steps, gpu_idx=0):
    demo_paths = [DEMO_PATH_PATTERNS[expt] for expt in expts]
    cmd_parts = [
        *("xvfb-run -a python -m mtil.algos.gail --no-omit-noop "
          "--disc-up-per-iter 16 --disc-replay-mult 5 --disc-lr 1e-4 "
          "--disc-use-act --disc-all-frames --sampler-time-steps 32 "
          "--sampler-batch-envs 24 --ppo-lr 2e-4 --ppo-gamma 0.9 "
          "--ppo-lambda 0.95 --ppo-ent 1e-4 --ppo-adv-clip 0.05 "
          "--no-ppo-norm-adv").split(),
        *demo_paths,
        # at the end to make it easy to change
        '--total-n-steps',
        str(n_steps),
        '--run-name',
        str(run_name),
        '--gpu-idx',
        str(gpu_idx),
    ]
    return cmd_parts


def make_eval_cmd(run_name, env_shorthand, env_name, gpu_idx):
    run_dir = f'./scratch/run_gail-{run_name}'
    new_cmd = [
        "xvfb-run",
        "-a",
        "python",
        "-m",
        "mtil.algos.mtbc",
        "testall",
        "--env-name",
        env_name,
        # the last occurrence of 'LATEST' will be replaced with the latest run
        # ID
        f"{run_dir}/itr_LATEST.pkl",
        "--load-latest",
        "--run-id",
        f"{run_name}-on-{env_shorthand}-epochs",
        "--write-latex",
        f"{run_dir}/eval-{env_shorthand}.tex",
        "--write-csv",
        f"{run_dir}/eval-{env_shorthand}.csv",
        "--n-rollouts",
        "30",
        "--gpu-idx",
        str(gpu_idx),
    ]
    return new_cmd


def gen_all_expts():
    # these will all have GPU 0, so I need to change things once I have the
    # shell script. (another reminder: there are 20 passes through the dataset
    # for each 'eval')
    name_opt_combos = [
        ('sweep-default', []),
        # TODO: make variants non-env-specific
        ('sweep-xtrain-shape-colour',
         ['--transfer-variant', 'MoveToCorner-TestShapeColour-LoResStack-v0']),
        ('sweep-xtrain-robot-pose',
         ['--transfer-variant', 'MoveToCorner-TestRobotPose-LoResStack-v0']),
        ('sweep-xtrain-all',
         ['--transfer-variant', 'MoveToCorner-TestAll-LoResStack-v0']),
    ]
    gpu_itr = itertools.cycle(range(NUM_GPUS))
    date = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')

    st_run_names = []
    st_cmds = []
    for task in DEMO_PATH_PATTERNS.keys():
        # single-task training (x num_envs)
        for opt_name, extra_opts in name_opt_combos:
            new_st_run_name = (task,
                               f"st-{task}-{opt_name}-{date}")
            st_run_names.append(new_st_run_name)
            new_st_cmd = gen_command(
                [task], new_st_run_name[1], N_STEPS, gpu_idx=next(gpu_itr)) \
                + extra_opts
            st_cmds.append(new_st_cmd)

    train_cmds = [*st_cmds]

    # test CMDs will write problem data to a problem-specific dataframe, then
    # combine the frames into a plot later on
    eval_cmds = []
    # single-task eval runs (one per env)
    for env_shorthand, run_name in st_run_names:
        env_name = ENV_NAMES[env_shorthand]
        new_cmd = make_eval_cmd(run_name,
                                env_shorthand,
                                env_name,
                                gpu_idx=next(gpu_itr))
        eval_cmds.append(new_cmd)

    return [*train_cmds, *eval_cmds]


def main():
    expt_cmd_parts = gen_all_expts()
    expt_commands = [
        subprocess.list2cmdline(parts) for parts in expt_cmd_parts
    ]
    print("Writing commands to commands_gail.sh")
    with open("commands_gail.sh", "w") as fp:
        print("#!/usr/bin/env bash\n", file=fp)
        for line in expt_commands:
            print(line + ' &', file=fp)
            print('\n', file=fp)


if __name__ == '__main__':
    main()

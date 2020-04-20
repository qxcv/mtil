#!/usr/bin/env python3
import datetime
import glob
import itertools
import os
import subprocess

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
# IDK why I thought this was a better idea than "python -m mtil.algos.mtbc"
MAIN_FILE = os.path.abspath(
    os.path.join(THIS_DIR, '../mtil/algos/mtbc/__main__.py'))
NUM_GPUS = 4
EPOCHS = 30


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
    # demo_paths = expand_patterns(*[DEMO_PATH_PATTERNS[expt] for expt in expts])
    demo_paths = [DEMO_PATH_PATTERNS[expt] for expt in expts]
    cmd_parts = [
        'xvfb-run',
        '-a',
        'python',
        MAIN_FILE,
        'train',
        '--run-name',
        run_name,
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


def make_eval_cmd(run_name, env_shorthand, env_name, itr, gpu_idx):
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
        f"{run_dir}/eval-{env_shorthand}-{itr}.tex",
        "--write-csv",
        f"{run_dir}/eval-{env_shorthand}-{itr}.csv",
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
        ('sweep-dropout', ['--net-dropout', '0.3']),
        ('sweep-coordconv', ['--net-coord-conv']),
        ('sweep-aug-trans-rot', ['--aug-mode', 'transrot']),
    ]
    gpu_itr = itertools.cycle(range(NUM_GPUS))
    date = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')

    mt_run_names = []
    st_run_names = []
    mt_cmds = []
    st_cmds = []
    for opt_name, extra_opts in name_opt_combos:
        # multi-task training (x 1)
        mt_run_name = f"multi-task-bc-{opt_name}-{date}"
        mt_run_names.append(mt_run_name)
        mt_cmd = gen_command(sorted(DEMO_PATH_PATTERNS.keys()),
                             mt_run_name,
                             EPOCHS,
                             gpu_idx=next(gpu_itr)) + extra_opts
        mt_cmds.append(mt_cmd)

    for task in DEMO_PATH_PATTERNS.keys():
        # single-task training (x num_envs)
        for opt_name, extra_opts in name_opt_combos:
            new_st_run_name = (task,
                               f"single-task-bc-{task}-{opt_name}-{date}")
            st_run_names.append(new_st_run_name)
            new_st_cmd = gen_command(
                [task], new_st_run_name[1], EPOCHS, gpu_idx=next(gpu_itr)) \
                + extra_opts
            st_cmds.append(new_st_cmd)

    train_cmds = [*mt_cmds, *st_cmds]

    # test CMDs will write problem data to a problem-specific dataframe, then
    # combine the frames into a plot later on
    target_itrs = [EPOCHS - 1]
    eval_cmds = []
    # single-task eval runs (one per env)
    # TODO: add GPUs to these eval runs
    for itr in target_itrs:
        for env_shorthand, run_name in st_run_names:
            env_name = ENV_NAMES[env_shorthand]
            new_cmd = make_eval_cmd(run_name,
                                    env_shorthand,
                                    env_name,
                                    itr=itr,
                                    gpu_idx=next(gpu_itr))
            eval_cmds.append(new_cmd)
    # multi-task eval runs (one eval run per env, even though there was only
    # one multi-task training run shared across all envs)
    for itr in target_itrs:
        for env_shorthand, env_name in ENV_NAMES.items():
            for mt_run_name in mt_run_names:
                new_cmd = make_eval_cmd(mt_run_name,
                                        env_shorthand,
                                        env_name,
                                        itr=itr,
                                        gpu_idx=next(gpu_itr))
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
        print("#!/usr/bin/env bash\n", file=fp)
        for line in expt_commands:
            print(line + ' &', file=fp)
            print('\n', file=fp)


if __name__ == '__main__':
    main()

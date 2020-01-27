"""Main entry point for GAIL algorithm. Separated from actual implementation so
that parts of the implementation can be pickled."""
import multiprocessing as mp
import readline  # noqa: F401

import click
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
import torch

from mtil.algos.gail.gail import (GAILMinibatchRl, GAILOptimiser,
                                  MILBenchDiscriminator,
                                  MILBenchPolicyValueNetwork, RewardModel)
from mtil.common import (MILBenchGymEnv, MILBenchTrajInfo, get_env_meta,
                         make_logger_ctx, sane_click_init, set_seeds)
from mtil.reward_injection_wrappers import CustomRewardPPO


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--add-preproc",
    default="LoResStack",
    type=str,
    help="add preprocessor to the demos and test env (default: 'LoResStack')")
@click.option("--use-gpu/--no-use-gpu", default=False, help="use GPU")
@click.option("--gpu-idx", default=0, help="index of GPU to use")
@click.option("--n-workers",
              default=None,
              type=int,
              help="number of rollout workers")
@click.option("--seed", default=42, help="PRNG seed")
@click.option("--epochs", default=1000, help="epochs of training to perform")
@click.option("--out-dir", default="scratch", help="dir for snapshots/logs")
@click.option(
    "--log-interval-steps",
    default=1e5,
    help="how many env transitions to take between writing log outputs")
@click.option("--n-envs",
              default=16,
              help="number of parallel envs to sample from")
@click.option("--n-steps-per-iter",
              default=64,
              help="number of timesteps to advance each env when sampling")
@click.option("--disc-batch-size",
              default=32,
              help="batch size for discriminator training")
@click.option("--disc-up-per-itr",
              default=1,
              help="number of discriminator steps per RL step")
@click.option("--total-n-steps",
              default=1e6,
              help="total number of steps to take in environment")
@click.option("--eval-n-traj",
              default=10,
              help="number of trajectories to roll out on each evaluation")
@click.option("--run-name",
              default=None,
              type=str,
              help="unique name for this run")
@click.argument("demos", nargs=-1, required=True)
def main(demos, use_gpu, add_preproc, seed, n_envs, n_steps_per_iter,
         disc_batch_size, epochs, out_dir, run_name, gpu_idx, eval_n_traj,
         disc_up_per_itr, total_n_steps, log_interval_steps, n_workers):
    # set up seeds & devices
    set_seeds(seed)
    # 'spawn' is necessary to use GL envs in subprocesses. For whatever reason
    # they don't play nice after a fork. (But what about set_seeds() in
    # subprocesses? May need to hack CpuSampler and GpuSampler.)
    mp.set_start_method('spawn')
    use_gpu = use_gpu and torch.cuda.is_available()
    dev = torch.device(["cpu", f"cuda:{gpu_idx}"][use_gpu])
    cpu_count = mp.cpu_count()
    n_workers = max(1, cpu_count // 2) if n_workers is None else n_workers
    assert n_workers <= cpu_count, \
        f"can't have n_workers={n_workers} > cpu_count={cpu_count}"
    # TODO: figure out a better way of assigning work to cores (why can't my OS
    # scheduler do it? Grumble grumble…).
    affinity = dict(
        cuda_idx=gpu_idx if use_gpu else None,
        # workers_cpus=list(np.random.permutation(cpu_count)[:n_workers])
        workers_cpus=list(range(n_workers)),
    )
    print(f"Using device {dev}, seed {seed}, affinity {affinity}")

    # register original envs
    import milbench
    milbench.register_envs()

    # load demos (this code copied from bc.py in original baselines)
    demo_dicts = load_demos(demos)
    orig_env_name = demo_dicts[0]['env_name']
    if add_preproc:
        env_name = splice_in_preproc_name(orig_env_name, add_preproc)
        print(f"Splicing preprocessor '{add_preproc}' into environment "
              f"'{orig_env_name}'. New environment is {env_name}")
    else:
        env_name = orig_env_name
    demo_trajs = [d['trajectory'] for d in demo_dicts]
    if add_preproc:
        demo_trajs = preprocess_demos_with_wrapper(demo_trajs, orig_env_name,
                                                   add_preproc)

    # local copy of Gym env, w/ args to create equivalent env in the sampler
    env_ctor = MILBenchGymEnv
    env_ctor_kwargs = dict(env_name=env_name)
    env_meta = get_env_meta(env_name)
    # number of transitions collected during each round of sampling will be
    # batch_T * batch_B = n_steps * n_envs
    batch_T = n_steps_per_iter
    batch_B = n_envs

    if use_gpu:
        sampler_ctor = GpuSampler
    else:
        sampler_ctor = CpuSampler
    sampler = sampler_ctor(
        env_ctor,
        env_ctor_kwargs,
        max_decorrelation_steps=env_meta.spec.max_episode_steps,
        TrajInfoCls=MILBenchTrajInfo,
        # seed=seed,  # TODO: make sampler seeding work
        batch_T=batch_T,
        batch_B=batch_B)

    # should be (H,W,C) even w/ frame stack
    assert len(env_meta.observation_space.shape) == 3, \
        env_meta.observation_space.shape
    in_chans = env_meta.observation_space.shape[-1]
    n_actions = env_meta.action_space.n  # categorical action space
    ppo_agent = CategoricalPgAgent(ModelCls=MILBenchPolicyValueNetwork,
                                   model_kwargs=dict(in_chans=in_chans,
                                                     n_actions=n_actions))

    discriminator = MILBenchDiscriminator(
            in_chans=in_chans, act_dim=n_actions).to(dev)
    reward_model = RewardModel(discriminator).to(dev)
    # TODO: figure out what pol_batch_size should be/do, and what relation it
    # should have with sampler batch size
    # TODO: also consider adding a BC loss to the policy (this will have to be
    # PPO-specific though)
    ppo_algo = CustomRewardPPO()
    ppo_algo.set_reward_model(reward_model)

    gail_optim = GAILOptimiser(
        expert_trajectories=demo_trajs,
        discrim_model=discriminator,
        buffer_num_samples=batch_T *
        # TODO: also update this once you've figured out what arg to give to
        # sampler
        max(batch_B, disc_batch_size),
        batch_size=disc_batch_size,
        updates_per_itr=disc_up_per_itr,
        dev=dev)

    # signature for arg: reward_model(obs_tensor, act_tensor) -> rewards
    runner = GAILMinibatchRl(
        gail_optim=gail_optim,
        algo=ppo_algo,
        agent=ppo_agent,
        sampler=sampler,
        # n_steps controls total number of environment steps we take
        n_steps=total_n_steps,
        # log_interval_steps controls how many environment steps we take
        # between making log outputs (doing N environment steps takes roughly
        # the same amount of time no matter what batch_B, batch_T, etc. are, so
        # this gives us a fairly constant interval between log outputs)
        log_interval_steps=log_interval_steps,
        affinity=affinity)

    with make_logger_ctx(out_dir, "gail", orig_env_name, run_name):
        runner.train()


if __name__ == '__main__':
    sane_click_init(main)

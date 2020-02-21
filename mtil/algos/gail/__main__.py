"""Main entry point for GAIL algorithm. Separated from actual implementation so
that parts of the implementation can be pickled."""

# 'import readline' is necessary to stop pdb.set_trace() from segfaulting in
# rl_initialize when importing readline. Problematic import is
# 'milbench.baselines.saved_trajectories' (putting this import after that means
# the segfault still happens). Haven't had time to chase down.
import multiprocessing as mp
import os
import readline  # noqa: F401

import click
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.logging import logger
import torch

from mtil.algos.gail.gail import (GAILMinibatchRl, GAILOptimiser,
                                  MILBenchDiscriminator, RewardModel)
from mtil.common import (MILBenchGymEnv, MILBenchTrajInfo, MultiHeadPolicyNet,
                         get_env_meta, load_demos_mt, make_logger_ctx,
                         sane_click_init, set_seeds, FixedTaskModelWrapper)
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
    default=1e4,
    help="how many env transitions to take between writing log outputs")
# FIXME: rename these --ppo-batch-b and --ppo-batch-t instead, since that's
# really what they are.
@click.option("--n-envs",
              default=16,
              help="number of parallel envs to sample from")
@click.option("--n-steps-per-iter",
              default=64,
              help="number of timesteps to advance each env when sampling")
@click.option("--disc-batch-size",
              default=32,
              help="batch size for discriminator training")
@click.option("--disc-up-per-iter",
              default=1,
              help="number of discriminator steps per RL step")
@click.option("--total-n-steps",
              default=1e7,
              help="total number of steps to take in environment")
@click.option("--eval-n-traj",
              default=10,
              help="number of trajectories to roll out on each evaluation")
@click.option("--run-name",
              default=None,
              type=str,
              help="unique name for this run")
@click.option("--snapshot-gap", default=10, help="evals between snapshots")
@click.argument("demos", nargs=-1, required=True)
def main(demos, add_preproc, seed, n_envs, n_steps_per_iter, disc_batch_size,
         epochs, out_dir, run_name, gpu_idx, eval_n_traj, disc_up_per_iter,
         total_n_steps, log_interval_steps, n_workers, snapshot_gap):
    # set up seeds & devices
    set_seeds(seed)
    # 'spawn' is necessary to use GL envs in subprocesses. For whatever reason
    # they don't play nice after a fork. (But what about set_seeds() in
    # subprocesses? May need to hack CpuSampler and GpuSampler.)
    mp.set_start_method('spawn')
    use_gpu = gpu_idx is not None and torch.cuda.is_available()
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
    dataset_mt, env_name_to_id, env_id_to_name, name_pairs \
        = load_demos_mt(demos, add_preproc)
    # loader_mt = make_loader_mt(dataset_mt, disc_batch_size // 2)
    env_ids_and_names = [(name, env_name_to_id[name])
                         for _, name in name_pairs]
    assert len(env_ids_and_names) == 1, \
        "GAIL doesn't support multi-task training yet"
    (env_name, env_id), = env_ids_and_names

    print("Getting env metadata")
    # local copy of Gym env, w/ args to create equivalent env in the sampler
    env_ctor = MILBenchGymEnv
    env_ctor_kwargs = dict(env_name=env_name)
    env_meta = get_env_meta(env_name)
    # number of transitions collected during each round of sampling will be
    # batch_T * batch_B = n_steps * n_envs
    batch_T = n_steps_per_iter
    batch_B = n_envs

    print("Setting up sampler")
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

    print("Setting up agent")
    # should be (H,W,C) even w/ frame stack
    assert len(env_meta.observation_space.shape) == 3, \
        env_meta.observation_space.shape
    in_chans = env_meta.observation_space.shape[-1]
    n_actions = env_meta.action_space.n  # categorical action space
    model_ctor = MultiHeadPolicyNet
    model_kwargs = dict(in_chans=in_chans,
                        n_actions=n_actions,
                        env_ids_and_names=env_ids_and_names)
    ppo_agent = CategoricalPgAgent(ModelCls=FixedTaskModelWrapper,
                                   model_kwargs=dict(
                                       model_ctor=model_ctor,
                                       task_id=env_id,
                                       model_kwargs=model_kwargs))

    print("Setting up discriminator/reward model")
    discriminator = MILBenchDiscriminator(in_chans=in_chans,
                                          act_dim=n_actions).to(dev)
    reward_model = RewardModel(discriminator).to(dev)
    # TODO: figure out what pol_batch_size should be/do, and what relation it
    # should have with sampler batch size
    # TODO: also consider adding a BC loss to the policy (this will have to be
    # PPO-specific though)

    # Stable Baselines hyperparams:
    # gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=0.00025,
    # vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4,
    # cliprange=0.2, cliprange_vf=None
    #
    # Default rlpyt hyperparams:
    # discount=0.99, learning_rate=0.001, value_loss_coeff=1.0,
    # entropy_loss_coeff=0.01, clip_grad_norm=1.0,
    # initial_optim_state_dict=None, gae_lambda=1, minibatches=4, epochs=4,
    # ratio_clip=0.1, linear_lr_schedule=True, normalize_advantage=False
    #
    # gae_lambda is probably doing a lot of work here, especially if we're
    # using some funky way of computing return for our partial traces. I doubt
    # value_loss_coeff and clip_grad_norm make much difference, since it's only
    # a factor of 2 change. cliprange difference might matter, but IDK. n_steps
    # will also matter a lot since it's so low by default in rlpyt (16).
    extra_ppo_kwargs = dict(
            learning_rate=0.00025,
            value_loss_coeff=0.5,
            clip_grad_norm=0.5,
            gae_lambda=0.95)
    ppo_algo = CustomRewardPPO(normalize_advantage=True, **extra_ppo_kwargs)
    ppo_algo.set_reward_model(reward_model)

    print("Setting up optimiser")
    gail_optim = GAILOptimiser(
        dataset_mt=dataset_mt,
        discrim_model=discriminator,
        buffer_num_samples=batch_T *
        # TODO: also update this once you've figured out what arg to give to
        # sampler
        max(batch_B, disc_batch_size),
        batch_size=disc_batch_size,
        updates_per_itr=disc_up_per_iter,
        dev=dev)

    print("Setting up RL algorithm")
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

    def save_model_cb(runner):
        """Callback which gets called once after Runner startup to save an
        initial policy model."""
        # get state of newly-initalised model
        wrapped_model = runner.algo.agent.model
        assert wrapped_model is not None, "has ppo_agent been initalised?"
        unwrapped_model = wrapped_model.model
        real_state = unwrapped_model.state_dict()

        # make a clone model so we can pickle it, and copy across weights
        policy_copy_mt = model_ctor(**model_kwargs).to('cpu')
        policy_copy_mt.load_state_dict(real_state)

        # save it here
        init_pol_snapshot_path = os.path.join(logger.get_snapshot_dir(),
                                              'full_model.pt')
        torch.save(policy_copy_mt, init_pol_snapshot_path)

    print("Training!")
    n_uniq_envs = len(env_ids_and_names)
    with make_logger_ctx(out_dir,
                         "gail",
                         f"mt{n_uniq_envs}",
                         run_name,
                         snapshot_gap=snapshot_gap):
        torch.save(
            discriminator,
            os.path.join(logger.get_snapshot_dir(), 'full_discrim_model.pt'))
        # note that periodic snapshots get saved by GAILMiniBatchRl, thanks to
        # the overridden get_itr_snapshot() method
        runner.train(cb_startup=save_model_cb)


if __name__ == '__main__':
    sane_click_init(main)

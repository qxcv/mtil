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
from rlpyt.utils.logging import logger
import torch

# TODO: factor load_state_dict_or_model from mtbc out into common
from mtil.algos.mtbc.mtbc import load_state_dict_or_model
from mtil.algos.mtgail.demos import load_demos_mtgail
from mtil.algos.mtgail.embedded_bc import BCCustomRewardPPO
from mtil.algos.mtgail.mtgail import (GAILMinibatchRl, GAILOptimiser,
                                      MILBenchDiscriminator, RewardModel)
from mtil.algos.mtgail.sample_mux import (MILBenchEnvMultiplexer,
                                          MuxCpuSampler, MuxGpuSampler,
                                          MuxTaskModelWrapper)
from mtil.augmentation import MILBenchAugmentations
from mtil.common import (MILBenchTrajInfo, MultiHeadPolicyNet, get_env_metas,
                         make_loader_mt, make_logger_ctx, sane_click_init,
                         set_seeds)
from mtil.reward_injection_wrappers import RewardEvaluator


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
@click.option("--out-dir", default="scratch", help="dir for snapshots/logs")
@click.option(
    "--log-interval-steps",
    default=1e4,
    help="how many env transitions to take between writing log outputs")
# FIXME: rename these --ppo-batch-b and --ppo-batch-t instead, since that's
# really what they are.
@click.option("-B",
              "--sampler-batch-envs",
              "sampler_batch_B",
              default=16,
              help="number of parallel envs to sample from")
@click.option("-T",
              "--sampler-time-steps",
              "sampler_batch_T",
              default=64,
              help="number of timesteps to advance each env when sampling")
@click.option("--disc-batch-size",
              default=32,
              help="batch size for discriminator training")
@click.option(
    "--disc-up-per-iter",
    default=4,  # IDK if this is too fast or slow or what
    help="number of discriminator steps per RL step")
@click.option('--disc-replay-mult',
              type=int,
              default=5,
              help="number of past epochs worth of interaction to save in "
              "discriminator replay buffer")
@click.option("--total-n-steps",
              default=10e6,
              help="total number of steps to take in environment")
@click.option("--bc-loss", default=0.0, help="behavioural cloning loss coeff")
@click.option("--run-name",
              default=None,
              type=str,
              help="unique name for this run")
@click.option("--snapshot-gap", default=1, help="evals between snapshots")
@click.option("--load-policy",
              default=None,
              type=str,
              help="path to a policy snapshot to load (e.g. from MTBC)")
@click.option("--omit-noop/--no-omit-noop",
              default=True,
              help="omit demonstration (s,a) pairs whenever a is a noop")
@click.option("--disc-aug/--no-disc-aug",
              default=True,
              help="enable/disable discriminator input data augmentation")
@click.option("--danger-debug-reward-weight",
              type=float,
              default=None,
              help="Replace some fraction of the IL reward with raw rewards "
              "from the env (DANGER!)")
@click.option("--danger-override-env-name",
              type=str,
              default=None,
              help="Do rollouts in this env instead of the demo one")
@click.option('--disc-lr', default=1e-3, help='discriminator learning rate')
@click.option('--disc-use-act/--no-disc-use-act',
              default=True,
              help='whether discriminator gets action inputs')
@click.option('--disc-all-frames/--no-disc-all-frames',
              default=True,
              help='whether discriminator gets full input frame stack')
@click.option('--ppo-lr', default=2.5e-4, help='PPO learning rate')
@click.option('--ppo-gamma', default=0.97, help='PPO discount factor (gamma)')
@click.option('--ppo-lambda', default=0.99, help='PPO GAE lamdba')
@click.option('--ppo-ent', default=1e-4, help='entropy bonus for PPO')
@click.option('--ppo-adv-clip', default=0.2, help='PPO advantage clip ratio')
@click.option('--ppo-norm-adv/--no-ppo-norm-adv',
              default=False,
              help='whether to normalise PPO advantages')
@click.option("--transfer-variant",
              'transfer_variants',
              default=[],
              multiple=True,
              help="name of transfer env for co-training (can be repeated)")
@click.argument("demos", nargs=-1, required=True)
def main(
        demos,
        add_preproc,
        seed,
        sampler_batch_B,
        sampler_batch_T,
        disc_batch_size,
        out_dir,
        run_name,
        gpu_idx,
        disc_up_per_iter,
        total_n_steps,
        log_interval_steps,
        n_workers,
        snapshot_gap,
        load_policy,
        bc_loss,
        omit_noop,
        disc_replay_mult,
        disc_aug,
        transfer_variants,
        danger_debug_reward_weight,
        danger_override_env_name,
        # new sweep hyperparams:
        disc_lr,
        disc_use_act,
        disc_all_frames,
        ppo_lr,
        ppo_gamma,
        ppo_lambda,
        ppo_ent,
        ppo_adv_clip,
        ppo_norm_adv,
):
    # set up seeds & devices
    # TODO: also seed child envs, when rlpyt supports it
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
    # (XXX: I suspect this will set torch_num_threads incorrectly, which sucks)
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
    dataset_mt, variant_groups = load_demos_mtgail(demos,
                                                   transfer_variants,
                                                   mb_preproc=add_preproc,
                                                   omit_noop=omit_noop)
    # (env_name, env_id), = env_ids_and_names
    if danger_override_env_name:
        raise NotImplementedError(
            "haven't re-implemeneted env name override for multi-task GAIL")

    print("Getting env metadata")
    # local copy of Gym env, w/ args to create equivalent env in the sampler
    env_mux = MILBenchEnvMultiplexer(variant_groups)
    new_sampler_batch_B, env_ctor_kwargs = env_mux.get_batch_size_and_kwargs(
        sampler_batch_B)
    all_env_names = sorted(variant_groups.task_variant_by_name.keys())
    if new_sampler_batch_B != sampler_batch_B:
        print(f"Increasing sampler batch size from '{sampler_batch_B}' to "
              f"'{new_sampler_batch_B}' to be divisible by number of "
              f"environments ({len(all_env_names)})")
        sampler_batch_B = new_sampler_batch_B

    env_metas = get_env_metas(*all_env_names)
    max_steps = max(
        [env_meta.spec.max_episode_steps for env_meta in env_metas])
    # number of transitions collected during each round of sampling will be
    # batch_T * batch_B = n_steps * n_envs

    print("Setting up sampler")
    if use_gpu:
        sampler_ctor = MuxGpuSampler
    else:
        sampler_ctor = MuxCpuSampler
    sampler = sampler_ctor(env_mux,
                           env_ctor_kwargs,
                           max_decorrelation_steps=max_steps,
                           TrajInfoCls=MILBenchTrajInfo,
                           batch_T=sampler_batch_T,
                           batch_B=sampler_batch_B)

    print("Setting up agent")
    # should be (H,W,C) even w/ frame stack
    # TODO: factor out getting max_steps, observation_space, action_space, etc.
    obs_space = env_metas[0].observation_space
    act_space = env_metas[0].action_space
    assert all(em.observation_space == obs_space for em in env_metas)
    assert all(em.action_space == act_space for em in env_metas)
    assert len(obs_space.shape) == 3, obs_space.shape
    in_chans = obs_space.shape[-1]
    n_actions = act_space.n  # categorical action space
    task_ids_and_demo_env_names = [
        (env_name[0], task_id)
        for env_name, (task_id,
                       _) in variant_groups.task_variant_by_name.items()
    ]
    model_ctor = MultiHeadPolicyNet
    model_kwargs = dict(in_chans=in_chans,
                        n_actions=n_actions,
                        env_ids_and_names=task_ids_and_demo_env_names)
    ppo_agent = CategoricalPgAgent(
        ModelCls=MuxTaskModelWrapper,
        model_kwargs=dict(
            model_ctor=model_ctor,
            # task_id=env_id,
            model_kwargs=model_kwargs))

    print("Setting up discriminator/reward model")
    # TODO: make all of these multi-task!
    raise NotImplementedError("need to make discirminator etc. multi-task")
    discriminator = MILBenchDiscriminator(
        in_chans=in_chans,
        act_dim=n_actions,
        use_all_chans=disc_all_frames,
        use_actions=disc_use_act,
    ).to(dev)
    reward_model = RewardModel(discriminator).to(dev)
    reward_evaluator = RewardEvaluator(
        reward_model,
        obs_dims=3,
        batch_size=disc_batch_size,
        normalise=True,
        # I think I had rewards in [0,0.01] in
        # the PPO run that I got to run with a
        # manually-defined reward.
        target_std=0.01)
    # TODO: figure out what pol_batch_size should be/do, and what relation it
    # should have with sampler batch size

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
    ppo_hyperparams = dict(
        learning_rate=ppo_lr,
        discount=ppo_gamma,
        entropy_loss_coeff=ppo_ent,  # was working at 0.003 and 0.001
        gae_lambda=ppo_lambda,
        ratio_clip=ppo_adv_clip,
        value_loss_coeff=1.0,
        clip_grad_norm=1.0,
        normalize_advantage=ppo_norm_adv,
    )
    if bc_loss:
        # TODO: make this configurable
        ppo_loader_mt = make_loader_mt(
            dataset_mt, max(16, min(64, sampler_batch_T * sampler_batch_B)))
    else:
        ppo_loader_mt = None
    ppo_algo = BCCustomRewardPPO(bc_loss_coeff=bc_loss,
                                 expert_traj_loader=ppo_loader_mt,
                                 true_reward_weight=danger_debug_reward_weight,
                                 **ppo_hyperparams)
    ppo_algo.set_reward_evaluator(reward_evaluator)

    print("Setting up optimiser")
    if disc_aug:
        print("Discriminator augmentations on")
        aug_model = MILBenchAugmentations(translate=True,
                                          rotate=True,
                                          noise=True)
    else:
        print("Discriminator augmentations off")
        aug_model = None
    gail_optim = GAILOptimiser(dataset_mt=dataset_mt,
                               discrim_model=discriminator,
                               buffer_num_samples=max(
                                   disc_batch_size, disc_replay_mult *
                                   sampler_batch_T * sampler_batch_B),
                               batch_size=disc_batch_size,
                               updates_per_itr=disc_up_per_iter,
                               dev=dev,
                               aug_model=aug_model,
                               lr=disc_lr)

    print("Setting up RL algorithm")
    # signature for arg: reward_model(obs_tensor, act_tensor) -> rewards
    runner = GAILMinibatchRl(
        seed=seed,
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

    def init_policy_cb(runner):
        """Callback which gets called once after Runner startup to save an
        initial policy model, and optionally load saved parameters."""
        # get state of newly-initalised model
        wrapped_model = runner.algo.agent.model
        assert wrapped_model is not None, "has ppo_agent been initalised?"
        unwrapped_model = wrapped_model.model

        if load_policy:
            print(f"Loading policy from '{load_policy}'")
            saved_model = load_state_dict_or_model(load_policy)
            saved_dict = saved_model.state_dict()
            unwrapped_model.load_state_dict(saved_dict)

        real_state = unwrapped_model.state_dict()

        # make a clone model so we can pickle it, and copy across weights
        policy_copy_mt = model_ctor(**model_kwargs).to('cpu')
        policy_copy_mt.load_state_dict(real_state)

        # save it here
        init_pol_snapshot_path = os.path.join(logger.get_snapshot_dir(),
                                              'full_model.pt')
        torch.save(policy_copy_mt, init_pol_snapshot_path)

    print("Training!")
    n_uniq_envs = len(all_env_names)
    with make_logger_ctx(out_dir,
                         "mtgail",
                         f"mt{n_uniq_envs}",
                         run_name,
                         snapshot_gap=snapshot_gap):
        torch.save(
            discriminator,
            os.path.join(logger.get_snapshot_dir(), 'full_discrim_model.pt'))
        # note that periodic snapshots get saved by GAILMiniBatchRl, thanks to
        # the overridden get_itr_snapshot() method
        runner.train(cb_startup=init_policy_cb)


if __name__ == '__main__':
    sane_click_init(main)
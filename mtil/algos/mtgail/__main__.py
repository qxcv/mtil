"""Main entry point for GAIL algorithm. Separated from actual implementation so
that parts of the implementation can be pickled."""

# 'import readline' is necessary to stop pdb.set_trace() from segfaulting in
# rl_initialize when importing readline. Problematic import is
# 'magical.saved_trajectories' (putting this import after that means
# the segfault still happens). Haven't had time to chase down.
import multiprocessing as mp
import os
import readline  # noqa: F401

import click
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.utils.logging import logger
import torch

from mtil.algos.mtgail.embedded_bc import BCCustomRewardPPO
from mtil.algos.mtgail.mtgail import (AETrainer, GAILMinibatchRl,
                                      GAILOptimiser, MILBenchDiscriminatorMT,
                                      RewardModel)
from mtil.augmentation import MILBenchAugmentations
from mtil.demos import get_demos_meta, make_loader_mt
from mtil.domain_transfer import BinaryDomainLossModule
from mtil.models import MultiHeadPolicyNet
from mtil.reward_injection_wrappers import RewardEvaluatorMT
from mtil.sample_mux import MuxTaskModelWrapper, make_mux_sampler
from mtil.utils.misc import (CPUListParamType, load_state_dict_or_model,
                             sample_cpu_list, sane_click_init, set_seeds)
from mtil.utils.rlpyt import get_policy_spec_magical, make_logger_ctx


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--add-preproc",
    default="LoRes4E",
    type=str,
    help="add preprocessor to the demos and test env (default: 'LoRes4E')")
@click.option("--gpu-idx", default=0, help="index of GPU to use")
@click.option("--cpu-list",
              default=None,
              type=CPUListParamType(),
              help="comma-separated list of CPUs to use")
@click.option("--seed", default=42, help="PRNG seed")
@click.option("--out-dir", default="scratch", help="dir for snapshots/logs")
@click.option(
    "--log-interval-steps",
    default=1e4,
    help="how many env transitions to take between writing log outputs")
# FIXME: rename these --ppo-batch-b and --ppo-batch-t instead, since that's
# really what they are.
# (also maybe turn default sampler_batch_B up? Seems too low for multitask.)
@click.option("-B",
              "--sampler-batch-envs",
              "sampler_batch_B",
              default=32,
              help="number of parallel envs to sample from")
@click.option("-T",
              "--sampler-time-steps",
              "sampler_batch_T",
              default=8,
              help="number of timesteps to advance each env when sampling")
@click.option("--disc-batch-size",
              default=24,
              help="batch size for discriminator training")
@click.option(
    "--disc-up-per-iter",
    default=12,  # surprisingly, benefits from MORE updates
    help="number of discriminator steps per RL step")
@click.option('--disc-replay-mult',
              type=int,
              default=4,
              help="number of past epochs worth of interaction to save in "
              "discriminator replay buffer")
@click.option("--total-n-steps",
              default=int(1e6),
              help="total number of steps to take in environment")
@click.option("--bc-loss", default=0.0, help="behavioural cloning loss coeff")
@click.option("--run-name",
              default=None,
              type=str,
              help="unique name for this run")
@click.option("--snapshot-gap", default=10, help="evals between snapshots")
@click.option("--load-policy",
              default=None,
              type=str,
              help="path to a policy snapshot to load (e.g. from MTBC)")
@click.option("--omit-noop/--no-omit-noop",
              default=False,
              help="omit demonstration (s,a) pairs whenever a is a noop")
@click.option(
    "--disc-aug",
    default="all",
    # TODO: add choices for this, like in mtbc/__main__.py
    help="choose augmentation for discriminator")
@click.option("--disc-al/--no-disc-al",
              default=False,
              help="use apprenticeship learning mode")
@click.option(
    "--disc-al-dim",
    default=16,
    # FIXME(sam): I have no fucking idea what this should be
    help="feature dimension for apprenticeship learning")
@click.option("--disc-al-nsamples",
              default=256,
              help="num data points to compute AL weights")
@click.option("--danger-debug-reward-weight",
              type=float,
              default=None,
              help="Replace some fraction of the IL reward with raw rewards "
              "from the env (DANGER!)")
@click.option("--danger-override-env-name",
              type=str,
              default=None,
              help="Do rollouts in this env instead of the demo one")
@click.option('--disc-lr', default=2e-5, help='discriminator learning rate')
@click.option('--disc-use-act/--no-disc-use-act',
              default=True,
              help='whether discriminator gets action inputs')
@click.option('--disc-all-frames/--no-disc-all-frames',
              default=True,
              help='whether discriminator gets full input frame stack')
@click.option('--disc-net-attn/--no-disc-net-attn',
              default=False,
              help='use attention for discriminator')
@click.option('--disc-use-bn/--no-disc-use-bn',
              default=True,
              help='should discriminator use batch norm?')
@click.option(
    "--ppo-aug",
    # TODO: add choices for this too (see --disc-aug note)
    default="none",
    help="augmentations to use for PPO (if any)")
@click.option("--ppo-use-bn/--no-ppo-use-bn",
              default=False,
              help="should policy/VF use batch norm?")
@click.option("--ppo-minibatches",
              default=4,
              help="number of minibatches to split each PPO batch into")
@click.option("--ppo-epochs",
              default=12,
              help="number of PPO 'epochs' to train for at each PPO update")
@click.option('--ppo-lr', default=6e-5, help='PPO learning rate')
@click.option('--ppo-gamma', default=0.8, help='PPO discount factor (gamma)')
@click.option('--ppo-lambda', default=0.8, help='PPO GAE lamdba')
@click.option('--ppo-ent', default=2.5e-5, help='entropy bonus for PPO')
@click.option('--ppo-adv-clip', default=0.01, help='PPO advantage clip ratio')
@click.option('--ppo-norm-adv/--no-ppo-norm-adv',
              default=False,
              help='whether to normalise PPO advantages')
@click.option("--transfer-variant",
              'transfer_variants',
              default=[],
              multiple=True,
              help="name of transfer env for co-training (can be repeated)")
@click.option("--transfer-pol-batch-weight",
              default=1.0,
              help="weight of transfer envs relative to demo envs when making "
              "trajectory sampler (>1 is more weight, <1 is less weight)")
@click.option("--transfer-disc-loss-weight",
              default=1e-1,
              help='transfer loss weight for disc. (with --transfer-variant)')
@click.option("--transfer-pol-loss-weight",
              default=0.05,
              help='transfer loss weight for disc. (with --transfer-variant)')
@click.option("--wgan/--no-wgan",
              default=False,
              help="use WGAN loss instead of GAN loss?")
@click.option("--disc-use-sn/--no-disc-use-sn",
              default=False,
              help="Use spectral norm for discriminator?")
@click.option("--disc-gp-weight",
              default=0.0,
              help="weight for discriminator gradient penalty")
@click.option("--disc-ae-pretrain-iters",
              default=0,
              help="num batches of autoencoder pretraining for discrim")
@click.option(
    "--transfer-disc-anneal/--no-transfer-disc-anneal",
    # TODO: make this also work for the policy
    default=False,
    help="anneal disc transfer loss from 0 to whatever "
    "--transfer-disc-weight is")
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
        cpu_list,
        snapshot_gap,
        load_policy,
        bc_loss,
        omit_noop,
        disc_replay_mult,
        disc_aug,
        ppo_aug,
        disc_use_bn,
        disc_net_attn,
        disc_use_sn,
        disc_gp_weight,
        disc_al,
        disc_al_dim,
        disc_al_nsamples,
        disc_ae_pretrain_iters,
        wgan,
        transfer_variants,
        transfer_disc_loss_weight,
        transfer_pol_loss_weight,
        transfer_disc_anneal,
        transfer_pol_batch_weight,
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
        ppo_use_bn,
        ppo_minibatches,
        ppo_epochs):
    # set up seeds & devices
    # TODO: also seed child envs, when rlpyt supports it
    set_seeds(seed)
    # 'spawn' is necessary to use GL envs in subprocesses. For whatever reason
    # they don't play nice after a fork. (But what about set_seeds() in
    # subprocesses? May need to hack CpuSampler and GpuSampler.)
    mp.set_start_method('spawn')
    use_gpu = gpu_idx is not None and torch.cuda.is_available()
    dev = torch.device(["cpu", f"cuda:{gpu_idx}"][use_gpu])
    if cpu_list is None:
        cpu_list = sample_cpu_list()
    # FIXME: I suspect current solution will set torch_num_threads suboptimally
    affinity = dict(cuda_idx=gpu_idx if use_gpu else None,
                    workers_cpus=cpu_list)
    print(f"Using device {dev}, seed {seed}, affinity {affinity}")

    # register original envs
    import magical
    magical.register_envs()

    if danger_override_env_name:
        raise NotImplementedError(
            "haven't re-implemeneted env name override for multi-task GAIL")

    demos_metas_dict = get_demos_meta(demo_paths=demos,
                                      omit_noop=omit_noop,
                                      transfer_variants=transfer_variants,
                                      preproc_name=add_preproc)
    dataset_mt = demos_metas_dict['dataset_mt']
    variant_groups = demos_metas_dict['variant_groups']
    env_metas = demos_metas_dict['env_metas']
    task_ids_and_demo_env_names = demos_metas_dict[
        'task_ids_and_demo_env_names']
    task_var_weights = {
        (task, variant): 1.0 if variant == 0 else transfer_pol_batch_weight
        for task, variant in variant_groups.env_name_by_task_variant
    }
    sampler, sampler_batch_B = make_mux_sampler(
        variant_groups=variant_groups,
        task_var_weights=task_var_weights,
        env_metas=env_metas,
        use_gpu=use_gpu,
        num_demo_sources=0,  # not important for now
        batch_B=sampler_batch_B,
        batch_T=sampler_batch_T)

    policy_kwargs = {
        'use_bn': ppo_use_bn,
        'env_ids_and_names': task_ids_and_demo_env_names,
        **get_policy_spec_magical(env_metas),
    }
    policy_ctor = MultiHeadPolicyNet
    ppo_agent = CategoricalPgAgent(ModelCls=MuxTaskModelWrapper,
                                   model_kwargs=dict(
                                       model_ctor=policy_ctor,
                                       model_kwargs=policy_kwargs))

    print("Setting up discriminator/reward model")
    disc_fc_dim = 256
    disc_final_feats_dim = disc_al_dim if disc_al else disc_fc_dim
    discriminator_mt = MILBenchDiscriminatorMT(
        task_ids_and_names=task_ids_and_demo_env_names,
        in_chans=policy_kwargs['in_chans'],
        act_dim=policy_kwargs['n_actions'],
        use_all_chans=disc_all_frames,
        use_actions=disc_use_act,
        # can supply any argument that goes to MILBenchFeatureNetwork (e.g.
        # dropout, use_bn, width, etc.)
        attention=disc_net_attn,
        use_bn=disc_use_bn,
        use_sn=disc_use_sn,
        fc_dim=disc_fc_dim,
        final_feats_dim=disc_final_feats_dim,
    ).to(dev)

    if (not transfer_variants
            and (transfer_disc_loss_weight or transfer_pol_loss_weight)):
        print("No xfer variants supplied, setting xfer disc loss term to zero")
        transfer_disc_loss_weight = 0.0
        transfer_pol_loss_weight = 0.0
    if transfer_pol_loss_weight > 0:
        assert transfer_disc_loss_weight > 0
    if transfer_variants and transfer_disc_loss_weight:
        xfer_adv_module = BinaryDomainLossModule(
            discriminator_mt.ret_feats_dim).to(dev)
    else:
        xfer_adv_module = None

    reward_model_mt = RewardModel(
        discriminator_mt,
        xfer_adv_module,
        transfer_pol_loss_weight,
        # In apprenticeship learning we can just pass
        # the model outputs straight through, just
        # like in WGAN.
        use_wgan=wgan or disc_al).to(dev)
    reward_evaluator_mt = RewardEvaluatorMT(
        task_ids_and_names=task_ids_and_demo_env_names,
        reward_model=reward_model_mt,
        obs_dims=3,
        batch_size=disc_batch_size,
        normalise=True,
        # I think I had rewards in [0,0.01] in
        # the PPO run that I got to run with a
        # manually-defined reward.
        target_std=0.01)

    ppo_hyperparams = dict(
        learning_rate=ppo_lr,
        discount=ppo_gamma,
        entropy_loss_coeff=ppo_ent,  # was working at 0.003 and 0.001
        gae_lambda=ppo_lambda,
        ratio_clip=ppo_adv_clip,
        minibatches=ppo_minibatches,
        epochs=ppo_epochs,
        value_loss_coeff=1.0,
        clip_grad_norm=1.0,
        normalize_advantage=ppo_norm_adv,
    )
    if bc_loss:
        # TODO: make this batch size configurable
        ppo_loader_mt = make_loader_mt(
            dataset_mt, max(16, min(64, sampler_batch_T * sampler_batch_B)))
    else:
        ppo_loader_mt = None

    # FIXME: abstract code for constructing augmentation model from presets
    try:
        ppo_aug_opts = MILBenchAugmentations.PRESETS[ppo_aug]
    except KeyError:
        raise ValueError(f"unsupported augmentation mode '{ppo_aug}'")
    if ppo_aug_opts:
        print("Policy augmentations:", ", ".join(ppo_aug_opts))
        ppo_aug_model = MILBenchAugmentations(
            **{k: True
               for k in ppo_aug_opts}).to(dev)
    else:
        print("No policy augmentations")
        ppo_aug_model = None

    ppo_algo = BCCustomRewardPPO(bc_loss_coeff=bc_loss,
                                 expert_traj_loader=ppo_loader_mt,
                                 true_reward_weight=danger_debug_reward_weight,
                                 aug_model=ppo_aug_model,
                                 **ppo_hyperparams)
    ppo_algo.set_reward_evaluator(reward_evaluator_mt)

    print("Setting up optimiser")
    try:
        aug_opts = MILBenchAugmentations.PRESETS[disc_aug]
    except KeyError:
        raise ValueError(f"unsupported augmentation mode '{disc_aug}'")
    if aug_opts:
        print("Discriminator augmentations:", ", ".join(aug_opts))
        aug_model = MILBenchAugmentations(**{k: True for k in aug_opts}) \
            .to(dev)
    else:
        print("No discriminator augmentations")
        aug_model = None
    gail_optim = GAILOptimiser(
        dataset_mt=dataset_mt,
        discrim_model=discriminator_mt,
        buffer_num_samples=max(
            disc_batch_size,
            disc_replay_mult * sampler_batch_T * sampler_batch_B),
        batch_size=disc_batch_size,
        updates_per_itr=disc_up_per_iter,
        gp_weight=disc_gp_weight,
        dev=dev,
        aug_model=aug_model,
        lr=disc_lr,
        xfer_adv_weight=transfer_disc_loss_weight,
        xfer_adv_anneal=transfer_disc_anneal,
        xfer_adv_module=xfer_adv_module,
        final_layer_only_mode=disc_al,
        final_layer_only_mode_n_samples=disc_al_nsamples,
        use_wgan=wgan)

    if disc_ae_pretrain_iters:
        # FIXME(sam): pass n_acts, obs_chans, lr to AETrainer
        ae_trainer = AETrainer(discriminator=discriminator_mt,
                               disc_out_size=disc_final_feats_dim,
                               data_batch_iter=gail_optim.expert_batch_iter,
                               dev=dev)

    print("Setting up RL algorithm")
    # signature for arg: reward_model(obs_tensor, act_tensor) -> rewards
    runner = GAILMinibatchRl(
        seed=seed,
        gail_optim=gail_optim,
        variant_groups=variant_groups,
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

    # TODO: factor out this callback
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
        policy_copy_mt = policy_ctor(**policy_kwargs).to('cpu')
        policy_copy_mt.load_state_dict(real_state)

        # save it here
        init_pol_snapshot_path = os.path.join(logger.get_snapshot_dir(),
                                              'full_model.pt')
        torch.save(policy_copy_mt, init_pol_snapshot_path)

    print("Training!")
    n_uniq_envs = variant_groups.num_tasks
    log_params = {
        'add_preproc': add_preproc,
        'seed': seed,
        'sampler_batch_T': sampler_batch_T,
        'sampler_batch_B': sampler_batch_B,
        'disc_batch_size': disc_batch_size,
        'disc_up_per_iter': disc_up_per_iter,
        'total_n_steps': total_n_steps,
        'bc_loss': bc_loss,
        'omit_noop': omit_noop,
        'disc_aug': disc_aug,
        'danger_debug_reward_weight': danger_debug_reward_weight,
        'disc_lr': disc_lr,
        'disc_use_act': disc_use_act,
        'disc_all_frames': disc_all_frames,
        'disc_net_attn': disc_net_attn,
        'disc_use_bn': disc_use_bn,
        'ppo_lr': ppo_lr,
        'ppo_gamma': ppo_gamma,
        'ppo_lambda': ppo_lambda,
        'ppo_ent': ppo_ent,
        'ppo_adv_clip': ppo_adv_clip,
        'ppo_norm_adv': ppo_norm_adv,
        'transfer_variants': transfer_variants,
        'transfer_pol_batch_weight': transfer_pol_batch_weight,
        'transfer_pol_loss_weight': transfer_pol_loss_weight,
        'transfer_disc_loss_weight': transfer_disc_loss_weight,
        'transfer_disc_anneal': transfer_disc_anneal,
        'ndemos': len(demos),
        'n_uniq_envs': n_uniq_envs,
    }
    with make_logger_ctx(out_dir,
                         "mtgail",
                         f"mt{n_uniq_envs}",
                         run_name,
                         snapshot_gap=snapshot_gap,
                         log_params=log_params):
        torch.save(
            discriminator_mt,
            os.path.join(logger.get_snapshot_dir(), 'full_discrim_model.pt'))

        if disc_ae_pretrain_iters:
            # FIXME(sam): come up with a better solution for creating these
            # montages (can I do it regularly? Should I put them somewhere
            # other than the snapshot dir?).
            ae_trainer.make_montage(
                os.path.join(logger.get_snapshot_dir(), 'ae-before.png'))
            ae_trainer.do_full_training(disc_ae_pretrain_iters)
            ae_trainer.make_montage(
                os.path.join(logger.get_snapshot_dir(), 'ae-after.png'))

        # note that periodic snapshots get saved by GAILMiniBatchRl, thanks to
        # the overridden get_itr_snapshot() method
        runner.train(cb_startup=init_policy_cb)


if __name__ == '__main__':
    sane_click_init(main)

import os

import click
import gym
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
import numpy as np
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger
import torch

from mtil.algos.bc import eval_model
from mtil.algos.mtbc.mtbc import (FixedTaskModelWrapper, MultiHeadPolicyNet,
                                  copy_model_into_sampler,
                                  do_epoch_training_mt, make_env_tag)
from mtil.common import (MILBenchGymEnv, make_logger_ctx, set_seeds,
                         trajectories_to_loader_mt)


@click.group()
def cli():
    pass


# TODO: abstract all these options out into a common set of options, possibly
# by using Sacred
@cli.command()
@click.option(
    "--add-preproc",
    default="LoResStack",
    type=str,
    help="add preprocessor to the demos and test env (default: 'LoResStack')")
@click.option("--use-gpu/--no-use-gpu", default=False, help="use GPU")
@click.option("--gpu-idx", default=0, help="index of GPU to use")
@click.option("--seed", default=42, help="PRNG seed")
@click.option("--batch-size", default=32, help="batch size")
@click.option("--epochs", default=100, help="epochs of training to perform")
@click.option("--out-dir", default="scratch", help="dir for snapshots/logs")
@click.option("--eval-n-traj",
              default=10,
              help="number of trajectories to roll out on each evaluation")
@click.option("--run-name",
              default=None,
              type=str,
              help="unique name for this run")
# set this to some big value if training on perceptron or something
@click.option(
    "--passes-per-eval",
    default=1,
    help="num training passes through full dataset between evaluations")
@click.argument("demos", nargs=-1, required=True)
def main(demos, use_gpu, add_preproc, seed, batch_size, epochs, out_dir,
         run_name, gpu_idx, eval_n_traj, passes_per_eval):
    # TODO: abstract this setup code (roughly: everything up to
    # 'trajectories_to_loader()') so that I don't have to keep rewriting it for
    # every IL method.

    # set up seeds & devices
    set_seeds(seed)
    use_gpu = use_gpu and torch.cuda.is_available()
    dev = torch.device(["cpu", f"cuda:{gpu_idx}"][use_gpu])
    print(f"Using device {dev}, seed {seed}")

    # register original envs
    import milbench
    milbench.register_envs()

    # load demos (this code copied from bc.py in original baselines)
    demo_dicts = load_demos(demos)
    orig_env_names = [d['env_name'] for d in demo_dicts]
    orig_names_uniq = sorted(set(orig_env_names))
    if add_preproc:
        env_names = [
            splice_in_preproc_name(orig_env_name, add_preproc)
            for orig_env_name in orig_env_names
        ]
        print(f"Splicing preprocessor '{add_preproc}' into environments "
              f"{orig_names_uniq}. New names are {sorted(set(env_names))}")
    else:
        env_names = orig_env_names
    # pair of (original name, name with preprocessor spliced in)
    name_pairs = sorted(set(zip(orig_env_names, env_names)))
    demo_trajs_by_env = {
        env_name: [
            demo_dict['trajectory'] for demo_dict in demo_dicts
            if demo_dict['env_name'] == orig_env_name
        ]
        for orig_env_name, env_name in name_pairs
    }
    assert sum(map(len, demo_trajs_by_env.values())) == len(demo_dicts)
    if add_preproc:
        demo_trajs_by_env = {
            env_name:
            preprocess_demos_with_wrapper(demo_trajs_by_env[env_name],
                                          orig_env_name, add_preproc)
            for orig_env_name, env_name in name_pairs
        }
    loader_mt, env_name_to_id, env_id_to_name = trajectories_to_loader_mt(
        demo_trajs_by_env, batch_size)
    dataset_len = len(loader_mt)
    env_ids_and_names = [(name, env_name_to_id[name])
                         for _, name in name_pairs]

    # model kwargs will be filled in when we start our first env
    model_kwargs = None
    model_ctor = MultiHeadPolicyNet
    env_ctor = MILBenchGymEnv

    samplers = []
    agents = []
    for orig_env_name, env_name in name_pairs:
        env_ctor_kwargs = dict(env_name=env_name)
        env = gym.make(env_name)
        max_steps = env.spec.max_episode_steps

        # set model kwargs if necessary
        if model_kwargs is None:
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                in_chans = obs_shape[-1]
            else:
                # frame stacking
                in_chans = obs_shape[-1] * obs_shape[0]
            n_actions = env.action_space.n

            model_kwargs = {
                'env_ids_and_names': env_ids_and_names,
                'in_chans': in_chans,
                'n_actions': n_actions,
            }

        env_sampler = SerialSampler(env_ctor,
                                    env_ctor_kwargs,
                                    batch_T=max_steps,
                                    max_decorrelation_steps=max_steps,
                                    batch_B=min(eval_n_traj, batch_size))
        env_agent = CategoricalPgAgent(ModelCls=FixedTaskModelWrapper,
                                       model_kwargs=dict(
                                           model_ctor=model_ctor,
                                           model_kwargs=model_kwargs,
                                           task_id=env_name_to_id[env_name]))
        env_sampler.initialize(env_agent, seed=np.random.randint(1 << 31))
        env_agent.to_device(dev.index if use_gpu else None)

        samplers.append(env_sampler)
        agents.append(env_agent)

    model_mt = MultiHeadPolicyNet(**model_kwargs).to(dev)
    opt_mt = torch.optim.Adam(model_mt.parameters(), lr=3e-4)

    n_uniq_envs = len(orig_names_uniq)
    with make_logger_ctx(out_dir, "mtbc", f"mt{n_uniq_envs}", run_name):
        # initial save
        torch.save(model_mt,
                   os.path.join(logger.get_snapshot_dir(), 'full_model.pt'))

        # train for a while
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs} ({dataset_len} batches "
                  f"* {passes_per_eval} passes between evaluations)")

            loss_ewma, losses, per_task_losses = do_epoch_training_mt(
                loader_mt, model_mt, opt_mt, dev, passes_per_eval)

            print(f"Evaluating {eval_n_traj} trajectories on "
                  f"{len(name_pairs)} envs")
            record_misc_calls = []
            for (orig_env_name,
                 env_name), sampler in zip(name_pairs, samplers):
                copy_model_into_sampler(model_mt, sampler)
                model_mt.eval()
                scores = eval_model(sampler, n_traj=eval_n_traj)
                tag = make_env_tag(orig_env_name)
                logger.record_tabular_misc_stat("Score%s" % tag, scores)
                env_id = env_name_to_id[env_name]
                env_losses = per_task_losses.get(env_id, [])
                record_misc_calls.append((f"Loss{tag}", env_losses))
            # we record score AFTER loss so that losses are all in one place,
            # and scores are all in another
            for args in record_misc_calls:
                logger.record_tabular_misc_stat(*args)

            # finish logging for this epoch
            logger.record_tabular("Epoch", epoch)
            logger.record_tabular("LossEWMA", loss_ewma)
            logger.record_tabular_misc_stat("Loss", losses)
            logger.dump_tabular()
            logger.save_itr_params(
                epoch, {
                    'model_state': model_mt.state_dict(),
                    'opt_state': opt_mt.state_dict(),
                })


if __name__ == '__main__':
    main()

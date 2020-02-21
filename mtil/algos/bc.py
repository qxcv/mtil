"""Single-task Behavioural Cloning (BC)."""
import os

import click
import gym
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
import numpy as np
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter
import torch
from torch import nn
import torch.nn.functional as F

from mtil.common import (AgentModelWrapper, MILBenchFeatureNetwork,
                         MILBenchGymEnv, MILBenchPreprocLayer, make_logger_ctx,
                         set_seeds, trajectories_to_loader)


class MILBenchPolicyNet(nn.Module):
    """Convolutional policy network that yields some action logits. Note this
    network is specific to MILBench (b/c of preproc layer and action count)."""
    def __init__(self,
                 in_chans=3,
                 n_actions=3 * 3 * 2,
                 ActivationCls=torch.nn.ReLU):
        super().__init__()
        # this asserts that input is 128x128, and ensures that it is in
        # [N,C,H,W] format with float32 values in [-1,1].
        self.preproc = MILBenchPreprocLayer()
        self.feature_extractor = MILBenchFeatureNetwork(
            in_chans=in_chans, ActivationCls=ActivationCls)
        self.logit_generator = nn.Sequential(
            nn.Linear(1024, 256),
            ActivationCls(),
            # now: flat 256-elem vector
            nn.Linear(256, 256),
            ActivationCls(),
            # now: flat 256-elem vector
            nn.Linear(256, n_actions),
            # now: n_actions-dimensional logit vector
        )

    def forward(self, x):
        preproc = self.preproc(x)
        features = self.feature_extractor(preproc)
        logits = self.logit_generator(features)
        return logits


def eval_model(sampler, n_traj=10):
    scores = []
    while len(scores) < n_traj:
        # can't see an obvious purpose to the 'itr' argument, so setting it to
        # None
        samples_pyt, _ = sampler.obtain_samples(None)
        eval_scores = samples_pyt.env.env_info.eval_score
        dones = samples_pyt.env.done
        done_scores = eval_scores.flatten()[dones.flatten()]
        scores.extend(done_scores)
    return scores


def do_epoch_training(loader, model, opt, dev):
    # TODO: replace this with do_epoch_training_mt from mtbc.py

    # @torch.jit.script
    def do_loss_forward_back(obs_batch, acts_batch):
        logits_flat = model(obs_batch)
        loss = F.cross_entropy(logits_flat, acts_batch.long())
        loss.backward()
        return loss.item()

    # make sure we're in train mode
    model.train()

    # for logging
    loss_ewma = None
    losses = []
    progress = ProgBarCounter(len(loader))
    for batches_done, (obs_batch, acts_batch) \
            in enumerate(loader, start=1):
        # copy to GPU
        obs_batch = obs_batch.to(dev)
        acts_batch = acts_batch.to(dev)

        # compute loss & take opt step
        opt.zero_grad()
        loss = do_loss_forward_back(obs_batch, acts_batch)
        opt.step()

        # for logging
        progress.update(batches_done)
        f_loss = loss
        loss_ewma = f_loss if loss_ewma is None \
            else 0.9 * loss_ewma + 0.1 * f_loss
        losses.append(f_loss)
    progress.stop()

    return loss_ewma, losses


def do_epoch_eval(model, sampler, fake_agent_model, eval_n_traj):
    # end-of-epoch evaluation
    model.eval()
    # TODO: change this so that it just removes the irrelevant key prefix from
    # the normal model state dict. Too much messing around otherwise.
    sampler.agent.load_state_dict(fake_agent_model.state_dict())
    scores = eval_model(sampler, n_traj=eval_n_traj)
    return scores


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
@click.argument("demos", nargs=-1, required=True)
def main(demos, use_gpu, add_preproc, seed, batch_size, epochs, out_dir,
         run_name, gpu_idx, eval_n_traj):
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
    all_names = set(d['env_name'] for d in demo_dicts)
    if len(all_names) != 1:
        raise ValueError(f"Supplied demos seem to come from {len(all_names)} "
                         f"envs, rather than 1. Names: {sorted(all_names)}")
    orig_env_name = next(iter(all_names))
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
    loader = trajectories_to_loader(demo_trajs, batch_size)

    # local copy of Gym env, w/ args to create equivalent env in the sampler
    env_ctor = MILBenchGymEnv
    env_ctor_kwargs = dict(env_name=env_name)
    env = gym.make(env_name)
    max_steps = env.spec.max_episode_steps

    # set up model & optimiser
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3:
        in_chans = obs_shape[-1]
    else:
        # frame stacking
        in_chans = obs_shape[-1] * obs_shape[0]
    model_ctor = MILBenchPolicyNet
    model_kwargs = dict(in_chans=in_chans)
    model = model_ctor(**model_kwargs) \
        .to(dev)
    # this is for syncing up weights appropriately
    fake_agent_model = AgentModelWrapper(None, None, model).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # set up sampler for evaluation (TODO: consider setting up complicated
    # parallel GPU sampler)
    sampler = SerialSampler(env_ctor,
                            env_ctor_kwargs,
                            batch_T=max_steps,
                            max_decorrelation_steps=max_steps,
                            batch_B=min(eval_n_traj, batch_size))
    agent = CategoricalPgAgent(ModelCls=AgentModelWrapper,
                               model_kwargs=dict(model_ctor=model_ctor,
                                                 model_kwargs=model_kwargs))
    sampler.initialize(agent, seed=np.random.randint(1 << 31))
    agent.to_device(dev.index if use_gpu else None)

    with make_logger_ctx(out_dir, "bc", orig_env_name, run_name):
        # save full model initially so we have something to put our saved
        # parameters in
        torch.save(model,
                   os.path.join(logger.get_snapshot_dir(), 'full_model.pt'))

        # train for a while
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs} ({len(loader)} batches)")

            loss_ewma, losses = do_epoch_training(loader, model, opt, dev)

            print(f"Evaluating {eval_n_traj} trajectories")
            scores = do_epoch_eval(model, sampler, fake_agent_model,
                                   eval_n_traj)
            logger.record_tabular_misc_stat("Score", scores)

            # finish logging for this epoch
            logger.record_tabular("Epoch", epoch)
            logger.record_tabular("LossEWMA", loss_ewma)
            logger.record_tabular_misc_stat("Loss", losses)
            logger.dump_tabular()
            logger.save_itr_params(epoch, {
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
            })


if __name__ == '__main__':
    main()

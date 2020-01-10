"""Single-task behavioural cloning (BC)."""

import datetime
import os
import uuid

import click
import gym
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
import numpy as np
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import context as log_ctx
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data

from mtil.common import MILBenchPolicyNet, VanillaGymEnv, set_seeds


class AgentModelWrapper(nn.Module):
    """Wraps a normal (observation -> logits) feedforward network so that (1)
    it deals gracefully with the variable-dimensional inputs that the rlpyt
    sampler gives it, (2) it produces action probabilities instead of logits,
    and (3) it produces some value 'values' to keep CategoricalPgAgent
    happy."""
    def __init__(self, model_ctor, model_kwargs, model=None):
        super().__init__()
        if model is not None:
            self.model = model
        else:
            self.model = model_ctor(**model_kwargs)

    def forward(self, obs, prev_act, prev_rew):
        # copied from AtariFfModel, then modified to match own situation
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        logits = self.model(obs.view(T * B, *img_shape))
        pi = F.softmax(logits, dim=-1)
        # fake values (BC doesn't use them)
        v = torch.zeros((T * B, ), device=pi.device, dtype=pi.dtype)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v


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


def trajectories_to_torch(demo_trajs, batch_size):
    """Re-format demonstration trajectories as a Torch DataLoader."""
    # convert dataset to Torch (always stored on CPU; we'll move one batch at a
    # time to the GPU)
    cpu_dev = torch.device("cpu")
    all_obs = torch.cat([
        torch.as_tensor(traj.obs[:-1], device=cpu_dev) for traj in demo_trajs
    ])
    all_acts = torch.cat(
        [torch.as_tensor(traj.acts, device=cpu_dev) for traj in demo_trajs])
    dataset = data.TensorDataset(all_obs, all_acts)
    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             pin_memory=True,
                             shuffle=True,
                             drop_last=True)
    return loader


def make_unique_run_name(orig_env_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    unique_suff = uuid.uuid4().hex[-6:]
    return f"bc-{orig_env_name}-{timestamp}-{unique_suff}"


def make_logger_ctx(out_dir, orig_env_name, custom_run_name=None):
    # for logging & model-saving
    if custom_run_name is None:
        run_name = make_unique_run_name(orig_env_name)
    else:
        run_name = custom_run_name
    logger.set_snapshot_gap(10)
    log_dir = os.path.abspath(out_dir)
    # this is irrelevant so long as it's a prefix of log_dir
    log_ctx.LOG_DIR = log_dir
    os.makedirs(out_dir, exist_ok=True)
    return log_ctx.logger_context(out_dir,
                                  run_ID=run_name,
                                  name="mtil",
                                  snapshot_mode="gap")


def do_epoch_training(loader, model, opt, dev):
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
    loader = trajectories_to_torch(demo_trajs, batch_size)

    # local copy of Gym env, w/ args to create equivalent env in the sampler
    env_ctor = VanillaGymEnv
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
                            max_decorrelation_steps=0,
                            batch_B=min(eval_n_traj, batch_size))
    agent = CategoricalPgAgent(ModelCls=AgentModelWrapper,
                               model_kwargs=dict(model_ctor=model_ctor,
                                                 model_kwargs=model_kwargs))
    sampler.initialize(agent, seed=np.random.randint(1 << 31))
    agent.to_device(dev.index if use_gpu else None)

    with make_logger_ctx(out_dir, orig_env_name, run_name):
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

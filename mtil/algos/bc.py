"""Single-task behavioural cloning (BC)."""

import datetime
import os
import uuid

# from rlpyt.samplers.serial.sampler import SerialSampler
import click
import gym
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
from rlpyt.utils.logging import logger
from rlpyt.utils.logging import context as log_ctx
from rlpyt.utils.prog_bar import ProgBarCounter
import torch
import torch.nn.functional as F
from torch.utils import data

from mtil.common import MILBenchPolicyNet, flat_to_md_action, md_to_flat_action


class BC:
    def __init__(self, env):
        pass

    def train(self, n_epochs=100):
        pass


def eval_model(env, model, n_traj=10, dev=None):
    scores = []
    prog_bar = ProgBarCounter(n_traj)
    dev = dev or next(model.parameters()).device
    for traj_num in range(1, n_traj + 1):
        obs = env.reset()
        done = False
        while not done:
            th_obs = torch.as_tensor(obs[None], device=dev)
            th_act_logits, = model(th_obs)
            th_best_logit = torch.argmax(th_act_logits, keepdim=True)
            th_action = flat_to_md_action(th_best_logit, env.action_space.nvec)
            action = th_action.detach().cpu().numpy()
            obs, rew, done, info = env.step(action)
        prog_bar.update(traj_num)
        scores.append(info['eval_score'])
    prog_bar.stop()
    logger.record_tabular_misc_stat("Score", scores)


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
@click.option("--seed", default=42, help="PRNG seed")
@click.option("--batch-size", default=32, help="batch size")
@click.option("--epochs", default=100, help="epochs of training to perform")
@click.option("--out-dir", default="scratch", help="dir for snapshots/logs")
@click.option("--run-name",
              default=None,
              type=str,
              help="unique name for this run")
@click.argument("demos", nargs=-1, required=True)
def main(demos, use_gpu, add_preproc, seed, batch_size, epochs, out_dir,
         run_name):
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
    env = gym.make(env_name)

    # set up torch
    torch.manual_seed(seed)
    use_gpu = use_gpu and torch.cuda.is_available()
    cpu_dev = torch.device("cpu")
    dev = torch.device(["cpu", "cuda"][use_gpu])
    print(f"Using device {dev}, seed {seed}")

    # convert dataset to Torch (always stored on CPU; we'll move one batch at a
    # time to the GPU)
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

    # set up model & optimiser
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3:
        in_chans = obs_shape[-1]
    else:
        # frame stacking
        in_chans = obs_shape[-1] * obs_shape[0]
    model = MILBenchPolicyNet(in_chans=in_chans) \
        .to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # set up sampler for evaluation (TODO: consider setting up complicated
    # parallel GPU sampler)
    # sampler = SerialSampler()
    # sampler.initialiaze()

    # some junk necessary to support TorchScript for forward-/back-prop
    nvec = torch.as_tensor(env.action_space.nvec, device=dev)

    # @torch.jit.script
    def do_loss_forward_back(obs_batch, acts_batch, nvec):
        logits_flat = model(obs_batch)
        labels_flat = md_to_flat_action(acts_batch, nvec)
        loss = F.cross_entropy(logits_flat, labels_flat)
        loss.backward()
        return loss.item()

    # for logging & model-saving
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        unique_suff = uuid.uuid4().hex[-6:]
        run_name = f"bc-{orig_env_name}-{timestamp}-{unique_suff}"
    logger.set_snapshot_gap(10)
    log_dir = os.path.abspath(out_dir)
    # this is irrelevant so long as it's a prefix of log_dir
    log_ctx.LOG_DIR = log_dir
    os.makedirs(out_dir, exist_ok=True)
    with log_ctx.logger_context(out_dir,
                                run_ID=run_name,
                                name="mtil",
                                snapshot_mode="gap"):
        # save full model initially so we have something to put our saved
        # parameters in
        torch.save(model,
                   os.path.join(logger.get_snapshot_dir(), 'full_model.pt'))

        # train for a while
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs} ({len(loader)} batches)")

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
                loss = do_loss_forward_back(obs_batch, acts_batch, nvec)
                opt.step()

                # for logging
                progress.update(batches_done)
                f_loss = loss
                loss_ewma = f_loss if loss_ewma is None \
                    else 0.9 * loss_ewma + 0.1 * f_loss
                losses.append(f_loss)
            progress.stop()

            # end-of-epoch evaluation
            eval_n_traj = 10
            print(f"Evaluating {eval_n_traj} trajectories")
            model.eval()
            eval_model(env, model, n_traj=eval_n_traj)

            # finish logging for this epoch
            logger.record_tabular("Epoch", epoch)
            logger.record_tabular("LossEWMA", loss_ewma)
            logger.record_tabular_misc_stat("Loss", losses)
            logger.dump_tabular()
            logger.save_itr_params(epoch, {
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
            })

            # TODO: three items:
            # - (1) Try doing rollouts with rlpyt to evaluate model.
            # - [DONE] (2) Integrate with rlpyt's logging stuff (needs viskit).
            # - [DONE] (3) Save the model after training.
            # - (4) Refactor everything :)


if __name__ == '__main__':
    main()

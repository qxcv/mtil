"""Multi-Task Behavioural Cloning (MTBC). Train on several environments, with
one "head" per environment. For now this only works with MILBench environments,
so it assumes that all environments have the same input & output spaces."""

import os
import re

import click
import gym
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
import numpy as np
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import nn
import torch.nn.functional as F

from mtil.algos.bc import AgentModelWrapper, eval_model
from mtil.common import (MILBenchFeatureNetwork, MILBenchGymEnv,
                         MILBenchPreprocLayer, make_logger_ctx, set_seeds,
                         trajectories_to_loader_mt)


class MultiHeadPolicyNet(nn.Module):
    """Like MILBenchPolicyNet in bc.py, but for multitask policies with one
    head per environment."""
    def __init__(self,
                 env_ids_and_names,
                 in_chans=3,
                 n_actions=3 * 3 * 2,
                 fc_dim=256,
                 ActivationCls=torch.nn.ReLU):
        super().__init__()
        self.preproc = MILBenchPreprocLayer()
        self.feature_extractor = MILBenchFeatureNetwork(
            in_chans=in_chans, ActivationCls=ActivationCls)
        self.fc_postproc = nn.Sequential(
            nn.Linear(1024, 256),
            ActivationCls(),
            # now: flat 256-elem vector
            nn.Linear(256, fc_dim),
            ActivationCls(),
            # now: flat 256-elem vector
        )
        # these "embeddings" are actually affine transformation parameters,
        # with weight matrix at the beginning and bias at the end
        embedding_size = fc_dim * n_actions + n_actions
        self.task_embeddings = nn.Embedding(len(env_ids_and_names),
                                            embedding_size)
        # now: n_actions-dimensional logit vector

        # save env IDs and names so that we know how to reconstruct, as well as
        # fc_dim and n_actionsso that we can do reshaping
        self.env_ids_and_names = sorted(env_ids_and_names)
        self.fc_dim = fc_dim
        self.n_actions = n_actions

    def forward(self, obs, task_ids):
        preproc = self.preproc(obs)
        features = self.feature_extractor(preproc)
        fc_features = self.fc_postproc(features)

        # slightly messy: have to reshape "embeddings" into weight matrices &
        # affine parameters, then apply them like a batch of different affine
        # layers
        embeddings = self.task_embeddings(task_ids)
        bias_begin = self.fc_dim * self.n_actions
        matrices_flat = embeddings[..., :bias_begin]
        mats_shape = matrices_flat.shape[:-1] + (self.n_actions, self.fc_dim)
        matrices = matrices_flat.view(mats_shape)
        biases = embeddings[..., bias_begin:]
        # TODO: make sure this does even remotely the right thing
        bc_fc_features = fc_features[..., None]
        mm_result = torch.squeeze(matrices @ bc_fc_features, dim=-1)
        assert mm_result.shape == biases.shape, (mm_result.shape, biases.shape)
        logits = mm_result + biases

        expected_shape = task_ids.shape + (self.n_actions, )
        assert logits.shape == expected_shape, \
            f"expected logits to be shape {expected_shape}, but got " \
            f"shape {logits.shape} (might be reshaping bug)"

        return logits


class FixedTaskModelWrapper(AgentModelWrapper):
    """Like AgentModelWrapper, but for multi-head policies that expect task IDs
    as input. Assumes that it is only ever getting applied to one task,
    identified by a given integer `task_id`. Good when you have one sampler per
    task."""
    def __init__(self, task_id, **kwargs):
        # This should be given 'model_ctor', 'model_kwargs', and optionally
        # 'model' kwargs.
        super().__init__(**kwargs)
        self.task_id = task_id

    def forward(self, obs, prev_act, prev_rew):
        # similar to AgentModelWrapper.forward(), but also constructs task IDs
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        task_ids = torch.full((T * B, ), self.task_id, dtype=torch.long) \
            .to(obs.device)
        logits = self.model(obs.view(T * B, *img_shape), task_ids)
        pi = F.softmax(logits, dim=-1)
        v = torch.zeros((T * B, ), device=pi.device, dtype=pi.dtype)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v


def copy_model_into_sampler(model, sampler, prefix='model'):
    """Update the `.agent` inside `sampler` so that it contains weights from
    `model`. Should call this before doing evaluation rollouts between epochs
    of training."""
    state_dict = model.state_dict()
    assert hasattr(sampler.agent, prefix)
    updated_state_dict = {
        prefix + '.' + key: value
        for key, value in state_dict.items()
    }
    sampler.agent.load_state_dict(updated_state_dict)


def do_epoch_training_mt(loader, model, opt, dev):
    # @torch.jit.script
    def do_loss_forward_back(task_ids_batch, obs_batch, acts_batch):
        logits_flat = model(obs_batch, task_ids_batch)
        loss = F.cross_entropy(logits_flat, acts_batch.long())
        loss.backward()
        return loss.item()

    # make sure we're in train mode
    model.train()

    # for logging
    loss_ewma = None
    losses = []
    progress = ProgBarCounter(len(loader))
    for batches_done, (task_ids_batch, obs_batch, acts_batch) \
            in enumerate(loader, start=1):
        # copy to GPU
        obs_batch = obs_batch.to(dev)
        acts_batch = acts_batch.to(dev)
        task_ids_batch = task_ids_batch.to(dev)

        # compute loss & take opt step
        opt.zero_grad()
        loss = do_loss_forward_back(task_ids_batch, obs_batch, acts_batch)
        opt.step()

        # for logging

        # TODO: do multi-task tracking of losses, not just single-task tracking
        # of losses

        progress.update(batches_done)
        f_loss = loss
        loss_ewma = f_loss if loss_ewma is None \
            else 0.9 * loss_ewma + 0.1 * f_loss
        losses.append(f_loss)

    progress.stop()

    return loss_ewma, losses


_no_version_re = re.compile(r'^(?P<env_name>.*?)(-v\d+)?$')
_alnum_re = re.compile(r'[a-zA-Z0-9]+')


def make_env_tag(env_name):
    """Take a Gym env name like 'fooBar-BazQux-v3' and return more concise string
    of the form 'FooBarBazQux' (no version string, no non-alphanumeric
    characters, letters that formerly separated words are always
    capitalised)."""
    no_version = _no_version_re.match(env_name).groupdict()['env_name']
    alnum_parts = _alnum_re.findall(no_version)
    final_name = ''.join(part[0].upper() + part[1:] for part in alnum_parts)
    return final_name


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
@click.argument("demos", nargs=-1, required=True)
def main(demos, use_gpu, add_preproc, seed, batch_size, epochs, out_dir,
         run_name, gpu_idx, eval_n_traj):
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
            print(f"Starting epoch {epoch+1}/{epochs} ({dataset_len} batches)")

            loss_ewma, losses = do_epoch_training_mt(loader_mt, model_mt,
                                                     opt_mt, dev)

            print(f"Evaluating {eval_n_traj} trajectories on "
                  f"{len(name_pairs)} envs")
            for (orig_env_name,
                 env_name), sampler in zip(name_pairs, samplers):
                copy_model_into_sampler(model_mt, sampler)
                model_mt.eval()
                scores = eval_model(sampler, n_traj=eval_n_traj)
                tag = make_env_tag(orig_env_name)
                logger.record_tabular_misc_stat("Score%s" % tag, scores)

            # TODO: again, this should be done separately for each task.
            # TODO: also make some rollout/qualitative eval code at some point.

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

"""Implementation of (single-task) Generative Adversarial Imitation Learning
(GAIL)."""
from collections import namedtuple
import itertools

import click
import gym
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
import numpy as np
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.buffer import (buffer_from_example, buffer_func,
                                get_leading_dims, torchify_buffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger
from rlpyt.utils.tensor import (infer_leading_dims, restore_leading_dims,
                                to_onehot)
import torch
from torch import nn
import torch.nn.functional as F

from mtil.common import (MILBenchFeatureNetwork, MILBenchPreprocLayer,
                         MILBenchTrajInfo, VanillaGymEnv, make_logger_ctx,
                         set_seeds, trajectories_to_loader)
from mtil.reward_injection_wrappers import CustomRewardPPO

DiscrimReplaySamples = namedarraytuple("DiscrimReplaySamples",
                                       ["all_observation", "all_action"])


class DiscrimTrainBuffer:
    """Cicular buffer of encountered samples for discriminator training. Unlike
    the replay buffers in rlpyt, this one does not try to keep sequences in
    order. It also does not compute returns."""
    def __init__(self, total_n_samples, example_samples):
        self.total_n_samples = total_n_samples
        replay_samples = DiscrimReplaySamples(
            all_observation=example_samples.env.observation,
            all_action=example_samples.agent.action)
        T, B = get_leading_dims(replay_samples, n_dim=2)
        assert total_n_samples > T * B > 0, (total_n_samples, T * B)
        self.circ_buf = buffer_from_example(replay_samples[0, 0],
                                            (total_n_samples, ))
        self.samples_in_buffer = 0
        self.ptr = 0

    def append_samples(self, samples):
        """Append samples drawn drawn from a sampler. Should be namedarraytuple
        with leading dimensions `(time_steps, batch_size)`."""
        replay_samples = DiscrimReplaySamples(
            all_observation=samples.env.observation,
            all_action=samples.agent.action)
        T, B = get_leading_dims(replay_samples, n_dim=2)
        # if there's not enough room for a single full round of sampling then
        # the buffer is _probably_ too small.
        assert T * B < self.total_n_samples, \
            f"There's not enough room in this buffer for a single full " \
            f"batch! T*B={T*B} > total_n_samples={self.total_n_samples}"
        flat_samples = buffer_func(
            replay_samples, lambda t: t.reshape((T * B, ) + t.shape[2:]))
        n_copied = 0
        while n_copied < T * B:
            # only copy to the end
            n_to_copy = min(T * B - n_copied, self.total_n_samples - self.ptr)
            self.circ_buf[self.ptr:self.ptr + n_to_copy] \
                = flat_samples[n_copied:n_copied + n_to_copy]
            n_copied += n_to_copy
            self.ptr = (self.ptr + n_to_copy) % self.total_n_samples
            self.samples_in_buffer = min(self.total_n_samples,
                                         self.samples_in_buffer + n_to_copy)

    def sample_batch(self, batch_size):
        """Samples a batch from internal replay. Will yield a namedarraytuple
        with a single leading dimension equal to `batch_size`."""
        inds_to_sample = np.random.randint(self.samples_in_buffer,
                                           size=(batch_size, ))
        return self.circ_buf[inds_to_sample]


class MILBenchDiscriminator(nn.Module):
    def __init__(self, in_chans, act_dim, ActivationCls=torch.nn.ReLU):
        super().__init__()
        self.act_dim = act_dim
        self.preproc = MILBenchPreprocLayer()
        # feature extractor gives us 1024-dim features
        self.feature_extractor = MILBenchFeatureNetwork(
            in_chans=in_chans, ActivationCls=ActivationCls)
        # the logit generator takes in *both* the 1024-dim features *and* the
        # act_dim-dimensional action (probably just a one-hot vector)
        self.logit_generator = nn.Sequential(
            nn.Linear(1024 + act_dim, 256),
            ActivationCls(),
            # now: flat 256-elem vector
            nn.Linear(256, 256),
            ActivationCls(),
            # now: flat 256-elem vector
            nn.Linear(256, 1),
            # now: 1-dimensional logits for a sigmoid activation
        )

    def forward(self, obs, act):
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        lead_dim_act, T_act, B_act, act_shape = infer_leading_dims(act, 0)
        assert (lead_dim, T, B) == (lead_dim_act, T_act, B_act)

        obs_preproc = self.preproc(obs.view((T * B, *img_shape)))
        obs_features = self.feature_extractor(obs_preproc)
        acts_one_hot = to_onehot(act.view((T * B, *act_shape)),
                                 self.act_dim,
                                 dtype=obs_features.dtype)
        all_features = torch.cat((obs_features, acts_one_hot), dim=1)
        logits = self.logit_generator(all_features)
        assert logits.dim() == 2, logits.shape
        flat_logits = logits.squeeze(1)

        flat_logits = restore_leading_dims(flat_logits, lead_dim, T, B)

        return flat_logits


class RewardModel(nn.Module):
    """Takes a binary discriminator module producing logits (with high = real
    demo)."""
    def __init__(self, discrim):
        super().__init__()
        self.discrim = discrim

    def forward(self, obs, act):
        """GAIL policy reward, without entropy bonus (should be introduced
        elsewhere). Policy should maximise this."""
        sigmoid_logits = self.discrim(obs, act)
        rewards = F.logsigmoid(sigmoid_logits)
        return rewards


class MILBenchPolicyValueNetwork(nn.Module):
    def __init__(self, in_chans, n_actions, ActivationCls=torch.nn.ReLU):
        super().__init__()
        self.preproc = MILBenchPreprocLayer()
        # feature extractor gives us 1024-dim features
        self.feature_extractor = MILBenchFeatureNetwork(
            in_chans=in_chans, ActivationCls=ActivationCls)
        # we then post-process those features with two FC layers
        self.feature_generator = nn.Sequential(
            nn.Linear(1024, 256),
            ActivationCls(),
            # now: flat 256-elem vector
            nn.Linear(256, 256),
            ActivationCls(),
            # finally: flat 256-elem vector
        )
        # finally, we add policy and value heads
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs, prev_act=None, prev_rew=None):
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)

        obs_preproc = self.preproc(obs.view((T * B, *img_shape)))
        obs_features = self.feature_extractor(obs_preproc)
        flat_features = self.feature_generator(obs_features)
        pi_logits = self.policy_head(flat_features)
        pi = F.softmax(pi_logits, dim=-1)
        v = self.value_head(flat_features).squeeze(-1)

        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v


GAILInfo = namedtuple(
    'GAILInfo',
    [
        # these are the names that should be written to the log
        'DiscMeanXEnt',
        'DiscAcc',
        'DiscAccExpert',
        'DiscAccNovice',
        'DiscFracExpertTrue',
        'DiscFracExpertPred',
        'DiscMeanLabelEnt',
    ])


def _compute_gail_stats(disc_logits, is_real_labels):
    """Returns dict for use in constructing GAILInfo later on. Provides a dict
    containing every field except DiscMeanXEnt."""
    pred_labels = (disc_logits > 0).to(dtype=torch.float32, device='cpu')
    real_labels = (is_real_labels > 0).to(dtype=torch.float32, device='cpu')
    torch_dict = dict(
        DiscAcc=torch.mean((pred_labels == real_labels).to(torch.float32)),
        DiscAccExpert=torch.mean(
            (pred_labels[real_labels.nonzero()] > 0).to(torch.float32)),
        DiscAccNovice=torch.mean(
            (pred_labels[(1 - real_labels).nonzero()] > 0).to(torch.float32)),
        DiscFracExpertTrue=(torch.sum(real_labels) / real_labels.shape[0]),
        DiscFracExpertPred=(torch.sum(pred_labels) / pred_labels.shape[0]),
        DiscMeanLabelEnt=-torch.mean(
            torch.sigmoid(disc_logits) * F.logsigmoid(disc_logits)),
    )
    np_dict = {k: v.item() for k, v in torch_dict.items()}
    return np_dict


class GAILOptimiser:
    def __init__(self, expert_trajectories, discrim_model, buffer_num_samples,
                 batch_size, updates_per_itr, dev):
        assert batch_size % 2 == 0, \
            "batch size must be even so we can split between real & fake"
        self.model = discrim_model
        self.batch_size = batch_size
        self.buffer_num_samples = buffer_num_samples
        self.updates_per_itr = updates_per_itr
        self.dev = dev
        self.expert_traj_loader = trajectories_to_loader(
            expert_trajectories, batch_size=batch_size // 2)
        self.expert_batch_iter = itertools.chain.from_iterable(
            itertools.repeat(self.expert_traj_loader))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # we'll set this up on the first pass
        self.pol_replay_buffer = None

    def optim_disc(self, itr, samples):
        # store new samples in replay buffer
        if self.pol_replay_buffer is None:
            self.pol_replay_buffer = DiscrimTrainBuffer(
                self.buffer_num_samples, samples)
        self.pol_replay_buffer.append_samples(samples)

        # switch to train mode before taking any steps
        self.model.train()
        info_dicts = []
        for _ in range(self.updates_per_itr):
            self.opt.zero_grad()

            expert_obs, expert_acts = next(self.expert_batch_iter)
            # grep for SamplesFromReplay to see what fields pol_replay_samples
            # has
            pol_replay_samples = self.pol_replay_buffer.sample_batch(
                self.batch_size // 2)
            pol_replay_samples = torchify_buffer(pol_replay_samples)
            all_obs = torch.cat(
                [expert_obs, pol_replay_samples.all_observation], dim=0) \
                .to(self.dev)
            all_acts = torch.cat([expert_acts.to(torch.int64),
                                  pol_replay_samples.all_action],
                                 dim=0) \
                .to(self.dev)
            is_real_label = torch.cat(
                [
                    # expert samples
                    torch.ones(self.batch_size // 2,
                               dtype=torch.float32,
                               device=self.dev),
                    # novice samples
                    torch.zeros(self.batch_size // 2,
                                dtype=torch.float32,
                                device=self.dev),
                ],
                dim=0)
            logits = self.model(all_obs, all_acts)

            # GAIL discriminator loss is E_fake[log D(s,a)] +
            # E_expert[log(1-D(s,a))].
            #
            # binary_cross_entropy_with_logits computes -labels *
            # log(sigmoid(logits)) - (1 - labels) * log(1-sigmoid(logits)) (per
            # PyTorch docs).
            #
            # Hence GAIL is like "reverse logistic regression" where you label
            # expert demonstrations as 1 and fake (novice) demonstrations as 0,
            # then flip the sign of the loss.
            loss = -F.binary_cross_entropy_with_logits(
                logits, is_real_label, reduction='mean')
            loss.backward()
            self.opt.step()

            # for logging; we'll average theses later
            info_dict = _compute_gail_stats(logits, is_real_label)
            info_dict['DiscMeanXEnt'] = loss.item()
            info_dicts.append(info_dict)

        # switch back to eval mode
        self.model.eval()

        opt_info = GAILInfo(**{
            k: np.mean([d[k] for d in info_dicts])
            for k in info_dicts[0].keys()
        })

        return opt_info


class GAILMinibatchRl(MinibatchRl):
    def __init__(self, *, gail_optim, **kwargs):
        super().__init__(**kwargs)
        self.gail_optim = gail_optim
        self.joint_info_cls = None

    def train(self):
        # copied from MinibatchRl.train() & extended to support GAIL update
        n_itr = self.startup()
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(
                    itr)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)

                # run GAIL & combine its output with RL algorithm output
                gail_info = self.gail_optim.optim_disc(itr, samples)
                if self.joint_info_cls is None:
                    self.joint_info_cls = namedtuple(
                        'joint_info_cls', gail_info._fields + opt_info._fields)
                opt_info = self.joint_info_cls(**gail_info._asdict(),
                                               **opt_info._asdict())

                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    self.log_diagnostics(itr)
        self.shutdown()

    def initialize_logging(self):
        super().initialize_logging()
        # make sure that logger knows how to handle GAIL info tuples, too
        self._opt_infos.update({field: [] for field in GAILInfo._fields})


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
         disc_up_per_itr, total_n_steps, log_interval_steps):
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

    # local copy of Gym env, w/ args to create equivalent env in the sampler
    env_ctor = VanillaGymEnv
    env_ctor_kwargs = dict(env_name=env_name)
    env = gym.make(env_name)
    # number of transitions collected during each round of sampling will be
    # batch_T * batch_B = n_steps * n_envs
    batch_T = n_steps_per_iter
    batch_B = n_envs

    # TODO: replace this with CpuSampler or GpuSampler, as appropriate for
    # device
    sampler = SerialSampler(env_ctor,
                            env_ctor_kwargs,
                            max_decorrelation_steps=env.spec.max_episode_steps,
                            TrajInfoCls=MILBenchTrajInfo,
                            batch_T=batch_T,
                            batch_B=batch_B)

    # should be (H,W,C) even w/ frame stack
    assert len(env.observation_space.shape) == 3, env.observation_space.shape
    in_chans = env.observation_space.shape[-1]
    n_actions = env.action_space.n  # categorical action space
    ppo_agent = CategoricalPgAgent(ModelCls=MILBenchPolicyValueNetwork,
                                   model_kwargs=dict(in_chans=in_chans,
                                                     n_actions=n_actions))

    discriminator = MILBenchDiscriminator(in_chans=in_chans, act_dim=n_actions)
    reward_model = RewardModel(discriminator)
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
        # TODO: figure out whether I need to set affinity to anything
        # affinity=affinity,
    )

    with make_logger_ctx(out_dir, "gail", orig_env_name, run_name):
        runner.train()


if __name__ == '__main__':
    main()

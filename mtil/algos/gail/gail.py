from collections import namedtuple
import itertools

import numpy as np
from rlpyt.runners.minibatch_rl import MinibatchRl
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
                         trajectories_to_loader)

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
    # Reminder: in GAIL, high output = predicted to be from expert. In arrays
    # below, 1 = expert and 0 = novice.
    pred_labels = (disc_logits > 0).to(dtype=torch.float32, device='cpu')
    real_labels = (is_real_labels > 0).to(dtype=torch.float32, device='cpu')
    torch_dict = dict(
        DiscAcc=torch.mean((pred_labels == real_labels).to(torch.float32)),
        DiscAccExpert=torch.mean(
            (pred_labels[real_labels.nonzero()] > 0).to(torch.float32)),
        DiscAccNovice=torch.mean(
            (pred_labels[(1 - real_labels).nonzero()] <= 0).to(torch.float32)),
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
            # PyTorch docs). In other words:
            #   -y * log D(s,a) - (1 - y) * log(1 - D(s, a))
            #
            # Hence GAIL is like "reverse logistic regression" where you label
            # expert demonstrations as 0 and fake (novice) demonstrations as 1,
            # then flip the sign of the loss. I doubt that part of their
            # algorithm is necessary, so I've just done things the normal way.
            loss = F.binary_cross_entropy_with_logits(
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
        new_fields = GAILInfo._fields + self.algo._custom_logging_fields
        self._opt_infos.update({field: [] for field in new_fields})

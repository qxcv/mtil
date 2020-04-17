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
                         MultiTaskAffineLayer, make_loader_mt, tree_map)

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
        assert total_n_samples >= T * B > 0, (total_n_samples, T * B)
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
        assert T * B <= self.total_n_samples, \
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


class MILBenchDiscriminatorMT(nn.Module):
    def __init__(self,
                 task_ids_and_names,
                 in_chans,
                 act_dim,
                 use_actions,
                 use_all_chans,
                 fc_dim=256,
                 ActivationCls=torch.nn.ReLU):
        super().__init__()
        self.task_ids_and_names = task_ids_and_names
        self.act_dim = act_dim
        self.in_chans = in_chans
        self.use_all_chans = use_all_chans
        self.use_actions = use_actions
        self.preproc = MILBenchPreprocLayer()
        # feature extractor gives us 1024-dim features
        if use_all_chans:
            feat_in_chans = in_chans
        else:
            # use just last channels
            assert in_chans >= 3 and (in_chans % 3) == 0, in_chans
            feat_in_chans = 3
        self.feature_extractor = MILBenchFeatureNetwork(
            in_chans=feat_in_chans,
            out_chans=fc_dim,
            ActivationCls=ActivationCls)
        extra_dim = act_dim if use_actions else 0
        # the logit generator takes in both the image features and the
        # act_dim-dimensional action (probably just a one-hot vector)
        self.postproc = nn.Sequential(
            nn.Linear(fc_dim + extra_dim, fc_dim),
            ActivationCls(),
            # now: flat fc_dim-elem vector
        )
        self.mt_logits = MultiTaskAffineLayer(fc_dim, 1,
                                              len(self.task_ids_and_names))
        # TODO: remove self.logit_generator, replace with self.postproc and
        # self.mt_logits

    def forward(self, obs, act):
        # this only handles images
        lead_dim, T, B, img_shape = infer_leading_dims(obs.observation, 3)
        lead_dim_act, T_act, B_act, act_shape = infer_leading_dims(act, 0)
        assert (lead_dim, T, B) == (lead_dim_act, T_act, B_act)

        if self.use_all_chans:
            obs_trimmed = obs.observation
            trimmed_img_shape = img_shape
        else:
            # don't use all channels, just use the last three (i.e. last image
            # in the stack, in RGB setting)
            assert obs.shape[-1] == self.in_chans
            obs_trimmed = obs.observation[..., -3:]
            trimmed_img_shape = (*img_shape[:-1], obs_trimmed.shape[-1])
        obs_reshape = obs_trimmed.view((T * B, *trimmed_img_shape))
        obs_preproc = self.preproc(obs_reshape)
        obs_features = self.feature_extractor(obs_preproc)
        if self.use_actions:
            acts_one_hot = to_onehot(act.view((T * B, *act_shape)),
                                     self.act_dim,
                                     dtype=obs_features.dtype)
            all_features = torch.cat((obs_features, acts_one_hot), dim=1)
        else:
            all_features = obs_features
        lin_features = self.postproc(all_features)
        logits = self.mt_logits(lin_features, obs.task_id)
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

    def forward(self, obs, act, *args, **kwargs):
        """GAIL policy reward, without entropy bonus (should be introduced
        elsewhere). Policy should maximise this."""
        sigmoid_logits = self.discrim(obs, act, *args, **kwargs)
        # In the GAIL paper they use this as the cost (i.e. they minimise it).
        # I'm maximising it to be consistent with the reference implementation,
        # where the conventions for discriminator output are reversed (so high
        # = expert & low = novice by default, although they do also have a
        # switch that can flip the convention for environments with early
        # termination).
        rewards = F.logsigmoid(sigmoid_logits)
        return rewards


GAILInfo = namedtuple(
    'GAILInfo',
    [
        # these are the names that should be written to the log
        'discMeanXEnt',
        'discAcc',
        'discAccExpert',
        'discAccNovice',
        'discFracExpertTrue',
        'discFracExpertPred',
        'discMeanLabelEnt',
    ])


def _compute_gail_stats(disc_logits, is_real_labels):
    """Returns dict for use in constructing GAILInfo later on. Provides a dict
    containing every field except discMeanXEnt."""
    # Reminder: in GAIL, high output = predicted to be from expert. In arrays
    # below, 1 = expert and 0 = novice.
    pred_labels = (disc_logits > 0).to(dtype=torch.float32, device='cpu')
    real_labels = (is_real_labels > 0).to(dtype=torch.float32, device='cpu')
    torch_dict = dict(
        discAcc=torch.mean((pred_labels == real_labels).to(torch.float32)),
        discAccExpert=torch.mean(
            (pred_labels[real_labels.nonzero()] > 0).to(torch.float32)),
        discAccNovice=torch.mean(
            (pred_labels[(1 - real_labels).nonzero()] <= 0).to(torch.float32)),
        discFracExpertTrue=(torch.sum(real_labels) / real_labels.shape[0]),
        discFracExpertPred=(torch.sum(pred_labels) / pred_labels.shape[0]),
        discMeanLabelEnt=-torch.mean(
            torch.sigmoid(disc_logits) * F.logsigmoid(disc_logits)),
    )
    np_dict = {k: v.item() for k, v in torch_dict.items()}
    return np_dict


class GAILOptimiser:
    def __init__(self, dataset_mt, discrim_model, buffer_num_samples,
                 batch_size, updates_per_itr, dev, aug_model, lr):
        assert batch_size % 2 == 0, \
            "batch size must be even so we can split between real & fake"
        self.model = discrim_model
        self.batch_size = batch_size
        self.buffer_num_samples = buffer_num_samples
        self.updates_per_itr = updates_per_itr
        self.dev = dev
        self.aug_model = aug_model
        self.expert_traj_loader = make_loader_mt(dataset_mt, batch_size // 2)
        self.expert_batch_iter = itertools.chain.from_iterable(
            itertools.repeat(self.expert_traj_loader))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # we'll set this up on the first pass
        self.pol_replay_buffer = None

    def state_dict(self):
        return {
            'disc_model_state': self.model.state_dict(),
            'disc_opt_state': self.opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['disc_model_state'])
        self.opt.load_state_dict(state_dict['disc_opt_state'])

    def optim_disc(self, itr, samples):
        # store new samples in replay buffer
        if self.pol_replay_buffer is None:
            self.pol_replay_buffer = DiscrimTrainBuffer(
                self.buffer_num_samples, samples)
        # keep ONLY the demo env samples
        sample_variant_ids = samples.env.observation.variant_id
        keep_mask = sample_variant_ids == 0
        # check that each batch index is "pure", in sense that e.g. all
        # elements at index k are always for the same task ID
        assert (keep_mask[:1] == keep_mask).all(), keep_mask
        filtered_samples = samples[:, keep_mask[0]]
        self.pol_replay_buffer.append_samples(filtered_samples)

        # switch to train mode before taking any steps
        self.model.train()
        info_dicts = []
        for _ in range(self.updates_per_itr):
            self.opt.zero_grad()

            expert_data = next(self.expert_batch_iter)
            expert_obs = expert_data['obs']
            expert_acts = expert_data['acts']
            # grep rlpyt source for "SamplesFromReplay" to see what fields
            # pol_replay_samples has
            pol_replay_samples = self.pol_replay_buffer.sample_batch(
                self.batch_size // 2)
            pol_replay_samples = torchify_buffer(pol_replay_samples)
            novice_obs = pol_replay_samples.all_observation
            all_obs = tree_map(lambda *args: torch.cat(args, 0), expert_obs,
                               novice_obs)
            if self.aug_model is not None:
                # augmentations
                all_obs = all_obs._replace(
                    observation=self.aug_model(all_obs.observation))
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

            # GAIL discriminator *objective* is E_fake[log D(s,a)] +
            # E_expert[log(1-D(s,a))]. You actually want to maximise this; in
            # reality the "loss" to be minimised is -E_fake[log D(s,a)] -
            # E_expert[log(1-D(s,a))].
            #
            # binary_cross_entropy_with_logits computes -labels *
            # log(sigmoid(logits)) - (1 - labels) * log(1-sigmoid(logits)) (per
            # PyTorch docs). In other words:
            #   -y * log D(s,a) - (1 - y) * log(1 - D(s, a))
            #
            # Hence, GAIL is like logistic regression with label 1 for the
            # novice and 0 for the expert. This is kind of weird, because you
            # actually want to *minimise* the discriminator's output. Indeed,
            # in the actual implementation, they flip this & use 1 for the
            # expert and 0 for the novice.

            # In light of all the above, I'm using the OPPOSITE convention to
            # the paper, but the same convention as the implementation. To wit:
            #
            # - 1 is the *expert* label, and high = more expert-like
            # - 0 is the *novice* label, and low = less expert-like
            loss = F.binary_cross_entropy_with_logits(logits,
                                                      is_real_label,
                                                      reduction='mean')
            loss.backward()
            self.opt.step()

            # for logging; we'll average theses later
            info_dict = _compute_gail_stats(logits, is_real_label)
            info_dict['discMeanXEnt'] = loss.item()
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

    def train(self, cb_startup=None):
        # copied from MinibatchRl.train() & extended to support GAIL update
        n_itr = self.startup()
        if cb_startup:
            # post-startup callback (cb)
            cb_startup(self)
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

    def get_itr_snapshot(self, itr):
        snap_dict = super().get_itr_snapshot(itr)

        # Save policy model state directly so that mtbc.py can pick it up. This
        # will only work with PPO & my hacky GAIL thing (ugh, inelegant).
        real_model = self.algo.agent.model.model
        assert 'model_state' not in snap_dict, snap_dict.keys()

        # Build full dict (also including discrim state)
        new_dict = {
            'model_state': real_model.state_dict(),
            'disc_state': self.gail_optim.state_dict(),
        }

        # make sure we're not overwriting anything
        assert not (new_dict.keys() & snap_dict.keys()), \
            (new_dict.keys(), snap_dict.keys())
        new_dict.update(snap_dict)

        return new_dict

    def initialize_logging(self):
        super().initialize_logging()
        # make sure that logger knows how to handle GAIL info tuples, too
        new_fields = GAILInfo._fields + self.algo._custom_logging_fields
        self._opt_infos.update({field: [] for field in new_fields})

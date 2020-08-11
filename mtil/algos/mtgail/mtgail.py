from collections import defaultdict, namedtuple
import copy
import functools
import itertools as it

from magical.benchmarks import EnvName
import numpy as np
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.buffer import (buffer_from_example, buffer_func,
                                get_leading_dims, torchify_buffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import nn
import torch.nn.functional as F

from mtil.demos import make_loader_mt
from mtil.models import (MILBenchFeatureNetwork, MILBenchPreprocLayer,
                         MultiTaskAffineLayer, SingleTaskAffineLayer)
from mtil.utils.misc import tree_map
from mtil.utils.torch import repeat_dataset

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


def hacky_interpolate(structure, eps, sub_batch_size):
    def inner_map(tens):
        tens_l = tens[:sub_batch_size]
        tens_r = tens[sub_batch_size:]
        if tens.ndim == 4 and tens.dtype in (torch.uint8, torch.float):
            # Float interpolation is easy. Otherwise we properly interpolate
            # byte image by (implicitly) converting to float, then back to
            # byte.
            eps_exp = eps[:, None, None, None]
            interp = eps_exp * tens_l + (1 - eps_exp) * tens_r
            if tens.type == torch.uint8:
                interp = interp.to(torch.uint8)
        elif tens.ndim == 1 and not torch.is_floating_point(tens):
            # "interpolate" some integers (assume action label or something, so
            # the output must be discrete)

            # FIXME(sam): what we really want to do is interpolate the
            # one-hot representation or something, but instead of that I'm
            # just going to do this. (whatever, Lipschitzness of discrete
            # functions doesn't make sense anyway.)

            tens_elems = []
            for eps_i, elem_l, elem_r in zip(eps, tens_l, tens_r):
                tens_elems.append(elem_l if eps_i > 0.5 else elem_r)
            interp = torch.stack(tens_elems, dim=0)

        else:
            raise ValueError("cannot handle shape/type of tensor", tens)

        return interp

    new_structure = tree_map(inner_map, structure)

    return new_structure


class MILBenchDiscriminatorMT(nn.Module):
    def __init__(
            self,
            task_ids_and_names,
            in_chans,
            act_dim,
            use_actions,
            use_all_chans,
            use_sn,
            fc_dim=256,
            act_rep_dim=32,
            ActivationCls=torch.nn.ReLU,
            **feat_gen_kwargs,
    ):
        super().__init__()
        self.task_ids_and_names = task_ids_and_names
        self.act_dim = act_dim
        self.in_chans = in_chans
        self.use_all_chans = use_all_chans
        self.use_actions = use_actions
        self.preproc = MILBenchPreprocLayer()
        self.use_sn = use_sn
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
            use_sn=use_sn,
            ActivationCls=ActivationCls,
            **feat_gen_kwargs)
        if self.use_actions:
            # action_preproc lets us learn a distinct embedding for each
            # action. Having learnt embeddings instead of one-hot doesn't make
            # increase model power, since the embedding later gets stuffed into
            # a linear layer into a linear layer in both cases. However, it
            # does let us apply a domain transfer loss to the embedding, which
            # can force the model to be invariant to actions to some extent.
            self.action_preproc = nn.Embedding(act_dim, act_rep_dim)
            extra_dim = act_rep_dim
        else:
            extra_dim = 0
        # the logit generator takes in both the image features and the
        # action encoding
        reduce_layer = nn.Linear(fc_dim + extra_dim, fc_dim)
        if use_sn:
            reduce_layer = nn.utils.spectral_norm(reduce_layer)
        self.postproc = nn.Sequential(
            reduce_layer,
            ActivationCls(),
            # now: flat fc_dim-elem vector
        )
        self.ret_feats_dim = fc_dim + extra_dim
        if use_sn:
            assert len(self.task_ids_and_names) == 1, \
                "SN only supported in single-task setting, for now"
            self.mt_logits = SingleTaskAffineLayer(fc_dim, 1, use_sn=use_sn)
        else:
            self.mt_logits = MultiTaskAffineLayer(fc_dim, 1,
                                                  len(self.task_ids_and_names))

    def forward_no_preproc(self, obs_preproc, act_preproc, task_id):
        obs_features = self.feature_extractor(obs_preproc)
        if self.use_actions:
            action_features = self.action_preproc(act_preproc)
            all_features = torch.cat((obs_features, action_features), dim=1)
        else:
            all_features = obs_features
        lin_features = self.postproc(all_features)
        logits = self.mt_logits(lin_features, task_id)
        assert logits.dim() == 2, logits.shape
        flat_logits = logits.squeeze(1)

        return flat_logits, all_features

    def preproc_obs_acts(self, obs, act):
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
        act_labels_flat = act.view((T * B, *act_shape))

        return obs_preproc, act_labels_flat, (lead_dim, T, B)

    def forward(self, obs, act, return_feats=False):
        obs_preproc, act_labels_flat, restore_dim_args = self.preproc_obs_acts(
            obs, act)

        # now do actual learnt transform
        flat_logits, all_features = self.forward_no_preproc(
            obs_preproc, act_labels_flat, obs.task_id)

        # finally, restore dims as necessary
        flat_logits = restore_leading_dims(flat_logits, *restore_dim_args)

        if return_feats:
            return flat_logits, all_features
        return flat_logits


class RewardModel(nn.Module):
    """Takes a binary discriminator module producing logits (with high = real
    demo)."""
    def __init__(self, discrim, domain_xfer_model, domain_xfer_weight,
                 use_wgan):
        super().__init__()
        self.discrim = discrim
        self.domain_xfer_model = domain_xfer_model
        self.domain_xfer_weight = domain_xfer_weight
        self.use_wgan = use_wgan
        if self.domain_xfer_weight:
            assert domain_xfer_model is not None

    def forward(self,
                obs,
                act,
                task_ids=None,
                variant_ids=None,
                *args,
                **kwargs):
        """GAIL policy reward, without entropy bonus (should be introduced
        elsewhere). Policy should maximise this."""
        if self.domain_xfer_weight:
            sigmoid_logits, discrim_features = self.discrim(obs,
                                                            act,
                                                            *args,
                                                            return_feats=True,
                                                            **kwargs)
            base_rewards = F.logsigmoid(sigmoid_logits)
            # want to max. loss of model that distinguishes demo from test
            # variants
            xfer_is_demo_labels = (obs.variant_id == 0).to(torch.float)
            xfer_losses, _ = self.domain_xfer_model(discrim_features,
                                                    xfer_is_demo_labels,
                                                    reduce_loss=False)
            assert xfer_losses.shape == base_rewards.shape, \
                (xfer_losses.shape, base_rewards.shape)
            rewards = base_rewards + self.domain_xfer_weight * xfer_losses
        else:
            sigmoid_logits = self.discrim(obs, act, *args, **kwargs)
            if self.use_wgan:
                # in WGAIL we just try to maximise the discriminator output
                rewards = sigmoid_logits
            else:
                # In the GAIL paper they use this as the cost (i.e. they
                # minimise it). I'm maximising it to be consistent with the
                # reference implementation, where the conventions for
                # discriminator output are reversed (so high = expert & low =
                # novice by default, although they do also have a switch that
                # can flip the convention for environments with early
                # termination).
                rewards = F.logsigmoid(sigmoid_logits)
        return rewards


GAILInfo = namedtuple(
    'GAILInfo',
    [
        # these are the names that should be written to the log
        'discLoss',
        'discXentLoss',
        'discWGANLoss',
        'discGPGradNorm',
        'discGPLoss',
        'discAcc',
        'discAccExpert',
        'discAccNovice',
        'discFracExpertTrue',
        'discFracExpertPred',
        'discMeanLabelEnt',
        # the next two are only filled in when we use a transfer loss
        'xferLoss',
        'xferAcc',
    ])


def _compute_gail_stats(disc_logits, is_real_labels, is_wgan):
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
            torch.sigmoid(disc_logits) * F.logsigmoid(disc_logits))
        if is_wgan else torch.as_tensor(float('nan')),
    )
    np_dict = {k: v.item() for k, v in torch_dict.items()}
    return np_dict


class GAILOptimiser:
    def __init__(self, *, dataset_mt, discrim_model, buffer_num_samples,
                 batch_size, updates_per_itr, dev, aug_model, xfer_adv_weight,
                 xfer_adv_anneal, xfer_adv_module, lr, use_wgan, gp_weight):
        assert batch_size % 2 == 0, \
            "batch size must be even so we can split between real & fake"
        self.model = discrim_model
        self.batch_size = batch_size
        self.buffer_num_samples = buffer_num_samples
        self.updates_per_itr = updates_per_itr
        self.dev = dev
        self.aug_model = aug_model
        self.expert_traj_loader = make_loader_mt(dataset_mt, batch_size // 2)
        self.expert_batch_iter = repeat_dataset(self.expert_traj_loader)
        self.use_wgan = use_wgan
        self.gp_weight = gp_weight
        self.xfer_adv_weight = xfer_adv_weight
        # if xfer_disc_anneal is True, then we anneal from 0 to 1
        self.xfer_disc_anneal = xfer_adv_anneal
        if self.xfer_adv_weight > 0:
            assert xfer_adv_module is not None
            self.xfer_adv_model = xfer_adv_module
            all_params = it.chain(self.model.parameters(),
                                  self.xfer_adv_model.parameters())
            self.xfer_replay_buffer = None
        else:
            self.xfer_adv_model = None
            all_params = self.model.parameters()
        self.opt = torch.optim.Adam(all_params, lr=lr)
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

    def optim_disc(self, itr, n_itr, samples):
        # TODO: refactor this method. Makes sense to split code that sets up
        # the replay buffer(s) from code that sets up each batch (and from the
        # code that computes losses and records metrics, etc.)

        # store new samples in replay buffer
        if self.pol_replay_buffer is None:
            self.pol_replay_buffer = DiscrimTrainBuffer(
                self.buffer_num_samples, samples)
        # keep ONLY the demo env samples
        sample_variant_ids = samples.env.observation.variant_id
        train_variant_mask = sample_variant_ids == 0
        # check that each batch index is "pure", in sense that e.g. all
        # elements at index k are always for the same task ID
        assert (train_variant_mask[:1] == train_variant_mask).all(), \
            train_variant_mask
        filtered_samples = samples[:, train_variant_mask[0]]
        self.pol_replay_buffer.append_samples(filtered_samples)

        if self.xfer_adv_model is not None:
            # if we have an adversarial domain adaptation model for transfer
            # learning, then we also keep samples that *don't* come from the
            # train variant so we can use them for the transfer loss
            if self.xfer_replay_buffer is None:
                # second replay buffer for off-task samples
                self.xfer_replay_buffer = DiscrimTrainBuffer(
                    self.buffer_num_samples, samples)
            xfer_variant_mask = ~train_variant_mask
            assert torch.any(xfer_variant_mask), \
                "xfer_adv_weight>0 supplied, but no xfer variants in batch?"
            assert (xfer_variant_mask[:1] == xfer_variant_mask).all()
            filtered_samples_xfer = samples[:, xfer_variant_mask[0]]
            self.xfer_replay_buffer.append_samples(filtered_samples_xfer)

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
            sub_batch_size = self.batch_size // 2
            pol_replay_samples = self.pol_replay_buffer.sample_batch(
                sub_batch_size)
            pol_replay_samples = torchify_buffer(pol_replay_samples)
            novice_obs = pol_replay_samples.all_observation

            if self.xfer_adv_model:
                # add a bunch of of domain transfer samples
                xfer_replay_samples = self.xfer_replay_buffer.sample_batch(
                    sub_batch_size)
                xfer_replay_samples = torchify_buffer(xfer_replay_samples)
                xfer_replay_obs = xfer_replay_samples.all_observation
                all_obs = tree_map(
                    lambda *args: torch.cat(args, 0).to(self.dev), expert_obs,
                    novice_obs, xfer_replay_obs)
                all_acts = torch.cat([expert_acts.to(torch.int64),
                                      pol_replay_samples.all_action,
                                      xfer_replay_samples.all_action],
                                     dim=0) \
                    .to(self.dev)
            else:
                all_obs = tree_map(
                    lambda *args: torch.cat(args, 0).to(self.dev), expert_obs,
                    novice_obs)
                all_acts = torch.cat([expert_acts.to(torch.int64),
                                      pol_replay_samples.all_action],
                                     dim=0) \
                    .to(self.dev)

            if self.aug_model is not None:
                # augmentations
                aug_frames = self.aug_model(all_obs.observation)
                all_obs = all_obs._replace(observation=aug_frames)

            make_ones = functools.partial(torch.ones,
                                          dtype=torch.float32,
                                          device=self.dev)
            make_zeros = functools.partial(torch.zeros,
                                           dtype=torch.float32,
                                           device=self.dev)
            is_real_label = torch.cat(
                [
                    # expert samples
                    make_ones(sub_batch_size),
                    # novice samples
                    make_zeros(sub_batch_size),
                ],
                dim=0)

            if self.xfer_adv_model:
                # apply domain transfer loss to the transfer samples, then
                # remove them
                logits_all, disc_feats_all = self.model(all_obs,
                                                        all_acts,
                                                        return_feats=True)
                # cut the expert samples out of the discriminator transfer
                # objective
                disc_feats_xfer = disc_feats_all[sub_batch_size:]
                xfer_labels = torch.cat(
                    [make_ones(sub_batch_size),
                     make_zeros(sub_batch_size)],
                    dim=0)
                xfer_loss, xfer_acc = self.xfer_adv_model(
                    disc_feats_xfer, xfer_labels)
                # cut the transfer env samples out of the logits
                logits = logits_all[:-sub_batch_size]
            else:
                logits = self.model(all_obs, all_acts)

            if self.use_wgan:
                # Now assign label +1 for novice, -1 for expert. This means we
                # are trying to push novice scores down, and expert scores up
                # (I don't know whether this matches the original paper).
                pm_labels = 1 - 2 * is_real_label
                wgan_loss = main_loss = torch.mean(pm_labels * logits)
            else:
                # GAIL discriminator *objective* is E_fake[log D(s,a)] +
                # E_expert[log(1-D(s,a))]. You actually want to maximise this;
                # in reality the "loss" to be minimised is -E_fake[log D(s,a)]
                # - E_expert[log(1-D(s,a))].
                #
                # binary_cross_entropy_with_logits computes -labels *
                # log(sigmoid(logits)) - (1 - labels) * log(1-sigmoid(logits))
                # (per PyTorch docs). In other words: -y * log D(s,a) - (1 - y)
                # * log(1 - D(s, a))
                #
                # Hence, GAIL is like logistic regression with label 1 for the
                # novice and 0 for the expert. This is kind of weird, because
                # you actually want to *minimise* the discriminator's output.
                # Indeed, in the actual implementation, they flip this & use 1
                # for the expert and 0 for the novice.

                # In light of all the above, I'm using the OPPOSITE convention
                # to the paper, but the same convention as the implementation.
                # To wit:
                #
                # - 1 is the *expert* label, and high = more expert-like
                # - 0 is the *novice* label, and low = less expert-like
                xent_loss = main_loss = F.binary_cross_entropy_with_logits(
                    logits, is_real_label, reduction='mean')

            loss = main_loss

            if self.gp_weight:
                eps = torch.rand((sub_batch_size, )).to(self.dev)
                preproc_obs, preproc_acts, _ = self.model.preproc_obs_acts(
                    all_obs, all_acts)
                interp_obs = hacky_interpolate(preproc_obs, eps,
                                               sub_batch_size)
                interp_acts = hacky_interpolate(preproc_acts, eps,
                                                sub_batch_size)
                interp_task_ids = hacky_interpolate(all_obs.task_id, eps,
                                                    sub_batch_size)

                # using the strategy from
                # https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
                # (also we only do this w.r.t images, not scalar inputs)
                interp_obs_var = torch.autograd.Variable(interp_obs,
                                                         requires_grad=True)
                interp_logits, _ = self.model.forward_no_preproc(
                    interp_obs_var, interp_acts, interp_task_ids)
                start_grad = torch.ones((sub_batch_size, ), device=self.dev)
                # FIXME(sam): I suspect retrain_graph and grad_outputs are not
                # actually required. Should dig deeper some time.
                grads_wrt_inputs, = torch.autograd.grad(
                    outputs=interp_logits,
                    inputs=[interp_obs_var],
                    grad_outputs=start_grad,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)
                grads_wrt_inputs_flat = torch.flatten(grads_wrt_inputs,
                                                      start_dim=1)
                grad_norms = torch.sqrt(
                    torch.sum(grads_wrt_inputs_flat**2, dim=1))
                grad_penalty = torch.mean((grad_norms - 1.0)**2)
                loss = loss + self.gp_weight * grad_penalty


            if self.xfer_adv_model:
                if self.xfer_disc_anneal:
                    progress = min(1, max(0, itr / float(n_itr)))
                    xfer_adv_weight = progress * self.xfer_adv_weight
                else:
                    xfer_adv_weight = self.xfer_adv_weight
                loss = loss + xfer_adv_weight * xfer_loss

            loss.backward()
            self.opt.step()

            # for logging; we'll average theses later
            info_dict = _compute_gail_stats(logits, is_real_label,
                                            self.use_wgan)
            info_dict['discLoss'] = loss.item()
            if self.use_wgan:
                info_dict['discXentLoss'] = float('nan')
                info_dict['discWGANLoss'] = wgan_loss.item()
            else:
                info_dict['discXentLoss'] = xent_loss.item()
                info_dict['discWGANLoss'] = float('nan')
            if self.gp_weight:
                info_dict['discGPGradNorm'] = grad_norms.mean().item()
                info_dict['discGPLoss'] = grad_penalty.item()
            else:
                info_dict['discGPGradNorm'] = float('nan')
                info_dict['discGPLoss'] = float('nan')
            if self.xfer_adv_model:
                info_dict['xferLoss'] = xfer_loss.item()
                info_dict['xferAcc'] = xfer_acc.item()
            else:
                info_dict['xferLoss'] = 0.0
                info_dict['xferAcc'] = 0.0
            info_dicts.append(info_dict)

        # switch back to eval mode
        self.model.eval()

        opt_info = GAILInfo(**{
            k: np.mean([d[k] for d in info_dicts])
            for k in info_dicts[0].keys()
        })

        return opt_info


def _simplify_env_name(env_name):
    e = EnvName(env_name)
    new_name = e.name_prefix.strip('-')
    dt_spec = e.demo_test_spec.strip('-')
    return new_name + dt_spec


def _label_traj_infos(traj_infos, variant_groups):
    new_traj_infos = []
    for traj_info in traj_infos:
        new_traj_info = copy.copy(traj_info)
        task_id = traj_info["Task"]
        variant_id = traj_info["Variant"]
        env_name = variant_groups.env_name_by_task_variant[(task_id,
                                                            variant_id)]
        env_tag = _simplify_env_name(env_name)
        # squish together task + variant, stripping "demo" and any suffix
        variant_groups.name_by_prefix.keys()
        new_traj_info["Score" + env_tag] = traj_info["Score"]
        del new_traj_info["Score"], new_traj_info["Task"], \
            new_traj_info["Variant"]
        new_traj_infos.append(new_traj_info)
    return new_traj_infos


class GAILMinibatchRl(MinibatchRl):
    def __init__(self, *, gail_optim, variant_groups, **kwargs):
        super().__init__(**kwargs)
        self.gail_optim = gail_optim
        self.variant_groups = variant_groups
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
                # label traj_infos with env IDs (this is specific to
                # magical/my multi-task thing)
                traj_infos = _label_traj_infos(traj_infos, self.variant_groups)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)

                # run GAIL & combine its output with RL algorithm output
                gail_info = self.gail_optim.optim_disc(itr, n_itr, samples)
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

    def _log_infos(self, traj_infos=None):
        """Customised version of _log_infos that supports having different keys
        in each `TrajInfo`."""
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            values = defaultdict(lambda: [])
            for traj_info in traj_infos:
                for k, v in traj_info.items():
                    if not k.startswith("_"):
                        values[k].append(v)
            for k, vs in values.items():
                logger.record_tabular_misc_stat(k, vs)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)

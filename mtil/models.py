import warnings

import numpy as np
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
from torch import nn
from torch.nn import functional as F

from mtil.utils.misc import save_my_kwargs


class MILBenchPreprocLayer(nn.Module):
    """Takes a uint8 image in format [N,T,H,W,C] (a batch of several time steps
    of H,W,C images) or [N,H,W,C] (i.e. T=1 time steps) and returns float image
    (elements in [-1,1]) in format [N,C*T,H,W] for Torch."""
    def forward(self, x):
        assert x.dtype == torch.uint8, \
            f"expected uint8 tensor but got {x.dtype} tensor"

        assert len(x.shape) == 4, x.shape
        N, H, W, C = x.shape
        # just transpose channels axis to front, do nothing else
        x = x.permute((0, 3, 1, 2))

        assert (H, W) == (96, 96), \
            f"(height,width)=({H},{W}), but should be (96,96) (try " \
            f"resizing)"

        # convert format and scale to [0,1]
        x = x.to(torch.float32) / 127.5 - 1.0

        return x


def _conv_out_d(d, kernel_size=1, stride=1, padding=0, dilation=1):
    numerator = d + 2 * padding - dilation * (kernel_size - 1) - 1
    return int(np.floor(numerator / stride + 1))


def _pair(maybe_tup):
    # turn either a pair of integers or a single integer into a pair of
    # integers
    if len(maybe_tup) == 2:
        return maybe_tup
    # force cast to int
    value = int(maybe_tup)
    assert value == maybe_tup, (value, maybe_tup)
    return (value, value)


def conv_out_hw(hw, kernel_size=(1, 1), stride=1, padding=0, dilation=1):
    # Return (h',w') for input feature map of size (h,w) after convolution with
    # something that has given parameters. Formula taken from
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    h_in, w_in = hw
    h_out = _conv_out_d(h_in,
                        kernel_size=kernel_size[0],
                        stride=stride[0],
                        padding=padding[0],
                        dilation=dilation[0])
    w_out = _conv_out_d(w_in,
                        kernel_size=kernel_size[1],
                        stride=stride[1],
                        padding=padding[1],
                        dilation=dilation[1])
    return h_out, w_out


def compute_convnet_out_size(in_size, layers):
    size = in_size
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            size = conv_out_hw(size,
                               kernel_size=layer.kernel_size,
                               stride=layer.stride,
                               padding=layer.padding,
                               dilation=layer.dilation)
        else:
            # This is meant to check that it's an op that doesn't change shape,
            # but I was too lazy to add all the operations meeting that
            # criterion. If you get an assertion error then you'll have to
            # extend the tuple :)
            known_shape_preserving_layers = (nn.ReLU, nn.BatchNorm2d,
                                             nn.Dropout2d, CoordConv,
                                             ConvAttentionLayer)
            assert isinstance(layer, known_shape_preserving_layers), \
                f"is {layer} a shape-preserving op? If not then this " \
                f"function will break."
    return size


class CoordConv(nn.Module):
    """Add coordinates in [-1,1] to a convolution layer's input."""
    def forward(self, x):
        # needs N,C,H,W inputs
        assert x.ndim == 4
        h, w = x.shape[2:]
        ones_h = x.new_ones((h, 1))
        lin_h = torch.linspace(-1, 1, h, dtype=x.dtype,
                               device=x.device)[:, None]
        ones_w = x.new_ones((1, w))
        lin_w = torch.linspace(-1, 1, w, dtype=x.dtype,
                               device=x.device)[None, :]
        new_maps_2d = torch.stack((lin_h * ones_w, lin_w * ones_h), dim=0)
        new_maps_4d = new_maps_2d[None]
        assert new_maps_4d.shape == (1, 2, h, w), (x.shape, new_maps_4d.shape)
        batch_size = x.size(0)
        new_maps_4d_batch = new_maps_4d.repeat(batch_size, 1, 1, 1)
        result = torch.cat((x, new_maps_4d_batch), dim=1)
        return result


class ConvAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.mha = nn.MultiheadAttention(out_dim, num_heads)

    def forward(self, in_nchw):
        b, c, h, w = in_nchw.shape
        assert c == self.in_dim
        in_tbc = in_nchw.reshape((b, c, h * w)).permute(2, 0, 1)
        in_tbc_flat = in_tbc.reshape((b * h * w, c))
        query = self.query(in_tbc_flat).reshape((h * w, b, self.out_dim))
        key = self.key(in_tbc_flat).reshape((h * w, b, self.out_dim))
        value = self.value(in_tbc_flat).reshape((h * w, b, self.out_dim))
        out_tbo, _ = self.mha(query, key, value)
        assert out_tbo.shape == (h * w, b, self.out_dim)
        out_nohw = out_tbo.permute(1, 2, 0).reshape((b, self.out_dim, h, w))
        assert out_nohw.shape == (b, self.out_dim, h, w)
        return out_nohw


class MILBenchFeatureNetwork(nn.Module):
    """Convolutional feature extractor to process 128x128 images down into
    1024-dimensional feature vectors."""
    def __init__(self,
                 in_chans,
                 out_chans,
                 use_bn=False,
                 dropout=None,
                 coord_conv=False,
                 attention=False,
                 use_sn=False,
                 width=2,
                 ActivationCls=torch.nn.ReLU):
        super().__init__()
        w = width
        # Batch norm after conv has its own bias, so disable as necessary.
        # Undecide on what the best padding is. May have to experiment. I
        # suspect 'replicate' will make it less likely to pick up on absolute
        # position.
        conv_kwargs = dict(bias=not use_bn, padding_mode='zeros')
        extra_in = 2 if coord_conv else 0
        conv_out_dim = 64 * w
        conv_layers = [
            # at input: (96, 96)
            nn.Conv2d(in_chans + extra_in,
                      32 * w,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      **conv_kwargs),
            ActivationCls(),
            # now: (96, 96)
            nn.Conv2d(32 * w + extra_in,
                      64 * w,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      **conv_kwargs),
            ActivationCls(),
            # now: (48, 48)
            nn.Conv2d(64 * w + extra_in,
                      64 * w,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      **conv_kwargs),
            ActivationCls(),
            # now: (24, 24)
            nn.Conv2d(64 * w + extra_in,
                      64 * w,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      **conv_kwargs),
            ActivationCls(),
            # now: (12, 12)
            # TODO: a 12x12 feature map is a good place for attention :)
            # (Maybe a bit big? Ask Dan to see what transformers use.)
            nn.Conv2d(64 * w + extra_in,
                      conv_out_dim,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      **conv_kwargs),
            ActivationCls(),
            # now (6,6)
        ]

        # add CoordConv, BN & channel-wise dropout if appropriate
        new_conv_layers = []
        # This "bn_next" thing ensures that we put BN *after* the activation. I
        # doubt it matters much, but see here for CONTROVERSY:
        # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
        bn_next = False
        bn_chans = None
        for layer in conv_layers:
            is_conv = isinstance(layer, nn.Conv2d)
            if is_conv and coord_conv:
                new_conv_layers.append(CoordConv())
            new_conv_layers.append(layer)
            if is_conv and dropout:
                # assert a channel-wise dropout layer after convolution
                new_conv_layers.append(nn.Dropout2d(dropout))
            if bn_next:
                assert isinstance(layer, nn.ReLU), layer  # or other activation
                new_conv_layers.append(nn.BatchNorm2d(bn_chans))
                bn_next = False
                bn_chans = None
            if is_conv and use_bn:
                # insert BN layer after convolution (and optionally after
                # dropout)
                bn_next = True
                bn_chans = layer.out_channels
        conv_layers = new_conv_layers

        # add attention to the last layer, if appropriate
        if attention:
            # TODO: play with number of heads; might need quite a few for,
            # e.g., multi-task learning (or maybe I only need one, and I'm
            # overthinking things!)
            conv_layers.append(
                ConvAttentionLayer(conv_out_dim, conv_out_dim, 4))

        # final FC layer to make feature maps the right size
        out_size = compute_convnet_out_size((96, 96), conv_layers)
        fc_in_size = np.prod(out_size) * conv_out_dim
        if fc_in_size >= 10 * out_chans:
            # warn if we have a huge input to the last layer
            # (I chose the size threshold arbitrarily)
            warnings.warn(
                f"Output of conv layers in {type(self)} is of size "
                f"{fc_in_size}, but only {out_chans} channels will be "
                f"produced after final FC layer. Is the network too wide for "
                f"the task?")
        reduction_layers = [
            nn.Flatten(),
            # now: (6*6*32*w,)=(1152*w,)
            nn.Linear(fc_in_size, out_chans),
            ActivationCls(),
            # now: (out_chans,)
        ]
        if dropout:
            # also include dropout on the FC layer
            reduction_layers.append(nn.Dropout(dropout))
        all_layers = [*conv_layers, *reduction_layers]
        if use_sn:
            new_layers = []
            for layer in all_layers:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    layer = nn.utils.spectral_norm(layer)
                new_layers.append(layer)
            all_layers = new_layers
        self.feature_generator = nn.Sequential(*all_layers)

    def forward(self, x):
        flat_feats = self.feature_generator(x)
        return flat_feats


class MultiTaskAffineLayer(nn.Module):
    """A multi-task version of torch.nn.Linear. It keeps a separate set of
    weights for each of `n_tasks` tasks. On the forward pass, it takes a batch
    of task IDs in addition to a batch of feature inputs, and uses those to
    look up the appropriate affine transform to apply to each batch element."""
    def __init__(self, in_chans, out_chans, n_tasks, use_sn=False):
        super().__init__()

        # these "embeddings" are actually affine transformation parameters,
        # with weight matrix at the beginning and bias at the end
        self._embedding_shapes = [(out_chans, in_chans), (out_chans, )]
        self._embedding_sizes = [
            np.prod(shape) for shape in self._embedding_shapes
        ]
        full_embed_size = sum(self._embedding_sizes)
        self.task_embeddings = nn.Embedding(n_tasks, full_embed_size)

    def _retrieve_embeddings(self, task_ids):
        embeddings = self.task_embeddings(task_ids)
        stops = list(np.cumsum(self._embedding_sizes))
        starts = [0] + stops[:-1]
        reshaped = []
        for start, stop, shape in zip(starts, stops, self._embedding_shapes):
            full_shape = embeddings.shape[:-1] + shape
            part = embeddings[..., start:stop].view(full_shape)
            reshaped.append(part)
        return reshaped

    def forward(self, inputs, task_ids):
        # messy: must reshape "embeddings" into weight matrices & affine
        # parameters, then apply them like a batch of different affine layers
        matrices, biases \
            = self._retrieve_embeddings(task_ids)
        bc_fc_features = inputs[..., None]
        mm_result = torch.squeeze(matrices @ bc_fc_features, dim=-1)
        assert mm_result.shape == biases.shape, \
            (mm_result.shape, biases.shape)
        result = mm_result + biases

        return result


class SingleTaskAffineLayer(nn.Module):
    """MultiTaskAffineLayer-compatible layer that works only for a _single_
    task."""
    def __init__(self, in_chans, out_chans, use_sn):
        super().__init__()
        lin_layer = nn.Linear(in_chans, out_chans)
        if use_sn:
            lin_layer = nn.utils.spectral_norm(lin_layer)
        self.lin_layer = lin_layer

    def forward(self, inputs, task_ids):
        return self.lin_layer(inputs)


class MultiHeadPolicyNet(nn.Module):
    """Like MILBenchPolicyNet in bc.py, but for multitask policies with one
    head per environment. Returns both logits and values, for use in algorithms
    other than BC."""
    def __init__(self,
                 env_ids_and_names,
                 in_chans=3,
                 n_actions=3 * 3 * 2,
                 width=2,
                 use_bn=False,
                 dropout=None,
                 coord_conv=False,
                 attention=False,
                 n_task_spec_layers=1,
                 ActivationCls=torch.nn.ReLU):
        # save kwargs for rebuild_net
        self.__kwargs = save_my_kwargs(self.__init__, locals())
        assert 'self' not in self.__kwargs

        # init module so we can add parameters
        super().__init__()

        assert n_task_spec_layers >= 1, \
            f"must have >=1 task specific layers; got {n_task_spec_layers} " \
            f"layers instead"
        # TODO: save kwargs here!
        self.preproc = MILBenchPreprocLayer()
        fc_dim = 128 * width
        self.feature_extractor = MILBenchFeatureNetwork(
            in_chans=in_chans,
            out_chans=fc_dim,
            use_bn=use_bn,
            dropout=dropout,
            width=width,
            coord_conv=coord_conv,
            attention=attention,
            ActivationCls=ActivationCls)
        # TODO: Maybe I should scale the output width of the layer by the
        # number of tasks? IDK whether width of the network needs to increase
        # with task count.
        # TODO: also decide whether these postproc_layers are even necessary :P
        postproc_layers = [nn.Linear(fc_dim, fc_dim), ActivationCls()]
        if dropout:
            # don't allow insane values of dropout by default
            # (also, reminder that `dropout` value is p(drop), not the other
            # way around)
            assert 0 < dropout < 0.7, dropout
            postproc_layers.append(torch.nn.Dropout(dropout))
        self.fc_postproc = nn.Sequential(*postproc_layers)
        # this produces both a single value output and a vector of policy
        # logits for each sample
        self.mt_fc_layers = []
        for _ in range(n_task_spec_layers - 1):
            self.mt_fc_layers.extend([
                MultiTaskAffineLayer(fc_dim, fc_dim, len(env_ids_and_names)),
                ActivationCls(),
            ])
        self.mt_fc_layers.append(
            MultiTaskAffineLayer(fc_dim, n_actions + 1,
                                 len(env_ids_and_names)))
        for layer_num, layer in enumerate(self.mt_fc_layers):
            self.add_module(f'mt_fc_layers_{layer_num}', layer)

        # save env IDs and names so that we know how to reconstruct, as well as
        # fc_dim and n_actions so that we can do reshaping
        self.env_ids_and_names = sorted(env_ids_and_names)
        self.fc_dim = fc_dim
        self.n_actions = n_actions

    def rebuild_net(self, new_env_ids_and_names):
        """Build a copy of this network"""
        new_kwargs = self.__kwargs.copy()
        eid_key = 'env_ids_and_names'
        del new_kwargs[eid_key]
        new_kwargs[eid_key] = new_env_ids_and_names
        new_model = type(self)(**new_kwargs)

        # Here we build a list mapping old slot numbers to new slot numbers.
        # For this to even be representable as a list, we need
        # new_env_ids_and_names to have sequentially-assigned task IDs.
        new_names, new_ids = zip(*new_env_ids_and_names)
        assert min(new_ids) == 0 and max(new_ids) == len(new_ids) - 1, new_ids
        new_id_to_ename = dict(zip(new_ids, new_names))
        ename_to_old_id = dict(self.env_ids_and_names)
        new_ids = [
            ename_to_old_id[new_id_to_ename[new_id]]
            for new_id in range(len(new_ids))
        ]
        n_old_tasks = len(self.env_ids_and_names)
        n_new_tasks = len(new_env_ids_and_names)

        # now we build a new state dict, with some special handling for FC
        # layers
        old_state_dict = self.state_dict()
        new_state_dict = type(old_state_dict)()
        for key, value in old_state_dict.items():
            value = value.cpu()
            if 'mt_fc_layers_' not in key:
                new_state_dict[key] = value
                continue

            # It's multi-task, so we do some conversions. Note that
            # MultiTaskAffineLayers have only one weight, and it's of shape
            # [n_tasks, out*(in+1)]
            assert key.endswith('.task_embeddings.weight'), \
                f"unexpected key '{key}'---is this for a multitask layer?"
            assert value.dim() == 2 and value.size(0) == n_old_tasks, \
                "weird shape {value.shape} for {n_old_tasks}-task layer)"
            new_value = value[new_ids]
            assert new_value.shape == (n_new_tasks, value.size(1))
            new_state_dict[key] = new_value

        # finally, load up the new dict
        new_model.load_state_dict(new_state_dict, strict=True)

        return new_model

    def forward(self, obs, task_ids=None):
        if task_ids is None:
            # if the task is unambiguous, then it's fine not to pass IDs
            n_tasks = len(self.env_ids_and_names)
            assert n_tasks == 1, \
                "no task_ids given, but have {n_tasks} tasks to choose from"
            task_ids = obs.new_zeros(obs.shape[1:], dtype=torch.long)

        preproc = self.preproc(obs)
        features = self.feature_extractor(preproc)
        fc_features = self.fc_postproc(features)
        # apply multi-task layers
        for mt_fc_layer in self.mt_fc_layers:
            if isinstance(mt_fc_layer, MultiTaskAffineLayer):
                fc_features = mt_fc_layer(fc_features, task_ids)
            else:
                fc_features = mt_fc_layer(fc_features)
        logits_and_values = fc_features
        assert logits_and_values.shape[-1] == self.n_actions + 1, \
            (logits_and_values.shape, self.n_actions + 1)
        logits = logits_and_values[..., :-1]
        values = logits_and_values[..., -1]

        l_expected_shape = task_ids.shape + (self.n_actions, )
        assert logits.shape == l_expected_shape, \
            f"expected logits to be shape {l_expected_shape}, but got " \
            f"shape {logits.shape}"
        assert values.shape == task_ids.shape, \
            f"expected values to be shape {task_ids.shape}, but got " \
            f"shape {values.shape}"

        return logits, values


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
        logits, v = self.model(obs.view(T * B, *img_shape), task_ids)
        pi = F.softmax(logits, dim=-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v

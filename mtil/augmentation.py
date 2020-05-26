"""Utilities for on-device image augmentation with Kornia."""
import collections

import kornia.augmentation as aug
import torch
from torch import nn

from mtil.utils.colour import apply_lab_jitter


class KorniaAugmentations(nn.Module):
    """Container for Kornia augmentations. It does something like this:

    1. User gives byte tensor of the form [...,H,W,C] (i.e. byte-based TF image
       format). Any leading dimensions are acceptable. The only limitation is
       that the tensor should be a stack of RGB sub-images along the last
       dimension; hence, C needs to be a multiple of 3.
    2. Reshapes array to [N,H,W,C].
    3. Permute to [N,C,H,W] (PyTorch format), then reshape to [N*C/3,3,H,W]
       (split out channels).
    4. Convert to float tensor with values in [0,1].
    5. Apply a bunch of Kornia ops to get another float tensor with values in
       [0,1]. These ops can be passed into the constructor. Note that _any_
       Torch module can be passed in, not just a Kornia op :)
    6. Undo steps (2)-(4) and give the user back a byte tensor of the same
       shape that they started with, but containing augmented images.

    This is unnecessarily expensive. At some point it might be smarter just to
    make all images [C,H,W] floats during loading. There will be a 4x memory
    cost, but might save on compute (especially the permute ops, which
    suck)."""
    def __init__(self, *kornia_ops):
        super().__init__()
        self.kornia_ops = nn.Sequential(*kornia_ops)

    def forward(self, byte_tensor):
        # check type & number of dims
        assert byte_tensor.dtype == torch.uint8, byte_tensor.dtype
        assert byte_tensor.dim() >= 4, byte_tensor.shape
        # make sure this is a channels-last stack of RGB images
        stack_depth = byte_tensor.size(-1) // 3
        # the upper limit of 4 is just a sanity check; if it's much larger then
        # we might have been passed an [N,C,H,W] image or something instead of
        # an [N,H,W,C] image
        assert 1 <= stack_depth <= 4 \
            and stack_depth * 3 == byte_tensor.size(-1), byte_tensor.shape

        with torch.no_grad():
            # make sure we don't build a backward graph or anything
            lead_dims = byte_tensor.shape[:-3]
            # reshape to [N,H,W,C]
            byte_tensor = byte_tensor.reshape((-1, ) + byte_tensor.shape[-3:])
            # permute to [N,C,H,W]
            byte_tensor = byte_tensor.permute((0, 3, 1, 2))
            # convert to float & put in range [0,1]
            float_tensor = byte_tensor.float() / 255
            del byte_tensor

            # apply actual ops
            from mtil.utils.torch import save_normalised_image_tensor
            save_normalised_image_tensor(float_tensor, "batch_no_aug.png")
            float_tensor = self.kornia_ops(float_tensor)
            save_normalised_image_tensor(float_tensor, "batch_with_aug.png")
            assert False

            # convert back to byte
            out_tensor = torch.round(float_tensor * 255).byte()
            del float_tensor
            # permute back to [N,H,W,C]
            out_tensor = out_tensor.permute((0, 2, 3, 1))
            # restore leading dims
            out_tensor = out_tensor.reshape(lead_dims + out_tensor.shape[-3:])

        return out_tensor


class GaussianNoise(nn.Module):
    """Apply zero-mean Gaussian noise with a given standard deviation to input
    tensor."""
    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.std * torch.randn_like(x)
        out.clamp_(0., 1.)
        return out


class CIELabJitter(nn.Module):
    """Apply 'jitter' in CIELab colour space."""
    def __init__(self, max_lum_scale, max_uv_rads):
        super().__init__()
        self.max_lum_scale = max_lum_scale
        self.max_uv_rads = max_uv_rads

    def forward(self, x):
        # we take in stacked [N,C,H,W] images, where C=3*T. We then reshape
        # into [N,T,C,H,W] like apply_lab_jitter expects.
        stack_depth = x.size(1) // 3
        assert x.size(1) == 3 * stack_depth, x.shape
        x_reshape = x.view(x.shape[:1] + (stack_depth, 3) + x.shape[2:])
        jittered_reshape = apply_lab_jitter(x_reshape, self.max_lum_scale,
                                            self.max_uv_rads)
        jittered = jittered_reshape.view(x.shape)
        return jittered


class MILBenchAugmentations(KorniaAugmentations):
    """Convenience class for data augmentation. Has a standard set of possible
    augmentations with sensible pre-set values."""
    PRESETS = collections.OrderedDict([
        ('all', ['colour_jitter', 'translate', 'rotate', 'noise']),
        ('col', ['colour_jitter']),
        ('trans', ['translate']),
        ('rot', ['rotate']),
        ('transrot', ['translate', 'rotate']),
        ('trn', ['translate', 'rotate', 'noise']),
        ('noise', ['noise']),
        ('none', []),
    ])

    def __init__(self,
                 colour_jitter=False,
                 translate=False,
                 rotate=False,
                 noise=False):
        transforms = []
        if colour_jitter:
            transforms.append(CIELabJitter(max_lum_scale=1.03,
                                           max_uv_rads=0.3))
        if translate or rotate:
            transforms.append(
                aug.RandomAffine(degrees=(-5, 5) if rotate else (0, 0),
                                 translate=(0.05, 0.05) if translate else None,
                                 padding_mode='border'))
        if noise:
            # Remember that values lie in [0,1], so std=0.01 (for example)
            # means there's a >99% chance that any given noise value will lie
            # in [-0.03,0.03]. I think any value <=0.03 will probably be
            # reasonable.
            noise_mod = GaussianNoise(std=0.01)
            # JIT doesn't make it any faster (unsurprisingly)
            # noise_mod = torch.jit.script(noise_mod)
            transforms.append(noise_mod)
        super().__init__(*transforms)

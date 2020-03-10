"""Utilities for on-device image augmentation with Kornia."""
import kornia.augmentation as aug
import torch
from torch import nn


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
            float_tensor = self.kornia_ops(float_tensor)

            # convert back to byte
            out_tensor = torch.round(float_tensor * 255).byte()
            del float_tensor
            # permute back to [N,H,W,C]
            out_tensor = out_tensor.permute((0, 2, 3, 1))
            # restore leading dims
            out_tensor = out_tensor.reshape(lead_dims + out_tensor.shape[-3:])

        return out_tensor


class UnstackWrapper(nn.Module):
    """Wrapper layer for passing n-image RGB stacks to layers that only expect
    batches of three-channel RGB layers. Takes in an [N,C,H,W] tensor, resizes
    is to [N*C/3,3,H,W], passes that to given layer, then reshapes the result
    back to [N,C',H,W]"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        assert x.dim() == 4, x.shape
        n, c, *rest = x.shape
        assert c >= 0 and (c % 3) == 0, c
        x_reshaped = x.reshape([n * c // 3, 3] + rest)
        result_reshaped = self.layer(x_reshaped)
        assert result_reshaped.shape[2:] == x.shape[2:]
        result = result_reshaped.reshape(x.shape)
        return result


class GaussianNoise(nn.Module):
    """Apply zero-mean Gaussian noise with a given standard deviation to input
    tensor."""
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        out = x + self.std * torch.randn_like(x)
        out.clamp_(0., 1.)
        return out


class MILBenchAugmentations(KorniaAugmentations):
    """Convenience class for data augmentation. Has a standard set of possible
    augmentations with sensible pre-set values."""
    def __init__(self,
                 colour_jitter=False,
                 translate=False,
                 rotate=False,
                 noise=False):
        transforms = []
        if colour_jitter:
            transforms.append(
                UnstackWrapper(
                    aug.ColorJitter(brightness=0.05,
                                    contrast=0.05,
                                    saturation=0.05,
                                    hue=0.01)))
        if translate or rotate:
            transforms.append(
                aug.RandomAffine(degrees=(-5, 5) if rotate else (0, 0),
                                 translate=(0.05, 0.05) if translate else None,
                                 border_mode='border'))
        if noise:
            # Remember that values lie in [0,1], so std=0.01 (for example)
            # means there's a >99% chance that any given noise value will lie
            # in [-0.03,0.03]. I think any value <=0.03 will probably be
            # reasonable.
            transforms.append(GaussianNoise(std=0.01))
        super().__init__(*transforms)

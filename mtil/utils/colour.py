"""This module contains optimised, JITted versions of Kornia's colour space
conversion routines. Code is originally from Kornia; I've just updated it to be
compatible with torch.jit etc."""
from typing import Tuple

import torch


@torch.jit.script
def _lab_f(t: torch.Tensor) -> torch.Tensor:
    delta = 6 / 29.0
    return torch.where(t > delta**3, torch.pow(t, 1 / 3.0),
                       t / (3 * delta**2) + 4 / 29.0)


@torch.jit.script
def _lab_f_inv(t: torch.Tensor) -> torch.Tensor:
    delta = 6 / 29.0
    return torch.where(t > delta, torch.pow(t, 3),
                       3 * delta**2 * (t - 4 / 29.0))


@torch.jit.script
def rgb_to_lab(
        r: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # originally copied from kornia.color.luv.rgb_to_luv, then adapted to
    # L*a*b*.

    # # Convert from Linear RGB to sRGB
    # rs = torch.where(r > 0.04045, torch.pow(((r + 0.055) / 1.055), 2.4),
    #                  r / 12.92)
    # gs = torch.where(g > 0.04045, torch.pow(((g + 0.055) / 1.055), 2.4),
    #                  g / 12.92)
    # bs = torch.where(b > 0.04045, torch.pow(((b + 0.055) / 1.055), 2.4),
    #                  b / 12.92)

    # # sRGB to XYZ
    # x = 0.412453 * rs + 0.357580 * gs + 0.180423 * bs
    # y = 0.212671 * rs + 0.715160 * gs + 0.072169 * bs
    # z = 0.019334 * rs + 0.119193 * gs + 0.950227 * bs

    # skipping linear RGB <-> sRGB conversion because I don't understand
    # whether/why it is necessary for my data
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    # XYZ to Lab
    X_n = 0.950489
    Y_n = 1.000
    Z_n = 1.088840
    x_frac = _lab_f(x / X_n)
    y_frac = _lab_f(y / Y_n)
    z_frac = _lab_f(z / Z_n)
    L = 116 * y_frac - 16
    a = 500 * (x_frac - y_frac)
    b = 200 * (y_frac - z_frac)

    return L, a, b


@torch.jit.script
def lab_to_rgb(
        L: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # convert from Lab to XYZ
    X_n = 0.950489
    Y_n = 1.000
    Z_n = 1.088840
    L_16_frac = (L + 16) / 116.0
    x = X_n * _lab_f_inv(L_16_frac + a / 500.0)
    y = Y_n * _lab_f_inv(L_16_frac)
    z = Z_n * _lab_f_inv(L_16_frac - b / 200)

    # inlined call to rgb_to_xyz(torch.stack((x, y, z)))
    rs = 3.2404813432005266 * x + -1.5371515162713185 * y \
        + -0.4985363261688878 * z
    gs = -0.9692549499965682 * x + 1.8759900014898907 * y \
        + 0.0415559265582928 * z
    bs = 0.0556466391351772 * x + -0.2040413383665112 * y \
        + 1.0573110696453443 * z

    # Convert from sRGB to RGB Linear
    # r = torch.where(rs > 0.0031308, 1.055 * torch.pow(rs, 1 / 2.4) - 0.055,
    #                 12.92 * rs)
    # g = torch.where(gs > 0.0031308, 1.055 * torch.pow(gs, 1 / 2.4) - 0.055,
    #                 12.92 * gs)
    # b = torch.where(bs > 0.0031308, 1.055 * torch.pow(bs, 1 / 2.4) - 0.055,
    #                 12.92 * bs)

    # return torch.stack((r, g, b), dim=-3)

    # skipping linear RGB <-> sRGB conversion because I don't understand
    # whether/why it is necessary for my data
    return rs, gs, bs


@torch.jit.script
def _unif_rand_range(lo, hi, size, device):
    # type: (float, float, int, Device) -> Tensor  # noqa: F821

    # Note that I'm using type comments instead of Py3K type annotations
    # because Torch 1.4.0 doesn't seem to expose the "Device" type in the
    # normal way (using "torch.device" raises an error). The "noqa: F821" is to
    # stop flake8 from complaining about the imaginary "Device" type.

    return (hi - lo) * torch.rand((size, ), device=device) + lo


def apply_lab_jitter(images: torch.Tensor, max_lum_scale: float,
                     max_uv_rads: float) -> torch.Tensor:
    """Apply random L*a*b* jitter to each element of a batch of images. The
    `images` tensor should be of shape `[B,...,C,H,W]`, where the ellipsis
    denotes extraneous dimensions which will not be transformed separately
    (e.g. there might be a time axis present after the batch axis)."""

    assert len(images.shape) >= 4 and images.shape[-3] == 3
    assert 2.0 >= max_lum_scale >= 1.0
    assert max_uv_rads >= 0.0

    L, a, b = rgb_to_lab(images[..., 0, :, :], images[..., 1, :, :],
                         images[..., 2, :, :])

    # random transforms
    batch_size = images.size(0)
    ab_angles = _unif_rand_range(-max_uv_rads, max_uv_rads, batch_size,
                                 images.device)
    lum_scale_factors = _unif_rand_range(1.0 / max_lum_scale, max_lum_scale,
                                         batch_size, images.device)
    sines = torch.sin(ab_angles)
    cosines = torch.cos(ab_angles)

    # resize transformations to take advantage of broadcasting
    new_shape = (batch_size, ) + (1, ) * (a.ndim - 1)
    sines = sines.view(new_shape)
    cosines = cosines.view(new_shape)
    lum_scale_factors = lum_scale_factors.view(new_shape)

    # now apply the transformations
    # (this is way faster than stacking and using torch.matmul())
    trans_L = torch.clamp(L * lum_scale_factors, 0.0, 100.0)
    trans_a = cosines * a - sines * b
    trans_b = sines * a + cosines * b

    trans_r, trans_g, trans_b = lab_to_rgb(trans_L, trans_a, trans_b)

    rgb_trans = torch.stack((trans_r, trans_g, trans_b), dim=-3)

    # throw out colours that can't be expressed as RGB (this is probably not a
    # good way of doing it, but whatever)
    rgb_trans.clamp_(0.0, 1.0)

    return rgb_trans

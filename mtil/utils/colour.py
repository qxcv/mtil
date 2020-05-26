"""This module contains optimised, JITted versions of Kornia's colour space
conversion routines. Code is originally from Kornia; I've just updated it to be
compatible with torch.jit etc."""

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
def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    # originally copied from kornia.color.luv.rgb_to_luv, then adapted to
    # L*a*b*.

    # Convert from Linear RGB to sRGB
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

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

    return torch.stack((L, a, b), dim=-3)


@torch.jit.script
def lab_to_rgb(image: torch.Tensor) -> torch.Tensor:
    L = image[..., 0, :, :]
    a = image[..., 1, :, :]
    b = image[..., 2, :, :]

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
    return torch.stack((rs, gs, bs), dim=-3)


@torch.jit.script
def _unif_rand_range(lo, hi, size, device):
    # type: (float, float, int, Device) -> Tensor  # noqa: F821

    # Note that I'm using type comments instead of Py3K type annotations
    # because Torch 1.4.0 doesn't seem to expose the "Device" type in the
    # normal way (using "torch.device" raises an error). The "noqa: F821" is to
    # stop flake8 from complaining about the imaginary "Device" type.

    return (hi - lo) * torch.rand((size, ), device=device) + lo


@torch.jit.script
def generate_luv_jitter_mats(max_lum_scale, max_uv_rads, batch_size, device):
    # type: (float, float, int, Device) -> Tensor  # noqa: F821
    uv_angles = _unif_rand_range(-max_uv_rads, max_uv_rads, batch_size, device)
    lum_scale_factors = _unif_rand_range(1.0 / max_lum_scale, max_lum_scale,
                                         batch_size, device)

    # final transform matrices
    trans_mats = torch.zeros((batch_size, 3, 3), device=device)

    # we just do a random scaling on the L* channel
    trans_mats[:, 0, 0] = lum_scale_factors

    # we rotate the (a*,b*) channels together
    # (2D CCW rotation matrix: [[cos, -sin], [sin, cos]])
    sines = torch.sin(uv_angles)
    cosines = torch.cos(uv_angles)
    trans_mats[:, 1, 1] = cosines
    trans_mats[:, 1, 2] = -sines
    trans_mats[:, 2, 2] = cosines
    trans_mats[:, 2, 1] = sines

    return trans_mats


# @torch.jit.script
def apply_luv_jitter(images: torch.Tensor, max_lum_scale: float,
                     max_uv_rads: float) -> torch.Tensor:
    assert len(images.shape) == 4 and images.shape[1] == 3
    assert 2.0 >= max_lum_scale >= 1.0
    assert max_uv_rads >= 0.0

    batch_size = images.size(0)
    ndim = images.dim()
    assert ndim >= 4
    excess_dims = ndim - 4

    rand_mats = generate_luv_jitter_mats(max_lum_scale, max_uv_rads,
                                         batch_size, images.device)
    lab_images = rgb_to_lab(images)

    # FIXME: no, don't use the excess_dims thing; instead, flatten it all down!
    # go from B*(excess dims)*C*H*W to B*(excess dims)*H*W*C*1 for the sake of broadcasting
    lab_images_nhwc = lab_images.permute((0, 2, 3, 1))
    lab_images_bcast = lab_images_nhwc[..., None]
    # go from B*3*3 to B*(excess dims)*1*1*3*3 for the sake of broadcasting
    rand_mats.reshape(rand_mats.shape[1:] + (1, ) * (excess_dims + 2) +
                      rand_mats.shape[1:])
    rand_mats_bcast = rand_mats[:, None, None, :]
    trans_lab_images_bcast = torch.matmul(rand_mats_bcast, lab_images_bcast)
    trans_lab_images_nhwc = torch.squeeze(trans_lab_images_bcast, dim=-1)
    # clip L channel so that it's in the right range
    trans_lab_images_nhwc[..., 0] = torch.clamp(trans_lab_images_nhwc[..., 0],
                                                0.0, 100.0)
    trans_lab_images = trans_lab_images_nhwc.permute((0, 3, 1, 2))

    # rgb_trans = torch.clamp(lab_to_rgb(trans_lab_images), 0, 1)
    rgb_trans = lab_to_rgb(trans_lab_images)

    # throw out colours that can't be expressed as RGB (this is probably not a
    # good way of doing it, but whatever)
    rgb_trans.clamp_(0.0, 1.0)

    return rgb_trans

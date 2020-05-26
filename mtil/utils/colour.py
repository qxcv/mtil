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
    # X_n = 95.0489
    # Y_n = 100.0
    # Z_n = 108.8840
    X_n = 0.950489
    Y_n = 1.000
    Z_n = 1.088840
    x_frac = _lab_f(x / X_n)
    y_frac = _lab_f(y / Y_n)
    z_frac = _lab_f(z / Z_n)
    L = 116 * y_frac - 16
    a = 500 * (x_frac - y_frac)
    b = 200 * (y_frac - z_frac)

    # print("Lab ranges:")
    # print("  L range:", L.min().item(), L.max().item())
    # print("  a range:", a.min().item(), a.max().item())
    # print("  b range:", b.min().item(), b.max().item())

    return torch.stack((L, a, b), dim=-3)


@torch.jit.script
def lab_to_rgb(image: torch.Tensor) -> torch.Tensor:
    L = image[..., 0, :, :]
    a = image[..., 1, :, :]
    b = image[..., 2, :, :]

    # convert from Lab to XYZ
    # X_n = 95.0489
    # Y_n = 100.0
    # Z_n = 108.8840
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
def rgb_to_luv(image: torch.Tensor) -> torch.Tensor:
    # originally copied from kornia.color.luv.rgb_to_luv

    # Convert from Linear RGB to sRGB
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    rs: torch.Tensor = torch.where(r > 0.04045,
                                   torch.pow(((r + 0.055) / 1.055), 2.4),
                                   r / 12.92)
    gs: torch.Tensor = torch.where(g > 0.04045,
                                   torch.pow(((g + 0.055) / 1.055), 2.4),
                                   g / 12.92)
    bs: torch.Tensor = torch.where(b > 0.04045,
                                   torch.pow(((b + 0.055) / 1.055), 2.4),
                                   b / 12.92)

    # here I'm manually inlining a call of the form
    # kornia.color.xyz.rgb_to_xyz(torch.stack((rs, gs, bs)))
    x: torch.Tensor = 0.412453 * rs + 0.357580 * gs + 0.180423 * bs
    y: torch.Tensor = 0.212671 * rs + 0.715160 * gs + 0.072169 * bs
    z: torch.Tensor = 0.019334 * rs + 0.119193 * gs + 0.950227 * bs

    L: torch.Tensor = torch.where(torch.gt(y, 0.008856),
                                  116. * torch.pow(y, 1. / 3.) - 16.,
                                  903.3 * y)

    # eps: float = torch.finfo(torch.float64).eps  # For numerical stability
    eps: float = 1e-15  # (close enough to float64 machine epsilon)

    # Compute reference white point
    xyz_ref_white: Tuple[float, float, float] = (.95047, 1., 1.08883)
    u_w: float = (4 * xyz_ref_white[0]) / (
        xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = (9 * xyz_ref_white[1]) / (
        xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])

    u_p: torch.Tensor = (4 * x) / (x + 15 * y + 3 * z + eps)
    v_p: torch.Tensor = (9 * y) / (x + 15 * y + 3 * z + eps)

    u: torch.Tensor = 13 * L * (u_p - u_w)
    v: torch.Tensor = 13 * L * (v_p - v_w)

    out = torch.stack((L, u, v), dim=-3)

    return out


@torch.jit.script
def luv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # copied from kornia.color.luv.luv_to_rgb

    L: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]
    # Convert from Luv to XYZ
    y: torch.Tensor = torch.where(L > 7.999625, torch.pow((L + 16) / 116, 3.0),
                                  L / 903.3)
    # Compute white point
    xyz_ref_white: Tuple[float, float, float] = (0.95047, 1., 1.08883)
    u_w: float = (4 * xyz_ref_white[0]) / (
        xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w: float = (9 * xyz_ref_white[1]) / (
        xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])

    # eps: float = torch.finfo(torch.float64).eps  # For numerical stability
    eps: float = 1e-15  # (close enough to float64 machine epsilon)

    a: torch.Tensor = u_w + u / (13 * L + eps)
    d: torch.Tensor = v_w + v / (13 * L + eps)
    c: torch.Tensor = 3 * y * (5 * d - 3)

    z: torch.Tensor = ((a - 4) * c - 15 * a * d * y) / (12 * d + eps)
    x: torch.Tensor = -(c / (d + eps) + 3. * z)

    # inlined call to rgb_to_xyz(torch.stack((x, y, z)))
    rs: torch.Tensor = 3.2404813432005266 * x + -1.5371515162713185 * y \
        + -0.4985363261688878 * z
    gs: torch.Tensor = -0.9692549499965682 * x + 1.8759900014898907 * y \
        + 0.0415559265582928 * z
    bs: torch.Tensor = 0.0556466391351772 * x + -0.2040413383665112 * y \
        + 1.0573110696453443 * z

    # Convert from sRGB to RGB Linear
    r: torch.Tensor = torch.where(rs > 0.0031308,
                                  1.055 * torch.pow(rs, 1 / 2.4) - 0.055,
                                  12.92 * rs)
    g: torch.Tensor = torch.where(gs > 0.0031308,
                                  1.055 * torch.pow(gs, 1 / 2.4) - 0.055,
                                  12.92 * gs)
    b: torch.Tensor = torch.where(bs > 0.0031308,
                                  1.055 * torch.pow(bs, 1 / 2.4) - 0.055,
                                  12.92 * bs)

    rgb_im: torch.Tensor = torch.stack((r, g, b), dim=-3)

    return rgb_im


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

    # linf_dist = torch.max(torch.abs(images - lab_to_rgb(rgb_to_lab(images))))
    # lab_images = rgb_to_lab(images)
    # flat_lab = lab_images.permute((0, 2, 3, 1)).reshape((-1, 3))
    # lab_min, _ = torch.min(flat_lab, dim=0)
    # lab_max, _ = torch.max(flat_lab, dim=0)
    # print("Lab min/max:", lab_min.cpu().numpy(), lab_max.cpu().numpy())
    # print("Linf dist:", linf_dist)

    batch_size = images.size(0)
    ndim = images.dim()
    assert ndim >= 4
    excess_dims = ndim - 4

    # rand_mats = generate_luv_jitter_mats(max_lum_scale, max_uv_rads,
    #                                      batch_size, images.device)
    # luv_images = rgb_to_luv(images)

    # # go from B*C*H*W to B*H*W*C*1 for the sake of broadcasting
    # luv_images_nhwc = luv_images.permute((0, 2, 3, 1))
    # luv_images_bcast = luv_images_nhwc[..., None]
    # # go from B*3*3 to B*1*1*3*3 for the sake of broadcasting
    # rand_mats_bcast = rand_mats[:, None, None, :]
    # trans_luv_images_bcast = torch.matmul(rand_mats_bcast, luv_images_bcast)
    # # FIXME: clip so that everything is still in range
    # trans_luv_images_nhwc = torch.squeeze(trans_luv_images_bcast, dim=-1)
    # # TODO: also avoid this transpose by modifying rgb_to_luv
    # trans_luv_images = trans_luv_images_nhwc.permute((0, 3, 1, 2))
    # luv_flat = luv_images_nhwc.reshape((-1, 3))
    # luv_min = luv_flat.min(axis=0)[0].cpu().numpy()
    # luv_max = luv_flat.max(axis=0)[0].cpu().numpy()

    # rgb_trans = luv_to_rgb(trans_luv_images)

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

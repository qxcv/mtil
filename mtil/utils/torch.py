import re

import numpy as np
from rlpyt.utils.tensor import infer_leading_dims
import torch
from torch._six import container_abcs, int_classes, string_classes
from torch.utils.data.dataloader import default_collate
from torchvision import utils as vutils

_default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
_np_str_obj_array_pattern = re.compile(r'[SaUO]')


def fixed_default_collate(batch):
    """Copy-paste of `default_collate` for Torch's `DataLoader` class, but
    constructs namedtuples using `._make` instead of constructor (this
    accommodates the weird duck typing in rlpyt)."""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, (float, torch.Tensor)) or \
       isinstance(elem, int_classes) or isinstance(elem, string_classes):
        # let the original impl. handle these non-recursive cases
        return default_collate(batch)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # COPIED
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if _np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    _default_collate_err_msg_format.format(elem.dtype))

            return fixed_default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, container_abcs.Mapping):
        # COPIED
        return {
            key: fixed_default_collate([d[key] for d in batch])
            for key in elem
        }
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        # CHANGED: now uses elem._make() instead of type(elem)()
        return elem._make(
            tuple(fixed_default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # COPIED
        transposed = zip(*batch)
        return [fixed_default_collate(samples) for samples in transposed]

    # defer to original impl. for error handling etc.
    return default_collate(batch)


def save_normalised_image_tensor(image_tensor, file_name):
    """Save a normalised tensor of stacked images. Tensor values should be in
    [-1, 1], and tensor shape should be [Ni*3, H, W], or [B, Ni*3, H, W], or
    [T, B, Ni*3, H, W]. Here Ni is the number of stacked images, each of which
    are three-channel RGB."""
    lead_dim, T, B, shape = infer_leading_dims(image_tensor, 3)
    # make sure channels axis is (probably) a stack of RGB frames
    assert len(shape) == 3 and shape[0] < 30 \
        and 0 == (shape[0] % 3), shape
    # reshaping this way separates out each stacked image into its own frame
    Ni = shape[0] // 3
    flat_tensor = image_tensor.reshape((T * B * Ni, 3) + shape[1:])
    assert torch.all((-1.1 <= flat_tensor) & (flat_tensor <= 1.1)), \
        f"this only takes normalised images, but range is " \
        f"[{flat_tensor.min()}, {flat_tensor.max()}]"
    flat_tensor = torch.clamp(flat_tensor, -1., 1.)
    nrow = max(1, int(np.sqrt(flat_tensor.shape[0])))
    vutils.save_image(flat_tensor, file_name, nrow, range=(-1, 1))


def mixup(*tensors, alpha=0.2):
    """Copy of mixup implementation in Avi's code:

    https://github.com/avisingh599/reward-learning-rl/blob/93bb52f75bea850bd01f3c3342539f0231a561f3/softlearning/misc/utils.py#L164-L175

    This is not "real" mixup because it only mixes between elements of one
    batch, so there will be some redundancy (and could even be duplicate
    entries!). I expect it will work just as well though."""
    dist = torch.distributions.Beta(alpha, alpha)
    batch_size = tensors[0].shape[0]
    coeffs = dist.sample((batch_size, ))
    perm = torch.randperm(batch_size)

    out_tensors = []
    for in_tensor in tensors:
        assert in_tensor.shape[0] == batch_size, (in_tensor.shape, batch_size)
        perm_tensor = in_tensor[perm]
        # make sure broadcasting works correctly by appending training "1" dims
        bc_coeffs = coeffs.view((batch_size, ) + (1, ) *
                                (len(in_tensor.shape) - 1))
        result = bc_coeffs * in_tensor + (1 - bc_coeffs) * perm_tensor
        out_tensors.append(result)

    return tuple(out_tensors)


def repeat_dataset(loader):
    # equivalent to it.chain.from_iterable(it.repeat(loader)), except it will
    # raise an error if the given iterable is empty (instead of cycling
    # forever)
    while True:
        n_yielded = 0
        for element in loader:
            n_yielded += 1
            yield element
        if n_yielded == 0:
            raise ValueError("there aren't actually any elements in the "
                             "given loader...")

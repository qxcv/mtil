#!/usr/bin/env python3
"""Benchmarking augmentations."""

import timeit

import torch

from mtil.augmentation import MILBenchAugmentations


def gen_images(size=128, batch_size=32, depth=4, device='cuda'):
    """Generate byte-type NHWC images."""
    tensor_size = (batch_size, size, size, depth * 3)
    # must make it float type so we can use uniform_()
    data_float = torch.empty(tensor_size, dtype=torch.float, device=device)
    data_float.uniform_(0, 256)
    data_byte = data_float.byte()
    return data_byte


def time_augs(aug_kwargs, gen_kwargs, trials=100):
    """Time repeated application of some augmentations"""
    images = gen_images(**gen_kwargs)
    augmentor = MILBenchAugmentations(**aug_kwargs)
    context = {'images': images, 'augmentor': augmentor}
    statement = 'augmentor(images)'
    # warmup
    print("  Doing warmup runs")
    timeit.timeit(statement, globals=context, number=5)
    # actual result
    print("  Warmup done, doing actual runs")
    time = timeit.timeit(statement, globals=context, number=trials)
    mean = time / float(trials)
    print(f"  Mean time {mean:.3}s")
    return mean


def main():
    all_aug_kwargs = [
        (
            ('translate', True),
            ('rotate', False),
            ('noise', False),
            ('colour_jitter', False),
        ),
        (
            ('translate', True),
            ('rotate', True),
            ('noise', False),
            ('colour_jitter', False),
        ),
        (
            ('translate', True),
            ('rotate', True),
            ('noise', True),
            ('colour_jitter', False),
        ),
        # colour_jitter is just insanely expensive b/c of HSV conversions
        # (
        #     ('translate', True),
        #     ('rotate', True),
        #     ('noise', True),
        #     ('colour_jitter', True),
        # ),
    ]
    for aug_kwargs in all_aug_kwargs:
        kwargs_str = ", ".join(f"{k} = {v}" for k, v in aug_kwargs)
        print(f"Timing with {kwargs_str}:")
        time_augs(aug_kwargs=dict(aug_kwargs), gen_kwargs={})


if __name__ == '__main__':
    main()

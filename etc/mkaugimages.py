#!/usr/bin/env python3

import os
import readline  # noqa: F401

import click
import imageio
from magical.saved_trajectories import load_demos
import numpy as np
import skimage
import skimage.transform as sktrans
import torch

import mtil.augmentation as aug

FRAMES_PER_DEMO = 1
AUGS_PER_FRAME = 5
ANIM_FPS = 4
OUT_RES = 192


def make_cont_dir(path):
    # TODO: factor this out of mkimages.py
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def write_gif(path, frames, fps, loop=0):
    # TODO: factor this out of mkimages.py
    make_cont_dir(path)
    assert path.endswith('.gif'), path
    imageio.mimwrite(path, frames, format='gif', fps=fps, loop=loop)


@click.command()
@click.option('--dest', default='aug-image-data', help='destination dir')
@click.argument("demos", nargs=-1, required=True)
def main(dest, demos):
    """Make images illustrating each available augmentation, applied to some
    randomly selected frames from the given demos."""
    all_demos = load_demos(demos)
    all_frames = []
    print("Selecting frames")
    for task_num, demo_dict in enumerate(all_demos, start=1):
        # format of demo_obs: (T, H, W, C)
        this_demo_frames = [od['allo'] for od in demo_dict['trajectory'].obs]
        sel_inds = np.random.permutation(FRAMES_PER_DEMO)
        sel_frames = [this_demo_frames[i] for i in sel_inds]
        all_frames.extend([
            skimage.img_as_ubyte(sktrans.resize(f, (OUT_RES, OUT_RES)))
            for f in sel_frames
        ])

    all_frames_np = np.asarray(all_frames, dtype='object')
    np.random.shuffle(all_frames_np)
    all_frames_np = np.stack(all_frames_np, axis=0).astype('uint8')
    all_frames_np = np.repeat(all_frames_np, AUGS_PER_FRAME, axis=0)
    torch_frame_stack = torch.from_numpy(all_frames_np)

    print("Writing augmentations")
    for preset, opts in aug.MILBenchAugmentations.PRESETS.items():
        out_p = os.path.join(dest, preset.lower() + '.gif')
        print(f"  Results for augmentation '{preset}' go to '{out_p}'")
        kwargs = {k: True for k in opts}
        aug_mod = aug.MILBenchAugmentations(**kwargs)
        augmented = aug_mod(torch_frame_stack)
        aug_out = augmented.cpu().numpy()
        write_gif(out_p, aug_out, ANIM_FPS)


if __name__ == '__main__':
    main()

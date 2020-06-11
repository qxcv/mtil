#!/usr/bin/env python3
"""Extract preprocessed egocentric and allocentric views from a demonstration
and save them individually to some directory."""

import os

import click
import imageio
import magical
from magical.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper)
import numpy as np


@click.command()
@click.option('--save-dir',
              default='split-ea-demo',
              help='output directory for rendered demos')
@click.option('--preproc',
              default=['LoRes4E', 'LoRes4A'],
              multiple=True,
              help='specify which preprocessor to use (repeat if necessary)')
@click.argument('demo')
def main(demo, save_dir, preproc):
    magical.register_envs()

    demo_dict, = load_demos([demo])
    # keys of the demo are env_name, trajectory, score
    orig_env_name = demo_dict['env_name']
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    print(f"Will save all demos to '{save_dir}'")
    for preproc_name in preproc:
        preprocessed, = preprocess_demos_with_wrapper(
            [demo_dict['trajectory']],
            orig_env_name,
            preproc_name=preproc_name)
        print(f"Working on preprocessor '{preproc_name}'")
        for frame_idx, frame in enumerate(preprocessed.obs):
            frame_fname = 'frame-' + preproc_name.lower() \
                + f'-{frame_idx:03}.png'
            frame_path = os.path.join(save_dir, frame_fname)
            assert frame.shape[-1] % 3 == 0
            frame_stacked = frame.reshape(frame.shape[:-1] + (-1, 3))
            frame_trans = frame_stacked.transpose((2, 0, 1, 3))
            frame_wide = np.concatenate(frame_trans, axis=1)
            print(f"  Writing frame to '{frame_path}'")
            # make much larger so we can see pixel boundaries
            frame_wide = np.repeat(np.repeat(frame_wide, 8, axis=0), 8, axis=1)
            imageio.imsave(frame_path, frame_wide)


if __name__ == '__main__':
    main()

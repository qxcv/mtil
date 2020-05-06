#!/usr/bin/env python3
"""Make necessary survey media/HTML from a demonstration. Requires you to `pip
install scikit-video`."""
import os

import click
import gym
import imageio
from milbench.baselines.saved_trajectories import load_demos
from milbench.benchmarks import (DEMO_ENVS_TO_TEST_ENVS_MAP, EnvName,
                                 register_envs)
import numpy as np
import skimage
import skimage.transform as sktrans

DEMO_FPS = 8
OUT_RES = 192
SWAP_FPS = 4
RAND_FRAMES = 8


def make_rand_frames(env_name):
    env = gym.make(env_name)
    try:
        frames = []
        for _ in range(RAND_FRAMES):
            frame = env.reset()
            resized = skimage.img_as_ubyte(
                sktrans.resize(frame, (OUT_RES, OUT_RES)))
            frames.append(resized)
    finally:
        env.close()
        del env
    return np.stack(frames, axis=0)


def get_all_variants(demo_env_name):
    return [demo_env_name] + list(DEMO_ENVS_TO_TEST_ENVS_MAP[demo_env_name])


def make_cont_dir(path):
    dirname = os.path.dirname(path)
    if dirname:
        os.path.makedirs(dirname, exist_ok=True)


def write_gif(path, frames, fps, loop=0):
    make_cont_dir(path)
    assert path.endswith('.gif'), path
    imageio.mimwrite(path, frames, format='gif', fps=fps, loop=loop)


def write_image(path, image):
    make_cont_dir(path)
    imageio.imwrite(path, image)


@click.command()
@click.option('--dest', default='cs287h-data', help='destination dir')
@click.argument("demos", nargs=-1, required=True)
def main(demos, dest):
    """Make survey media/HTML from a given demonstration `demo`."""
    register_envs()
    all_demos = load_demos(demos)
    os.makedirs(dest, exist_ok=True)
    os.chdir(dest)
    for task_num, demo_dict in enumerate(all_demos, start=1):
        name_data = EnvName(demo_dict['env_name'])
        all_variants = get_all_variants(name_data.env_name)
        print('Env name', name_data.env_name, 'with variants', all_variants)

        # make static images and gifs
        for variant in all_variants:
            print('  Variant', variant)
            frames = make_rand_frames(variant)
            static_out = 'static-' + variant.lower() + '.png'
            write_image(static_out, frames[0])
            gif_out = 'anim-' + variant.lower() + '.gif'
            write_gif(gif_out, frames, SWAP_FPS)

        # format of demo_obs: (T, H, W, C)
        print('  Writing demo')
        demo_frames = np.stack([
            skimage.img_as_ubyte(sktrans.resize(f, (OUT_RES, OUT_RES)))
            for f in demo_dict['trajectory'].obs
        ],
                               axis=0)
        demo_out = 'demo-' + name_data.env_name.lower() + '.gif'
        write_gif(demo_out, demo_frames, DEMO_FPS)


if __name__ == '__main__':
    main()

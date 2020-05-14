#!/usr/bin/env python3
"""Make necessary survey media/HTML from a demonstration. Requires you to `pip
install scikit-video`."""
import os
import readline  # noqa: F401

import click
import gym
import imageio
from milbench.baselines.saved_trajectories import load_demos
from milbench.benchmarks import (DEMO_ENVS_TO_TEST_ENVS_MAP, EnvName,
                                 register_envs)
import numpy as np
import skimage
import skimage.morphology
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


def blend_frames(frames, subsample=1, med_pixel_only=False):
    """Blend together a T*H*W*C (type uint8) volume of video frames taken in
    front of a static background. Should yield decent results for my
    environments, but no guarantees outside of there.

    Original: https://gist.github.com/qxcv/99b859232fed10fdb0d462ed14f6b90b"""
    frames = np.stack([skimage.img_as_float(f) for f in frames], axis=0)
    if subsample:
        # subsample in convoluted way to ensure we always get last frame
        frames = frames[::-1][::subsample][::-1]
    med_frame = np.median(frames, axis=0)
    if med_pixel_only:
        med_pixel = np.median(frames.reshape((-1, frames.shape[-1])))
        med_frame[..., :] = med_pixel
    # our job is to find weights for frames st frames average out in the end
    frame_weights = np.zeros(frames.shape[:3] + (1, ))
    for frame_idx, frame in enumerate(frames):
        pixel_dists = np.linalg.norm(frame - med_frame, axis=2)
        diff_pixel_mask = pixel_dists > 0.05
        # fade in by a few pixels
        for i in range(4):
            eroded = skimage.morphology.erosion(diff_pixel_mask)
            diff_pixel_mask = 0.5 * eroded + 0.5 * diff_pixel_mask
        frame_weights[frame_idx, :, :, 0] = diff_pixel_mask
    # give later frames a bonus for blending over top of others
    # (edit: removed this because it led to ugly artefacts in some places)
    # frame_range = np.arange(len(frames)) \
    #     .reshape(-1, 1, 1, 1).astype('float32')
    # frame_weights *= 1 + frame_range
    # normalise frame weights while avoiding division by zero
    frame_weight_sums = frame_weights.sum(axis=0)[None, ...]
    frame_weights = np.divide(frame_weights,
                              frame_weight_sums,
                              where=frame_weight_sums > 0)
    # now denormalize so that later frames get brighter than earlier
    # ones
    n = len(frame_weights)
    min_alpha = 0.6
    age_descale = min_alpha + (1 - min_alpha) * np.arange(n) / (n - 1)
    age_descale = age_descale.reshape((-1, 1, 1, 1))
    frame_weights = frame_weights * age_descale
    frame_weight_sums = frame_weights.sum(axis=0)
    # finally blend
    combined_frames = np.sum(frames * frame_weights, axis=0)
    combined_frames += med_frame * (1 - frame_weight_sums)
    # clip bad pixel values
    combined_frames[combined_frames < 0] = 0
    combined_frames[combined_frames > 1] = 1
    byte_frames = np.stack([skimage.img_as_ubyte(f) for f in combined_frames],
                           axis=0)
    return byte_frames


def remove_dupes(old_frames, thresh=1e-3):
    old_frames = [skimage.img_as_float(f) for f in old_frames]
    new_frames = [old_frames[0]]
    for frame in old_frames[1:]:
        prev_frame = new_frames[-1]
        mean_diff = np.mean(np.abs(frame - prev_frame))
        if mean_diff >= thresh:
            new_frames.append(frame)
    result = np.stack([skimage.img_as_ubyte(f) for f in new_frames], axis=0)
    dropped = len(old_frames) - len(result)
    if dropped > 0:
        print(f'  Dropped {dropped} redundant frames ({len(result)} left)')
    return result


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

        # static_demo_anim_out = 'demo-sanim-' + name_data.env_name.lower(
        # ) + '.png'
        # trunc_demo_frames = remove_dupes(demo_frames)
        # subsample = max(1, int(np.floor(len(trunc_demo_frames) / 3.0)))
        # static_demo_anim = blend_frames(trunc_demo_frames,
        #                                 subsample,
        #                                 med_pixel_only=False)
        # write_image(static_demo_anim_out, static_demo_anim)

        static_demo_anim_out_root = 'demo-sanim-' \
            + name_data.env_name.lower()
        static_demo_anim_out_start = static_demo_anim_out_root + '-start.png'
        static_demo_anim_out_end = static_demo_anim_out_root + '-end.png'
        trunc_demo_frames = remove_dupes(demo_frames)
        trunc_demo_frames_start = trunc_demo_frames[:8]
        trunc_demo_frames_end = trunc_demo_frames[-8:]
        static_demo_anim_start = blend_frames(trunc_demo_frames_start, 2)
        static_demo_anim_end = blend_frames(trunc_demo_frames_end, 2)
        write_image(static_demo_anim_out_start, static_demo_anim_start)
        write_image(static_demo_anim_out_end, static_demo_anim_end)

        demo_montage_out = 'demo-mont-' + name_data.env_name.lower() + '.png'
        skip = int(np.ceil(len(trunc_demo_frames) / 5.0))
        indices = np.arange(0, len(trunc_demo_frames), skip)
        indices[-1] = len(trunc_demo_frames) - 1  # always include last frame
        sel_frames = trunc_demo_frames[indices]
        # stack them horizontally
        demo_montage = np.concatenate(sel_frames, axis=1)
        write_image(demo_montage_out, demo_montage)


if __name__ == '__main__':
    main()

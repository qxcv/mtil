#!/usr/bin/env python3
"""Make necessary survey media/HTML from a demonstration. Requires you to `pip
install scikit-video`."""
import json
import os
import random

import click
import gym
from milbench.baselines.saved_trajectories import load_demos
from milbench.benchmarks import EnvName, register_envs
import numpy as np
import skimage
import skimage.io as skio
import skimage.transform as sktrans
import skvideo.io as vidio

DEMO_FPS = 8
OUT_RES = 192
MAX_DEMOS = 10


@click.command()
@click.option('--ntest', default=3, help='number of test frames to generate')
@click.option('--dest', default='survey-data', help='destination dir')
@click.argument("demos", nargs=-1, required=True)
def main(demos, ntest, dest):
    """Make survey media/HTML from a given demonstration `demo`."""
    random.seed(42)
    data_prefix = 'data'
    print("Loading demos")
    all_demos = load_demos(demos)
    print("Done")
    demos_by_env = {}
    for dd in all_demos:
        demos_by_env.setdefault(dd['env_name'],
                                []).append(dd['trajectory'].obs)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(dest, exist_ok=True)
    os.chdir(dest)
    task_data = []
    items = list(demos_by_env.items())
    random.shuffle(items)
    items = iter(items)  # so things get evicted from memory faster
    for task_num, (env_name, all_demo_obs) in enumerate(items, start=1):
        name_data = EnvName(env_name)
        test_name = name_data.name_prefix + '-TestAll' \
            + name_data.version_suffix
        print('Env name', name_data.env_name, 'with test variant', test_name)

        # make out dir
        out_dir = os.path.join(data_prefix, name_data.name_prefix.lower())
        os.makedirs(out_dir, exist_ok=True)

        # make test observations
        register_envs()
        test_env = gym.make(test_name)
        test_files = []
        for test_num in range(ntest):
            test_frame = test_env.reset()
            test_frame = sktrans.resize(test_frame, (OUT_RES, OUT_RES))
            out_file = os.path.join(out_dir, f'test-{test_num}.png')
            print("Writing test frame to", out_file)
            skio.imsave(out_file, test_frame)
            test_files.append({
                'num': test_num + 1,
                'path': out_file,
                'task_num': task_num,
            })
        test_env.close()
        del test_env

        # format of demo_obs: (T, H, W, C)
        random.shuffle(all_demo_obs)
        demo_obs = np.concatenate(all_demo_obs[:MAX_DEMOS], axis=0)
        demo_obs = np.stack([
            skimage.img_as_ubyte(sktrans.resize(f, (OUT_RES, OUT_RES)))
            for f in demo_obs
        ],
                            axis=0)
        demo_path = os.path.join(out_dir, f'demo-0.mp4')
        print('Writing demo to', demo_path)
        vidio.vwrite(demo_path,
                     demo_obs,
                     outputdict={
                         '-r': str(DEMO_FPS),
                         '-vcodec': 'libx264',
                         '-pix_fmt': 'yuv420p',
                     })

        # save this for later
        task_data.append({
            'env_name': name_data.env_name,
            'task_num': task_num,
            'demo_video': demo_path,
            'test_images': test_files,
            'res': OUT_RES,
        })

    # do pjson hack to write out data
    with open('data.js', 'w') as out_fp:
        print(
            '// registers pre-created tasks; for use with survey '
            'HTML/javascript only',
            file=out_fp)
        json_data = json.dumps(task_data)
        print(f'registerData({json_data});', file=out_fp)

    with open(os.path.join(script_dir, 'survey.html'), 'r') as fp:
        survey_html = fp.read()

    with open('index.html', 'w') as fp:
        fp.write(survey_html)


if __name__ == '__main__':
    main()

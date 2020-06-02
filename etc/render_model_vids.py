#!/usr/bin/env python3
import os

import click
import gym
from milbench import DEMO_ENVS_TO_TEST_ENVS_MAP, EnvName, register_envs
import numpy as np
import skvideo.io as vidio
import torch
import torch.nn.functional as F

from mtil.algos.mtbc.mtbc import get_latest_path
from mtil.utils.misc import load_state_dict_or_model, set_seeds

DEFAULT_FPS = 120
DEFAULT_NTRAJ = 10
DEFAULT_SEED = 42


def unwrap_env(env):
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    return inner


def generate_video(model,
                   dev,
                   env_name,
                   out_path,
                   ntraj=DEFAULT_NTRAJ,
                   seed=DEFAULT_SEED,
                   fps=DEFAULT_FPS):
    # figure out the right task ID
    task_id = get_task_id(env_name, model.env_ids_and_names)
    task_ids = np.asarray([task_id], dtype='int64')
    torch_task_ids = torch.from_numpy(task_ids).to(dev)

    # use the right env name (grandfathering in old models)
    # TODO: remove this after you're done with the data from NeurIPS 2020; this
    # is only for old models
    env_name = fix_old_env_name(env_name)

    # load the env & set all seeds
    set_seeds(seed)
    env = gym.make(env_name)
    env.seed(seed)
    unwrapped = unwrap_env(env)
    act_range = np.arange(env.action_space.n)

    # we'll iteratively write the video instead of storing intermediate frames
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = vidio.FFmpegWriter(out_path,
                                outputdict={
                                    '-r': str(fps),
                                    '-vcodec': 'libx264',
                                    '-pix_fmt': 'yuv420p',
                                })

    traj_done = 0
    while traj_done < ntraj:
        # reset env
        pol_obs = env.reset()

        # record first observation
        rend_obs = unwrapped.render(mode='rgb_array')
        frame = np.concatenate((rend_obs['allo'], rend_obs['ego']), axis=1)
        writer.writeFrame(frame)

        # step until done
        done = False
        while not done:
            # get an action
            with torch.no_grad():
                torch_pol_obs = torch.from_numpy(pol_obs).to(dev)
                (logits_torch,), _ = model(torch_pol_obs[None], torch_task_ids)
                pi = F.softmax(logits_torch, dim=0).cpu().numpy()
                pi = pi / pi.sum()
                action = np.random.choice(act_range, p=pi)

            # step the env
            pol_obs, _, done, infos = env.step(action)

            # write next frame
            rend_obs = unwrapped.render(mode='rgb_array')
            frame = np.concatenate((rend_obs['allo'], rend_obs['ego']), axis=1)
            writer.writeFrame(frame)

            if done:
                print(f'  Done traj {traj_done+1}/{ntraj}, score is',
                      infos['eval_score'])

        traj_done += 1

    writer.close()


def get_task_id(env_name, task_ids_and_demo_env_names):
    parsed = EnvName(env_name)
    for train_env_name, task_id in task_ids_and_demo_env_names:
        if train_env_name == parsed.demo_env_name:
            return task_id
    raise ValueError(
        f"can't find ID for train version of env '{env_name}' from list "
        f"'{task_ids_and_demo_env_names}'")


def get_test_envs(task_ids_and_demo_env_names):
    test_envs = []
    for train_env_name, task_id in sorted(set(task_ids_and_demo_env_names)):
        # en = EnvName(train_env_name)
        test_envs.append(train_env_name)
        test_envs.extend(DEMO_ENVS_TO_TEST_ENVS_MAP[train_env_name])
    return test_envs


def generate_vid_name(env_name):
    parsed = EnvName(env_name)
    short_name = parsed.name_prefix + parsed.demo_test_spec
    return 'vid-' + short_name.lower() + '.mp4'


def fix_old_env_name(old_env_name):
    # TODO: can remove this after I'm done with the runs for NeurIPS 2020; it's
    # just to grandfather in some old models
    parsed = EnvName(old_env_name)
    if parsed.env_name_suffix == '-LoResStack':
        # need to change this to LoRes4E to match new code
        new_name = parsed.name_prefix + parsed.demo_test_spec \
                   + '-LoRes4E' + parsed.version_suffix
        return new_name
    return old_env_name


@click.command()
@click.option("--out-dir", default=None, help="output dir")
@click.option("--gpu-idx", default=None, type=int, help="GPU index to use")
@click.argument("model-dir")
def main(model_dir, out_dir, gpu_idx):
    """Load the latest snapshot in `model-dir` (if any), and render a video for
    each test variant corresponding to the train variants in the model."""
    set_seeds(DEFAULT_SEED)
    register_envs()

    if gpu_idx is None:
        dev = torch.device('cpu')
    else:
        dev = torch.device(f'cuda:{gpu_idx}')

    latest_path = os.path.join(model_dir, "itr_LATEST.pkl")
    latest_path = get_latest_path(latest_path)
    model = load_state_dict_or_model(latest_path)
    model = model.to(dev)

    test_envs = get_test_envs(model.env_ids_and_names)

    if out_dir is None:
        out_dir = model_dir

    for test_env in test_envs:
        # will be something like, e.g., "movetocorner-testjitter.mp4"
        vid_name = generate_vid_name(test_env)
        out_path = os.path.join(out_dir, vid_name)
        print(f"Writing video for '{test_env}' to '{out_path}'")
        generate_video(model=model,
                       env_name=test_env,
                       out_path=out_path,
                       dev=dev)


if __name__ == '__main__':
    main()

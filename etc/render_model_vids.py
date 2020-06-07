#!/usr/bin/env python3
import collections
import os
import re
import readline  # noqa: F401 (saves the debugger!)

import click
import gym
from magical import DEMO_ENVS_TO_TEST_ENVS_MAP, EnvName, register_envs
import numpy as np
import ray
import skvideo.io as vidio
import torch
import torch.nn.functional as F

from mtil.algos.mtbc.mtbc import get_latest_path
from mtil.utils.misc import load_state_dict_or_model, set_seeds

DEFAULT_FPS = 120
NTRAJ_PER_MODEL = 3
DEFAULT_SEED = 42


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


def unwrap_env(env):
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    return inner


def combine_frame(full_obs, pol_obs):
    """Combine full observation (a dict produced by the .render() method of an
    unwrapped env) with the stacked observation that the policy actually sees
    (e.g. produced by LoRes4E or similar)."""

    # left of the picture shows allo view (left) and ego view (right)
    frame_left = np.concatenate((full_obs['allo'], full_obs['ego']), axis=1)

    # right of the picture shows the actual frames observed by the agent,
    # stacked vertically too
    frame_right_t = pol_obs.reshape(pol_obs.shape[:2] +
                                    (pol_obs.shape[-1] // 3, ) + (3, ))
    frame_right = np.concatenate(frame_right_t.transpose((2, 0, 1, 3)), axis=0)

    # now we stack them horizontally (the maths here only works out for
    # preprocessors that give four frames to the agent, each downsampled at a
    # /2 rate along each dimension)
    frame = np.concatenate((frame_left, frame_right), axis=1)

    return frame


# TODO: convert this to a generator so I can deal with multiple dirs
def generate_frames(*, model, dev, env_name, ntraj, seed, fps):
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
    traj_done = 0
    while traj_done < ntraj:
        # reset env
        pol_obs = env.reset()

        # record first observation
        # FIXME: unroll code for generating frame. Also, make sure that you
        # include the agent's observation in the generated frame, perhaps
        # stacked horizontally.
        rend_obs = unwrapped.render(mode='rgb_array')
        frame = combine_frame(rend_obs, pol_obs)
        yield frame

        # step until done
        done = False
        while not done:
            # get an action
            with torch.no_grad():
                torch_pol_obs = torch.from_numpy(pol_obs).to(dev)
                (logits_torch, ), _ = model(torch_pol_obs[None],
                                            torch_task_ids)
                pi = F.softmax(logits_torch, dim=0).cpu().numpy()
            pi = pi / pi.sum()
            action = np.random.choice(act_range, p=pi)

            # step the env
            pol_obs, _, done, infos = env.step(action)

            # write next frame
            rend_obs = unwrapped.render(mode='rgb_array')
            frame = combine_frame(rend_obs, pol_obs)
            yield frame

            if done:
                print(f'    Done traj {traj_done+1}/{ntraj}, score is',
                      infos['eval_score'])

        traj_done += 1

    env.close()


def process_directories(model_dirs, out_dir, gpu_idx=0):
    """Load the latest snapshot in each of the given model directories, and
    render a video for each test variant corresponding to the train variants in
    the model. Note that the video will show all models in sequence."""
    set_seeds(DEFAULT_SEED)
    register_envs()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if gpu_idx is None:
        dev = torch.device('cpu')
    else:
        dev = torch.device(f'cuda:{gpu_idx}')

    # we're going to keep a video writer for each env
    writers_by_env = {}

    for model_idx, model_dir in enumerate(model_dirs):
        print(f"Loading model from '{model_dir}' "
              f"(model {model_idx+1}/{len(model_dirs)})")
        latest_path = get_latest_path(os.path.join(model_dir,
                                                   "itr_LATEST.pkl"))
        model = load_state_dict_or_model(latest_path)
        model = model.to(dev)
        test_envs = get_test_envs(model.env_ids_and_names)

        for test_env in test_envs:
            if test_env not in writers_by_env:
                vid_name = generate_vid_name(test_env)
                out_path = os.path.join(out_dir, vid_name)
                print(f"  Writing video for '{test_env}' to '{out_path}'")
                writer = vidio.FFmpegWriter(out_path,
                                            outputdict={
                                                '-r': str(DEFAULT_FPS),
                                                '-vcodec': 'libx264',
                                                '-pix_fmt': 'yuv420p',
                                            })
                writers_by_env[test_env] = writer
            else:
                print(f"  Writing video for '{test_env}'")
                writer = writers_by_env.get(test_env)
            # will be something like, e.g., "movetocorner-testjitter.mp4"
            for frame in generate_frames(model=model,
                                         env_name=test_env,
                                         dev=dev,
                                         seed=DEFAULT_SEED + model_idx,
                                         ntraj=NTRAJ_PER_MODEL,
                                         fps=DEFAULT_FPS):
                writer.writeFrame(frame)

    for writer in writers_by_env.values():
        writer.close()


_ALG_NAME_RE = re.compile(r'^run_(?P<alg_name>.+)-s(?P<seed>\d+)$')


def parse_run_dir(run_dir):
    # run dir format is "run_<alg-name>-s<seed>". Want to return just the
    # algorithm name
    match = _ALG_NAME_RE.match(run_dir)
    if match is None:
        raise ValueError(f"Don't know how to parse run dir name '{run_dir}'")
    alg_name = match.group('alg_name')
    return alg_name


@click.command()
@click.option("--out-root", default="videos", help="output dir")
@click.option("--nprocs", default=4, help="number of processes to spin up")
@click.option("--job-ngpus", default=1.0, help="number of GPUs per job")
@click.argument("model-dirs", nargs=-1, required=True)
def main(model_dirs, out_root, nprocs, job_ngpus):
    """Load the latest snapshot in `model-dir` (if any), and render a video for
    each test variant corresponding to the train variants in the model."""
    register_envs()

    ray.init(num_cpus=nprocs)
    remote_process_dirs = ray.remote(num_gpus=job_ngpus)(process_directories)

    grouped_dirs = collections.OrderedDict()
    for model_dir in sorted(set(model_dirs)):
        base_dir = os.path.basename(model_dir.strip('/'))
        parsed_alg = parse_run_dir(base_dir)
        grouped_dirs.setdefault(parsed_alg, []).append(model_dir)

    handles = []
    for group_alg, model_dirs in grouped_dirs.items():
        out_dir = os.path.join(out_root, group_alg)
        print(f"Writing videos for group '{group_alg}' ({len(model_dirs)} "
              f"directories) to '{out_dir}'")
        remote_handle = remote_process_dirs.remote(model_dirs, out_dir)
        handles.append(remote_handle)
    ray.get(handles)


if __name__ == '__main__':
    main()

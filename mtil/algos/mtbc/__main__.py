import os
import sys
import time

import click
import gym
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
from milbench.evaluation import EvaluationProtocol, latexify_results
import numpy as np
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.logging import logger
import torch

from mtil.algos.bc import eval_model
from mtil.algos.mtbc.mtbc import (FixedTaskModelWrapper, MultiHeadPolicyNet,
                                  copy_model_into_sampler,
                                  do_epoch_training_mt, make_env_tag)
from mtil.common import (MILBenchGymEnv, make_logger_ctx, set_seeds,
                         trajectories_to_loader_mt)


def load_state_dict_or_model(state_dict_or_model_path):
    """Load a model from a path to either a state dict or a full PyTorch model.
    If it's just a state dict path then the corresponding model path will be
    inferred (assuming the state dict was produced by `train`). This is useful
    for the `test` and `testall` commands."""
    state_dict_or_model_path = os.path.abspath(state_dict_or_model_path)
    cpu_dev = torch.device('cpu')
    state_dict_or_model = torch.load(state_dict_or_model_path,
                                     map_location=cpu_dev)
    if not isinstance(state_dict_or_model, dict):
        print(f"Treating supplied path '{state_dict_or_model_path}' as "
              f"policy (type {type(state_dict_or_model)})")
        model = state_dict_or_model
    else:
        state_dict = state_dict_or_model['model_state']
        state_dict_dir = os.path.dirname(state_dict_or_model_path)
        # we save full model once at beginning of training so that we have
        # architecture saved; not sure how to support loading arbitrary models
        fm_path = os.path.join(state_dict_dir, 'full_model.pt')
        print(f"Treating supplied path '{state_dict_or_model_path}' as "
              f"state dict to insert into model at '{fm_path}'")
        model = torch.load(fm_path, map_location=cpu_dev)
        model.load_state_dict(state_dict)

    return model


def wrap_model_for_fixed_task(model, env_name):
    """Wrap a loaded multi-task model in a `FixedTaskModelWrapper` that _only_
    uses the weights for the given env. Useful for `test` and `testall`."""
    # contra its name, .env_ids_and_names is list of tuples of form
    # (environment name, numeric environment ID)
    env_name_to_id = dict(model.env_ids_and_names)
    if env_name not in env_name_to_id:
        env_names = ', '.join(
            [f'{name} ({eid})' for name, eid in model.env_ids_and_names])
        raise ValueError(
            f"Supplied environment name '{env_name}' is not supported by "
            f"model. Supported names (& IDs) are: {env_names}")
    env_id = env_name_to_id[env_name]
    # this returns (pi, v)
    ft_wrapper = FixedTaskModelWrapper(task_id=env_id,
                                       model_ctor=None,
                                       model_kwargs=None,
                                       model=model)
    return ft_wrapper


@click.group()
def cli():
    pass


# TODO: abstract all these options out into a common set of options, possibly
# by using Sacred
@cli.command()
@click.option(
    "--add-preproc",
    default="LoResStack",
    type=str,
    help="add preprocessor to the demos and test env (default: 'LoResStack')")
@click.option("--use-gpu/--no-use-gpu", default=False, help="use GPU")
@click.option("--gpu-idx", default=0, help="index of GPU to use")
@click.option("--seed", default=42, help="PRNG seed")
@click.option("--batch-size", default=32, help="batch size")
@click.option("--epochs", default=100, help="epochs of training to perform")
@click.option("--out-dir", default="scratch", help="dir for snapshots/logs")
@click.option("--eval-n-traj",
              default=10,
              help="number of trajectories to roll out on each evaluation")
@click.option("--run-name",
              default=None,
              type=str,
              help="unique name for this run")
# set this to some big value if training on perceptron or something
@click.option(
    "--passes-per-eval",
    default=1,
    help="num training passes through full dataset between evaluations")
@click.argument("demos", nargs=-1, required=True)
def train(demos, use_gpu, add_preproc, seed, batch_size, epochs, out_dir,
          run_name, gpu_idx, eval_n_traj, passes_per_eval):
    # TODO: abstract this setup code (roughly: everything up to
    # 'trajectories_to_loader()') so that I don't have to keep rewriting it for
    # every IL method.

    # set up seeds & devices
    set_seeds(seed)
    use_gpu = use_gpu and torch.cuda.is_available()
    dev = torch.device(["cpu", f"cuda:{gpu_idx}"][use_gpu])
    print(f"Using device {dev}, seed {seed}")

    # register original envs
    import milbench
    milbench.register_envs()

    # load demos (this code copied from bc.py in original baselines)
    demo_dicts = load_demos(demos)
    orig_env_names = [d['env_name'] for d in demo_dicts]
    orig_names_uniq = sorted(set(orig_env_names))
    if add_preproc:
        env_names = [
            splice_in_preproc_name(orig_env_name, add_preproc)
            for orig_env_name in orig_env_names
        ]
        print(f"Splicing preprocessor '{add_preproc}' into environments "
              f"{orig_names_uniq}. New names are {sorted(set(env_names))}")
    else:
        env_names = orig_env_names
    # pair of (original name, name with preprocessor spliced in)
    name_pairs = sorted(set(zip(orig_env_names, env_names)))
    demo_trajs_by_env = {
        env_name: [
            demo_dict['trajectory'] for demo_dict in demo_dicts
            if demo_dict['env_name'] == orig_env_name
        ]
        for orig_env_name, env_name in name_pairs
    }
    assert sum(map(len, demo_trajs_by_env.values())) == len(demo_dicts)
    if add_preproc:
        demo_trajs_by_env = {
            env_name:
            preprocess_demos_with_wrapper(demo_trajs_by_env[env_name],
                                          orig_env_name, add_preproc)
            for orig_env_name, env_name in name_pairs
        }
    loader_mt, env_name_to_id, env_id_to_name = trajectories_to_loader_mt(
        demo_trajs_by_env, batch_size)
    dataset_len = len(loader_mt)
    env_ids_and_names = [(name, env_name_to_id[name])
                         for _, name in name_pairs]

    # model kwargs will be filled in when we start our first env
    model_kwargs = None
    model_ctor = MultiHeadPolicyNet
    env_ctor = MILBenchGymEnv

    samplers = []
    agents = []
    for orig_env_name, env_name in name_pairs:
        env_ctor_kwargs = dict(env_name=env_name)
        env = gym.make(env_name)
        max_steps = env.spec.max_episode_steps

        # set model kwargs if necessary
        if model_kwargs is None:
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                in_chans = obs_shape[-1]
            else:
                # frame stacking
                in_chans = obs_shape[-1] * obs_shape[0]
            n_actions = env.action_space.n

            model_kwargs = {
                'env_ids_and_names': env_ids_and_names,
                'in_chans': in_chans,
                'n_actions': n_actions,
            }

        env_sampler = SerialSampler(env_ctor,
                                    env_ctor_kwargs,
                                    batch_T=max_steps,
                                    max_decorrelation_steps=max_steps,
                                    batch_B=min(eval_n_traj, batch_size))
        env_agent = CategoricalPgAgent(ModelCls=FixedTaskModelWrapper,
                                       model_kwargs=dict(
                                           model_ctor=model_ctor,
                                           model_kwargs=model_kwargs,
                                           task_id=env_name_to_id[env_name]))
        env_sampler.initialize(env_agent, seed=np.random.randint(1 << 31))
        env_agent.to_device(dev.index if use_gpu else None)

        samplers.append(env_sampler)
        agents.append(env_agent)

    model_mt = MultiHeadPolicyNet(**model_kwargs).to(dev)
    opt_mt = torch.optim.Adam(model_mt.parameters(), lr=3e-4)

    n_uniq_envs = len(orig_names_uniq)
    with make_logger_ctx(out_dir, "mtbc", f"mt{n_uniq_envs}", run_name):
        # initial save
        torch.save(model_mt,
                   os.path.join(logger.get_snapshot_dir(), 'full_model.pt'))

        # train for a while
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs} ({dataset_len} batches "
                  f"* {passes_per_eval} passes between evaluations)")

            loss_ewma, losses, per_task_losses = do_epoch_training_mt(
                loader_mt, model_mt, opt_mt, dev, passes_per_eval)

            print(f"Evaluating {eval_n_traj} trajectories on "
                  f"{len(name_pairs)} envs")
            record_misc_calls = []
            for (orig_env_name,
                 env_name), sampler in zip(name_pairs, samplers):
                copy_model_into_sampler(model_mt, sampler)
                model_mt.eval()
                scores = eval_model(sampler, n_traj=eval_n_traj)
                tag = make_env_tag(orig_env_name)
                logger.record_tabular_misc_stat("Score%s" % tag, scores)
                env_id = env_name_to_id[env_name]
                env_losses = per_task_losses.get(env_id, [])
                record_misc_calls.append((f"Loss{tag}", env_losses))
            # we record score AFTER loss so that losses are all in one place,
            # and scores are all in another
            for args in record_misc_calls:
                logger.record_tabular_misc_stat(*args)

            # finish logging for this epoch
            logger.record_tabular("Epoch", epoch)
            logger.record_tabular("LossEWMA", loss_ewma)
            logger.record_tabular_misc_stat("Loss", losses)
            logger.dump_tabular()
            logger.save_itr_params(
                epoch, {
                    'model_state': model_mt.state_dict(),
                    'opt_state': opt_mt.state_dict(),
                })


@cli.command()
@click.option("--env-name",
              default="MoveToCorner-Demo-LoResStack-v0",
              help="name of env to get policy for")
@click.option("--transfer-to",
              default=None,
              help="optionally specify a different env name to instantiate")
@click.option("--det-pol/--no-det-pol",
              default=False,
              help="should actions be sampled deterministically?")
# @click.option("--use-gpu/--no-use-gpu", default=False, help="use GPU")
# @click.option("--gpu-idx", default=0, help="index of GPU to use")
@click.option("--seed", default=42, help="PRNG seed")
@click.option("--fps",
              default=None,
              type=int,
              help="force frames per second to this value instead of default")
@click.argument('state_dict_or_model_path')
def test(state_dict_or_model_path, env_name, det_pol, seed, fps, transfer_to):
    """Repeatedly roll out a policy on a given environment. Mostly useful for
    visual debugging; see `testall` for quantitative evaluation."""
    set_seeds(seed)

    import milbench
    milbench.register_envs()

    # build env
    if transfer_to:
        env = gym.make(transfer_to)
    else:
        env = gym.make(env_name)

    model = load_state_dict_or_model(state_dict_or_model_path)
    ft_wrapper = wrap_model_for_fixed_task(model, env_name)

    spf = 1.0 / (env.fps if fps is None else fps)
    act_range = np.arange(env.action_space.n)
    obs = env.reset()
    try:
        while env.viewer.isopen:
            # for limiting FPS
            frame_start = time.time()
            # return value is actions, values, states, neglogp
            torch_obs = torch.from_numpy(obs)
            with torch.no_grad():
                (pi_torch, ), _ = ft_wrapper(torch_obs[None], None, None)
                pi = pi_torch.cpu().numpy()
            if det_pol:
                action = np.argmax(pi)
            else:
                # numpy is super complain-y about things "not summing to 1"
                pi = pi / sum(pi)
                action = np.random.choice(act_range, p=pi)
            obs, rew, done, info = env.step(action)
            obs = np.asarray(obs)
            env.render(mode='human')
            if done:
                print(f"Done, score {info['eval_score']:.4g}/1.0")
                obs = env.reset()
            elapsed = time.time() - frame_start
            if elapsed < spf:
                time.sleep(spf - elapsed)
    finally:
        env.viewer.close()


class MTBCEvalProtocol(EvaluationProtocol):
    def __init__(self, ft_wrapper, run_id, seed, det_pol, **kwargs):
        super().__init__(**kwargs)
        self.ft_wrapper = ft_wrapper
        self._run_id = run_id
        self.seed = seed
        self.det_pol = det_pol

    @property
    def run_id(self):
        return self._run_id

    def obtain_scores(self, env_name):
        print(f"Testing on {env_name}")

        env = gym.make(env_name)
        # use same seed for each instantiated test env (maybe this is a bad
        # idea? IDK.)
        rng = np.random.RandomState(self.seed)
        env.seed(rng.randint(0, 1 << 31 - 1))
        # alternative:
        # env_hash_digest = hashlib.md5(env_name.encode('utf8')).digest()
        # env_seed = struct.unpack('>I', env_hash_digest[:4])
        # env.seed(self.seed ^ env_seed)

        scores = []
        for _ in range(self.n_rollouts):
            act_range = np.arange(env.action_space.n)
            obs = env.reset()
            done = False
            while not done:
                torch_obs = torch.from_numpy(obs)
                with torch.no_grad():
                    (pi_torch, ), _ = self.ft_wrapper(torch_obs[None], None,
                                                      None)
                    pi = pi_torch.cpu().numpy()
                if self.det_pol:
                    action = np.argmax(pi)
                else:
                    pi = pi / sum(pi)
                    action = rng.choice(act_range, p=pi)
                obs, rew, done, info = env.step(action)
                obs = np.asarray(obs)
                if done:
                    scores.append(info['eval_score'])

        env.close()

        return scores


@cli.command()
@click.option("--env-name",
              default="MoveToCorner-Demo-LoResStack-v0",
              help="name of env to get policy for")
@click.option("--det-pol/--no-det-pol",
              default=False,
              help="should actions be sampled deterministically?")
# @click.option("--use-gpu/--no-use-gpu", default=False, help="use GPU")
# @click.option("--gpu-idx", default=0, help="index of GPU to use")
@click.option("--seed", default=42, help="PRNG seed")
@click.option("--fps",
              default=None,
              type=int,
              help="force frames per second to this value instead of default")
@click.option("--write-latex",
              default=None,
              help="write LaTeX table to this file")
@click.option("--latex-alg-name",
              default="UNK",
              help="algorithm name for LaTeX")
@click.option("--n-rollouts",
              default=10,
              help="number of rollouts to execute in each test config")
@click.argument('state_dict_or_model_path')
def testall(state_dict_or_model_path, env_name, det_pol, seed, fps,
            write_latex, latex_alg_name, n_rollouts):
    """Run quantitative evaluation on all test variants of a given
    environment."""
    # TODO: is there some way of factoring this init code out? Maybe put into
    # Click base command so that it gets run for `train`, `testall`, etc.
    set_seeds(seed)
    import milbench
    milbench.register_envs()

    model = load_state_dict_or_model(state_dict_or_model_path)
    ft_wrapper = wrap_model_for_fixed_task(model, env_name)

    eval_protocol = MTBCEvalProtocol(ft_wrapper=ft_wrapper,
                                     seed=seed,
                                     det_pol=det_pol,
                                     run_id=state_dict_or_model_path,
                                     demo_env_name=env_name,
                                     n_rollouts=n_rollouts)

    # next bit copied from testall() in bc.py
    frame = eval_protocol.do_eval(verbose=True)
    frame['latex_alg_name'] = latex_alg_name

    if write_latex:
        latex_str = latexify_results(frame, id_column='latex_alg_name')
        dir_path = os.path.dirname(write_latex)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(write_latex, 'w') as fp:
            fp.write(latex_str)

    return frame


if __name__ == '__main__':
    try:
        with cli.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            result = cli.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        if e.exit_code == 0:
            sys.exit(e.exit_code)
        raise

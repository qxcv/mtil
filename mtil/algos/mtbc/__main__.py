import contextlib
import multiprocessing as mp
import os
import readline  # noqa: F401
import time

import click
import gym
from milbench.evaluation import EvaluationProtocol, latexify_results
import numpy as np
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.logging import logger
import torch

from mtil.algos.mtbc.mtbc import (copy_model_into_agent_eval,
                                  do_epoch_training_mt, eval_model,
                                  eval_model_st, get_latest_path, make_env_tag,
                                  saved_model_loader_ft, strip_mb_preproc_name,
                                  wrap_model_for_fixed_task)
from mtil.augmentation import MILBenchAugmentations
from mtil.demos import get_demos_meta, make_loader_mt
from mtil.sample_mux import make_mux_sampler
from mtil.utils.misc import (load_state_dict_or_model, sane_click_init,
                             set_seeds)
from mtil.utils.rlpyt import (MILBenchGymEnv, make_agent_policy_mt,
                              make_logger_ctx)


@click.group()
def cli():
    pass


# TODO: abstract some of these options according to the logical role they play
# (perhaps by using Sacred).
@cli.command()
@click.option(
    "--add-preproc",
    default="LoResStack",
    type=str,
    help="add preprocessor to the demos and test env (default: 'LoResStack')")
@click.option("--gpu-idx", default=None, help="index of GPU to use")
@click.option("--seed", default=42, help="PRNG seed")
@click.option("--batch-size", default=32, help="batch size")
@click.option("--epochs", default=50, help="epochs of training to perform")
@click.option("--out-dir", default="scratch", help="dir for snapshots/logs")
@click.option("--eval-n-traj",
              default=10,
              help="number of trajectories to roll out on each evaluation")
@click.option("--run-name",
              default=None,
              type=str,
              help="unique name for this run")
@click.option("--omit-noop/--no-omit-noop",
              default=True,
              help="omit demonstration (s,a) pairs whenever a is a noop")
@click.option("--net-width-mul", default=2, help="width multiplier for net")
@click.option("--net-use-bn/--no-net-use-bn",
              default=True,
              help="use batch norm in net?")
@click.option("--net-dropout",
              default=0.0,
              help="dropout p(drop) for all layers of policy (default: 0)")
@click.option("--net-coord-conv/--no-net-coord-conv",
              default=False,
              help="enable (x,y) coordinate inputs for the policy conv layers")
@click.option("--net-attention/--no-net-attention",
              default=False,
              help="enable attention over final conv layer of policy net")
@click.option("--aug-mode",
              type=click.Choice([
                  "none",
                  "recol",
                  "trans",
                  "rot",
                  "noise",
                  "transrot",
                  "trn",
                  "all",
              ]),
              default="trn",
              help="augmentations to use")
# set this to some big value if training on perceptron or something
@click.option(
    "--passes-per-eval",
    default=20,
    help="num training passes through full dataset between evaluations")
@click.option("--snapshot-gap",
              default=10,
              help="how many evals to wait for before saving snapshot")
@click.argument("demos", nargs=-1, required=True)
def train(demos, add_preproc, seed, batch_size, epochs, out_dir, run_name,
          gpu_idx, eval_n_traj, passes_per_eval, snapshot_gap, omit_noop,
          net_width_mul, net_use_bn, net_dropout, net_coord_conv,
          net_attention, aug_mode):
    # TODO: abstract setup code. Seeds & GPUs should go in one function. Env
    # setup should go in another function (or maybe the same function). Dataset
    # loading should be simplified by having a single class that can provide
    # whatever form of data the current IL method needs, without having to do
    # unnecessary copies in memory. Maybe also just use Sacred, because YOLO.
    with contextlib.ExitStack() as exit_stack:
        # set up seeds & devices
        set_seeds(seed)
        mp.set_start_method('spawn')
        use_gpu = gpu_idx is not None and torch.cuda.is_available()
        dev = torch.device(["cpu", f"cuda:{gpu_idx}"][use_gpu])
        print(f"Using device {dev}, seed {seed}")
        cpu_count = mp.cpu_count()
        n_workers = max(1, cpu_count // 2)
        affinity = dict(
            cuda_idx=gpu_idx if use_gpu else None,
            # workers_cpus=list(np.random.permutation(cpu_count)[:n_workers])
            workers_cpus=list(range(n_workers)),
        )

        # register original envs
        import milbench
        milbench.register_envs()

        # TODO: maybe make this a class so that I don't have to pass around a
        # zillion attrs and use ~5 lines just to load some demos?
        # TODO: split out part of the dataset for validation. (IDK whether to
        # do this trajectory-wise or what)
        # dataset_mt, env_name_to_id, env_id_to_name, name_pairs \
        #     = load_demos_mt(demos, add_preproc, omit_noop=omit_noop)
        # loader_mt = make_loader_mt(dataset_mt, batch_size)

        demos_metas_dict = get_demos_meta(demo_paths=demos,
                                          omit_noop=omit_noop,
                                          transfer_variants=[],
                                          preproc_name=add_preproc)
        dataset_mt = demos_metas_dict['dataset_mt']
        loader_mt = make_loader_mt(dataset_mt, batch_size)
        variant_groups = demos_metas_dict['variant_groups']
        env_metas = demos_metas_dict['env_metas']
        task_ids_and_demo_env_names = demos_metas_dict[
            'task_ids_and_demo_env_names']
        sampler_batch_B = batch_size
        # this doesn't really matter
        sampler_batch_T = 5
        sampler, sampler_batch_B = make_mux_sampler(
            variant_groups=variant_groups,
            env_metas=env_metas,
            use_gpu=use_gpu,
            batch_B=sampler_batch_B,
            batch_T=sampler_batch_T)
        agent, policy_ctor, policy_kwargs = make_agent_policy_mt(
            env_metas, task_ids_and_demo_env_names)
        sampler.initialize(agent=agent,
                           seed=np.random.randint(1 << 31),
                           affinity=affinity)
        exit_stack.callback(lambda: sampler.shutdown())

        model_mt = policy_ctor(**policy_kwargs).to(dev)
        # Adam mostly works fine, but in very loose informal tests it seems
        # like SGD had fewer weird failures where mean loss would jump up by a
        # factor of 2x for a period (?). (I don't think that was solely due to
        # high LR; probably an architectural issue.) opt_mt =
        # torch.optim.Adam(model_mt.parameters(), lr=3e-4)
        opt_mt = torch.optim.SGD(model_mt.parameters(), lr=1e-3, momentum=0.1)

        try:
            aug_opts = MILBenchAugmentations.PRESETS[aug_mode]
        except KeyError:
            raise ValueError(f"unsupported mode '{aug_mode}'")
        if aug_opts:
            print("Augmentations:", ", ".join(aug_opts))
            aug_model = MILBenchAugmentations(**{k: True for k in aug_opts}) \
                .to(dev)
        else:
            print("No augmentations")
            aug_model = None

        n_uniq_envs = len(task_ids_and_demo_env_names)
        log_params = {
            'n_uniq_envs': n_uniq_envs,
            'n_demos': len(demos),
            'net_use_bn': net_use_bn,
            'net_width_mul': net_width_mul,
            'net_dropout': net_dropout,
            'net_coord_conv': net_coord_conv,
            'net_attention': net_attention,
            'aug_mode': aug_mode,
            'seed': seed,
            'eval_n_traj': eval_n_traj,
            'passes_per_eval': passes_per_eval,
            'omit_noop': omit_noop,
            'batch_size': batch_size,
            'epochs': epochs,
            'snapshot_gap': snapshot_gap,
            'add_preproc': add_preproc,
        }
        with make_logger_ctx(out_dir,
                             "mtbc",
                             f"mt{n_uniq_envs}",
                             run_name,
                             snapshot_gap=snapshot_gap,
                             log_params=log_params):
            # initial save
            torch.save(
                model_mt,
                os.path.join(logger.get_snapshot_dir(), 'full_model.pt'))

            # train for a while
            for epoch in range(epochs):
                dataset_len = len(dataset_mt)
                print(
                    f"Starting epoch {epoch+1}/{epochs} ({dataset_len} "
                    f"batches * {passes_per_eval} passes between evaluations)")

                model_mt.train()
                loss_ewma, losses, per_task_losses = do_epoch_training_mt(
                    loader_mt, model_mt, opt_mt, dev, passes_per_eval,
                    aug_model)

                # TODO: record accuracy on a random subset of the train and
                # validation sets (both in eval mode, not train mode)

                print(f"Evaluating {eval_n_traj} trajectories on "
                      f"{variant_groups.num_tasks} tasks")
                record_misc_calls = []
                model_mt.eval()

                copy_model_into_agent_eval(model_mt, sampler.agent)
                scores_by_tv = eval_model(sampler,
                                          itr=epoch,
                                          n_traj=eval_n_traj)
                for (task_id, variant_id), scores in scores_by_tv.items():
                    tv_id = (task_id, variant_id)
                    env_name = variant_groups.env_name_by_task_variant[tv_id]
                    tag = make_env_tag(strip_mb_preproc_name(env_name))
                    logger.record_tabular_misc_stat("Score%s" % tag, scores)
                    env_losses = per_task_losses.get(tv_id, [])
                    record_misc_calls.append((f"Loss{tag}", env_losses))

                # we record score AFTER loss so that losses are all in one
                # place, and scores are all in another
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


# TODO: factor this out into mtbc.py
class MTBCEvalProtocol(EvaluationProtocol):
    def __init__(self, demo_env_name, state_dict_or_model_path, run_id, seed,
                 gpu_idx, affinity, batch_size, **kwargs):
        super().__init__(demo_env_name=demo_env_name, **kwargs)
        self._run_id = run_id
        self.seed = seed
        self.gpu_idx = gpu_idx
        self.affinity = affinity
        self.batch_size = batch_size
        self.demo_env_name = demo_env_name
        self.state_dict_or_model_path = state_dict_or_model_path

    @property
    def run_id(self):
        return self._run_id

    def obtain_scores(self, env_name):
        print(f"Testing on {env_name}")

        use_gpu = self.gpu_idx is not None
        if use_gpu:
            SamplerCls = GpuSampler
        else:
            SamplerCls = CpuSampler

        env_ctor = MILBenchGymEnv
        env_ctor_kwargs = dict(env_name=env_name)
        env = gym.make(env_name)
        max_steps = env.spec.max_episode_steps
        env.close()
        del env

        env_sampler = SamplerCls(
            env_ctor,
            env_ctor_kwargs,
            batch_T=max_steps,
            # don't decorrelate, it will fuck up the
            # scores
            max_decorrelation_steps=0,
            batch_B=min(self.n_rollouts, self.batch_size))
        env_agent = CategoricalPgAgent(
            ModelCls=saved_model_loader_ft,
            model_kwargs=dict(
                state_dict_or_model_path=self.state_dict_or_model_path,
                env_name=self.demo_env_name))
        env_sampler.initialize(env_agent,
                               seed=self.seed,
                               affinity=self.affinity)
        dev = torch.device(["cpu", f"cuda:{self.gpu_idx}"][use_gpu])
        env_agent.to_device(dev.index if use_gpu else None)
        try:
            scores = eval_model_st(env_sampler, 0, self.n_rollouts)
        finally:
            env_sampler.shutdown()

        return scores


@cli.command()
@click.option("--env-name",
              default="MoveToCorner-Demo-LoResStack-v0",
              help="name of env to get policy for")
# @click.option("--det-pol/--no-det-pol",
#               default=False,
#               help="should actions be sampled deterministically?")
@click.option("--gpu-idx", default=None, help="index of GPU to use (if any)")
@click.option("--seed", default=42, help="PRNG seed")
@click.option("--fps",
              default=None,
              type=int,
              help="force frames per second to this value instead of default")
@click.option("--write-latex",
              default=None,
              help="write LaTeX table to this file")
@click.option("--run-id",
              default=None,
              type=str,
              help="override default run identifier in Pandas frame")
@click.option("--write-csv",
              default=None,
              type=str,
              help="write Pandas frame to this file as CSV")
@click.option("--latex-alg-name",
              default=None,
              help="algorithm name for LaTeX")
@click.option("--n-rollouts",
              default=10,
              help="number of rollouts to execute in each test config")
@click.option("--batch-size", default=32, help="batch size for eval")
@click.option("--load-latest/--no-load-latest",
              default=False,
              help="replaces the last '*' in the state_dict_or_model_path "
              "basename with the highest integer that yields a valid path")
@click.argument('state_dict_or_model_path')
def testall(state_dict_or_model_path, env_name, seed, fps, write_latex,
            latex_alg_name, n_rollouts, run_id, write_csv, gpu_idx, batch_size,
            load_latest):
    """Run quantitative evaluation on all test variants of a given
    environment."""
    # TODO: is there some way of factoring this init code out? Maybe put into
    # Click base command so that it gets run for `train`, `testall`, etc.
    set_seeds(seed)
    import milbench
    milbench.register_envs()

    # for parallel GPU/CPU sampling
    mp.set_start_method('spawn')
    use_gpu = gpu_idx is not None and torch.cuda.is_available()
    dev = torch.device(["cpu", f"cuda:{gpu_idx}"][use_gpu])
    print(f"Using device {dev}, seed {seed}")
    cpu_count = mp.cpu_count()
    n_workers = max(1, cpu_count // 2)
    affinity = dict(cuda_idx=gpu_idx if use_gpu else None,
                    workers_cpus=list(range(n_workers)))

    if load_latest:
        state_dict_or_model_path = get_latest_path(state_dict_or_model_path)

    if run_id is None:
        run_id = state_dict_or_model_path

    eval_protocol = MTBCEvalProtocol(
        demo_env_name=env_name,
        state_dict_or_model_path=state_dict_or_model_path,
        seed=seed,
        # det_pol=det_pol,
        run_id=run_id,
        gpu_idx=gpu_idx,
        affinity=affinity,
        n_rollouts=n_rollouts,
        batch_size=batch_size,
    )

    # next bit copied from testall() in bc.py
    frame = eval_protocol.do_eval(verbose=True)
    if latex_alg_name is None:
        latex_alg_name = run_id
    frame['latex_alg_name'] = latex_alg_name

    if write_latex:
        latex_str = latexify_results(frame, id_column='latex_alg_name')
        dir_path = os.path.dirname(write_latex)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(write_latex, 'w') as fp:
            fp.write(latex_str)

    if write_csv:
        dir_path = os.path.dirname(write_csv)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        frame.to_csv(write_csv)

    return frame


if __name__ == '__main__':
    sane_click_init(cli)

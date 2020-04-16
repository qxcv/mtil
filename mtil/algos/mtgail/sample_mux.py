"""Multi-task samplers for rlpyt. Useful for performing round-robin sampling
from some specified set of environments. To do that, it makes the following
modifications to the normal sampler & environments:

(1) Modifies sampler so that it can use a different environment class for each
    element of any given batch.
(2) Creates an environment wrapper that includes task ID along with each
    observation.
(3) Has some helpers for constructing an appropriate array of class IDs &
    instantiating envs, samplers, etc."""
import multiprocessing as mp

from gym import Wrapper
import numpy as np
from rlpyt.envs.base import EnvSpaces
from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.worker import initialize_worker
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.collections import (AttrDict, NamedArrayTupleSchema,
                                     NamedTupleSchema)
from rlpyt.utils.logging import logger
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
import torch
import torch.nn.functional as F

from mtil.common import AgentModelWrapper, MILBenchGymEnv


def _mux_sampler(common_kwargs, worker_kwargs):
    """Variant of `rlpyt.samplers.parallel.worker.sampling_process` that is
    able to supply different environment keyword arguments to each environment
    that makes up a batch."""
    c, w = AttrDict(**common_kwargs), AttrDict(**worker_kwargs)
    initialize_worker(w.rank, w.seed, w.cpus, c.torch_threads)
    # vvv CHANGED LINES vvv
    if isinstance(c.env_kwargs, (list, tuple)):
        env_ranks = w["env_ranks"]
        envs = [c.EnvCls(**c.env_kwargs[rank]) for rank in env_ranks]
    else:
        envs = [c.EnvCls(**c.env_kwargs) for _ in range(w.n_envs)]
    # ^^^ CHANGED LINES ^^^
    collector = c.CollectorCls(
        rank=w.rank,
        envs=envs,
        samples_np=w.samples_np,
        batch_T=c.batch_T,
        TrajInfoCls=c.TrajInfoCls,
        agent=c.get("agent", None),  # Optional depending on parallel setup.
        sync=w.get("sync", None),
        step_buffer_np=w.get("step_buffer_np", None),
        global_B=c.get("global_B", 1),
        env_ranks=w.get("env_ranks", None),
    )
    agent_inputs, traj_infos = collector.start_envs(c.max_decorrelation_steps)
    collector.start_agent()

    if c.get("eval_n_envs", 0) > 0:
        eval_envs = [
            c.EnvCls(**c.eval_env_kwargs) for _ in range(c.eval_n_envs)
        ]
        eval_collector = c.eval_CollectorCls(
            rank=w.rank,
            envs=eval_envs,
            TrajInfoCls=c.TrajInfoCls,
            traj_infos_queue=c.eval_traj_infos_queue,
            max_T=c.eval_max_T,
            agent=c.get("agent", None),
            sync=w.get("sync", None),
            step_buffer_np=w.get("eval_step_buffer_np", None),
        )
    else:
        eval_envs = list()

    ctrl = c.ctrl
    ctrl.barrier_out.wait()
    while True:
        collector.reset_if_needed(agent_inputs)  # Outside barrier?
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        if ctrl.do_eval.value:
            eval_collector.collect_evaluation(
                ctrl.itr.value)  # Traj_infos to queue inside.
        else:
            (agent_inputs, traj_infos,
             completed_infos) = collector.collect_batch(
                 agent_inputs, traj_infos, ctrl.itr.value)
            for info in completed_infos:
                c.traj_infos_queue.put(info)
        ctrl.barrier_out.wait()

    for env in envs + eval_envs:
        env.close()


class MuxParallelSampler(ParallelSamplerBase):
    """Variant of `ParallelSamplerBase` that allows different env kwargs for
    each of the environments that constitutes a batch."""
    def initialize(self,
                   agent,
                   affinity,
                   seed,
                   bootstrap_value=False,
                   traj_info_kwargs=None,
                   world_size=1,
                   rank=0,
                   worker_process=None):
        n_envs_list = self._get_n_envs_list(affinity=affinity)
        self.n_worker = n_worker = len(n_envs_list)
        B = self.batch_spec.B
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        self.world_size = world_size
        self.rank = rank

        if self.eval_n_envs > 0:
            self.eval_n_envs_per = max(1, self.eval_n_envs // n_worker)
            self.eval_n_envs = eval_n_envs = self.eval_n_envs_per * n_worker
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}.")
            self.eval_max_T = int(self.eval_max_steps // eval_n_envs)

        # vvv CHANGED LINES vvv
        if isinstance(self.env_kwargs, (list, tuple)):
            assert len(self.env_kwargs) == global_B, \
                (len(self.env_kwargs), global_B)
            env = self.EnvCls(**self.env_kwargs[0])
        else:
            env = self.EnvCls(**self.env_kwargs)
        # ^^^ CHANGED LINES ^^^
        self._agent_init(agent, env, global_B=global_B, env_ranks=env_ranks)
        examples = self._build_buffers(env, bootstrap_value)
        env.close()
        del env

        self._build_parallel_ctrl(n_worker)

        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k,
                        v)  # Avoid passing every init.

        common_kwargs = self._assemble_common_kwargs(affinity, global_B)
        workers_kwargs = self._assemble_workers_kwargs(affinity, seed,
                                                       n_envs_list)

        # vvv CHANGED LINES vvv
        target = _mux_sampler if worker_process is None else worker_process
        # ^^^ CHANGED LINES ^^^
        self.workers = [
            mp.Process(target=target,
                       kwargs=dict(common_kwargs=common_kwargs,
                                   worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs
        ]
        for w in self.workers:
            w.start()

        self.ctrl.barrier_out.wait(
        )  # Wait for workers ready (e.g. decorrelate).
        return examples  # e.g. In case useful to build replay buffer.


class MuxCpuSampler(MuxParallelSampler, CpuSampler):
    pass


class MuxGpuSampler(MuxParallelSampler, GpuSampler):
    pass


EnvIDObs = NamedTupleSchema(
            'EnvIDObs', ('observation', 'task_id', 'variant_id'))
EnvIDObsArray = NamedArrayTupleSchema(EnvIDObs._typename, EnvIDObs._fields)


class EnvIDWrapper(Wrapper):
    def __init__(self, env, task_id, variant_id, num_tasks, max_num_variants):
        super().__init__(env)
        assert isinstance(task_id, int) and isinstance(variant_id, int)
        self._task_id_np = np.asarray([task_id]).reshape(())
        self._variant_id_np = np.asarray([variant_id]).reshape(())
        task_space = IntBox(0, num_tasks)
        variant_space = IntBox(0, max_num_variants)
        self.observation_space = Composite(
            (env.observation_space, task_space, variant_space),
            self.obs_schema)

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return self.wrap_obs(obs)

    def step(self, *args, **kwargs):
        env_step = super().step(*args, **kwargs)
        new_obs = self.wrap_obs(env_step.observation)
        return env_step._replace(observation=new_obs)

    def wrap_obs(self, obs):
        new_obs = EnvIDObs._make(
            (obs, self._task_id_np, self._variant_id_np))
        return new_obs

    @property
    def spaces(self):
        return EnvSpaces(observation=self.observation_space,
                         action=self.action_space)


class MILBenchEnvMultiplexer:
    """Pseudo-constructor for environments that is able to instantiate
    environments based on a *task ID* and *variant ID*. Useful for passing to
    rlpyt sampler constructors in order to get a different environment in each
    sampler process."""
    def __init__(self, variant_groups):
        """Construct the multiplexer.

        Arguments:
            variant_groups (GroupdVariantNames): structure holding all of the
                known tasks and task variant names. Contains task IDs too."""
        self.variant_groups = variant_groups

    def get_batch_size_and_kwargs(self, min_batch_size):
        """Compute an sampler batch size that is >= min_batch_size
        and divisible by the number of environments (so total number of unique
        (task, variant) pairs). Return both the batch size and a list of
        environment kwargs for instantiating environments with
        CpuSampler/GpuSampler/etc."""
        assert self.variant_groups.num_tasks > 0 \
            and self.variant_groups.max_num_variants > 0 \
            and min_batch_size > 0
        n_envs = len(self.variant_groups.task_variant_by_name)
        batch_size = min_batch_size
        remainder = min_batch_size % n_envs
        if remainder > 0:
            batch_size += min_batch_size - remainder
        assert batch_size % n_envs == 0
        env_keys = sorted(self.variant_groups.task_variant_by_name.values())
        multiplier = int(np.round(batch_size / n_envs))
        batch_env_kwargs = sum([[dict(env_key=env_key)] * multiplier
                                for env_key in env_keys], [])
        assert len(batch_env_kwargs) == batch_size
        return batch_size, batch_env_kwargs

    def __call__(self, env_key):
        task_id, variant_id = env_key
        env_name = self.variant_groups.env_name_by_task_variant[env_key]
        mb_env = MILBenchGymEnv(env_name)
        id_env = EnvIDWrapper(mb_env, task_id, variant_id,
                              self.variant_groups.num_tasks,
                              self.variant_groups.max_num_varaints)
        return id_env


class MuxTaskModelWrapper(AgentModelWrapper):
    """Like AgentModelWrapper, but for multi-head policies that expect task IDs
    as input. Assumes that it is only ever getting applied to one task,
    identified by a given integer `task_id`. Good when you have one sampler per
    task."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, obs, prev_act, prev_rew):
        # similar to AgentModelWrapper.forward(), but also constructs task IDs
        obs_image, env_id = obs.observation, obs.env_id
        env_id = 0 * env_id  # FIXME: actually use the env_id :)
        del obs  # defensive, avoid using tuple obs
        lead_dim, T, B, img_shape = infer_leading_dims(obs_image, 3)
        task_ids = env_id.view((T * B, )).to(device=obs_image.device,
                                             dtype=torch.long)
        logits, v = self.model(obs_image.view(T * B, *img_shape), task_ids)
        pi = F.softmax(logits, dim=-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v

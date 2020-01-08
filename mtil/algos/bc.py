"""Single-task behavioural cloning (BC)."""

import click
from milbench.baselines.saved_trajectories import (
    load_demos, preprocess_demos_with_wrapper, splice_in_preproc_name)
from tensorflow import keras
from tf_agents.agents.behavioral_cloning.behavioral_cloning_agent import \
    BehavioralCloningAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_network import \
    ActorDistributionNetwork
from tf_agents.policies.actor_policy import ActorPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer

from mtil.common import ImageScaleStackLayer, MultiCategoricalProjectionNetwork


class ActorBCAgent(BehavioralCloningAgent):
    """Overrides BehaviouralCloningAgent's policy-creator to use an
    ActorDistributionNetwork instead (wouldn't this have been the obvious way
    to create the network? Ugh.)."""
    def _get_policies(self, time_step_spec, action_spec, cloning_network):
        collect_policy = ActorPolicy(time_step_spec,
                                     action_spec,
                                     actor_network=self._cloning_network)
        # Original code used a greedy policy as the "policy" (as opposed to the
        # "collect_policy"); I'm going to use same policy for both b/c IDK what
        # the distinction is yet.
        # policy = GreedyPolicy(collect_policy)
        policy = collect_policy
        return policy, collect_policy


class BC:
    def __init__(self, env_name, optimiser=None):
        # should use xent loss so long as the action spec is correct
        if optimiser is None:
            optimiser = keras.optimizers.Adam(lr=3e-4)
        self.tf_env = TFPyEnvironment(suite_gym.load(env_name))
        # conv_layer_params is list of tuples like (filts, kern_size, stride)
        self.bc_net = ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            preprocessing_layers=ImageScaleStackLayer(),
            # FIXME: make this a little resnet or something fancier
            conv_layer_params=[(64, 5, 1), (128, 3, 2), (128, 3, 2),
                               (128, 3, 2), (128, 3, 2), (128, 3, 2)],
            fc_layer_params=(256, 128),
            discrete_projection_net=MultiCategoricalProjectionNetwork)
        self.agent = ActorBCAgent(self.tf_env.time_step_spec(),
                                  self.tf_env.action_spec(),
                                  self.bc_net,
                                  optimiser,
                                  epsilon_greedy=0.0)
        self.step_replay_buffer = TFUniformReplayBuffer(
            self.agent.collect_data_spec, batch_size=1)
        self.driver = DynamicStepDriver(
            self.tf_env,
            self.agent.collect_policy,
            # TODO: figure out how to collect score stats in
            # a more sensible way. I don't actually need to
            # save full trajectories.
            observers=[self.step_replay_buffer.add_batch],
            num_steps=1024)

    def train(self, n_epochs=100):
        pass


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--add-preproc",
    default="LoResStack",
    type=str,
    help="add preprocessor to the demos and test env (default: 'LoResStack')")
@click.argument("demos", nargs=-1, required=True)
def main(demos, add_preproc):
    # register original envs
    import milbench
    milbench.register_envs()

    # load demos (this code copied from bc.py in original baselines)
    demo_dicts = load_demos(demos)
    orig_env_name = demo_dicts[0]['env_name']
    if add_preproc:
        env_name = splice_in_preproc_name(orig_env_name, add_preproc)
        print(f"Splicing preprocessor '{add_preproc}' into environment "
              f"'{orig_env_name}'. New environment is {env_name}")
    else:
        env_name = orig_env_name
    demo_trajs = [d['trajectory'] for d in demo_dicts]
    if add_preproc:
        demo_trajs = preprocess_demos_with_wrapper(demo_trajs, orig_env_name,
                                                   add_preproc)

    # instantiate & train algorithm
    bc = BC(env_name)
    print("Collecting some data… (could take a few minutes)")
    bc.driver.run()
    print("Done!")


if __name__ == '__main__':
    main()

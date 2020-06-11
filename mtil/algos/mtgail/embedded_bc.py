"""PPO variant that includes a behavioural cloning loss (call it 'PPO from
demonstrations'). Doesn't work for multitask or recurrent policies as of
2020-02-29 (making that work will require rewriting the data-loading code)."""
from rlpyt.agents.base import AgentInputs
from rlpyt.algos.pg.base import OptInfo
from rlpyt.algos.pg.ppo import PPO, LossInputs
from rlpyt.utils.buffer import buffer_method, buffer_to
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.tensor import valid_mean
import torch

from mtil.reward_injection_wrappers import CustomRewardMixinPg
from mtil.utils.torch import repeat_dataset


class BehaviouralCloningPPOMixin:
    """Mixin for PPO that supports behavioural cloning (+ augmentation, which
    is necessary to make BC work...)."""
    def __init__(self, bc_loss_coeff, expert_traj_loader, aug_model, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_model = aug_model
        self.bc_loss_coeff = bc_loss_coeff
        if bc_loss_coeff:
            self.expert_batch_iter = repeat_dataset(expert_traj_loader)
        else:
            self.expert_batch_iter = None

    def augment_loss_inputs(self, loss_inputs):
        if not self.aug_model:
            return loss_inputs

        agent_inputs = loss_inputs.agent_inputs
        outer_obs = agent_inputs.observation
        inner_obs = outer_obs.observation
        new_inner_obs = self.aug_model(inner_obs)
        new_loss_inputs = loss_inputs._replace(
            agent_inputs=agent_inputs._replace(observation=outer_obs._replace(
                observation=new_inner_obs)))

        return new_loss_inputs

    def augment_bc_obs(self, bc_obs):
        if not self.aug_model:
            return bc_obs

        inner_obs = bc_obs.observation
        new_inner_obs = self.aug_model(inner_obs)
        new_obs = bc_obs._replace(observation=new_inner_obs)

        return new_obs

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        return_, advantage, valid = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle
        # all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            # we apply different augmentations for each "epoch"
            aug_loss_inputs = self.augment_loss_inputs(loss_inputs)
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be
                # OK.
                if self.expert_batch_iter:
                    bc_batch_dict = next(self.expert_batch_iter)
                    bc_obs = self.augment_bc_obs(bc_batch_dict['obs'])
                    bc_acts = bc_batch_dict['acts']
                    assert not torch.is_floating_point(bc_acts), bc_acts
                    bc_acts = bc_acts.long()
                else:
                    bc_obs = None
                    bc_acts = None
                loss, entropy, perplexity = self.loss(
                    *aug_loss_inputs[T_idxs, B_idxs],
                    bc_observations=bc_obs,
                    bc_actions=bc_acts,
                    init_rnn_state=rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) \
                / self.n_itr

        return opt_info

    def loss(self,
             agent_inputs,
             action,
             return_,
             advantage,
             valid,
             old_dist_info,
             bc_observations,
             bc_actions,
             init_rnn_state=None):
        """
        Compute the BC-augmented training loss:
            policy_loss + value_loss + entropy_loss + bc_loss
        Policy loss: min(likelhood-ratio * advantage,
                         clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        BC loss: xent(policy(demo_states), action_labels)
        Calls the agent to compute forward pass on training data, and uses the
        ``agent.distribution`` to compute likelihoods and entropies. Valid for
        feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs,
                                                      init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action,
                                      old_dist_info=old_dist_info,
                                      new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
                                    1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = -valid_mean(surrogate, valid)

        # TODO: log the value error and correlation
        value_error = 0.5 * (value - return_)**2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = -self.entropy_loss_coeff * entropy

        # BC loss (this is the only new part)
        if self.bc_loss_coeff:
            if init_rnn_state is not None:
                raise NotImplementedError("doesn't quite work with RNNs yet")
                # bc_dist_info, _, _ = self.agent(*bc_agent_inputs,
                #                                 init_rnn_state)
            else:
                # This will break if I have an agent/model that actually needs
                # the previous action and reward. (IIRC that only includes
                # recurrent agents in rlpyt, though)
                dummy_prev_action = bc_actions
                dummy_prev_reward = torch.zeros(bc_actions.shape[0],
                                                device=bc_actions.device)
                bc_dist_info, _ = self.agent(bc_observations,
                                             dummy_prev_action,
                                             dummy_prev_reward)
            expert_ll = dist.log_likelihood(bc_actions, bc_dist_info)
            # bc_loss = -self.bc_loss_coeff * valid_mean(expert_ll, bc_valid)
            # TODO: also log BC accuracy (or maybe do it somewhere else, IDK)
            bc_loss = -self.bc_loss_coeff * expert_ll.mean()
        else:
            bc_loss = 0.0

        loss = pi_loss + value_loss + entropy_loss + bc_loss

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity


class BCCustomRewardPPO(CustomRewardMixinPg, BehaviouralCloningPPOMixin, PPO):
    pass

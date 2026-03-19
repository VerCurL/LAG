import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppo_policy import PPOMoEPolicy
from ..utils.buffer import SharedReplayBuffer
from ..utils.utils import check, get_gard_norm


class PPOMoETrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.expert_out_loss_coef = args.expert_out_loss_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length

    def ppo_update(self, policy: PPOMoEPolicy, sample):

        obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy, actor_experts_out, actor_record_info, critic_experts_out, critic_record_info \
            = policy.evaluate_actions(
                share_obs_batch,
                obs_batch,
                rnn_states_actor_batch,
                rnn_states_critic_batch,
                actions_batch,
                masks_batch,
            )

        actor_gate_entropy, actor_gate_max_prob, actor_expert_usage = self._summarize_gate(actor_record_info)
        critic_gate_entropy, critic_gate_max_prob, critic_expert_usage = self._summarize_gate(critic_record_info)

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
        policy_loss = -policy_loss.mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()

        actor_expert_out_loss = self._compute_expert_out_loss(
            actor_experts_out,
            actor_record_info,
            policy.actor.num_general_experts,
            policy.actor.top_k,
        )
        critic_expert_out_loss = self._compute_expert_out_loss(
            critic_experts_out,
            critic_record_info,
            policy.critic.num_general_experts,
            policy.critic.top_k,
        )
        expert_out_loss = actor_expert_out_loss + critic_expert_out_loss

        loss = (
            policy_loss
            + value_loss * self.value_loss_coef
            + policy_entropy_loss * self.entropy_coef
            + expert_out_loss * self.expert_out_loss_coef
        )

        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return (
            policy_loss,
            value_loss,
            policy_entropy_loss,
            expert_out_loss,
            actor_expert_out_loss,
            critic_expert_out_loss,
            actor_gate_entropy,
            actor_gate_max_prob,
            actor_expert_usage,
            critic_gate_entropy,
            critic_gate_max_prob,
            critic_expert_usage,
            ratio,
            actor_grad_norm,
            critic_grad_norm,
        )

    def _summarize_gate(self, record_info):
        gate_probs = record_info["gate_probs"]
        gate_entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=1).mean().detach()
        gate_max_prob = gate_probs.max(dim=1)[0].mean().detach()

        expert_usage = torch.zeros(gate_probs.size(1), device=self.device, dtype=torch.float32)
        unique, counts = record_info["top_k_idx"].flatten().unique(return_counts=True)
        expert_usage[unique] = counts.float()
        return gate_entropy, gate_max_prob, expert_usage.detach()

    def _compute_expert_out_loss(self, experts_out, record_info, num_general_experts, top_k):
        if top_k <= 1:
            return torch.zeros((), device=self.device, dtype=experts_out.dtype)

        selected_experts = record_info["top_k_idx"] + num_general_experts
        sel_out = experts_out.gather(1, selected_experts.unsqueeze(-1).expand(-1, -1, experts_out.size(-1)))
        normed = F.normalize(sel_out, p=2, dim=-1)
        inner = torch.matmul(normed, normed.transpose(-1, -2))
        mask = 1 - torch.eye(inner.size(-1), device=inner.device)
        return (inner * mask).pow(2).sum() / (experts_out.size(0) * top_k * (top_k - 1))

    def train(self, policy: PPOMoEPolicy, buffer: SharedReplayBuffer):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['expert_out_loss'] = 0
        train_info['actor_expert_out_loss'] = 0
        train_info['critic_expert_out_loss'] = 0
        train_info['actor_gate_entropy'] = 0
        train_info['actor_gate_max_prob'] = 0
        train_info['actor_expert_usage'] = []
        train_info['critic_gate_entropy'] = 0
        train_info['critic_gate_max_prob'] = 0
        train_info['critic_expert_usage'] = []
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = buffer.recurrent_generator(buffer.advantages, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:
                (
                    policy_loss,
                    value_loss,
                    policy_entropy_loss,
                    expert_out_loss,
                    actor_expert_out_loss,
                    critic_expert_out_loss,
                    actor_gate_entropy,
                    actor_gate_max_prob,
                    actor_expert_usage,
                    critic_gate_entropy,
                    critic_gate_max_prob,
                    critic_expert_usage,
                    ratio,
                    actor_grad_norm,
                    critic_grad_norm,
                ) = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['expert_out_loss'] += expert_out_loss.item()
                train_info['actor_expert_out_loss'] += actor_expert_out_loss.item()
                train_info['critic_expert_out_loss'] += critic_expert_out_loss.item()
                train_info['actor_gate_entropy'] += actor_gate_entropy.item()
                train_info['actor_gate_max_prob'] += actor_gate_max_prob.item()
                if len(train_info['actor_expert_usage']) == 0:
                    train_info['actor_expert_usage'] = actor_expert_usage.tolist()
                else:
                    train_info['actor_expert_usage'] = [
                        x + y for x, y in zip(train_info['actor_expert_usage'], actor_expert_usage.tolist())
                    ]
                train_info['critic_gate_entropy'] += critic_gate_entropy.item()
                train_info['critic_gate_max_prob'] += critic_gate_max_prob.item()
                if len(train_info['critic_expert_usage']) == 0:
                    train_info['critic_expert_usage'] = critic_expert_usage.tolist()
                else:
                    train_info['critic_expert_usage'] = [
                        x + y for x, y in zip(train_info['critic_expert_usage'], critic_expert_usage.tolist())
                    ]
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k, v in train_info.items():
            if isinstance(v, list):
                train_info[k] = [x / num_updates for x in v]
            else:
                train_info[k] = v / num_updates

        return train_info

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List
from .ppo_policy import PPOPCANPolicy
from ..utils.buffer import SharedReplayBuffer
from ..utils.utils import check, get_gard_norm


class PPOPCANTrainer():
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
        self.rewards_pred_coef = args.rewards_pred_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length

        self.num_agents = args.num_agents
        # 上一时刻credit
        self._prev_credit_snapshot = None

    def pcan_update(self, policy: PPOPCANPolicy, sample):
        # -------- 收集buffer_size缓冲值 --------
        rewards_batch, obs_batch, masks_batch, rnn_states_actor_batch = sample
        rewards_batch = check(rewards_batch).to(**self.tpdv)

        # -------- 评估pcan网络获得预测奖励和credit --------
        rewards_pred, credit, pcan_record_info = policy.pcan(obs_batch, rnn_states_actor_batch, masks_batch)

        # -------- 计算损失函数 --------
        # 奖励预测损失
        rewards_pred_loss = F.mse_loss(rewards_pred, rewards_batch)           # [L, D] -loss_mean-> 1

        # -------- 开始梯度下降 --------
        policy.optimizer.zero_grad()
        rewards_pred_loss.backward()
        policy.optimizer.step()

        # -------- 记录信息 --------
        credit_diag_mean, credit_entropy = self._summarize_credit(credit)
        credit_change_rate = self._compute_credit_change_rate(credit)
        head_diversity, pcan_output_norm = self._summarize_pcan_record(pcan_record_info)

        return rewards_pred_loss, credit_diag_mean, credit_entropy, credit_change_rate, head_diversity, pcan_output_norm

    def ppo_update(self, policy: PPOPCANPolicy, sample):
        # -------- 收集buffer_size缓冲值 --------
        obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # -------- 评估行为获得所需值 --------
        values, action_log_probs, dist_entropy \
            = policy.evaluate_actions(share_obs_batch, obs_batch, rnn_states_actor_batch, rnn_states_critic_batch, actions_batch, masks_batch)

        # -------- 计算损失函数 --------
        # 策略损失
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
        policy_loss = -policy_loss.mean()

        # 价值损失
        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        # 策略熵损失
        policy_entropy_loss = -dist_entropy.mean()

        # 损失加权求和
        loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

        # -------- 开始梯度下降 --------
        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPCANPolicy, buffer: SharedReplayBuffer):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['rewards_pred_loss'] = 0
        train_info["credit_diag_mean"] = 0
        train_info["credit_entropy"] = 0
        train_info["head_diversity"] = 0
        train_info["pcan_output_norm"] = 0
        train_info["credit_change_rate"] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            # --------- 训练 pcan 网络 ---------
            pcan_generator = buffer.pcan_generator(self.num_mini_batch, self.data_chunk_length)
            for sample in pcan_generator:
                rewards_pred_loss, credit_diag_mean, credit_entropy, credit_change_rate, head_diversity, pcan_output_norm\
                    = self.pcan_update(policy, sample)
                train_info['rewards_pred_loss'] += rewards_pred_loss.item()
                train_info["credit_diag_mean"] += credit_diag_mean.item()
                train_info["credit_entropy"] += credit_entropy.item()
                train_info["credit_change_rate"] += credit_change_rate.item()
                train_info["head_diversity"] += head_diversity.item()
                train_info["pcan_output_norm"] += pcan_output_norm.item()

            # --------- 训练 actor-critic 网络 ---------
            if self.use_recurrent_policy:
                data_generator = buffer.recurrent_generator(buffer.advantages, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:

                policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm \
                    = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def _summarize_credit(self, credit: torch.Tensor):
        """
        credit shape: [B, N, N]
        B: batch/env dimension
        N: num_agents

        credit_diag_mean:
            对角线元素均值，表示每个 agent 给自己分配 credit 的平均强度

        credit_entropy:
            对每一行做熵，再对 batch 和 agent 求均值
        """
        eps = 1e-8

        # [B, N]
        diag = torch.diagonal(credit, dim1=-2, dim2=-1)
        credit_diag_mean = diag.mean().detach()

        # 每一行应近似是概率分布，shape [B, N]
        row_entropy = -(credit * torch.log(credit + eps)).sum(dim=-1)
        credit_entropy = row_entropy.mean().detach()

        return credit_diag_mean, credit_entropy

    def _summarize_pcan_record(self, pcan_record_info: dict):
        attn_weights = pcan_record_info["attn_weights"]
        pcan_output_norm = pcan_record_info["pcan_output_norm"]

        # attn_weights: [B, H, N, N]
        # 用 head 维标准差来衡量不同 head 的差异程度
        head_diversity = attn_weights.std(dim=1, unbiased=False).mean().detach()

        return head_diversity, pcan_output_norm

    def _compute_credit_change_rate(self, credit: torch.Tensor):
        # 用当前 batch 的平均 credit 矩阵和上一 batch 做差
        current_credit_snapshot = credit.detach().mean(dim=0)

        if self._prev_credit_snapshot is None:
            credit_change_rate = torch.zeros(
                (), device=current_credit_snapshot.device, dtype=current_credit_snapshot.dtype
            )
        else:
            credit_change_rate = (
                    current_credit_snapshot - self._prev_credit_snapshot
            ).abs().mean()

        self._prev_credit_snapshot = current_credit_snapshot
        return credit_change_rate.detach()
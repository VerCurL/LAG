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
        self.obs_pred_coef = args.obs_pred_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length
        # 上一时刻credit
        self._prev_credit_snapshot = None

    def ppo_update(self, policy: PPOPCANPolicy, sample):
        # -------- 收集buffer_size缓冲值 --------
        obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # -------- 评估行为获得所需值 --------
        values, action_log_probs, dist_entropy, obs_next, credit, pcan_record_info\
            = policy.evaluate_actions(share_obs_batch, obs_batch, rnn_states_actor_batch,
                        rnn_states_critic_batch, actions_batch, masks_batch)

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

        # 态势预测损失
        time_batch, dim = obs_batch.shape
        obs_batch = torch.from_numpy(obs_batch).to(self.device)
        masks_batch = torch.from_numpy(masks_batch).to(self.device)
        # print("obs_batch: ", obs_batch.shape)
        # print("obs_next: ", obs_next.shape)
        # print("masks_batch: ", masks_batch.shape)
        obs_target = obs_batch.view(self.data_chunk_length, time_batch // self.data_chunk_length, dim)
        obs_pred = obs_next.view(self.data_chunk_length, time_batch // self.data_chunk_length, dim)
        masks_batch = masks_batch.view(self.data_chunk_length, time_batch // self.data_chunk_length, masks_batch.shape[-1])
        # print("obs_target: ", obs_target.shape)
        # print("obs_target[:-1]: ", obs_target[:-1].shape)
        # print("obs_pred: ", obs_pred.shape)
        # print("masks_batch: ", masks_batch.shape)
        valid_transition = masks_batch[1:]                  # [L-1, N, 1]
        per_dim_obs_loss = F.mse_loss(obs_pred[:-1], obs_target[1:], reduction='none')          # [L-1, N, D]
        per_transition_obs_loss = per_dim_obs_loss.mean(dim=-1, keepdim=True)                   # [L-1, N, 1]
        masked_transition_obs_loss = per_transition_obs_loss * valid_transition
        valid_count = valid_transition.sum().clamp(min=1.0)
        obs_pred_loss = masked_transition_obs_loss.sum() / valid_count

        # 损失加权求和
        loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef + obs_pred_loss * self.obs_pred_coef

        # 记录
        credit_diag_mean, credit_entropy = self._summarize_credit(credit)
        head_diversity, pcan_output_norm = self._summarize_pcan_record(pcan_record_info)
        credit_change_rate = self._compute_credit_change_rate(credit)

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

        return (policy_loss, value_loss, policy_entropy_loss, obs_pred_loss, credit_diag_mean, credit_entropy,
                head_diversity, pcan_output_norm, credit_change_rate, ratio, actor_grad_norm, critic_grad_norm)

    def train(self, policy: PPOPCANPolicy, buffer: SharedReplayBuffer):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['obs_pred_loss'] = 0
        train_info["credit_diag_mean"] = 0
        train_info["credit_entropy"] = 0
        train_info["head_diversity"] = 0
        train_info["pcan_output_norm"] = 0
        train_info["credit_change_rate"] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = buffer.recurrent_generator(buffer.advantages, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:

                (policy_loss, value_loss, policy_entropy_loss, obs_pred_loss, credit_diag_mean, credit_entropy,
                 head_diversity, pcan_output_norm, credit_change_rate, ratio, actor_grad_norm, critic_grad_norm) \
                    = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['obs_pred_loss'] += obs_pred_loss.item()
                train_info["credit_diag_mean"] += credit_diag_mean.item()
                train_info["credit_entropy"] += credit_entropy.item()
                train_info["head_diversity"] += head_diversity.item()
                train_info["pcan_output_norm"] += pcan_output_norm.item()
                train_info["credit_change_rate"] += credit_change_rate.item()
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
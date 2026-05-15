import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List
from .ppo_policy import PPOPCANPolicy
from ..utils.buffer import SharedReplayBuffer, PCANRolloutBuffer
from ..utils.utils import check, get_gard_norm

def _t2n(x):
    return x.detach().cpu().numpy()

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
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length

        self.num_agents = args.num_agents

    def pcan_update(self, policy: PPOPCANPolicy, sample):
        obs_batch, actions_batch, threat_target_batch, attack_target_batch = sample

        B, M = obs_batch.shape[:2]

        obs_batch = check(obs_batch.reshape(B * M, -1)).to(**self.tpdv)
        actions_batch = check(actions_batch.reshape(B * M, -1)).to(**self.tpdv)
        threat_target_batch = check(threat_target_batch).to(**self.tpdv)
        attack_target_batch = check(attack_target_batch).to(**self.tpdv)

        threat_pred, attack_pred, _ = policy.evaluate_pcan(obs_batch, actions_batch)

        threat_loss = F.mse_loss(threat_pred, threat_target_batch)
        attack_loss = F.mse_loss(attack_pred, attack_target_batch)

        pcan_loss = threat_loss + attack_loss

        policy.pcan_optimizer.zero_grad()
        pcan_loss.backward()
        if self.use_max_grad_norm:
            pcan_grad_norm = nn.utils.clip_grad_norm_(policy.pcan.parameters(), self.max_grad_norm).item()
        else:
            pcan_grad_norm = get_gard_norm(policy.pcan.parameters())

        policy.pcan_optimizer.step()

        return pcan_loss, threat_loss, attack_loss, pcan_grad_norm

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

    def train(self, policy: PPOPCANPolicy, buffer: SharedReplayBuffer, pcan_buffer: PCANRolloutBuffer, field_calculator):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['pcan_loss'] = 0
        train_info['pcan_grad_norm'] = 0
        train_info['threat_loss'] = 0
        train_info['attack_loss'] = 0

        # --------- 1. 训练 pcan 网络 ---------
        policy.prep_training()
        field_samples = pcan_buffer.build_field_samples(buffer, field_calculator)
        for _ in range(self.ppo_epoch):
            data_generator = pcan_buffer.pcan_generator(field_samples, self.num_mini_batch)

            for sample in data_generator:
                pcan_loss, threat_loss, attack_loss, pcan_grad_norm = self.pcan_update(policy, sample)

                train_info['pcan_loss'] += pcan_loss.item()
                train_info['pcan_grad_norm'] += pcan_grad_norm
                train_info['threat_loss'] += threat_loss.item()
                train_info['attack_loss'] += attack_loss.item()

        # --------- 2. 用训练好的 PCAN 重分配 reward ---------
        fact_threat_mean, fact_attack_mean, threat_delta_mean, attack_delta_mean, contribution_mean, contribution_std, weight_min, weight_max\
            = self.redistribute_rewards_by_pcan(policy, buffer, threat_coef=1.0, attack_coef=1.0, softmax_tau=0.2)

        # --------- 3. 更新buffer中的累计奖励值 ---------
        self.compute(policy, buffer)

        # --------- 4. 训练 actor-critic 网络 ---------
        for _ in range(self.ppo_epoch):
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

        train_info['fact_threat_mean'] = fact_threat_mean
        train_info['fact_attack_mean'] = fact_attack_mean
        train_info['threat_delta_mean'] = threat_delta_mean
        train_info['attack_delta_mean'] = attack_delta_mean
        train_info['contribution_mean'] = contribution_mean
        train_info['contribution_std'] = contribution_std
        train_info['weight_min'] = weight_min
        train_info['weight_max'] = weight_max

        return train_info

    @torch.no_grad()
    def compute(self, policy: PPOPCANPolicy, buffer: SharedReplayBuffer):
        policy.prep_rollout()
        next_values = policy.get_values(np.concatenate(buffer.share_obs[-1]),
                                        np.concatenate(buffer.rnn_states_critic[-1]),
                                        np.concatenate(buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), buffer.n_rollout_threads))
        buffer.compute_returns(next_values)
        policy.prep_training()

    @torch.no_grad()
    def redistribute_rewards_by_pcan(self, policy, buffer, threat_coef=1.0, attack_coef=1.0, softmax_tau=0.2):
        """
        使用 PCAN 事实 / 反事实场值进行 reward redistribution。

        buffer.obs[:-1]:  [T, n_env, num_agents, obs_dim]
        buffer.actions:   [T, n_env, num_agents, act_dim]
        buffer.rewards:   [T, n_env, num_agents, 1]
        buffer.masks:     [T + 1, n_env, num_agents, 1]
        """
        policy.prep_rollout()

        obs = buffer.obs[:-1]
        actions = buffer.actions
        masks = buffer.masks

        T, n_envs, num_agents = actions.shape[:3]

        # 转成 joint sample: [S, M, dim], S = T * n_env
        obs_joint = obs.transpose(1, 0, 2, 3).reshape(n_envs * T * num_agents, -1)
        act_joint = actions.transpose(1, 0, 2, 3).reshape(n_envs * T * num_agents, -1)

        # ----------- 1. 事实场值 -----------
        fact_threat, fact_attack, _ = policy.evaluate_pcan(obs_joint, act_joint)
        fact_threat = _t2n(fact_threat)
        fact_attack = _t2n(fact_attack)

        # size: [T, N, 1]
        fact_threat = fact_threat.reshape(n_envs, T, -1).transpose(1, 0, 2)
        fact_attack = fact_attack.reshape(n_envs, T, -1).transpose(1, 0, 2)

        # ----------- 2. 反事实动作 -----------
        cf_actions = self.build_counterfactual_actions(actions, masks)

        cf_threat = np.zeros((T, n_envs, num_agents, 1), dtype=np.float32)
        cf_attack = np.zeros((T, n_envs, num_agents, 1), dtype=np.float32)

        # ----------- 3. 对每个 agent 单独构建一次反事实输入 -----------
        for agent_i in range(num_agents):
            cf_act_i = cf_actions[agent_i]
            cf_act_joint = cf_act_i.transpose(1, 0, 2, 3).reshape(n_envs * T * num_agents, -1)

            threat_i, attack_i, _ = policy.evaluate_pcan(obs_joint, cf_act_joint)
            threat_i = _t2n(threat_i)
            attack_i = _t2n(attack_i)

            # size: (N * T, 1) -r-> (N, T, 1) -T-> (T, N, 1)
            threat_i = threat_i.reshape(n_envs, T, -1).transpose(1, 0, 2)
            attack_i = attack_i.reshape(n_envs, T, -1).transpose(1, 0, 2)

            cf_threat[:, :, agent_i, :] = threat_i
            cf_attack[:, :, agent_i, :] = attack_i

        # ----------- 4. 贡献度 -----------
        # 威胁越低越好：fact_threat - cf_threat < 0 表示当前真实动作降低了威胁
        # 进攻越高越好：fact_attack - cf_attack > 0 表示当前真实动作提升了进攻
        # size: (T, N, M, 1)
        threat_delta = cf_threat - fact_threat[:, :, None, :]
        attack_delta = fact_attack[:, :, None, :] - cf_attack
        contribution = threat_coef * threat_delta + attack_coef * attack_delta

        # ----------- 5. reward pool -----------
        # 这里用 sum 是为了保持原先所有 agent 的 reward 总量不变。
        # [T, n_env, 1, 1]
        reward_pool = np.sum(buffer.rewards, axis=2, keepdims=True)

        # 正奖励：贡献越大，分得越多
        positive_weights = self.masked_softmax(contribution, tau=softmax_tau)

        # 负奖励：贡献越小，承担越多惩罚
        negative_weights = self.masked_softmax(-contribution, tau=softmax_tau)

        # 根据reward_pool选择正奖励或负奖励
        # [T, n_env, M, 1]
        weights = np.where(reward_pool >= 0.0, positive_weights, negative_weights)

        # 广播 -> [T, n_env, M, 1]
        redistributed_rewards = reward_pool * weights

        # 如果某些时刻没有 active agent，保持原 reward，避免全 0 或 nan
        buffer.rewards[:] = redistributed_rewards

        # ----------- 6. 返回需要记录的值 -----------
        fact_threat_mean = float(np.mean(fact_threat))
        fact_attack_mean = float(np.mean(fact_attack))
        threat_delta_mean = float(np.mean(threat_delta))
        attack_delta_mean = float(np.mean(attack_delta))
        contribution_mean = float(np.mean(contribution))
        contribution_std = float(np.std(contribution))
        weight_max = float(np.max(positive_weights))
        weight_min = float(np.min(positive_weights))

        policy.prep_training()
        return fact_threat_mean, fact_attack_mean, threat_delta_mean, attack_delta_mean, contribution_mean, contribution_std, weight_min, weight_max

    def build_counterfactual_actions(self, actions, masks):
        """
        actions: [T, n_env, num_agents, act_dim]
        masks:   [T + 1, n_env, num_agents, 1]

        Returns:
            cf_actions: [num_agents, T, n_env, num_agents, act_dim]

        cf_actions[i, t, env] 表示：
        - 只有飞机 i 的动作替换成上一时刻动作
        - 其他飞机保持当前时刻动作
        - t == 0 或 t 是新 episode 起点时，不替换
        """
        T, n_envs, num_agents, act_dim = actions.shape

        cf_actions = np.repeat(actions[None, ...], repeats=num_agents, axis=0).copy()

        for agent_i in range(num_agents):
            for t in range(1, T):
                # masks[t] == 0 表示 obs[t] 是 reset 后的新开局
                # 此时 action[t - 1] 属于上一局，不能拿来当反事实动作
                same_episode = np.all(masks[t] > 0.0, axis=(1, 2))  # [n_env]

                if not np.any(same_episode):
                    continue

                cf_actions[agent_i, t, same_episode, agent_i, :] = actions[t - 1, same_episode, agent_i, :]

        return cf_actions

    def masked_softmax(self, x, tau=0.2, axis=2, eps=1e-8):
        """
        x:  [T, n_env, num_agents, 1]
        """
        tau = max(float(tau), eps)
        logits = x / tau

        logits = logits - np.max(logits, axis=axis, keepdims=True)
        exp_logits = np.exp(logits)

        denom = np.sum(exp_logits, axis=axis, keepdims=True)
        return exp_logits / (denom + eps)
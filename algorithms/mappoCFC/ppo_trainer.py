from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.JSBSim.situation.field import FieldCalculator
from scripts.AeroTAF.data.schema import CATEGORY_NAMES
from ..utils.utils import check, get_gard_norm
from .online_dataset import parse_counterfactual_actions


def _t2n(x):
    return x.detach().cpu().numpy()


class PPOAeroTAFTrainer:
    def __init__(self, args, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length
        self.num_agents = args.num_agents

        self.counterfactual_actions = parse_counterfactual_actions(args.CFC_counterfactual_actions)
        self._validate_args(args)
        self.field_calculator = FieldCalculator(
            k_step=args.AeroTAF_kstep,
            gamma=args.AeroTAF_field_gamma,
            ego_team=args.AeroTAF_ego_team,
        )
        self.train_calls = 0
        self.aerotaf_gradient_updates = 0

    @staticmethod
    def _validate_args(args):
        positive = {
            "AeroTAF_history_windows": args.AeroTAF_history_windows,
            "AeroTAF_epoch": args.AeroTAF_epoch,
            "AeroTAF_mini_batch_size": args.AeroTAF_mini_batch_size,
            "AeroTAF_inference_batch_size": args.AeroTAF_inference_batch_size,
        }
        invalid = {key: value for key, value in positive.items() if int(value) <= 0}
        if invalid:
            raise ValueError(f"mappoCFC parameters must be positive: {invalid}")
        if not 0.0 <= args.AeroTAF_stable_sample_ratio <= 1.0:
            raise ValueError("--AeroTAF-stable-sample-ratio must be in [0, 1]")
        if not 0.0 <= args.CFC_reward_blend <= 1.0:
            raise ValueError("--CFC-reward-blend must be in [0, 1]")
        if args.CFC_softmax_tau <= 0.0:
            raise ValueError("--CFC-softmax-tau must be positive")

    def _forward_aerotaf(self, policy, dataset, indices):
        obs, actions = dataset.stack_windows(indices)
        batch_size, seq_len, num_agents = obs.shape[:3]
        obs = check(obs.reshape(batch_size * seq_len * num_agents, -1)).to(**self.tpdv)
        actions = check(actions.reshape(batch_size * seq_len * num_agents, -1)).to(**self.tpdv)
        threat_target, attack_target, categories = dataset.targets(indices)
        threat_target = check(threat_target).to(**self.tpdv)
        attack_target = check(attack_target).to(**self.tpdv)
        threat_pred, attack_pred, _ = policy.evaluate_AeroTAF(obs, actions, seq_len=seq_len)
        return threat_pred, attack_pred, threat_target, attack_target, categories

    def _aerotaf_update(self, policy, dataset, indices):
        threat_pred, attack_pred, threat_target, attack_target, categories = self._forward_aerotaf(
            policy, dataset, indices
        )
        threat_loss = F.mse_loss(threat_pred, threat_target)
        attack_loss = F.mse_loss(attack_pred, attack_target)
        loss = (
            self.args.AeroTAF_threat_loss_weight * threat_loss
            + self.args.AeroTAF_attack_loss_weight * attack_loss
        )

        policy.AeroTAF_optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(policy.AeroTAF.parameters(), self.max_grad_norm).item()
        else:
            grad_norm = get_gard_norm(policy.AeroTAF.parameters())
        policy.AeroTAF_optimizer.step()
        self.aerotaf_gradient_updates += 1

        return {
            "loss": float(loss.item()),
            "threat_loss": float(threat_loss.item()),
            "attack_loss": float(attack_loss.item()),
            "grad_norm": float(grad_norm),
            "threat_squared_error": _t2n((threat_pred - threat_target).square()).reshape(-1),
            "attack_squared_error": _t2n((attack_pred - attack_target).square()).reshape(-1),
            "categories": np.asarray(categories, dtype=np.int16),
        }

    def train_aerotaf(self, policy, dataset):
        selected = dataset.sampled_training_indices(
            self.args.AeroTAF_stable_sample_ratio,
            self.args.seed + self.train_calls * 1009,
        )
        totals = defaultdict(float)
        category_sse = {
            name: {"threat": 0.0, "attack": 0.0, "count": 0}
            for name in CATEGORY_NAMES
        }
        updates = 0
        if selected.size == 0:
            return dict(totals), selected

        policy.prep_training()
        for epoch in range(self.args.AeroTAF_epoch):
            batches = dataset.grouped_batches(
                selected,
                self.args.AeroTAF_mini_batch_size,
                seed=self.args.seed + self.train_calls * 1009 + epoch,
            )
            for indices, _ in batches:
                result = self._aerotaf_update(policy, dataset, indices)
                updates += 1
                totals["AeroTAF_loss"] += result["loss"]
                totals["AeroTAF_threat_loss"] += result["threat_loss"]
                totals["AeroTAF_attack_loss"] += result["attack_loss"]
                totals["AeroTAF_grad_norm"] += result["grad_norm"]
                for category_id, category_name in enumerate(CATEGORY_NAMES):
                    mask = result["categories"] == category_id
                    if np.any(mask):
                        category_sse[category_name]["threat"] += float(
                            result["threat_squared_error"][mask].sum()
                        )
                        category_sse[category_name]["attack"] += float(
                            result["attack_squared_error"][mask].sum()
                        )
                        category_sse[category_name]["count"] += int(mask.sum())

        for key in list(totals):
            totals[key] /= max(updates, 1)
        totals["AeroTAF_updates"] = float(updates)
        totals["AeroTAF_train_samples"] = float(len(selected))
        for category_name, values in category_sse.items():
            count = max(values["count"], 1)
            totals[f"AeroTAF_{category_name}_threat_loss"] = values["threat"] / count
            totals[f"AeroTAF_{category_name}_attack_loss"] = values["attack"] / count
        return dict(totals), selected

    @torch.no_grad()
    def _predict_fields(self, policy, dataset, counterfactual_kind=None, counterfactual_agent=None):
        threat = np.zeros(len(dataset), dtype=np.float32)
        attack = np.zeros(len(dataset), dtype=np.float32)
        batches = dataset.grouped_batches(
            dataset.all_indices,
            self.args.AeroTAF_inference_batch_size,
            seed=self.args.seed,
        )
        for indices, seq_len in batches:
            obs, actions = dataset.stack_windows(indices, counterfactual_kind, counterfactual_agent)
            batch_size, _, num_agents = obs.shape[:3]
            obs_flat = obs.reshape(batch_size * seq_len * num_agents, -1)
            action_flat = actions.reshape(batch_size * seq_len * num_agents, -1)
            threat_pred, attack_pred, _ = policy.evaluate_AeroTAF(
                obs_flat,
                action_flat,
                seq_len=seq_len,
            )
            threat[indices] = _t2n(threat_pred).reshape(-1)
            attack[indices] = _t2n(attack_pred).reshape(-1)
        return threat, attack

    @staticmethod
    def _masked_softmax(values, active_masks, tau, eps=1e-8):
        active = np.asarray(active_masks, dtype=np.float32) > 0.5
        logits = values / max(float(tau), eps)
        logits = np.where(active, logits, -np.inf)
        maximum = np.max(logits, axis=2, keepdims=True)
        maximum = np.where(np.isfinite(maximum), maximum, 0.0)
        exp_logits = np.where(active, np.exp(logits - maximum), 0.0)
        denominator = exp_logits.sum(axis=2, keepdims=True)
        return np.divide(
            exp_logits,
            denominator,
            out=np.zeros_like(exp_logits),
            where=denominator > eps,
        )

    @torch.no_grad()
    def redistribute_rewards_by_cfc(self, policy, buffer, dataset):
        policy.prep_rollout()
        fact_threat, fact_attack = self._predict_fields(policy, dataset)
        fact_threat = fact_threat.reshape(dataset.n_envs, dataset.time_steps).T
        fact_attack = fact_attack.reshape(dataset.n_envs, dataset.time_steps).T

        cf_threat = np.zeros((dataset.time_steps, dataset.n_envs, self.num_agents, 1), dtype=np.float32)
        cf_attack = np.zeros_like(cf_threat)
        for agent_i in range(self.num_agents):
            agent_threat = []
            agent_attack = []
            for action_name in self.counterfactual_actions:
                threat, attack = self._predict_fields(policy, dataset, action_name, agent_i)
                agent_threat.append(threat.reshape(dataset.n_envs, dataset.time_steps).T)
                agent_attack.append(attack.reshape(dataset.n_envs, dataset.time_steps).T)
            cf_threat[:, :, agent_i, 0] = np.mean(agent_threat, axis=0)
            cf_attack[:, :, agent_i, 0] = np.mean(agent_attack, axis=0)

        threat_delta = cf_threat - fact_threat[:, :, None, None]
        attack_delta = fact_attack[:, :, None, None] - cf_attack
        contribution = (
            self.args.CFC_threat_coef * threat_delta
            + self.args.CFC_attack_coef * attack_delta
        )

        original_rewards = buffer.rewards.copy()
        reward_pool = original_rewards.sum(axis=2, keepdims=True)
        active_masks = buffer.active_masks[:-1]
        positive_weights = self._masked_softmax(
            contribution, active_masks, self.args.CFC_softmax_tau
        )
        negative_weights = self._masked_softmax(
            -contribution, active_masks, self.args.CFC_softmax_tau
        )
        weights = np.where(reward_pool >= 0.0, positive_weights, negative_weights)
        redistributed = reward_pool * weights
        has_active = np.any(active_masks > 0.5, axis=2, keepdims=True)
        redistributed = np.where(has_active, redistributed, original_rewards)
        blend = float(self.args.CFC_reward_blend)
        buffer.rewards[:] = (1.0 - blend) * original_rewards + blend * redistributed
        policy.prep_training()

        active_contribution = contribution[active_masks > 0.5]
        return {
            "CFC_applied": 1.0,
            "CFC_fact_threat_mean": float(fact_threat.mean()),
            "CFC_fact_attack_mean": float(fact_attack.mean()),
            "CFC_threat_delta_mean": float(threat_delta.mean()),
            "CFC_attack_delta_mean": float(attack_delta.mean()),
            "CFC_contribution_mean": float(active_contribution.mean()) if active_contribution.size else 0.0,
            "CFC_contribution_std": float(active_contribution.std()) if active_contribution.size else 0.0,
            "CFC_weight_min": float(weights.min()),
            "CFC_weight_max": float(weights.max()),
            "CFC_reward_sum_error": float(
                np.max(np.abs(original_rewards.sum(axis=2) - buffer.rewards.sum(axis=2)))
            ),
        }

    def ppo_update(self, policy, sample):
        (
            obs_batch,
            share_obs_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            advantages_batch,
            returns_batch,
            value_preds_batch,
            rnn_states_actor_batch,
            rnn_states_critic_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        values, action_log_probs, dist_entropy = policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_actor_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
        )

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_loss = 0.5 * torch.max(
                (values - returns_batch).pow(2),
                (value_pred_clipped - returns_batch).pow(2),
            ).mean()
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2).mean()
        policy_entropy_loss = -dist_entropy.mean()
        loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * policy_entropy_loss

        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_entropy_loss": float(policy_entropy_loss.item()),
            "ratio": float(ratio.mean().item()),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
        }

    @torch.no_grad()
    def compute(self, policy, buffer):
        policy.prep_rollout()
        next_values = policy.get_values(
            np.concatenate(buffer.share_obs[-1]),
            np.concatenate(buffer.rnn_states_critic[-1]),
            np.concatenate(buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), buffer.n_rollout_threads))
        buffer.compute_returns(next_values)
        policy.prep_training()

    def train(self, policy, buffer, cfc_buffer):
        self.train_calls += 1
        dataset = cfc_buffer.build_dataset(buffer, self.field_calculator, self.args)
        train_info, selected = self.train_aerotaf(policy, dataset)

        for category_name, count in dataset.category_counts(eligible_only=True).items():
            train_info[f"AeroTAF_{category_name}_points"] = float(count)
        train_info["AeroTAF_episode_segments"] = float(
            sum(1 for env in dataset.segment_starts for t, start in enumerate(env) if t == start)
        )
        train_info["AeroTAF_train_episodes"] = float(dataset.train_episode_count)
        train_info.update(
            {f"AeroTAF_threshold_{key}": float(value) for key, value in dataset.thresholds.items() if isinstance(value, (int, float))}
        )
        train_info.update({f"AeroTAF_event_{key}": float(value) for key, value in dataset.event_counts.items()})

        can_apply_cfc = (
            selected.size > 0
            and self.train_calls > self.args.CFC_warmup_rollouts
            and self.aerotaf_gradient_updates > 0
        )
        if can_apply_cfc:
            train_info.update(self.redistribute_rewards_by_cfc(policy, buffer, dataset))
        else:
            train_info["CFC_applied"] = 0.0

        self.compute(policy, buffer)
        ppo_totals = defaultdict(float)
        ppo_updates = 0
        for _ in range(self.ppo_epoch):
            if not self.use_recurrent_policy:
                raise NotImplementedError("mappoCFC currently requires --use-recurrent-policy")
            generator = buffer.recurrent_generator(
                buffer.advantages,
                self.num_mini_batch,
                self.data_chunk_length,
            )
            for sample in generator:
                result = self.ppo_update(policy, sample)
                ppo_updates += 1
                for key, value in result.items():
                    ppo_totals[key] += value
        for key, value in ppo_totals.items():
            train_info[key] = value / max(ppo_updates, 1)
        train_info["ppo_updates"] = float(ppo_updates)
        return train_info

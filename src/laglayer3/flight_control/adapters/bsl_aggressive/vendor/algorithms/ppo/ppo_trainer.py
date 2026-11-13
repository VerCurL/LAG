import torch
import torch.nn as nn
import math
from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm


def analytic_teacher_kl(action_dist, teacher_actions, teacher_valid, teacher_std):
    """Return KL(teacher || student) in their pre-tanh Gaussian space."""
    if not hasattr(action_dist, "base_dist"):
        raise TypeError("Teacher KL requires a squashed diagonal Normal policy")
    teacher_std = float(teacher_std)
    if not math.isfinite(teacher_std) or teacher_std <= 0.0:
        raise ValueError("teacher_kl_std must be finite and positive")
    student_mean = action_dist.base_dist.loc
    student_std = action_dist.base_dist.scale
    teacher_actions = check(teacher_actions).to(
        dtype=student_mean.dtype, device=student_mean.device
    )
    teacher_valid = check(teacher_valid).to(
        dtype=student_mean.dtype, device=student_mean.device
    )
    if teacher_actions.shape != student_mean.shape:
        raise ValueError("Teacher action shape does not match student mean shape")
    if teacher_valid.shape != student_mean.shape[:-1] + (1,):
        raise ValueError("teacher_valid must have one scalar per action vector")

    eps = 1e-6
    teacher_mean = torch.atanh(
        teacher_actions.clamp(-1.0 + eps, 1.0 - eps)
    )
    teacher_scale = torch.full_like(student_std, teacher_std)
    per_channel = (
        torch.log(student_std / teacher_scale)
        + (teacher_scale.pow(2) + (teacher_mean - student_mean).pow(2))
        / (2.0 * student_std.pow(2))
        - 0.5
    )
    per_sample = per_channel.sum(dim=-1, keepdim=True)
    mask = (teacher_valid > 0.5) & torch.isfinite(per_sample)
    if not torch.any(mask):
        raise RuntimeError("Teacher KL minibatch has no valid finite labels")
    return per_sample[mask].mean(), mask.float().mean()


class PPOTrainer():
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
        self.teacher_kl_coef = float(getattr(args, "teacher_kl_coef", 0.0))
        self.teacher_kl_final_coef = float(
            getattr(args, "teacher_kl_final_coef", 0.0)
        )
        self.teacher_kl_std = float(getattr(args, "teacher_kl_std", 0.02))
        for name, value in (
            ("teacher_kl_coef", self.teacher_kl_coef),
            ("teacher_kl_final_coef", self.teacher_kl_final_coef),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.teacher_kl_std) or self.teacher_kl_std <= 0.0:
            raise ValueError("teacher_kl_std must be finite and positive")
        self.use_teacher_kl = max(
            self.teacher_kl_coef, self.teacher_kl_final_coef
        ) > 0.0
        self.teacher_kl_progress = 0.0

    def set_training_progress(self, progress):
        progress = float(progress)
        if not math.isfinite(progress):
            raise ValueError("teacher KL progress must be finite")
        self.teacher_kl_progress = min(max(progress, 0.0), 1.0)

    @property
    def current_teacher_kl_coef(self):
        return self.teacher_kl_coef + self.teacher_kl_progress * (
            self.teacher_kl_final_coef - self.teacher_kl_coef
        )

    def ppo_update(self, policy: PPOPolicy, sample):
        teacher_aux_batch = None
        if self.use_teacher_kl:
            if len(sample) != 10:
                raise ValueError("Teacher KL requires a teacher sidecar minibatch")
            sample, teacher_aux_batch = sample[:-1], sample[-1]
        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = policy.evaluate_actions(obs_batch,
                                                                         rnn_states_actor_batch,
                                                                         rnn_states_critic_batch,
                                                                         actions_batch,
                                                                         masks_batch)

        # Obtain the loss function
        log_ratio = action_log_probs - old_action_log_probs_batch
        ratio = torch.exp(log_ratio)
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

        # ACTLayer returns transition-wise entropy already normalized by the
        # active batch size. Summing recovers the requested batch mean;
        # another mean here would weaken entropy by the minibatch size.
        policy_entropy_loss = -dist_entropy.sum()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                torch.abs(ratio - 1.0) > self.clip_param
            ).float().mean()
            returns_variance = torch.var(returns_batch, unbiased=False)
            residual_variance = torch.var(
                returns_batch - values, unbiased=False
            )
            explained_variance = torch.where(
                returns_variance > 1e-8,
                1.0 - residual_variance / returns_variance,
                torch.full_like(returns_variance, float("nan")),
            )

        teacher_kl_loss = torch.zeros((), **self.tpdv)
        teacher_valid_ratio = torch.zeros((), **self.tpdv)
        teacher_kl_coef = self.current_teacher_kl_coef
        if self.use_teacher_kl:
            action_dist = policy.actor.get_action_distribution(
                obs_batch, rnn_states_actor_batch, masks_batch
            )
            teacher_kl_loss, teacher_valid_ratio = analytic_teacher_kl(
                action_dist,
                teacher_aux_batch["teacher_actions"],
                teacher_aux_batch["teacher_valid"],
                self.teacher_kl_std,
            )

        loss = (
            policy_loss
            + value_loss * self.value_loss_coef
            + policy_entropy_loss * self.entropy_coef
            + teacher_kl_loss * teacher_kl_coef
        )

        # Optimize the loss function
        policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(policy.actor.parameters())
            critic_grad_norm = get_gard_norm(policy.critic.parameters())
        policy.optimizer.step()

        result = (
            policy_loss,
            value_loss,
            policy_entropy_loss,
            ratio,
            actor_grad_norm,
            critic_grad_norm,
            approx_kl,
            clip_fraction,
            explained_variance,
        )
        if self.use_teacher_kl:
            result = result + (
                teacher_kl_loss,
                teacher_valid_ratio,
                teacher_kl_coef,
            )
        return result

    def train(self, policy: PPOPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['approx_kl'] = 0
        train_info['clip_fraction'] = 0
        train_info['explained_variance'] = 0
        train_info['teacher_kl_loss'] = 0
        train_info['teacher_valid_ratio'] = 0
        train_info['teacher_kl_coef'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = ReplayBuffer.recurrent_generator(
                    buffer,
                    self.num_mini_batch,
                    self.data_chunk_length,
                    include_teacher_aux=self.use_teacher_kl,
                )
            else:
                raise NotImplementedError

            for sample in data_generator:

                update_result = self.ppo_update(policy, sample)
                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm, approx_kl, \
                    clip_fraction, explained_variance = update_result[:9]
                if self.use_teacher_kl:
                    teacher_kl_loss, teacher_valid_ratio, teacher_kl_coef = (
                        update_result[9:]
                    )
                else:
                    teacher_kl_loss = torch.zeros((), **self.tpdv)
                    teacher_valid_ratio = torch.zeros((), **self.tpdv)
                    teacher_kl_coef = 0.0

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()
                train_info['approx_kl'] += approx_kl.item()
                train_info['clip_fraction'] += clip_fraction.item()
                train_info['explained_variance'] += explained_variance.item()
                train_info['teacher_kl_loss'] += teacher_kl_loss.item()
                train_info['teacher_valid_ratio'] += teacher_valid_ratio.item()
                train_info['teacher_kl_coef'] += teacher_kl_coef

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

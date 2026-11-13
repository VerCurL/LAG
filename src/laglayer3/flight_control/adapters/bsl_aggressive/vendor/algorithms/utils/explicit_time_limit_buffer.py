"""Replay buffer with exact terminal-observation bootstrap for truncations.

This is opt-in for tasks that expose exact terminated/truncated semantics.  It
keeps environmental rewards unchanged while adding the critic value of the
last observable state to return/GAE equations at an administrative time limit.
"""

import numpy as np

from .buffer import ReplayBuffer


class ExplicitTimeLimitReplayBuffer(ReplayBuffer):
    """ReplayBuffer with transition-aligned truncation sidecars."""

    def __init__(self, args, num_agents, obs_space, act_space):
        super().__init__(args, num_agents, obs_space, act_space)
        self.truncation_masks = np.zeros_like(self.rewards, dtype=np.float32)
        self.terminal_bootstrap_values = np.zeros_like(
            self.rewards, dtype=np.float32
        )

    @property
    def time_limit_sidecar_nbytes(self):
        return int(
            self.truncation_masks.nbytes
            + self.terminal_bootstrap_values.nbytes
        )

    def insert(
        self,
        obs,
        actions,
        rewards,
        masks,
        action_log_probs,
        value_preds,
        rnn_states_actor,
        rnn_states_critic,
        bad_masks=None,
        qp_aux=None,
        truncation_masks=None,
        terminal_bootstrap_values=None,
        **kwargs
    ):
        mask_target = self.truncation_masks[self.step]
        value_target = self.terminal_bootstrap_values[self.step]
        mask_target.fill(0.0)
        value_target.fill(0.0)

        if truncation_masks is not None:
            truncation_masks = np.asarray(
                truncation_masks, dtype=np.float32
            )
            if truncation_masks.shape != mask_target.shape:
                raise ValueError(
                    "truncation_masks has shape {}, expected {}".format(
                        truncation_masks.shape, mask_target.shape
                    )
                )
            if not np.all(
                np.logical_or(truncation_masks == 0.0, truncation_masks == 1.0)
            ):
                raise ValueError("truncation_masks must contain only 0 or 1")
            mask_target[...] = truncation_masks

        if terminal_bootstrap_values is not None:
            terminal_bootstrap_values = np.asarray(
                terminal_bootstrap_values, dtype=np.float32
            )
            if terminal_bootstrap_values.shape != value_target.shape:
                raise ValueError(
                    "terminal_bootstrap_values has shape {}, expected {}".format(
                        terminal_bootstrap_values.shape, value_target.shape
                    )
                )
            if not np.all(np.isfinite(terminal_bootstrap_values)):
                raise ValueError(
                    "terminal_bootstrap_values must be finite"
                )
            value_target[...] = terminal_bootstrap_values

        if np.any(
            np.logical_and(mask_target == 0.0, value_target != 0.0)
        ):
            raise ValueError(
                "terminal bootstrap values require a truncation mask"
            )

        return super().insert(
            obs,
            actions,
            rewards,
            masks,
            action_log_probs,
            value_preds,
            rnn_states_actor,
            rnn_states_critic,
            bad_masks=bad_masks,
            qp_aux=qp_aux,
            **kwargs
        )

    def clear(self):
        super().clear()
        self.truncation_masks.fill(0.0)
        self.terminal_bootstrap_values.fill(0.0)

    def compute_returns(self, next_value):
        """Compute return/GAE without propagating a trace across reset."""

        self.value_preds[-1] = next_value
        if self.use_gae:
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                continuation = self.masks[step + 1]
                terminal_bootstrap = (
                    self.truncation_masks[step]
                    * self.terminal_bootstrap_values[step]
                )
                td_delta = (
                    self.rewards[step]
                    + self.gamma
                    * (
                        continuation * self.value_preds[step + 1]
                        + terminal_bootstrap
                    )
                    - self.value_preds[step]
                )
                gae = (
                    td_delta
                    + self.gamma
                    * self.gae_lambda
                    * continuation
                    * gae
                )
                self.returns[step] = gae + self.value_preds[step]
            return

        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.shape[0])):
            continuation = self.masks[step + 1]
            terminal_bootstrap = (
                self.truncation_masks[step]
                * self.terminal_bootstrap_values[step]
            )
            self.returns[step] = (
                self.rewards[step]
                + self.gamma
                * (
                    continuation * self.returns[step + 1]
                    + terminal_bootstrap
                )
            )

"""C23-S1: frozen C23 low-level Actor inside the analytic 1v1 hierarchy."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from ....algorithms.ppo.ppo_actor import PPOActor

from .analytic_hierarchical_singlecombat_task import (
    AnalyticHierarchicalSingleCombatTask,
)
from ..core.catalog import Catalog as c
from ..utils.coordinated_action import (
    apply_structured_feedback_action_mask,
    compose_structured_feedback_gain_residual,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class C23HybridHierarchicalSingleCombatTask(
    AnalyticHierarchicalSingleCombatTask
):
    """Use a hash-locked deterministic C23 Actor as a narrow gain residual."""

    def __init__(self, config):
        super().__init__(config)
        self._low_level_actor = self._load_low_level_actor()
        self._low_level_rnn_states = {}

    def reset(self, env):
        result = super().reset(env)
        self._low_level_rnn_states = {
            agent_id: np.zeros(
                (
                    1,
                    int(getattr(self.config, "C23LowLevel_recurrent_hidden_layers", 1)),
                    int(getattr(self.config, "C23LowLevel_recurrent_hidden_size", 128)),
                ),
                dtype=np.float32,
            )
            for agent_id in env.ego_ids
        }
        return result

    def _analytic_command(
        self,
        agent_id,
        sim,
        *,
        altitude_offset_m,
        heading_offset_deg,
        speed_offset_mps,
    ):
        residual, observation = self._infer_low_level_residual(
            agent_id,
            sim,
            altitude_offset_m=altitude_offset_m,
            heading_offset_deg=heading_offset_deg,
            speed_offset_mps=speed_offset_mps,
        )
        masked, mask_log = apply_structured_feedback_action_mask(
            residual,
            getattr(
                self.config,
                "C23LowLevel_action_mask",
                [1.0, 0.0, 0.0, 1.0],
            ),
        )
        multipliers, gain_log = compose_structured_feedback_gain_residual(
            masked,
            feedback_gain_fractions=getattr(
                self.config,
                "C23LowLevel_feedback_gain_fractions",
                [0.03, 0.15, 0.15],
            ),
            energy_exchange_gain_fraction=float(
                getattr(
                    self.config,
                    "C23LowLevel_energy_exchange_gain_fraction",
                    0.0125,
                )
            ),
        )

        names = (
            "CoordinatedResidual_course_tau_sec",
            "CoordinatedResidual_altitude_tau_sec",
            "CoordinatedResidual_total_energy_tau_sec",
            "CoordinatedResidual_fixed_energy_exchange_gain",
        )
        original = {name: getattr(self.config, name) for name in names}
        try:
            self.config.CoordinatedResidual_course_tau_sec = (
                float(original[names[0]]) / float(multipliers[0])
            )
            self.config.CoordinatedResidual_altitude_tau_sec = (
                float(original[names[1]]) / float(multipliers[1])
            )
            self.config.CoordinatedResidual_total_energy_tau_sec = (
                float(original[names[2]]) / float(multipliers[2])
            )
            self.config.CoordinatedResidual_fixed_energy_exchange_gain = (
                float(original[names[3]])
                * float(gain_log["energy_exchange_gain_multiplier"])
            )
            command, log = super()._analytic_command(
                agent_id,
                sim,
                altitude_offset_m=altitude_offset_m,
                heading_offset_deg=heading_offset_deg,
                speed_offset_mps=speed_offset_mps,
            )
        finally:
            for name, value in original.items():
                setattr(self.config, name, value)

        log.update(mask_log)
        log.update({f"c23_{key}": float(value) for key, value in gain_log.items()})
        log.update(
            {
                "c23_low_level_enabled": 1.0,
                "c23_low_level_observation_finite": float(
                    np.all(np.isfinite(observation))
                ),
                "c23_low_level_target_rates_are_zero": 1.0,
                "c23_low_level_residual_abs_mean": float(
                    np.mean(np.abs(residual))
                ),
                "c23_low_level_residual_abs_max": float(
                    np.max(np.abs(residual))
                ),
            }
        )
        for index, value in enumerate(residual):
            log[f"c23_low_level_action_{index}"] = float(value)
        return command, log

    def _infer_low_level_residual(
        self,
        agent_id,
        sim,
        *,
        altitude_offset_m,
        heading_offset_deg,
        speed_offset_mps,
    ):
        observation = self._build_low_level_observation(
            agent_id,
            sim,
            altitude_offset_m=altitude_offset_m,
            heading_offset_deg=heading_offset_deg,
            speed_offset_mps=speed_offset_mps,
        )
        state = self._low_level_rnn_states.get(agent_id)
        if state is None:
            state = np.zeros(
                (
                    1,
                    int(getattr(self.config, "C23LowLevel_recurrent_hidden_layers", 1)),
                    int(getattr(self.config, "C23LowLevel_recurrent_hidden_size", 128)),
                ),
                dtype=np.float32,
            )
        with torch.no_grad():
            action, _, next_state = self._low_level_actor(
                observation[None, :],
                state,
                np.ones((1, 1), dtype=np.float32),
                deterministic=True,
            )
        self._low_level_rnn_states[agent_id] = (
            next_state.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        residual = action.detach().cpu().numpy().reshape(4).astype(np.float32)
        if not np.all(np.isfinite(residual)):
            raise RuntimeError("C23 low-level Actor produced non-finite action")
        return residual, observation

    def _build_low_level_observation(
        self,
        agent_id,
        sim,
        *,
        altitude_offset_m,
        heading_offset_deg,
        speed_offset_mps,
    ):
        obs = np.zeros(32, dtype=np.float32)
        roll = float(sim.get_property_value(c.attitude_roll_rad))
        pitch = float(sim.get_property_value(c.attitude_pitch_rad))
        obs[0] = float(altitude_offset_m) / 1000.0
        obs[1] = np.deg2rad(float(heading_offset_deg))
        obs[2] = float(speed_offset_mps) / 340.0
        obs[3] = float(sim.get_property_value(c.position_h_sl_m)) / 5000.0
        obs[4:8] = (np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch))
        obs[8] = float(sim.get_property_value(c.velocities_u_mps)) / 340.0
        obs[9] = float(sim.get_property_value(c.velocities_v_mps)) / 340.0
        obs[10] = float(sim.get_property_value(c.velocities_w_mps)) / 340.0
        obs[11] = float(sim.get_property_value(c.velocities_vc_mps)) / 340.0
        obs[12] = float(sim.get_property_value(c.velocities_p_rad_sec)) / (2.0 * np.pi)
        obs[13] = float(sim.get_property_value(c.velocities_q_rad_sec)) / (2.0 * np.pi)
        obs[14] = float(sim.get_property_value(c.velocities_r_rad_sec)) / (2.0 * np.pi)
        obs[15] = float(sim.get_property_value(c.aero_alpha_deg)) / 15.0
        obs[16] = float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048 / 10.0
        physical = np.asarray(sim.get_property_values(self.action_var), dtype=np.float32)
        obs[17:21] = self._physical_to_policy(physical)

        state = self._projector.get_observation_state(agent_id)
        if state is None:
            raise RuntimeError("C23 low-level QP state is unavailable")
        previous = self._physical_to_policy(state["prev"])
        mode = str(getattr(self.config, "C23LowLevel_qp_state_mode", "repeat_prev"))
        history = previous if mode == "repeat_prev" else self._physical_to_policy(state["prev2"])
        if mode not in ("repeat_prev", "real"):
            raise ValueError("C23LowLevel_qp_state_mode must be repeat_prev or real")
        obs[21:25] = previous
        obs[25:29] = history

        # C17 was trained with explicit arc rates. The 1v1 hierarchy instead
        # issues recentered point offsets, whose exogenous target derivatives
        # are zero. Keep that semantic distinction explicit and auditable.
        obs[29:32] = 0.0
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("C23 low-level observation is non-finite")
        return np.clip(obs, -10.0, 10.0)

    def _physical_to_policy(self, action):
        result = np.asarray(action, dtype=np.float32).reshape(4).copy()
        center = float(
            getattr(self.config, "CoordinatedResidual_policy_throttle_center", 0.625)
        )
        half_range = float(
            getattr(self.config, "CoordinatedResidual_policy_throttle_half_range", 0.275)
        )
        result[3] = (result[3] - center) / half_range
        return result

    def _load_low_level_actor(self):
        relative = str(getattr(self.config, "C23LowLevel_actor_path", "")).strip()
        expected_hash = str(
            getattr(self.config, "C23LowLevel_actor_sha256", "")
        ).strip().lower()
        if not relative or len(expected_hash) != 64:
            raise ValueError("C23 low-level Actor path/hash must be frozen")
        path = (REPO_ROOT / relative).resolve()
        if REPO_ROOT not in path.parents or not path.is_file():
            raise ValueError(f"C23 low-level Actor path is invalid: {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"C23 low-level Actor hash mismatch: expected={expected_hash} actual={actual_hash}"
            )
        args = SimpleNamespace(
            gain=0.01,
            initial_log_std=None,
            initial_action_std=0.25,
            use_squashed_action=True,
            hidden_size="128 128",
            act_hidden_size="128 128",
            activation_id=1,
            use_feature_normalization=False,
            use_recurrent_policy=True,
            recurrent_hidden_size=int(
                getattr(self.config, "C23LowLevel_recurrent_hidden_size", 128)
            ),
            recurrent_hidden_layers=int(
                getattr(self.config, "C23LowLevel_recurrent_hidden_layers", 1)
            ),
            use_prior=False,
        )
        actor = PPOActor(
            args,
            spaces.Box(-10.0, 10.0, shape=(32,), dtype=np.float32),
            spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            device=torch.device("cpu"),
        )
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(path), map_location="cpu")
        actor.load_state_dict(state, strict=True)
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        return actor


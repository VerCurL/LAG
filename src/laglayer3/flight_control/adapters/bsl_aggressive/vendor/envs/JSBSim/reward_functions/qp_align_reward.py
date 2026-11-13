import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class QPAlignReward(BaseRewardFunction):
    """
    Small positive reward for policy actions that already agree with the QP projector.

    This reward treats the QP projector as a weak teacher: if the policy's normalized
    action is close to the projected action, it receives a small bounded bonus. It
    never penalizes disagreement directly, so task tracking remains the primary
    objective.
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = bool(getattr(config, f"{cls}_enable", False))
        self.d_max = max(float(getattr(config, f"{cls}_d_max", 0.25)), 1e-6)
        self.channel_weights = self._make_weights(getattr(config, f"{cls}_channel_weights", [0.25, 0.40, 0.10, 0.25]))
        self.gate_enable = bool(getattr(config, f"{cls}_gate_enable", True))
        self.gate_heading_deg = float(getattr(config, f"{cls}_gate_heading_deg", 60.0))
        self.gate_altitude_m = float(getattr(config, f"{cls}_gate_altitude_m", 600.0))
        self.gate_speed_mps = float(getattr(config, f"{cls}_gate_speed_mps", 35.0))
        self.gate_pitch_deg = float(getattr(config, f"{cls}_gate_pitch_deg", 20.0))
        self.gate_roll_deg = float(getattr(config, f"{cls}_gate_roll_deg", 75.0))
        self.health_gate_enable = bool(getattr(config, f"{cls}_health_gate_enable", False))
        self.health_box_active_limit = float(getattr(config, f"{cls}_health_box_active_limit", 0.0))
        self.health_box_active_fade = max(float(getattr(config, f"{cls}_health_box_active_fade", 1.0)), 1e-6)
        self.health_residual_limit = float(getattr(config, f"{cls}_health_residual_limit", self.d_max))
        self.health_residual_fade = max(float(getattr(config, f"{cls}_health_residual_fade", self.d_max)), 1e-6)
        self.health_fail_streak_limit = int(getattr(config, f"{cls}_health_fail_streak_limit", 0))
        self.curriculum_enable = bool(getattr(config, f"{cls}_curriculum_enable", True))
        self.curriculum_levels = list(getattr(config, f"{cls}_curriculum_levels", [0.0, 1.0, 3.0, 5.0]))
        self.curriculum_scales = list(getattr(config, f"{cls}_curriculum_scales", [0.0, 0.5, 1.0, 1.5]))
        self.reward_item_names = [
            self.__class__.__name__ + item
            for item in [
                "",
                "_align",
                "_gate",
                "_residual",
                "_scale",
                "_health_gate",
                "_box_gate",
                "_residual_gate",
            ]
        ]

    @staticmethod
    def _make_weights(value):
        weights = np.asarray(value, dtype=np.float64).reshape(-1)
        if weights.size == 0:
            weights = np.ones(4, dtype=np.float64)
        out = np.ones(4, dtype=np.float64)
        out[: min(4, weights.size)] = weights[: min(4, weights.size)]
        total = float(np.sum(np.abs(out)))
        if total <= 1e-9:
            return np.full(4, 0.25, dtype=np.float64)
        return np.abs(out) / total

    def reset(self, task, env):
        env._last_qp_align_reward_log = {}
        return super().reset(task, env)

    def _curriculum_scale(self, env):
        if not self.curriculum_enable:
            return 1.0
        route_state = getattr(env, "heading_route_state", None)
        if not isinstance(route_state, dict):
            return 1.0
        level = float(route_state.get("level", 0.0))
        levels = np.asarray(self.curriculum_levels, dtype=np.float64)
        scales = np.asarray(self.curriculum_scales, dtype=np.float64)
        if levels.size != scales.size or levels.size == 0:
            return 1.0
        if levels.size == 1:
            return float(scales[0])
        order = np.argsort(levels)
        return float(np.interp(level, levels[order], scales[order]))

    def _gate(self, sim):
        if not self.gate_enable:
            return 1.0
        checks = (
            abs(float(sim.get_property_value(c.delta_heading))) <= self.gate_heading_deg,
            abs(float(sim.get_property_value(c.delta_altitude))) <= self.gate_altitude_m,
            abs(float(sim.get_property_value(c.delta_velocities_u))) <= self.gate_speed_mps,
            abs(float(sim.get_property_value(c.attitude_theta_deg))) <= self.gate_pitch_deg,
            abs(float(sim.get_property_value(c.attitude_phi_deg))) <= self.gate_roll_deg,
        )
        return 1.0 if all(checks) else 0.0

    def _projector_log(self, env, agent_id):
        logs = getattr(env, "_last_action_projector_log", None)
        if not isinstance(logs, dict):
            return {}
        log = logs.get(agent_id, {})
        return log if isinstance(log, dict) else {}

    @staticmethod
    def _soft_upper_gate(value, limit, fade):
        if not np.isfinite(value):
            return 0.0
        if not np.isfinite(limit):
            return 1.0
        if value <= limit:
            return 1.0
        return float(1.0 - np.clip((value - limit) / max(fade, 1e-6), 0.0, 1.0))

    def _health_gate(self, log, weighted_residual):
        if not self.health_gate_enable:
            return 1.0, 1.0, 1.0

        fail_streak = int(log.get("qp_projector_fail_streak", 0))
        if fail_streak > self.health_fail_streak_limit:
            return 0.0, 0.0, 0.0

        box_active = float(log.get("qp_projector_box_active_count", 0.0))
        box_gate = self._soft_upper_gate(
            box_active,
            self.health_box_active_limit,
            self.health_box_active_fade,
        )
        residual_gate = self._soft_upper_gate(
            float(weighted_residual),
            self.health_residual_limit,
            self.health_residual_fade,
        )
        return float(box_gate * residual_gate), box_gate, residual_gate

    def get_reward(self, task, env, agent_id):
        zero_items = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        log = self._projector_log(env, agent_id)
        if not bool(log.get("qp_projector_enabled", False)):
            env._last_qp_align_reward_log = getattr(env, "_last_qp_align_reward_log", {})
            env._last_qp_align_reward_log[agent_id] = {
                "qp_align_reward": 0.0,
                "qp_align": np.nan,
                "qp_align_gate": np.nan,
                "qp_align_residual": np.nan,
                "qp_align_scale": np.nan,
                "qp_align_health_gate": np.nan,
                "qp_align_box_gate": np.nan,
                "qp_align_residual_gate": np.nan,
                "qp_align_applicable": False,
                "qp_align_reward_enabled": bool(self.enable),
            }
            return self._process(0.0, agent_id, zero_items)

        residual = np.array(
            [
                abs(float(log.get("qp_residual_aileron", 0.0))),
                abs(float(log.get("qp_residual_elevator", 0.0))),
                abs(float(log.get("qp_residual_rudder", 0.0))),
                abs(float(log.get("qp_residual_throttle", 0.0))),
            ],
            dtype=np.float64,
        )
        residual = np.where(np.isfinite(residual), residual, 0.0)
        weighted_residual = float(np.dot(self.channel_weights, residual))
        align = float(1.0 - np.clip(weighted_residual / self.d_max, 0.0, 1.0))
        state_gate = self._gate(env.agents[agent_id])
        health_gate, box_gate, residual_gate = self._health_gate(log, weighted_residual)
        gate = float(state_gate * health_gate)
        curriculum_scale = self._curriculum_scale(env)
        raw_reward = float(gate * curriculum_scale * align)
        applied_raw_reward = raw_reward if self.enable else 0.0
        scaled_reward = applied_raw_reward * float(self.reward_scale)

        env._last_qp_align_reward_log = getattr(env, "_last_qp_align_reward_log", {})
        env._last_qp_align_reward_log[agent_id] = {
            "qp_align_reward": scaled_reward,
            "qp_align": align,
            "qp_align_gate": gate,
            "qp_align_residual": weighted_residual,
            "qp_align_scale": curriculum_scale,
            "qp_align_state_gate": state_gate,
            "qp_align_health_gate": health_gate,
            "qp_align_box_gate": box_gate,
            "qp_align_residual_gate": residual_gate,
            "qp_align_applicable": True,
            "qp_align_reward_enabled": bool(self.enable),
        }
        return self._process(
            applied_raw_reward,
            agent_id,
            (align, gate, weighted_residual, curriculum_scale, health_gate, box_gate, residual_gate),
        )

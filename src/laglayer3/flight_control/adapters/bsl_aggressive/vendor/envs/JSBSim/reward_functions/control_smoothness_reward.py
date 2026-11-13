import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class ControlSmoothnessReward(BaseRewardFunction):
    """
    Penalize abrupt control command changes to encourage smooth maneuvering.

    Reward (before scale):
        -(w_a * |d_aileron| + w_e * |d_elevator| + w_r * |d_rudder| + w_t * |d_throttle|)
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = getattr(config, f"{cls}_enable", False)
        self.aileron_weight = getattr(config, f"{cls}_aileron_weight", 1.0)
        self.elevator_weight = getattr(config, f"{cls}_elevator_weight", 1.0)
        self.rudder_weight = getattr(config, f"{cls}_rudder_weight", 1.0)
        self.throttle_weight = getattr(config, f"{cls}_throttle_weight", 0.5)
        self.deadband = getattr(config, f"{cls}_deadband", 0.0)
        # Decision gate (from tactical pipeline): make smoothness stricter in calm
        # and more permissive in high-threat tactical windows.
        self.use_decision_gate = getattr(config, f"{cls}_use_decision_gate", True)
        self.calm_gain = float(getattr(config, f"{cls}_calm_gain", 1.2))
        self.tactical_relief = float(getattr(config, f"{cls}_tactical_relief", 0.8))
        self.min_multiplier = float(getattr(config, f"{cls}_min_multiplier", 0.35))
        self.max_multiplier = float(getattr(config, f"{cls}_max_multiplier", 2.5))
        # Heading-control gate: keep commands smooth near a settled target, but
        # relax the penalty when the target error is large enough to need a hard
        # maneuver.
        self.tracking_relief_enable = getattr(config, f"{cls}_tracking_relief_enable", False)
        self.tracking_relief_min_multiplier = float(
            getattr(config, f"{cls}_tracking_relief_min_multiplier", 0.25)
        )
        self.tracking_relief_error_scale = float(getattr(config, f"{cls}_tracking_relief_error_scale", 4.0))
        self.heading_error_scale_deg = float(getattr(config, f"{cls}_heading_error_scale_deg", 5.0))
        self.altitude_error_scale_m = float(getattr(config, f"{cls}_altitude_error_scale_m", 15.24))
        self.speed_error_scale_mps = float(getattr(config, f"{cls}_speed_error_scale_mps", 24.0))

        self._prev_cmd = {}
        self.reward_item_names = [
            self.__class__.__name__ + item for item in ["", "_delta", "_multiplier", "_raw_delta"]
        ]

    def _get_control_cmd(self, sim):
        return np.array([
            sim.get_property_value(c.fcs_aileron_cmd_norm),
            sim.get_property_value(c.fcs_elevator_cmd_norm),
            sim.get_property_value(c.fcs_rudder_cmd_norm),
            sim.get_property_value(c.fcs_throttle_cmd_norm),
        ], dtype=np.float32)

    def _decision_multiplier(self, task, agent_id):
        if not self.use_decision_gate or not hasattr(task, "get_decision_ctx"):
            return 1.0
        ctx = task.get_decision_ctx(agent_id) or {}
        permit = float(np.clip(ctx.get("permit", 0.0), 0.0, 1.0))
        s_threat = float(np.clip(ctx.get("s_threat", 0.0), 0.0, 1.0))
        multiplier = 1.0 + self.calm_gain * (1.0 - permit) - self.tactical_relief * (permit * s_threat)
        return float(np.clip(multiplier, self.min_multiplier, self.max_multiplier))

    def _tracking_relief_multiplier(self, task, env, agent_id):
        if not self.tracking_relief_enable:
            return 1.0
        sim = env.agents[agent_id]
        heading_norm = abs(sim.get_property_value(c.delta_heading)) / max(self.heading_error_scale_deg, 1e-6)
        altitude_norm = abs(sim.get_property_value(c.delta_altitude)) / max(self.altitude_error_scale_m, 1e-6)
        if hasattr(task, "get_speed_error_mps"):
            speed_error_mps = float(task.get_speed_error_mps(sim))
        else:
            speed_error_mps = float(sim.get_property_value(c.delta_velocities_u))
        speed_norm = abs(speed_error_mps) / max(self.speed_error_scale_mps, 1e-6)
        tracking_error = max(heading_norm, altitude_norm, speed_norm)
        if tracking_error <= 1.0:
            return 1.0
        relief_span = max(self.tracking_relief_error_scale, 1e-6)
        relief = float(np.clip((tracking_error - 1.0) / relief_span, 0.0, 1.0))
        min_multiplier = float(np.clip(self.tracking_relief_min_multiplier, 0.0, 1.0))
        return 1.0 - relief * (1.0 - min_multiplier)

    def reset(self, task, env):
        self._prev_cmd = {
            agent_id: self._get_control_cmd(sim)
            for agent_id, sim in env.agents.items()
        }
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        if not self.enable:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0))

        cur = self._get_control_cmd(env.agents[agent_id])
        prev = self._prev_cmd.get(agent_id, cur)
        delta = np.abs(cur - prev)
        if self.deadband > 0:
            delta = np.maximum(delta - self.deadband, 0.0)

        weighted_delta = (
            self.aileron_weight * delta[0]
            + self.elevator_weight * delta[1]
            + self.rudder_weight * delta[2]
            + self.throttle_weight * delta[3]
        )
        raw_weighted_delta = float(weighted_delta)
        multiplier = self._decision_multiplier(task, agent_id) * self._tracking_relief_multiplier(task, env, agent_id)
        weighted_delta *= multiplier

        self._prev_cmd[agent_id] = cur
        return self._process(-weighted_delta, agent_id, (weighted_delta, float(multiplier), raw_weighted_delta))

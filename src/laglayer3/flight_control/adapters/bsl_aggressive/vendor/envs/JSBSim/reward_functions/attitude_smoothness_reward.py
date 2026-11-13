import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class AttitudeSmoothnessReward(BaseRewardFunction):
    """
    Penalize abrupt changes in roll/pitch rates.

    The core cost is a squared finite difference on p/q:
        cost = (dp / p_scale)^2 + (dq / q_scale)^2

    Optionally, an additional pitch-angle oscillation term penalizes three
    consecutive large pitch changes whose directions alternate.

    This targets short-horizon attitude oscillation without directly punishing
    roll/pitch angle magnitude.
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = getattr(config, f"{cls}_enable", False)
        self.p_scale = float(getattr(config, f"{cls}_p_scale_rad_sec", 0.5))
        self.q_scale = float(getattr(config, f"{cls}_q_scale_rad_sec", 0.5))
        self.pitch_osc_enable = getattr(config, f"{cls}_pitch_osc_enable", False)
        self.pitch_osc_weight = float(getattr(config, f"{cls}_pitch_osc_weight", 1.0))
        self.pitch_osc_deadband_deg = float(getattr(config, f"{cls}_pitch_osc_deadband_deg", 2.0))
        self.pitch_osc_scale_deg = float(getattr(config, f"{cls}_pitch_osc_scale_deg", 5.0))
        self.pitch_rate_weight = float(getattr(config, f"{cls}_pitch_rate_weight", 0.0))
        self.pitch_rate_scale = float(getattr(config, f"{cls}_pitch_rate_scale_rad_sec", 0.5))
        self.tracking_relief_enable = getattr(config, f"{cls}_tracking_relief_enable", False)
        self.tracking_relief_min_multiplier = float(
            getattr(config, f"{cls}_tracking_relief_min_multiplier", 0.25)
        )
        self.tracking_relief_error_scale = float(getattr(config, f"{cls}_tracking_relief_error_scale", 4.0))
        self.heading_error_scale_deg = float(getattr(config, f"{cls}_heading_error_scale_deg", 5.0))
        self.altitude_error_scale_m = float(getattr(config, f"{cls}_altitude_error_scale_m", 15.24))
        self.speed_error_scale_mps = float(getattr(config, f"{cls}_speed_error_scale_mps", 24.0))

        self._prev_rates = {}
        self._pitch_history = {}
        self.reward_item_names = [
            self.__class__.__name__,
            self.__class__.__name__ + "_delta_p_rad_sec",
            self.__class__.__name__ + "_delta_q_rad_sec",
            self.__class__.__name__ + "_smooth_cost",
            self.__class__.__name__ + "_pitch_osc_cost",
            self.__class__.__name__ + "_pitch_osc_trigger",
            self.__class__.__name__ + "_pitch_rate_cost",
            self.__class__.__name__ + "_multiplier",
        ]

    def reset(self, task, env):
        self._prev_rates = {
            agent_id: self._get_rates(sim)
            for agent_id, sim in env.agents.items()
        }
        self._pitch_history = {
            agent_id: [self._get_pitch_deg(sim)]
            for agent_id, sim in env.agents.items()
        }
        return super().reset(task, env)

    @staticmethod
    def _get_rates(sim):
        return np.array([
            sim.get_property_value(c.velocities_p_rad_sec),
            sim.get_property_value(c.velocities_q_rad_sec),
        ], dtype=np.float64)

    @staticmethod
    def _get_pitch_deg(sim):
        return float(np.rad2deg(sim.get_property_value(c.attitude_pitch_rad)))

    def _get_pitch_oscillation_cost(self, sim, agent_id):
        history = self._pitch_history.setdefault(agent_id, [])
        history.append(self._get_pitch_deg(sim))
        if len(history) > 4:
            del history[:-4]

        if not self.pitch_osc_enable or len(history) < 4:
            return 0.0, 0.0

        d1 = history[1] - history[0]
        d2 = history[2] - history[1]
        d3 = history[3] - history[2]
        deadband = max(self.pitch_osc_deadband_deg, 0.0)
        large_change = abs(d1) > deadband and abs(d2) > deadband and abs(d3) > deadband
        alternating = d1 * d2 < 0.0 and d2 * d3 < 0.0
        if not (large_change and alternating):
            return 0.0, 0.0

        scale = max(self.pitch_osc_scale_deg, 1e-6)
        avg_change = (abs(d1) + abs(d2) + abs(d3)) / 3.0
        return float((avg_change / scale) ** 2), 1.0

    def _tracking_relief_multiplier(self, env, agent_id):
        if not self.tracking_relief_enable:
            return 1.0
        sim = env.agents[agent_id]
        heading_norm = abs(sim.get_property_value(c.delta_heading)) / max(self.heading_error_scale_deg, 1e-6)
        altitude_norm = abs(sim.get_property_value(c.delta_altitude)) / max(self.altitude_error_scale_m, 1e-6)
        speed_norm = abs(sim.get_property_value(c.delta_velocities_u)) / max(self.speed_error_scale_mps, 1e-6)
        tracking_error = max(heading_norm, altitude_norm, speed_norm)
        if tracking_error <= 1.0:
            return 1.0
        relief_span = max(self.tracking_relief_error_scale, 1e-6)
        relief = float(np.clip((tracking_error - 1.0) / relief_span, 0.0, 1.0))
        min_multiplier = float(np.clip(self.tracking_relief_min_multiplier, 0.0, 1.0))
        return 1.0 - relief * (1.0 - min_multiplier)

    def get_reward(self, task, env, agent_id):
        if not self.enable:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        cur = self._get_rates(env.agents[agent_id])
        prev = self._prev_rates.get(agent_id, cur)
        delta = cur - prev

        p_scale = max(self.p_scale, 1e-6)
        q_scale = max(self.q_scale, 1e-6)
        smooth_cost = float((delta[0] / p_scale) ** 2 + (delta[1] / q_scale) ** 2)
        pitch_osc_cost, pitch_osc_trigger = self._get_pitch_oscillation_cost(env.agents[agent_id], agent_id)
        pitch_rate_scale = max(self.pitch_rate_scale, 1e-6)
        pitch_rate_cost = float((cur[1] / pitch_rate_scale) ** 2)
        multiplier = self._tracking_relief_multiplier(env, agent_id)
        total_cost = multiplier * (
            smooth_cost + self.pitch_osc_weight * pitch_osc_cost + self.pitch_rate_weight * pitch_rate_cost
        )

        self._prev_rates[agent_id] = cur
        return self._process(
            -total_cost,
            agent_id,
            (
                float(delta[0]),
                float(delta[1]),
                smooth_cost,
                pitch_osc_cost,
                pitch_osc_trigger,
                pitch_rate_cost,
                float(multiplier),
            ),
        )

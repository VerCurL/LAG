import math
import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class TargetDeltaSmoothnessReward(BaseRewardFunction):
    """
    Penalize abrupt changes of the three tracking errors.

    This is not a hold reward. It encourages the aircraft to reduce heading,
    altitude, and speed errors through smooth trajectories instead of bouncing
    around the target zone.
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = bool(getattr(config, f"{cls}_enable", False))
        self.heading_weight = float(getattr(config, f"{cls}_heading_weight", 1.0))
        self.altitude_weight = float(getattr(config, f"{cls}_altitude_weight", 0.7))
        self.speed_weight = float(getattr(config, f"{cls}_speed_weight", 0.8))
        self.heading_step_scale_deg = float(getattr(config, f"{cls}_heading_step_scale_deg", 4.0))
        self.altitude_step_scale_m = float(getattr(config, f"{cls}_altitude_step_scale_m", 35.0))
        self.speed_step_scale_mps = float(getattr(config, f"{cls}_speed_step_scale_mps", 4.0))
        self.deadband = float(getattr(config, f"{cls}_deadband", 0.0))
        self.tracking_relief_enable = bool(getattr(config, f"{cls}_tracking_relief_enable", True))
        self.tracking_relief_min_multiplier = float(
            getattr(config, f"{cls}_tracking_relief_min_multiplier", 0.35)
        )
        self.tracking_relief_error_scale = float(getattr(config, f"{cls}_tracking_relief_error_scale", 4.0))
        self.heading_error_scale_deg = float(getattr(config, f"{cls}_heading_error_scale_deg", 20.0))
        self.altitude_error_scale_m = float(getattr(config, f"{cls}_altitude_error_scale_m", 150.0))
        self.speed_error_scale_mps = float(getattr(config, f"{cls}_speed_error_scale_mps", 35.0))
        self._prev_delta = {}
        self.reward_item_names = [
            self.__class__.__name__ + item
            for item in ["", "_penalty", "_heading_step", "_altitude_step", "_speed_step", "_multiplier"]
        ]

    @staticmethod
    def _pseudo_huber(x):
        return math.sqrt(1.0 + x * x) - 1.0

    def _get_delta(self, sim):
        return np.array(
            [
                sim.get_property_value(c.delta_heading),
                sim.get_property_value(c.delta_altitude),
                sim.get_property_value(c.delta_velocities_u),
            ],
            dtype=np.float64,
        )

    def _tracking_relief_multiplier(self, sim):
        if not self.tracking_relief_enable:
            return 1.0
        heading_norm = abs(sim.get_property_value(c.delta_heading)) / max(self.heading_error_scale_deg, 1e-6)
        altitude_norm = abs(sim.get_property_value(c.delta_altitude)) / max(self.altitude_error_scale_m, 1e-6)
        speed_norm = abs(sim.get_property_value(c.delta_velocities_u)) / max(self.speed_error_scale_mps, 1e-6)
        tracking_error = max(heading_norm, altitude_norm, speed_norm)
        if tracking_error <= 1.0:
            return 1.0
        relief = float(np.clip((tracking_error - 1.0) / max(self.tracking_relief_error_scale, 1e-6), 0.0, 1.0))
        min_multiplier = float(np.clip(self.tracking_relief_min_multiplier, 0.0, 1.0))
        return 1.0 - relief * (1.0 - min_multiplier)

    def reset(self, task, env):
        self._prev_delta = {
            agent_id: self._get_delta(sim)
            for agent_id, sim in env.agents.items()
        }
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        if not self.enable:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0, 0.0, 1.0))

        sim = env.agents[agent_id]
        cur = self._get_delta(sim)
        prev = self._prev_delta.get(agent_id, cur)
        step = np.abs(cur - prev)
        if self.deadband > 0:
            step = np.maximum(step - self.deadband, 0.0)

        heading_step = self._pseudo_huber(step[0] / max(self.heading_step_scale_deg, 1e-6))
        altitude_step = self._pseudo_huber(step[1] / max(self.altitude_step_scale_m, 1e-6))
        speed_step = self._pseudo_huber(step[2] / max(self.speed_step_scale_mps, 1e-6))
        penalty = (
            self.heading_weight * heading_step
            + self.altitude_weight * altitude_step
            + self.speed_weight * speed_step
        )
        multiplier = self._tracking_relief_multiplier(sim)
        penalty *= multiplier

        self._prev_delta[agent_id] = cur
        return self._process(
            -penalty,
            agent_id,
            (penalty, heading_step, altitude_step, speed_step, multiplier),
        )

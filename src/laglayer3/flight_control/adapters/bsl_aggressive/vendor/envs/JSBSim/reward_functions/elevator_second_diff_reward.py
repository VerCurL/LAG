import math

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class ElevatorSecondDiffReward(BaseRewardFunction):
    """
    Penalize elevator command second difference.

    This targets repeated elevator reversals more directly than a generic
    attitude-rate penalty:

        de_acc = de_t - 2 * de_{t-1} + de_{t-2}

    If the elevator is held or changes with a smooth trend, de_acc is small.
    If the elevator keeps reversing direction, de_acc becomes large.
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = bool(getattr(config, f"{cls}_enable", False))
        self.acc_scale = float(getattr(config, f"{cls}_acc_scale", 0.35))
        self.deadband = float(getattr(config, f"{cls}_deadband", 0.01))
        self.tracking_relief_enable = bool(getattr(config, f"{cls}_tracking_relief_enable", True))
        self.tracking_relief_min_multiplier = float(getattr(config, f"{cls}_tracking_relief_min_multiplier", 0.35))
        self.tracking_relief_error_scale = float(getattr(config, f"{cls}_tracking_relief_error_scale", 4.0))
        self.heading_error_scale_deg = float(getattr(config, f"{cls}_heading_error_scale_deg", 20.0))
        self.altitude_error_scale_m = float(getattr(config, f"{cls}_altitude_error_scale_m", 150.0))
        self.speed_error_scale_mps = float(getattr(config, f"{cls}_speed_error_scale_mps", 35.0))
        self._prev_elevator = {}
        self._prev_prev_elevator = {}
        self.reward_item_names = [
            self.__class__.__name__ + item
            for item in ["", "_cost", "_second_diff", "_multiplier"]
        ]

    @staticmethod
    def _pseudo_huber(x):
        return math.sqrt(1.0 + x * x) - 1.0

    @staticmethod
    def _get_elevator(sim):
        return float(sim.get_property_value(c.fcs_elevator_cmd_norm))

    def _tracking_relief_multiplier(self, sim):
        if not self.tracking_relief_enable:
            return 1.0
        heading_norm = abs(sim.get_property_value(c.delta_heading)) / max(self.heading_error_scale_deg, 1e-6)
        altitude_norm = abs(sim.get_property_value(c.delta_altitude)) / max(self.altitude_error_scale_m, 1e-6)
        speed_norm = abs(sim.get_property_value(c.delta_velocities_u)) / max(self.speed_error_scale_mps, 1e-6)
        tracking_error = max(heading_norm, altitude_norm, speed_norm)
        if tracking_error <= 1.0:
            return 1.0
        relief = min(max((tracking_error - 1.0) / max(self.tracking_relief_error_scale, 1e-6), 0.0), 1.0)
        min_multiplier = min(max(self.tracking_relief_min_multiplier, 0.0), 1.0)
        return 1.0 - relief * (1.0 - min_multiplier)

    def reset(self, task, env):
        self._prev_elevator = {}
        self._prev_prev_elevator = {}
        for agent_id, sim in env.agents.items():
            elevator = self._get_elevator(sim)
            self._prev_elevator[agent_id] = elevator
            self._prev_prev_elevator[agent_id] = elevator
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        if not self.enable:
            return self._process(0.0, agent_id, (0.0, 0.0, 1.0))

        sim = env.agents[agent_id]
        cur = self._get_elevator(sim)
        prev = self._prev_elevator.get(agent_id, cur)
        prev_prev = self._prev_prev_elevator.get(agent_id, prev)

        second_diff = cur - 2.0 * prev + prev_prev
        abs_second_diff = abs(second_diff)
        if self.deadband > 0.0:
            abs_second_diff = max(abs_second_diff - self.deadband, 0.0)

        cost = self._pseudo_huber(abs_second_diff / max(self.acc_scale, 1e-6))
        multiplier = self._tracking_relief_multiplier(sim)
        cost *= multiplier

        self._prev_prev_elevator[agent_id] = prev
        self._prev_elevator[agent_id] = cur
        return self._process(-cost, agent_id, (cost, second_diff, multiplier))

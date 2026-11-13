import math

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class LongitudinalDampingReward(BaseRewardFunction):
    """
    Penalize under-damped longitudinal motion.

    This reward targets aircraft state oscillation, not just elevator motion:
      - pitch rate q is always damped;
      - vertical speed is damped more strongly near the altitude target;
      - elevator first difference is mildly damped to discourage bang-bang trim.

    The altitude gate keeps climb/descent freedom when the aircraft is far from
    the altitude target, then asks for vertical speed to settle near the target.
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = bool(getattr(config, f"{cls}_enable", False))
        self.q_scale_deg_s = float(getattr(config, f"{cls}_q_scale_deg_s", 8.0))
        self.vs_scale_mps = float(getattr(config, f"{cls}_vs_scale_mps", 8.0))
        self.delta_elevator_scale = float(getattr(config, f"{cls}_delta_elevator_scale", 0.25))
        self.q_weight = float(getattr(config, f"{cls}_q_weight", 0.45))
        self.vs_weight = float(getattr(config, f"{cls}_vs_weight", 0.35))
        self.delta_elevator_weight = float(getattr(config, f"{cls}_delta_elevator_weight", 0.20))
        self.alt_gate_scale_m = float(getattr(config, f"{cls}_alt_gate_scale_m", 180.0))
        self.min_gate = float(getattr(config, f"{cls}_min_gate", 0.25))
        self.deadband_q_deg_s = float(getattr(config, f"{cls}_deadband_q_deg_s", 0.1))
        self.deadband_vs_mps = float(getattr(config, f"{cls}_deadband_vs_mps", 0.05))
        self.deadband_delta_elevator = float(getattr(config, f"{cls}_deadband_delta_elevator", 0.01))
        self._prev_elevator = {}
        self.reward_item_names = [
            self.__class__.__name__ + item
            for item in ["", "_cost", "_q_cost", "_vs_cost", "_de_cost", "_alt_gate"]
        ]

    @staticmethod
    def _pseudo_huber(x):
        return math.sqrt(1.0 + x * x) - 1.0

    @staticmethod
    def _deadband_abs(value, deadband):
        return max(abs(float(value)) - max(float(deadband), 0.0), 0.0)

    @staticmethod
    def _get_elevator(sim):
        return float(sim.get_property_value(c.fcs_elevator_cmd_norm))

    def _altitude_gate(self, sim):
        altitude_error = abs(float(sim.get_property_value(c.delta_altitude)))
        near = math.exp(-self._pseudo_huber(altitude_error / max(self.alt_gate_scale_m, 1e-6)))
        min_gate = min(max(self.min_gate, 0.0), 1.0)
        return min_gate + (1.0 - min_gate) * near

    def reset(self, task, env):
        self._prev_elevator = {}
        for agent_id, sim in env.agents.items():
            self._prev_elevator[agent_id] = self._get_elevator(sim)
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        if not self.enable:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0, 0.0, 0.0))

        sim = env.agents[agent_id]
        q_deg_s = math.degrees(float(sim.get_property_value(c.velocities_q_rad_sec)))
        vertical_speed_mps = float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048
        elevator = self._get_elevator(sim)
        prev_elevator = self._prev_elevator.get(agent_id, elevator)
        delta_elevator = elevator - prev_elevator

        q_abs = self._deadband_abs(q_deg_s, self.deadband_q_deg_s)
        vs_abs = self._deadband_abs(vertical_speed_mps, self.deadband_vs_mps)
        de_abs = self._deadband_abs(delta_elevator, self.deadband_delta_elevator)

        alt_gate = self._altitude_gate(sim)
        q_cost = self._pseudo_huber(q_abs / max(self.q_scale_deg_s, 1e-6))
        vs_cost = self._pseudo_huber(vs_abs / max(self.vs_scale_mps, 1e-6)) * alt_gate
        de_cost = self._pseudo_huber(de_abs / max(self.delta_elevator_scale, 1e-6))

        weight_sum = max(self.q_weight + self.vs_weight + self.delta_elevator_weight, 1e-6)
        cost = (
            self.q_weight * q_cost
            + self.vs_weight * vs_cost
            + self.delta_elevator_weight * de_cost
        ) / weight_sum

        self._prev_elevator[agent_id] = elevator
        return self._process(-cost, agent_id, (cost, q_cost, vs_cost, de_cost, alt_gate))

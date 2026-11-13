import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class LongitudinalMIMOStabilityReward(BaseRewardFunction):
    """
    Soft MIMO-aware longitudinal stability regularizer for continuous heading control.

    This reward does not penalize non-zero pitch, AOA, or elevator trim. It keeps
    HeadingReward responsible for route/heading tracking and only discourages
    longitudinally non-physical behavior: vertical-speed mismatch, speed/altitude
    tracking residuals, persistent pitch-rate drift, elevator saturation, and fast
    elevator changes near low maneuver demand.
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = bool(getattr(config, f"{cls}_enable", False))

        self.alt_scale_m = float(getattr(config, f"{cls}_alt_scale_m", 300.0))
        self.speed_scale_mps = float(getattr(config, f"{cls}_speed_scale_mps", 35.0))
        self.k_h = float(getattr(config, f"{cls}_k_h", 0.03))
        self.vs_max_mps = float(getattr(config, f"{cls}_vs_max_mps", 25.0))
        self.vs_scale_mps = float(getattr(config, f"{cls}_vs_scale_mps", 10.0))
        self.q_scale_rad_sec = float(getattr(config, f"{cls}_q_scale_rad_sec", 0.35))
        self.sat_start = float(getattr(config, f"{cls}_sat_start", 0.85))
        self.sat_scale = float(getattr(config, f"{cls}_sat_scale", 0.10))
        self.de_scale = float(getattr(config, f"{cls}_de_scale", 0.15))

        self.heading_scale_deg = float(getattr(config, f"{cls}_heading_scale_deg", 45.0))
        self.alt_maneuver_scale_m = float(getattr(config, f"{cls}_alt_maneuver_scale_m", 800.0))
        self.speed_maneuver_scale_mps = float(getattr(config, f"{cls}_speed_maneuver_scale_mps", 60.0))

        self.w_alt = float(getattr(config, f"{cls}_w_alt", 0.5))
        self.w_speed = float(getattr(config, f"{cls}_w_speed", 0.6))
        self.w_vs = float(getattr(config, f"{cls}_w_vs", 1.0))
        self.w_q = float(getattr(config, f"{cls}_w_q", 0.35))
        self.w_sat = float(getattr(config, f"{cls}_w_sat", 0.25))
        self.w_de = float(getattr(config, f"{cls}_w_de", 0.20))
        self.huber_delta = float(getattr(config, f"{cls}_huber_delta", 1.5))

        self._prev_elevator = {}
        self.reward_item_names = [
            self.__class__.__name__,
            self.__class__.__name__ + "_alt_cost",
            self.__class__.__name__ + "_speed_cost",
            self.__class__.__name__ + "_vs_cost",
            self.__class__.__name__ + "_q_cost",
            self.__class__.__name__ + "_sat_cost",
            self.__class__.__name__ + "_de_cost",
            self.__class__.__name__ + "_stability_weight",
            self.__class__.__name__ + "_vs_ref_mps",
            self.__class__.__name__ + "_elevator_delta",
        ]

    def reset(self, task, env):
        self._prev_elevator = {
            agent_id: float(sim.get_property_value(c.fcs_elevator_cmd_norm))
            for agent_id, sim in env.agents.items()
        }
        return super().reset(task, env)

    def _huber(self, value):
        delta = max(self.huber_delta, 1e-6)
        abs_value = abs(float(value))
        if abs_value <= delta:
            return 0.5 * abs_value ** 2
        return delta * (abs_value - 0.5 * delta)

    def _stability_weight(self, delta_heading_deg, delta_altitude_m, delta_speed_mps):
        h_need = abs(delta_altitude_m) / max(self.alt_maneuver_scale_m, 1e-6)
        v_need = abs(delta_speed_mps) / max(self.speed_maneuver_scale_mps, 1e-6)
        psi_need = abs(delta_heading_deg) / max(self.heading_scale_deg, 1e-6)
        maneuver_demand = max(h_need, v_need, psi_need)
        return float(1.0 / (1.0 + maneuver_demand ** 2))

    def get_reward(self, task, env, agent_id):
        zero_items = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if not self.enable or getattr(task, "action_type", "discrete") != "continuous":
            return self._process(0.0, agent_id, zero_items)

        sim = env.agents[agent_id]
        delta_altitude_m = float(sim.get_property_value(c.delta_altitude))
        delta_heading_deg = float(sim.get_property_value(c.delta_heading))
        delta_speed_mps = float(sim.get_property_value(c.delta_velocities_u))
        vertical_speed_mps = float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048
        q_rad_sec = float(sim.get_property_value(c.velocities_q_rad_sec))
        elevator_cmd = float(sim.get_property_value(c.fcs_elevator_cmd_norm))

        vs_ref_mps = float(np.clip(self.k_h * delta_altitude_m, -self.vs_max_mps, self.vs_max_mps))
        prev_elevator = self._prev_elevator.get(agent_id, elevator_cmd)
        elevator_delta = elevator_cmd - prev_elevator
        self._prev_elevator[agent_id] = elevator_cmd

        alt_cost = self._huber(delta_altitude_m / max(self.alt_scale_m, 1e-6))
        speed_cost = self._huber(delta_speed_mps / max(self.speed_scale_mps, 1e-6))
        vs_cost = self._huber((vertical_speed_mps - vs_ref_mps) / max(self.vs_scale_mps, 1e-6))
        q_cost = self._huber(q_rad_sec / max(self.q_scale_rad_sec, 1e-6))
        sat_excess = max(abs(elevator_cmd) - self.sat_start, 0.0)
        sat_cost = self._huber(sat_excess / max(self.sat_scale, 1e-6))
        de_cost = self._huber(elevator_delta / max(self.de_scale, 1e-6))

        stability_weight = self._stability_weight(delta_heading_deg, delta_altitude_m, delta_speed_mps)
        tracking_cost = self.w_alt * alt_cost + self.w_speed * speed_cost + self.w_vs * vs_cost
        stability_cost = stability_weight * (self.w_q * q_cost + self.w_sat * sat_cost + self.w_de * de_cost)
        total_cost = tracking_cost + stability_cost

        return self._process(
            -total_cost,
            agent_id,
            (
                float(alt_cost),
                float(speed_cost),
                float(vs_cost),
                float(q_cost),
                float(sat_cost),
                float(de_cost),
                float(stability_weight),
                float(vs_ref_mps),
                float(elevator_delta),
            ),
        )

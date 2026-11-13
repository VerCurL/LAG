import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class TrimBalanceReward(BaseRewardFunction):
    """
    Encourage the continuous heading policy to find a stable elevator trim
    center by penalizing rapid changes of the elevator fast component.

    The reward is only active inside a conservative quasi-level-flight gate.
    It does not penalize non-zero elevator, AOA, or pitch directly.
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = bool(getattr(config, f"{cls}_enable", False))
        self.alpha = float(getattr(config, f"{cls}_alpha", 0.02))
        self.fast_delta_scale = float(getattr(config, f"{cls}_fast_delta_scale", 0.10))
        self.huber_delta = float(getattr(config, f"{cls}_huber_delta", 1.5))
        self.min_target_age_sec = float(getattr(config, f"{cls}_min_target_age_sec", 5.0))
        self.heading_error_gate_deg = float(getattr(config, f"{cls}_heading_error_gate_deg", 8.0))
        self.altitude_error_gate_m = float(getattr(config, f"{cls}_altitude_error_gate_m", 40.0))
        self.speed_error_gate_mps = float(getattr(config, f"{cls}_speed_error_gate_mps", 20.0))
        self.roll_gate_deg = float(getattr(config, f"{cls}_roll_gate_deg", 25.0))
        self.pitch_rate_gate_deg_s = float(getattr(config, f"{cls}_pitch_rate_gate_deg_s", 12.0))
        self.vertical_speed_gate_mps = float(getattr(config, f"{cls}_vertical_speed_gate_mps", 4.0))
        self.tracking_relief_enable = bool(getattr(config, f"{cls}_tracking_relief_enable", True))
        self.tracking_relief_min_multiplier = float(
            getattr(config, f"{cls}_tracking_relief_min_multiplier", 0.25)
        )
        self.tracking_relief_error_scale = float(getattr(config, f"{cls}_tracking_relief_error_scale", 4.0))
        self.heading_error_scale_deg = float(getattr(config, f"{cls}_heading_error_scale_deg", 5.0))
        self.altitude_error_scale_m = float(getattr(config, f"{cls}_altitude_error_scale_m", 15.24))
        self.speed_error_scale_mps = float(getattr(config, f"{cls}_speed_error_scale_mps", 24.0))

        self._state = {}
        self.reward_item_names = [
            self.__class__.__name__,
            self.__class__.__name__ + "_gate",
            self.__class__.__name__ + "_elevator_trim_est",
            self.__class__.__name__ + "_fast_part",
            self.__class__.__name__ + "_fast_delta",
            self.__class__.__name__ + "_huber_cost",
            self.__class__.__name__ + "_multiplier",
        ]

    def reset(self, task, env):
        self._state = {}
        for agent_id, sim in env.agents.items():
            target = self._get_target(sim)
            self._state[agent_id] = {
                "elevator_trim_est": 0.0,
                "prev_fast_part": 0.0,
                "initialized": False,
                "prev_fast_initialized": False,
                "last_target": target,
                "last_target_change_time": float(sim.get_property_value(c.simulation_sim_time_sec)),
            }
        return super().reset(task, env)

    @staticmethod
    def _get_target(sim):
        return (
            float(sim.get_property_value(c.target_heading_deg)),
            float(sim.get_property_value(c.target_altitude_ft)),
            float(sim.get_property_value(c.target_velocities_u_mps)),
        )

    def _update_target_age(self, sim, agent_id):
        state = self._state[agent_id]
        target = self._get_target(sim)
        if not np.allclose(np.asarray(target), np.asarray(state["last_target"]), atol=1e-6):
            state["last_target"] = target
            state["last_target_change_time"] = float(sim.get_property_value(c.simulation_sim_time_sec))
            state["prev_fast_initialized"] = False
        return float(sim.get_property_value(c.simulation_sim_time_sec)) - state["last_target_change_time"]

    def _is_gate_on(self, sim, target_age_sec):
        vertical_speed_mps = float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048
        pitch_rate_deg_s = float(np.rad2deg(sim.get_property_value(c.velocities_q_rad_sec)))
        return bool(
            target_age_sec >= self.min_target_age_sec
            and abs(float(sim.get_property_value(c.delta_heading))) <= self.heading_error_gate_deg
            and abs(float(sim.get_property_value(c.delta_altitude))) <= self.altitude_error_gate_m
            and abs(float(sim.get_property_value(c.delta_velocities_u))) <= self.speed_error_gate_mps
            and abs(float(sim.get_property_value(c.attitude_phi_deg))) <= self.roll_gate_deg
            and abs(pitch_rate_deg_s) <= self.pitch_rate_gate_deg_s
            and abs(vertical_speed_mps) <= self.vertical_speed_gate_mps
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
        relief_span = max(self.tracking_relief_error_scale, 1e-6)
        relief = float(np.clip((tracking_error - 1.0) / relief_span, 0.0, 1.0))
        min_multiplier = float(np.clip(self.tracking_relief_min_multiplier, 0.0, 1.0))
        return 1.0 - relief * (1.0 - min_multiplier)

    def _huber(self, value):
        delta = max(self.huber_delta, 1e-6)
        abs_value = abs(float(value))
        if abs_value <= delta:
            return 0.5 * abs_value ** 2
        return delta * (abs_value - 0.5 * delta)

    def get_reward(self, task, env, agent_id):
        zero_items = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if not self.enable or getattr(task, "action_type", "discrete") != "continuous":
            return self._process(0.0, agent_id, zero_items)

        sim = env.agents[agent_id]
        if agent_id not in self._state:
            self.reset(task, env)
        state = self._state[agent_id]
        target_age_sec = self._update_target_age(sim, agent_id)
        gate_on = self._is_gate_on(sim, target_age_sec)
        elevator_cmd = float(sim.get_property_value(c.fcs_elevator_cmd_norm))

        if not gate_on:
            state["prev_fast_initialized"] = False
            return self._process(
                0.0,
                agent_id,
                (0.0, state["elevator_trim_est"], 0.0, 0.0, 0.0, 0.0),
            )

        alpha = float(np.clip(self.alpha, 0.0, 1.0))
        if not state["initialized"]:
            state["elevator_trim_est"] = elevator_cmd
            state["initialized"] = True
        else:
            state["elevator_trim_est"] = (
                (1.0 - alpha) * state["elevator_trim_est"]
                + alpha * elevator_cmd
            )

        fast_part = elevator_cmd - state["elevator_trim_est"]
        if not state["prev_fast_initialized"]:
            fast_delta = 0.0
            state["prev_fast_initialized"] = True
        else:
            fast_delta = fast_part - state["prev_fast_part"]
        state["prev_fast_part"] = fast_part

        normalized_delta = fast_delta / max(self.fast_delta_scale, 1e-6)
        huber_cost = self._huber(normalized_delta)
        multiplier = self._tracking_relief_multiplier(sim)
        total_cost = huber_cost * multiplier
        return self._process(
            -total_cost,
            agent_id,
            (
                1.0,
                float(state["elevator_trim_est"]),
                float(fast_part),
                float(fast_delta),
                float(huber_cost),
                float(multiplier),
            ),
        )

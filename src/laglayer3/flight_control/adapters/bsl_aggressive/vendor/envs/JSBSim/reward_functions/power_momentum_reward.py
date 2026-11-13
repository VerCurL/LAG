import math

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class PowerMomentumReward(BaseRewardFunction):
    """
    Lightweight physical reward for heading control.

    Components:
        r_energy: total specific mechanical energy close to target
        r_split: altitude/speed energy allocation balance
        r_momentum: horizontal velocity aligned with target heading
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = bool(getattr(config, f"{cls}_enable", False))
        self.g = float(getattr(config, f"{cls}_gravity", 9.80665))
        self.energy_height_scale_m = float(getattr(config, f"{cls}_energy_height_scale_m", 300.0))
        self.split_sigma = float(getattr(config, f"{cls}_split_sigma", 0.6))
        self.split_eps = float(getattr(config, f"{cls}_split_eps", 0.05))
        self.velocity_gate_mps = float(getattr(config, f"{cls}_velocity_gate_mps", 40.0))
        self.velocity_eps = float(getattr(config, f"{cls}_velocity_eps", 1e-6))
        self.energy_weight = float(getattr(config, f"{cls}_energy_weight", 0.05))
        self.split_weight = float(getattr(config, f"{cls}_split_weight", 0.05))
        self.momentum_weight = float(getattr(config, f"{cls}_momentum_weight", 0.08))
        self.reward_item_names = [
            self.__class__.__name__ + item
            for item in [
                "",
                "_energy",
                "_split",
                "_momentum",
                "_energy_error_norm",
                "_split_error",
                "_momentum_alignment",
                "_momentum_gate",
            ]
        ]

    @staticmethod
    def _pseudo_huber(x):
        return math.sqrt(1.0 + x * x) - 1.0

    def get_reward(self, task, env, agent_id):
        if not self.enable:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        sim = env.agents[agent_id]
        h = float(sim.get_property_value(c.position_h_sl_m))
        h_target = float(sim.get_property_value(c.target_altitude_ft)) * 0.3048
        if hasattr(task, "get_current_speed_mps"):
            v = float(task.get_current_speed_mps(sim))
        else:
            v = float(sim.get_property_value(c.velocities_u_mps))
        if hasattr(task, "get_target_speed_mps"):
            v_target = float(task.get_target_speed_mps(sim))
        else:
            v_target = float(
                sim.get_property_value(c.target_velocities_u_mps)
            )

        height_scale = max(self.energy_height_scale_m, 1e-6)
        sigma_energy = max(self.g * height_scale, 1e-6)

        energy_error = self.g * (h - h_target) + 0.5 * (v * v - v_target * v_target)
        energy_error_norm = energy_error / sigma_energy
        r_energy = math.exp(-self._pseudo_huber(energy_error_norm))

        z_h = (h - h_target) / height_scale
        z_v = (v * v - v_target * v_target) / (2.0 * self.g * height_scale)
        split_error = (z_h - z_v) / (abs(z_h) + abs(z_v) + max(self.split_eps, 1e-9))
        r_split = math.exp(-self._pseudo_huber(split_error / max(self.split_sigma, 1e-6)))

        v_n = float(sim.get_property_value(c.velocities_v_north_mps))
        v_e = float(sim.get_property_value(c.velocities_v_east_mps))
        v_h = math.sqrt(v_n * v_n + v_e * v_e)
        target_heading = math.radians(float(sim.get_property_value(c.target_heading_deg)))
        target_n = math.cos(target_heading)
        target_e = math.sin(target_heading)
        alignment = (v_n * target_n + v_e * target_e) / (v_h + max(self.velocity_eps, 1e-9))
        alignment = max(-1.0, min(1.0, alignment))
        momentum_gate = v_h / (v_h + max(self.velocity_gate_mps, 1e-6))
        r_momentum = 0.5 + 0.5 * momentum_gate * alignment

        reward = (
            self.energy_weight * r_energy
            + self.split_weight * r_split
            + self.momentum_weight * r_momentum
        )
        return self._process(
            reward,
            agent_id,
            (
                r_energy,
                r_split,
                r_momentum,
                energy_error_norm,
                split_error,
                alignment,
                momentum_gate,
            ),
        )

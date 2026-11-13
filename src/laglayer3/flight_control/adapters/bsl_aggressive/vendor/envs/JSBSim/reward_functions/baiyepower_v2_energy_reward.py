"""Isolated BaiyePower v2 total and allocation energy reward."""

import math

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
from ..utils.baiyepower_v2_physics import compute_energy_cost_terms


class BaiyePowerV2EnergyReward(BaseRewardFunction):
    """Post-action ``C_T/C_h/C_k`` costs against one frozen target."""

    def __init__(self, config):
        super().__init__(config)
        self.gravity_mps2 = self._positive_config(
            config,
            "BaiyePowerV2_gravity_mps2",
            9.80665,
        )
        self.height_cost_scale_m = self._positive_config(
            config,
            "BaiyePowerV2_height_cost_scale_m",
            90.0,
        )
        self.speed_cost_scale_mps = self._positive_config(
            config,
            "BaiyePowerV2_speed_cost_scale_mps",
            20.0,
        )
        self.total_rate_per_sec = self._nonnegative_config(
            config,
            "BaiyePowerV2_total_reward_rate_per_sec",
            0.20,
        )
        self.height_rate_per_sec = self._nonnegative_config(
            config,
            "BaiyePowerV2_height_shortfall_reward_rate_per_sec",
            0.075,
        )
        self.kinetic_rate_per_sec = self._nonnegative_config(
            config,
            "BaiyePowerV2_kinetic_shortfall_reward_rate_per_sec",
            0.075,
        )
        self.enable_total = bool(
            getattr(config, "BaiyePowerV2_enable_total", True)
        )
        self.enable_height = bool(
            getattr(
                config,
                "BaiyePowerV2_enable_height_shortfall",
                True,
            )
        )
        self.enable_kinetic = bool(
            getattr(
                config,
                "BaiyePowerV2_enable_kinetic_shortfall",
                True,
            )
        )
        self._validate_base_semantics()
        self.reward_item_names = [
            "BaiyePowerV2EnergyReward",
            "BaiyePowerV2_C_T",
            "BaiyePowerV2_C_h",
            "BaiyePowerV2_C_k",
            "BaiyePowerV2_e_T_m",
            "BaiyePowerV2_e_h_m",
            "BaiyePowerV2_e_k_m",
            "BaiyePowerV2_total_reward_nominal",
            "BaiyePowerV2_height_reward_nominal",
            "BaiyePowerV2_kinetic_reward_nominal",
            "BaiyePowerV2_energy_reward_candidate",
            "BaiyePowerV2_target_lifecycle_valid",
        ]

    @staticmethod
    def _finite_config(config, name, default):
        value = float(getattr(config, name, default))
        if not math.isfinite(value):
            raise ValueError("{} must be finite".format(name))
        return value

    @classmethod
    def _positive_config(cls, config, name, default):
        value = cls._finite_config(config, name, default)
        if value <= 0.0:
            raise ValueError("{} must be positive".format(name))
        return value

    @classmethod
    def _nonnegative_config(cls, config, name, default):
        value = cls._finite_config(config, name, default)
        if value < 0.0:
            raise ValueError("{} cannot be negative".format(name))
        return value

    def _validate_base_semantics(self):
        if self.is_potential:
            raise ValueError(
                "BaiyePower v2 rewards cannot use BaseReward potential mode"
            )
        if not math.isfinite(float(self.reward_scale)):
            raise ValueError("BaiyePower v2 reward scale must be finite")
        if float(self.reward_scale) != 1.0:
            raise ValueError(
                "BaiyePower v2 reward scale must remain 1; use rate configs"
            )

    @staticmethod
    def _decision_dt(context):
        decision_dt_sec = float(context.decision_dt_sec)
        if not math.isfinite(decision_dt_sec) or decision_dt_sec <= 0.0:
            raise ValueError("decision_dt_sec must be finite and positive")
        return decision_dt_sec

    @staticmethod
    def _mark_failure(task, agent_id, error):
        task.mark_baiyepower_v2_numerical_failure(
            agent_id,
            "BaiyePowerV2EnergyReward",
            error,
        )

    def _zero_after_failure(self, task, agent_id, error):
        self._mark_failure(task, agent_id, error)
        return self._process(
            0.0,
            agent_id,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

    def get_reward(self, task, env, agent_id):
        try:
            context = task.get_baiyepower_v2_transition_context(agent_id)
            decision_dt_sec = self._decision_dt(context)
            lifecycle_valid = bool(context.target_lifecycle_valid)
            sim = env.agents[agent_id]
            post = compute_energy_cost_terms(
                altitude_m=sim.get_property_value(c.position_h_sl_m),
                target_altitude_m=context.target_altitude_m,
                u_mps=sim.get_property_value(c.velocities_u_mps),
                v_mps=sim.get_property_value(c.velocities_v_mps),
                w_mps=sim.get_property_value(c.velocities_w_mps),
                target_speed_mps=context.target_speed_mps,
                height_cost_scale_m=self.height_cost_scale_m,
                speed_cost_scale_mps=self.speed_cost_scale_mps,
                gravity_mps2=self.gravity_mps2,
            )
            pre = context.pre_energy_terms

            total_reward_nominal = (
                -self.total_rate_per_sec * decision_dt_sec * post.C_T
            )
            height_reward_nominal = (
                -self.height_rate_per_sec * decision_dt_sec * post.C_h
            )
            kinetic_reward_nominal = (
                -self.kinetic_rate_per_sec * decision_dt_sec * post.C_k
            )
            total_reward_candidate = (
                total_reward_nominal if self.enable_total else 0.0
            )
            height_reward_candidate = (
                height_reward_nominal if self.enable_height else 0.0
            )
            kinetic_reward_candidate = (
                kinetic_reward_nominal if self.enable_kinetic else 0.0
            )
            reward_candidate = (
                total_reward_candidate
                + height_reward_candidate
                + kinetic_reward_candidate
            )
            reward = reward_candidate if lifecycle_valid else 0.0
            energy_height_delta_m = (
                post.energy_height_m - pre.energy_height_m
            )
            if not all(
                math.isfinite(value)
                for value in (
                    total_reward_nominal,
                    height_reward_nominal,
                    kinetic_reward_nominal,
                    reward_candidate,
                    reward,
                    energy_height_delta_m,
                )
            ):
                raise ValueError("energy reward contribution is non-finite")

            task.record_baiyepower_v2_diag(
                agent_id,
                {
                    "altitude_m": post.altitude_m,
                    "target_altitude_m": post.target_altitude_m,
                    "speed_mps": post.speed_mps,
                    "target_speed_mps": post.target_speed_mps,
                    "delta_speed_mps": post.delta_speed_mps,
                    "kinetic_height_m": post.kinetic_height_m,
                    "target_kinetic_height_m": (
                        post.target_kinetic_height_m
                    ),
                    "energy_height_m": post.energy_height_m,
                    "target_energy_height_m": (
                        post.target_energy_height_m
                    ),
                    "e_h_m": post.e_h_m,
                    "e_k_m": post.e_k_m,
                    "e_T_m": post.e_T_m,
                    "e_B_m": post.e_B_m,
                    "S_h_m": post.S_h_m,
                    "S_k_m": post.S_k_m,
                    "S_T_m": post.S_T_m,
                    "z_T": post.z_T,
                    "z_h_shortage": post.z_h_shortage,
                    "z_k_shortage": post.z_k_shortage,
                    "C_T": post.C_T,
                    "C_h": post.C_h,
                    "C_k": post.C_k,
                    "energy_cost_pre": pre.C_T,
                    "energy_cost_post_exec": post.C_T,
                    "energy_progress_exec": pre.C_T - post.C_T,
                    "energy_height_delta_m": energy_height_delta_m,
                    "energy_height_rate_mps": (
                        energy_height_delta_m / decision_dt_sec
                    ),
                    "energy_error_delta_m": post.e_T_m - pre.e_T_m,
                    "energy_valid": True,
                    "total_enabled": self.enable_total,
                    "height_shortfall_enabled": self.enable_height,
                    "kinetic_shortfall_enabled": self.enable_kinetic,
                    "total_reward_nominal": total_reward_nominal,
                    "height_reward_nominal": height_reward_nominal,
                    "kinetic_reward_nominal": kinetic_reward_nominal,
                    "total_reward_candidate": total_reward_candidate,
                    "height_reward_candidate": height_reward_candidate,
                    "kinetic_reward_candidate": kinetic_reward_candidate,
                    "energy_reward_candidate": reward_candidate,
                    "energy_reward_contribution": reward,
                    "target_lifecycle_valid": lifecycle_valid,
                },
            )
            return self._process(
                reward,
                agent_id,
                (
                    post.C_T,
                    post.C_h,
                    post.C_k,
                    post.e_T_m,
                    post.e_h_m,
                    post.e_k_m,
                    total_reward_nominal,
                    height_reward_nominal,
                    kinetic_reward_nominal,
                    reward_candidate,
                    float(lifecycle_valid),
                ),
            )
        except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
            return self._zero_after_failure(task, agent_id, error)

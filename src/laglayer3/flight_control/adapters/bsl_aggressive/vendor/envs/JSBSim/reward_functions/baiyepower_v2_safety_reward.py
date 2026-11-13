"""Isolated BaiyePower v2 low-altitude soft-warning reward."""

import math

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
from ..utils.baiyepower_v2_physics import (
    compute_low_altitude_warning_terms,
)


class BaiyePowerV2SafetyReward(BaseRewardFunction):
    """Bounded low-altitude warning; hard termination remains separate."""

    def __init__(self, config):
        super().__init__(config)
        self.warning_altitude_m = self._finite_config(
            config,
            "BaiyePowerV2_low_altitude_warning_m",
            4000.0,
        )
        self.terminal_altitude_m = self._finite_config(
            config,
            "BaiyePowerV2_low_altitude_terminal_m",
            2500.0,
        )
        if self.warning_altitude_m <= self.terminal_altitude_m:
            raise ValueError(
                "BaiyePowerV2 warning altitude must exceed terminal altitude"
            )
        self.sink_speed_scale_mps = self._positive_config(
            config,
            "BaiyePowerV2_sink_speed_scale_mps",
            20.0,
        )
        self.sink_eta = self._bounded_config(
            config,
            "BaiyePowerV2_sink_eta",
            0.5,
            0.0,
            1.0,
        )
        self.safety_rate_per_sec = self._nonnegative_config(
            config,
            "BaiyePowerV2_safety_reward_rate_per_sec",
            0.10,
        )
        self.enable_safety = bool(
            getattr(config, "BaiyePowerV2_enable_safety", True)
        )
        self._validate_base_semantics()
        self.reward_item_names = [
            "BaiyePowerV2SafetyReward",
            "BaiyePowerV2_safety_cost",
            "BaiyePowerV2_altitude_barrier",
            "BaiyePowerV2_sink_speed_mps",
            "BaiyePowerV2_sink_factor",
            "BaiyePowerV2_safety_reward_nominal",
            "BaiyePowerV2_safety_reward_candidate",
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

    @classmethod
    def _bounded_config(cls, config, name, default, lower, upper):
        value = cls._finite_config(config, name, default)
        if not lower <= value <= upper:
            raise ValueError(
                "{} must lie in [{}, {}]".format(name, lower, upper)
            )
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
            "BaiyePowerV2SafetyReward",
            error,
        )

    def _zero_after_failure(self, task, agent_id, error):
        self._mark_failure(task, agent_id, error)
        return self._process(
            0.0,
            agent_id,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

    def get_reward(self, task, env, agent_id):
        try:
            context = task.get_baiyepower_v2_transition_context(agent_id)
            decision_dt_sec = self._decision_dt(context)
            lifecycle_valid = bool(context.target_lifecycle_valid)
            sim = env.agents[agent_id]
            safety = compute_low_altitude_warning_terms(
                altitude_m=sim.get_property_value(c.position_h_sl_m),
                altitude_rate_mps=(
                    sim.get_property_value(c.velocities_h_dot_fps) * 0.3048
                ),
                warning_altitude_m=self.warning_altitude_m,
                terminal_altitude_m=self.terminal_altitude_m,
                sink_speed_scale_mps=self.sink_speed_scale_mps,
                sink_eta=self.sink_eta,
            )
            safety_reward_nominal = (
                -self.safety_rate_per_sec
                * decision_dt_sec
                * safety.cost
            )
            safety_reward_candidate = (
                safety_reward_nominal if self.enable_safety else 0.0
            )
            reward = (
                safety_reward_candidate if lifecycle_valid else 0.0
            )
            if not all(
                math.isfinite(value)
                for value in (
                    safety_reward_nominal,
                    safety_reward_candidate,
                    reward,
                )
            ):
                raise ValueError("safety reward contribution is non-finite")

            task.record_baiyepower_v2_diag(
                agent_id,
                {
                    "safety_altitude_m": safety.altitude_m,
                    "safety_altitude_rate_mps": safety.altitude_rate_mps,
                    "safety_normalized_depth": safety.normalized_depth,
                    "safety_altitude_barrier": safety.altitude_barrier,
                    "safety_sink_speed_mps": safety.sink_speed_mps,
                    "safety_sink_factor": safety.sink_factor,
                    "safety_cost": safety.cost,
                    "safety_enabled": self.enable_safety,
                    "safety_reward_nominal": safety_reward_nominal,
                    "safety_reward_candidate": safety_reward_candidate,
                    "safety_reward_contribution": reward,
                    "target_lifecycle_valid": lifecycle_valid,
                },
            )
            return self._process(
                reward,
                agent_id,
                (
                    safety.cost,
                    safety.altitude_barrier,
                    safety.sink_speed_mps,
                    safety.sink_factor,
                    safety_reward_nominal,
                    safety_reward_candidate,
                    float(lifecycle_valid),
                ),
            )
        except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
            return self._zero_after_failure(task, agent_id, error)

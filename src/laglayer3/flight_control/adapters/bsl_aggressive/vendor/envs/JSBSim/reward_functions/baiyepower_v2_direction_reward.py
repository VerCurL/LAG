"""Isolated BaiyePower v2 yaw and course reward."""

import math

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
from ..utils.baiyepower_v2_physics import (
    compute_course_cost_terms,
    compute_upright_envelope_terms,
    compute_yaw_kernel_terms,
)


class BaiyePowerV2DirectionReward(BaseRewardFunction):
    """Primary yaw reward plus a small actual-course auxiliary.

    The target is read only from the task's immutable transition context.
    Current yaw and inertial horizontal velocity are post-action simulator
    values, so the contribution describes the result of the executed action.
    """

    def __init__(self, config):
        super().__init__(config)
        self.yaw_rate_per_sec = self._nonnegative_config(
            config,
            "BaiyePowerV2_yaw_reward_rate_per_sec",
            2.10,
        )
        self.course_rate_per_sec = self._nonnegative_config(
            config,
            "BaiyePowerV2_course_reward_rate_per_sec",
            0.10,
        )
        self.yaw_gaussian_scale_deg = self._positive_config(
            config,
            "BaiyePowerV2_yaw_gaussian_scale_deg",
            20.0,
        )
        self.yaw_narrow_weight = self._bounded_config(
            config,
            "BaiyePowerV2_yaw_narrow_weight",
            0.55,
            0.0,
            1.0,
        )
        self.course_half_cost_angle_deg = self._positive_config(
            config,
            "BaiyePowerV2_course_half_cost_angle_deg",
            15.0,
        )
        if self.course_half_cost_angle_deg >= 180.0:
            raise ValueError(
                "BaiyePowerV2_course_half_cost_angle_deg must be < 180"
            )
        self.course_reliability_mode = str(
            getattr(
                config,
                "BaiyePowerV2_course_reliability_mode",
                "hard",
            )
        ).strip().lower()
        if self.course_reliability_mode not in ("hard", "smoothstep"):
            raise ValueError(
                "BaiyePowerV2_course_reliability_mode must be hard or "
                "smoothstep"
            )
        self.course_min_speed_mps = self._positive_config(
            config,
            "BaiyePowerV2_course_min_speed_mps",
            40.0,
        )
        self.course_fade_start_mps = self._nonnegative_config(
            config,
            "BaiyePowerV2_course_fade_start_mps",
            30.0,
        )
        self.course_fade_end_mps = self._positive_config(
            config,
            "BaiyePowerV2_course_fade_end_mps",
            50.0,
        )
        if self.course_fade_end_mps <= self.course_fade_start_mps:
            raise ValueError(
                "BaiyePowerV2_course_fade_end_mps must exceed fade start"
            )
        self.enable_course = bool(
            getattr(config, "BaiyePowerV2_enable_course", True)
        )
        self.upright_yaw_gate_enable = bool(
            getattr(config, "BaiyePowerUpright_yaw_gate_enable", False)
        )
        self.upright_free_tilt_deg = self._positive_config(
            config,
            "BaiyePowerUpright_free_tilt_deg",
            90.0,
        )
        if self.upright_free_tilt_deg > 90.0:
            raise ValueError(
                "BaiyePowerUpright_free_tilt_deg must be <= 90"
            )
        self._validate_base_semantics()
        self.reward_item_names = [
            "BaiyePowerV2DirectionReward",
            "BaiyePowerV2_yaw_kernel",
            "BaiyePowerV2_yaw_error_deg",
            "BaiyePowerV2_course_cost",
            "BaiyePowerV2_course_error_deg",
            "BaiyePowerV2_course_reliability",
            "BaiyePowerV2_yaw_reward_nominal",
            "BaiyePowerV2_yaw_reward_ungated",
            "BaiyePowerV2_yaw_upright_gate",
            "BaiyePowerV2_yaw_upright_gate_loss",
            "BaiyePowerV2_course_reward_nominal",
            "BaiyePowerV2_direction_reward_candidate",
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
            "BaiyePowerV2DirectionReward",
            error,
        )

    def _zero_after_failure(self, task, agent_id, error):
        self._mark_failure(task, agent_id, error)
        return self._process(
            0.0,
            agent_id,
            (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        )

    def get_reward(self, task, env, agent_id):
        try:
            context = task.get_baiyepower_v2_transition_context(agent_id)
            decision_dt_sec = self._decision_dt(context)
            lifecycle_valid = bool(context.target_lifecycle_valid)
            sim = env.agents[agent_id]

            yaw_terms = compute_yaw_kernel_terms(
                current_yaw_rad=sim.get_property_value(c.attitude_psi_rad),
                target_yaw_rad=context.target_heading_rad,
                gaussian_scale_deg=self.yaw_gaussian_scale_deg,
                narrow_weight=self.yaw_narrow_weight,
            )
            course_terms = compute_course_cost_terms(
                north_velocity_mps=sim.get_property_value(
                    c.velocities_v_north_mps
                ),
                east_velocity_mps=sim.get_property_value(
                    c.velocities_v_east_mps
                ),
                target_course_rad=context.target_heading_rad,
                half_cost_angle_deg=self.course_half_cost_angle_deg,
                reliability_mode=self.course_reliability_mode,
                minimum_speed_mps=self.course_min_speed_mps,
                fade_start_mps=self.course_fade_start_mps,
                fade_end_mps=self.course_fade_end_mps,
            )

            if self.upright_yaw_gate_enable:
                upright = compute_upright_envelope_terms(
                    roll_rad=sim.get_property_value(c.attitude_roll_rad),
                    pitch_rad=sim.get_property_value(c.attitude_pitch_rad),
                    free_tilt_deg=self.upright_free_tilt_deg,
                )
                yaw_upright_gate = upright.yaw_gate
            else:
                yaw_upright_gate = 1.0
            yaw_reward_ungated = (
                self.yaw_rate_per_sec
                * decision_dt_sec
                * yaw_terms.kernel
            )
            yaw_reward_nominal = (
                yaw_reward_ungated * yaw_upright_gate
            )
            yaw_upright_gate_loss = (
                yaw_reward_ungated - yaw_reward_nominal
            )
            course_reward_nominal = (
                -self.course_rate_per_sec
                * decision_dt_sec
                * course_terms.cost
            )
            course_reward_candidate = (
                course_reward_nominal if self.enable_course else 0.0
            )
            reward_candidate = (
                yaw_reward_nominal + course_reward_candidate
            )
            reward = reward_candidate if lifecycle_valid else 0.0
            if not all(
                math.isfinite(value)
                for value in (
                    yaw_reward_nominal,
                    yaw_reward_ungated,
                    yaw_upright_gate,
                    yaw_upright_gate_loss,
                    course_reward_nominal,
                    reward_candidate,
                    reward,
                )
            ):
                raise ValueError("direction reward contribution is non-finite")

            yaw_alive_baseline = self.yaw_rate_per_sec * decision_dt_sec
            task.record_baiyepower_v2_diag(
                agent_id,
                {
                    "target_heading_rad": float(context.target_heading_rad),
                    "yaw_rad": yaw_terms.current_yaw_rad,
                    "yaw_error_rad": yaw_terms.error_rad,
                    "yaw_error_deg": math.degrees(yaw_terms.error_rad),
                    "yaw_kernel": yaw_terms.kernel,
                    "yaw_narrow_component": yaw_terms.narrow_component,
                    "yaw_wide_component": yaw_terms.wide_component,
                    "yaw_alive_baseline": yaw_alive_baseline,
                    "yaw_tracking_regret": (
                        yaw_alive_baseline * (1.0 - yaw_terms.kernel)
                    ),
                    "yaw_reward_nominal": yaw_reward_nominal,
                    "yaw_reward_ungated": yaw_reward_ungated,
                    "yaw_upright_gate": yaw_upright_gate,
                    "yaw_upright_gate_loss": yaw_upright_gate_loss,
                    "yaw_upright_gate_enabled": (
                        self.upright_yaw_gate_enable
                    ),
                    "course_rad": course_terms.course_rad,
                    "target_course_rad": course_terms.target_course_rad,
                    "course_error_rad": course_terms.error_rad,
                    "course_error_deg": math.degrees(course_terms.error_rad),
                    "course_horizontal_speed_mps": (
                        course_terms.horizontal_speed_mps
                    ),
                    "course_raw_cost": course_terms.raw_cost,
                    "course_reliability": course_terms.reliability,
                    "course_cost": course_terms.cost,
                    "course_valid": course_terms.valid,
                    "course_enabled": self.enable_course,
                    "course_reward_nominal": course_reward_nominal,
                    "course_reward_candidate": course_reward_candidate,
                    "direction_reward_candidate": reward_candidate,
                    "direction_reward_contribution": reward,
                    "target_lifecycle_valid": lifecycle_valid,
                },
            )
            return self._process(
                reward,
                agent_id,
                (
                    yaw_terms.kernel,
                    math.degrees(yaw_terms.error_rad),
                    course_terms.cost,
                    math.degrees(course_terms.error_rad),
                    course_terms.reliability,
                    yaw_reward_nominal,
                    yaw_reward_ungated,
                    yaw_upright_gate,
                    yaw_upright_gate_loss,
                    course_reward_nominal,
                    reward_candidate,
                    float(lifecycle_valid),
                ),
            )
        except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
            return self._zero_after_failure(task, agent_id, error)

"""Isolated BaiyePower U1 upright-envelope reward."""

import math

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
from ..utils.baiyepower_v2_physics import (
    compute_upright_envelope_terms,
    speed_magnitude_mps,
)


class BaiyePowerUprightReward(BaseRewardFunction):
    """Upright-envelope diagnostics with an optional later-stage soft cost.

    This is deliberately not a roll-to-zero reward.  The free envelope allows
    the complete upright hemisphere, while the body-up cosine makes the
    calculation valid for combined roll and pitch.  Frozen U1 sets the
    additive rate to zero and uses only the positive-yaw gate implemented by
    the direction reward.  There is no inverted-state termination.
    """

    def __init__(self, config):
        super().__init__(config)
        self.free_tilt_deg = self._finite_config(
            config,
            "BaiyePowerUpright_free_tilt_deg",
            90.0,
        )
        if not 0.0 < self.free_tilt_deg <= 90.0:
            raise ValueError(
                "BaiyePowerUpright_free_tilt_deg must lie in (0, 90]"
            )
        self.reward_rate_per_sec = self._nonnegative_config(
            config,
            "BaiyePowerUpright_reward_rate_per_sec",
            0.0,
        )
        self.enable_upright = bool(
            getattr(config, "BaiyePowerUpright_enable", True)
        )
        self._validate_base_semantics()
        self.reward_item_names = [
            "BaiyePowerUprightReward",
            "BaiyePowerUpright_upright_cos",
            "BaiyePowerUpright_tilt_angle_deg",
            "BaiyePowerUpright_normalized_depth",
            "BaiyePowerUpright_cost",
            "BaiyePowerUpright_yaw_gate",
            "BaiyePowerUpright_active",
            "BaiyePowerUpright_inverted",
            "BaiyePowerUpright_deep_inverted",
            "BaiyePowerUpright_reward_nominal",
            "BaiyePowerUpright_enabled",
        ]

    @staticmethod
    def _finite_config(config, name, default):
        value = float(getattr(config, name, default))
        if not math.isfinite(value):
            raise ValueError("{} must be finite".format(name))
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
                "BaiyePower upright reward cannot use potential mode"
            )
        if not math.isfinite(float(self.reward_scale)):
            raise ValueError("BaiyePower upright reward scale must be finite")
        if float(self.reward_scale) != 1.0:
            raise ValueError(
                "BaiyePower upright reward scale must remain 1; "
                "use BaiyePowerUpright_reward_rate_per_sec"
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
            "BaiyePowerUprightReward",
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
            ),
        )

    def get_reward(self, task, env, agent_id):
        try:
            context = task.get_baiyepower_v2_transition_context(agent_id)
            decision_dt_sec = self._decision_dt(context)
            sim = env.agents[agent_id]
            upright = compute_upright_envelope_terms(
                roll_rad=sim.get_property_value(c.attitude_roll_rad),
                pitch_rad=sim.get_property_value(c.attitude_pitch_rad),
                free_tilt_deg=self.free_tilt_deg,
            )
            body_u_mps = float(
                sim.get_property_value(c.velocities_u_mps)
            )
            body_speed_mps = speed_magnitude_mps(
                body_u_mps,
                sim.get_property_value(c.velocities_v_mps),
                sim.get_property_value(c.velocities_w_mps),
            )
            body_forward_valid = body_speed_mps > 1.0
            body_forward_ratio = (
                body_u_mps / body_speed_mps
                if body_forward_valid
                else 0.0
            )
            reward_nominal = (
                -self.reward_rate_per_sec
                * decision_dt_sec
                * upright.cost
            )
            reward = reward_nominal if self.enable_upright else 0.0
            if not math.isfinite(reward_nominal) or not math.isfinite(reward):
                raise ValueError("upright reward contribution is non-finite")

            # Unlike target-tracking rewards, attitude validity is independent
            # of target lifecycle.  Do not gate this contribution during a
            # point/arc transition.
            task.record_baiyepower_v2_diag(
                agent_id,
                {
                    "upright_cos": upright.upright_cos,
                    "upright_tilt_angle_deg": upright.tilt_angle_deg,
                    "upright_free_cos": upright.free_cos,
                    "upright_normalized_depth": upright.normalized_depth,
                    "upright_cost": upright.cost,
                    "upright_yaw_gate": upright.yaw_gate,
                    "upright_active": upright.active,
                    "upright_inverted": upright.inverted,
                    "upright_deep_inverted": upright.deep_inverted,
                    # Diagnostic only: U1 prevents inverted orientation but
                    # does not itself guarantee forward flight.  Keep this
                    # sidecar so a tail-slide/large-sideslip replacement
                    # strategy is visible without adding another reward.
                    "body_forward_ratio": body_forward_ratio,
                    "body_forward_valid": body_forward_valid,
                    "negative_body_forward": (
                        body_forward_valid and body_forward_ratio < 0.0
                    ),
                    "upright_reward_nominal": reward_nominal,
                    "upright_reward_contribution": reward,
                    "upright_enabled": self.enable_upright,
                },
            )
            return self._process(
                reward,
                agent_id,
                (
                    upright.upright_cos,
                    upright.tilt_angle_deg,
                    upright.normalized_depth,
                    upright.cost,
                    upright.yaw_gate,
                    float(upright.active),
                    float(upright.inverted),
                    float(upright.deep_inverted),
                    reward_nominal,
                    float(self.enable_upright),
                ),
            )
        except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
            return self._zero_after_failure(task, agent_id, error)

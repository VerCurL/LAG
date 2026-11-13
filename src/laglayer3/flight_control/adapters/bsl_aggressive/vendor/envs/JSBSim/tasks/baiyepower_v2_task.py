"""Isolated BaiyePower v2 task.

The v2 task deliberately reuses only the observation/action plumbing and route
generator of :class:`BaiyePowerTask`.  Its reward whitelist, transition target
snapshot, and termination arbitration are owned here so experiments cannot
silently re-enable the legacy reward stack.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .baiyepower_task import BaiyePowerTask
from ..core.catalog import Catalog as c
from ..reward_functions import (
    BaiyePowerV2DirectionReward,
    BaiyePowerV2EnergyReward,
    BaiyePowerV2SafetyReward,
)
from ..termination_conditions import (
    BaiyePowerV2FixedAttemptRoute,
    ExtremeState,
    LowAltitude,
    Overload,
    Timeout,
)
from ..utils.baiyepower_v2_physics import (
    BaiyePowerV2EnergyTerms,
    compute_energy_cost_terms,
)


@dataclass(frozen=True)
class BaiyePowerV2TransitionContext:
    """Immutable data defining which target generated one transition."""

    decision_step: int
    decision_dt_sec: float
    episode_id: int
    target_epoch: int
    route_attempt_id: Optional[int]
    target_lifecycle_valid: bool
    target_heading_rad: float
    target_altitude_m: float
    target_speed_mps: float
    pre_energy_terms: BaiyePowerV2EnergyTerms


class BaiyePowerV2Task(BaiyePowerTask):
    """BaiyePower v2 reward-and-lifecycle experiment.

    QP and minimum-jerk remain permanent action-pipeline components.  The
    The frozen U0/U1 policy observation remains the paired 29D real/sham
    design.  An explicitly flagged isolated child experiment may append
    analytic target rates without changing this reward or lifecycle boundary.
    """

    _DIAG_PREFIX = "baiyepower_v2_"
    route_initialized_on_reset = True
    metrics_use_route_attempt_boundaries = True

    def __init__(self, config):
        super().__init__(config)
        self._validate_isolation_config()

        # Explicit reward whitelist.  Replacing (rather than appending to) the
        # inherited list is the core isolation boundary.
        self.reward_functions = [
            BaiyePowerV2DirectionReward(config),
            BaiyePowerV2EnergyReward(config),
            BaiyePowerV2SafetyReward(config),
        ]

        self._route_condition = BaiyePowerV2FixedAttemptRoute(config)
        self._extreme_condition = ExtremeState(config)
        self._overload_condition = Overload(config)
        self._low_altitude_condition = LowAltitude(config)
        self._timeout_condition = Timeout(config)
        self.termination_conditions = [
            self._route_condition,
            self._extreme_condition,
            self._overload_condition,
            self._low_altitude_condition,
            self._timeout_condition,
        ]
        configured_terminal_altitude = float(
            getattr(
                config,
                "BaiyePowerV2_low_altitude_terminal_m",
                2500.0,
            )
        )
        if not math.isclose(
            configured_terminal_altitude,
            float(self._low_altitude_condition.altitude_limit),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "BaiyePowerV2 low-altitude reward and hard termination "
                "thresholds must match"
            )

        self.height_cost_scale_m = self._positive_config(
            "BaiyePowerV2_height_cost_scale_m", 90.0
        )
        self.speed_cost_scale_mps = self._positive_config(
            "BaiyePowerV2_speed_cost_scale_mps", 20.0
        )
        self.gravity_mps2 = self._positive_config(
            "BaiyePowerV2_gravity_mps2", 9.80665
        )

        self.task_terminal_penalty = self._nonpositive_config(
            "BaiyePowerV2_task_terminal_penalty", 0.0
        )
        self.unsafe_terminal_penalty = self._nonpositive_config(
            "BaiyePowerV2_unsafe_terminal_penalty", 0.0
        )
        self.numerical_terminal_penalty = self._nonpositive_config(
            "BaiyePowerV2_numerical_terminal_penalty",
            self.unsafe_terminal_penalty,
        )

        self._episode_id = -1
        self._target_epoch = 0
        self._transition_contexts: Dict[str, BaiyePowerV2TransitionContext] = {}
        self._transition_diag: Dict[str, dict] = {}
        self._numerical_failures: Dict[str, dict] = {}
        self._last_target_signatures: Dict[str, tuple] = {}
        self._last_finite_observations: Dict[str, np.ndarray] = {}
        self._external_target_mode = False
        self._external_target_rates = None

    def _validate_isolation_config(self):
        if str(getattr(self.config, "action_type", "")).lower() != "continuous":
            raise ValueError("BaiyePowerV2 requires action_type=continuous")
        obs_version = str(
            getattr(self.config, "obs_version", "")
        ).lower()
        target_rates_enabled = bool(
            getattr(
                self.config,
                "BaiyePowerTargetRates_enable",
                False,
            )
        )
        if obs_version == "v4_qp_state":
            if target_rates_enabled:
                raise ValueError(
                    "29D v4_qp_state cannot enable target-rate observation"
                )
        elif obs_version == "v5_target_rates":
            if not target_rates_enabled:
                raise ValueError(
                    "32D v5_target_rates requires "
                    "BaiyePowerTargetRates_enable=true"
                )
            target_rate_tasks = {
                "baiyepower_target_rates_o1",
                "baiyepower_coordinated_command_c3_v2",
                "baiyepower_energy_command_c10_r2",
                "baiyepower_energy_balance_probe_c10_d6",
                "baiyepower_authority_routed_c10_d10",
                "baiyepower_authority_routed_c10_s1",
                "baiyepower_authority_routed_c10_r3",
                "baiyepower_direction_invariant_c10_s2",
                "baiyepower_direction_invariant_c10_r4",
                "baiyepower_sign_constrained_c11_s0",
                "baiyepower_sign_constrained_c11_r0",
                "baiyepower_extended_gain_c12_d0",
                "baiyepower_fixed_gain_c12_r0",
                "baiyepower_fixed_gain_c12_r1",
                "baiyepower_structured_residual_c13_s0",
                "baiyepower_structured_feedback_c13_s1",
                "baiyepower_structured_feedback_c13_r0",
                "baiyepower_rate_climb_structured_c14_s0",
                "baiyepower_rate_climb_structured_c14_r0",
                "baiyepower_three_authority_c15_s0",
                "baiyepower_three_authority_c15_r0",
                "baiyepower_two_authority_c16_s0",
                "baiyepower_two_authority_c16_r0",
                "baiyepower_extended_exchange_c16_d4",
                "baiyepower_narrow_exchange_c16_d5",
                "baiyepower_exact_anchor_c16_d6",
                "baiyepower_exchange_holdout_c16_v1",
                "baiyepower_recentered_residual_c17_s0",
                "baiyepower_recentered_residual_c17_r0",
                "baiyepower_pitch_response_c26_sid0",
                "baiyepower_pitch_response_c26_sid1",
                "baiyepower_direct_flight_path_c26_n6",
                "baiyepower_maneuver_c28",
                "baiyepower_energy_momentum_c41",
                "baiyepower_maneuver_c31_v1",
                "baiyepower_maneuver_c31_v2",
                "baiyepower_maneuver_c31_v3",
                "baiyepower_maneuver_c31_v4",
                "baiyepower_maneuver_c31_v5",
                "baiyepower_direct_student_c21_s0",
                "baiyepower_direct_student_c21_s1",
                "baiyepower_direct_student_c21_s2",
                "baiyepower_direct_student_c21_s3",
                "baiyepower_direct_student_c21_s4",
                "baiyepower_direct_student_c21_s5",
                "baiyepower_direct_student_c21_s6",
                "baiyepower_direct_student_c21_s7",
                "baiyepower_direct_student_c21_s17",
                "baiyepower_direct_student_c21_d26",
            }
            if str(getattr(self.config, "task", "")) not in (
                target_rate_tasks
            ):
                raise ValueError(
                    "v5_target_rates is isolated to "
                    "an explicit target-rate task"
                )
        else:
            raise ValueError(
                "BaiyePowerV2 requires v4_qp_state or isolated "
                "v5_target_rates"
            )
        qp_mode = str(
            getattr(self.config, "QPStateObservation_mode", "")
        ).strip().lower()
        if qp_mode not in ("real", "repeat_prev"):
            raise ValueError(
                "BaiyePowerV2 requires QPStateObservation_mode=real or repeat_prev"
            )
        if not bool(getattr(self.config, "ActionProjector_enable", False)):
            raise ValueError("BaiyePowerV2 requires ActionProjector_enable=true")
        if not bool(getattr(self.config, "ActionSmoother_enable", False)):
            raise ValueError("BaiyePowerV2 requires ActionSmoother_enable=true")
        smoother_mode = str(
            getattr(self.config, "ActionSmoother_mode", "")
        ).strip().lower()
        if smoother_mode != "min_jerk":
            raise ValueError("BaiyePowerV2 requires ActionSmoother_mode=min_jerk")
        if not bool(getattr(self.config, "BaiyePowerV2_enable_yaw", True)):
            raise ValueError(
                "BaiyePowerV2 yaw objective is mandatory in this pilot"
            )
        if not bool(
            getattr(self.config, "HeadingRouteCurriculum_enable", False)
        ):
            raise ValueError(
                "BaiyePowerV2 requires HeadingRouteCurriculum_enable=true "
                "for fixed-duration attempt scheduling"
            )

    def _positive_config(self, name, default):
        value = float(getattr(self.config, name, default))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
        return value

    def _nonpositive_config(self, name, default):
        value = float(getattr(self.config, name, default))
        if not math.isfinite(value) or value > 0.0:
            raise ValueError(
                "{} must be finite and non-positive".format(name)
            )
        return value

    def reset(self, env):
        self._episode_id += 1
        self._target_epoch = 0
        self._transition_contexts.clear()
        self._transition_diag.clear()
        self._numerical_failures.clear()
        self._last_target_signatures.clear()
        self._last_finite_observations.clear()
        self._external_target_mode = False
        self._external_target_rates = None

        # A VecEnv may reuse one environment object after an automatic reset.
        # Route lifecycle state must never leak into the next episode.
        for name in ("heading_route_state", "heading_route_attempt_index"):
            if hasattr(env, name):
                delattr(env, name)

        super().reset(env)
        agent_id = next(iter(env.agents.keys()))
        self._route_condition._reset_route_state(
            env,
            agent_id,
            force=True,
            reset_attempt_sequence=True,
        )

    def set_baiyepower_v2_external_target_mode(self, enabled=True):
        """Let an evaluation script own target updates without disabling reward."""

        self._external_target_mode = bool(enabled)
        if not self._external_target_mode:
            self._external_target_rates = None

    def set_baiyepower_v2_external_target_rates(
        self,
        heading_rate_deg_s,
        altitude_rate_mps,
        speed_rate_mps2,
    ):
        """Set exact rates when an evaluation script owns the target."""

        rates = np.asarray(
            [
                heading_rate_deg_s,
                altitude_rate_mps,
                speed_rate_mps2,
            ],
            dtype=np.float32,
        )
        if rates.shape != (3,) or not np.all(np.isfinite(rates)):
            raise ValueError(
                "external target rates must be a finite 3D vector"
            )
        self._external_target_rates = rates.copy()

    def begin_transition(self, env):
        if self._transition_contexts:
            raise RuntimeError(
                "BaiyePowerV2 transition context was not finalized before "
                "the next action"
            )
        decision_dt = float(getattr(env, "time_interval", np.nan))
        if not math.isfinite(decision_dt) or decision_dt <= 0.0:
            raise ValueError("BaiyePowerV2 decision timestep must be positive")

        route_state = getattr(env, "heading_route_state", None)
        route_attempt_id = (
            int(route_state["attempt_id"])
            if isinstance(route_state, dict) and "attempt_id" in route_state
            else None
        )
        target_valid = (
            route_attempt_id is not None or self._external_target_mode
        )

        for agent_id, sim in env.agents.items():
            try:
                pre_observation = np.asarray(
                    self.get_obs(env, agent_id), dtype=np.float32
                ).copy()
                if (
                    pre_observation.shape != self.observation_space.shape
                    or not np.all(np.isfinite(pre_observation))
                ):
                    raise ValueError(
                        "pre-action policy observation is not finite"
                    )
                self._last_finite_observations[agent_id] = pre_observation
                target_heading_rad = math.radians(
                    float(sim.get_property_value(c.target_heading_deg))
                )
                target_altitude_m = (
                    float(sim.get_property_value(c.target_altitude_ft)) * 0.3048
                )
                target_speed_mps = self.get_target_speed_mps(sim)
                target_signature = (
                    target_heading_rad,
                    target_altitude_m,
                    target_speed_mps,
                )
                previous_signature = self._last_target_signatures.get(
                    agent_id
                )
                if previous_signature is not None and not np.allclose(
                    np.asarray(previous_signature, dtype=np.float64),
                    np.asarray(target_signature, dtype=np.float64),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    self._target_epoch += 1
                self._last_target_signatures[agent_id] = target_signature
                pre_terms = compute_energy_cost_terms(
                    altitude_m=sim.get_property_value(c.position_h_sl_m),
                    target_altitude_m=target_altitude_m,
                    u_mps=sim.get_property_value(c.velocities_u_mps),
                    v_mps=sim.get_property_value(c.velocities_v_mps),
                    w_mps=sim.get_property_value(c.velocities_w_mps),
                    target_speed_mps=target_speed_mps,
                    height_cost_scale_m=self.height_cost_scale_m,
                    speed_cost_scale_mps=self.speed_cost_scale_mps,
                    gravity_mps2=self.gravity_mps2,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                # Build no partially valid context.  A numerical simulator
                # state is a true terminal event, handled by get_termination.
                self.mark_baiyepower_v2_numerical_failure(
                    agent_id, "transition_pre_state", exc
                )
                raise RuntimeError(
                    "BaiyePowerV2 could not snapshot a finite pre-action state"
                ) from exc

            context = BaiyePowerV2TransitionContext(
                decision_step=int(env.current_step),
                decision_dt_sec=decision_dt,
                episode_id=int(self._episode_id),
                target_epoch=int(self._target_epoch),
                route_attempt_id=route_attempt_id,
                target_lifecycle_valid=bool(target_valid),
                target_heading_rad=target_heading_rad,
                target_altitude_m=target_altitude_m,
                target_speed_mps=target_speed_mps,
                pre_energy_terms=pre_terms,
            )
            self._transition_contexts[agent_id] = context
            self._transition_diag[agent_id] = {
                "active": 1.0,
                "episode_id": float(context.episode_id),
                "target_epoch": float(context.target_epoch),
                # target_epoch is retained for log compatibility.  In an arc
                # it is a command revision, not a stable segment identity.
                "target_revision": float(context.target_epoch),
                "decision_step": float(context.decision_step),
                "target_lifecycle_valid": float(
                    context.target_lifecycle_valid
                ),
                "external_target_mode": float(self._external_target_mode),
                "route_attempt_id": (
                    float(context.route_attempt_id)
                    if context.route_attempt_id is not None
                    else -1.0
                ),
                "target_heading_deg": math.degrees(
                    context.target_heading_rad
                ),
                "target_altitude_m": context.target_altitude_m,
                "target_speed_mps": context.target_speed_mps,
            }

    def validate_post_transition(self, env):
        required_properties = tuple(
            dict.fromkeys(
                tuple(self.state_var)
                + (
            c.detect_extreme_state,
            c.position_h_sl_m,
            c.velocities_u_mps,
            c.velocities_v_mps,
            c.velocities_w_mps,
            c.attitude_psi_rad,
            c.velocities_v_north_mps,
            c.velocities_v_east_mps,
            c.velocities_h_dot_fps,
            c.simulation_sim_time_sec,
            c.accelerations_n_pilot_x_norm,
            c.accelerations_n_pilot_y_norm,
            c.accelerations_n_pilot_z_norm,
                )
            )
        )
        for agent_id, sim in env.agents.items():
            try:
                values = np.asarray(
                    [
                        float(sim.get_property_value(prop))
                        for prop in required_properties
                    ],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        "post-action simulator state contains NaN or Inf"
                    )
                qp_state = np.asarray(
                    env.get_qp_state_observation(agent_id),
                    dtype=np.float64,
                )
                if qp_state.shape != (8,) or not np.all(
                    np.isfinite(qp_state)
                ):
                    raise ValueError(
                        "post-action QP observation contains NaN or Inf"
                    )
            except (TypeError, ValueError, OverflowError) as exc:
                self.mark_baiyepower_v2_numerical_failure(
                    agent_id, "post_transition_state", exc
                )

    def has_transition_numerical_failure(self, agent_id):
        return agent_id in self._numerical_failures

    def get_terminal_observation_fallback(self, agent_id):
        observation = self._last_finite_observations.get(agent_id)
        if observation is None:
            raise RuntimeError(
                "BaiyePowerV2 has no finite observation for terminal fallback"
            )
        return observation.copy()

    def get_baiyepower_v2_transition_context(self, agent_id):
        try:
            return self._transition_contexts[agent_id]
        except KeyError as exc:
            raise RuntimeError(
                "BaiyePowerV2 reward requested without begin_transition"
            ) from exc

    def record_baiyepower_v2_diag(self, agent_id, values):
        if not isinstance(values, dict):
            raise TypeError("BaiyePowerV2 diagnostics must be a dict")
        destination = self._transition_diag.setdefault(agent_id, {})
        for key, value in values.items():
            normalized_key = str(key)
            if normalized_key.startswith(self._DIAG_PREFIX):
                normalized_key = normalized_key[len(self._DIAG_PREFIX):]
            if isinstance(value, (bool, np.bool_)):
                destination[normalized_key] = float(bool(value))
                continue
            if value is None:
                continue
            if isinstance(value, str):
                destination[normalized_key] = value
                continue
            scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError(
                    "non-finite BaiyePowerV2 diagnostic: {}".format(key)
                )
            destination[normalized_key] = scalar

    def mark_baiyepower_v2_numerical_failure(
        self, agent_id, source, error
    ):
        self._numerical_failures[agent_id] = {
            "source": str(source),
            "error": "{}: {}".format(type(error).__name__, error),
        }
        self._transition_diag.setdefault(agent_id, {}).update(
            {
                "energy_valid": 0.0,
                "numerical_failure": 1.0,
                "numerical_failure_source": str(source),
            }
        )

    def get_transition_diagnostics(self, env, agent_id):
        diag = dict(self._transition_diag.get(agent_id, {}))
        failure = self._numerical_failures.get(agent_id)
        if failure is not None:
            diag["numerical_failure_error"] = failure["error"]
        return {
            self._DIAG_PREFIX + str(key): value
            for key, value in diag.items()
        }

    def get_reward(self, env, agent_id, info=None):
        info = {} if info is None else info
        reward, info = super().get_reward(env, agent_id, info)
        reward = float(reward)
        if not math.isfinite(reward):
            raise RuntimeError("BaiyePowerV2 aggregate reward is non-finite")
        if self.has_transition_numerical_failure(agent_id):
            # A numerically invalid transition is not a partially valid yaw or
            # safety transition.  Keep nominal/candidate diagnostics for root
            # cause analysis, but suppress every ordinary applied
            # contribution.  Only the explicit numerical terminal penalty may
            # reach PPO.
            destination = self._transition_diag.setdefault(agent_id, {})
            for key in tuple(destination):
                if str(key).endswith("_reward_contribution"):
                    destination[key] = 0.0
            destination["ordinary_reward_suppressed"] = 1.0
            reward = 0.0
        self.record_baiyepower_v2_diag(
            agent_id, {"reward_contribution_sum": reward}
        )
        return reward, info

    def _collect_safety_flags(self, env, agent_id):
        sim = env.agents[agent_id]
        numerical = agent_id in self._numerical_failures
        if numerical:
            # validate_post_transition already established that the state is
            # unusable.  Numerical has the highest termination priority, so
            # do not risk a second read from the failed simulator merely to
            # collect lower-priority flags.
            return {
                "numerical": True,
                "extreme": False,
                "overload": False,
                "low_altitude": False,
            }
        extreme = bool(sim.get_property_value(c.detect_extreme_state))
        altitude_m = float(sim.get_property_value(c.position_h_sl_m))
        low_altitude = bool(
            math.isfinite(altitude_m)
            and altitude_m <= float(self._low_altitude_condition.altitude_limit)
        )

        overload = False
        sim_time = float(sim.get_property_value(c.simulation_sim_time_sec))
        if math.isfinite(sim_time) and sim_time > 10.0:
            accelerations = (
                float(sim.get_property_value(c.accelerations_n_pilot_x_norm)),
                float(sim.get_property_value(c.accelerations_n_pilot_y_norm)),
                float(sim.get_property_value(c.accelerations_n_pilot_z_norm))
                + 1.0,
            )
            if not all(math.isfinite(value) for value in accelerations):
                numerical = True
                self.mark_baiyepower_v2_numerical_failure(
                    agent_id,
                    "termination_acceleration",
                    ValueError("non-finite pilot acceleration"),
                )
            else:
                overload = bool(
                    abs(accelerations[0])
                    > float(self._overload_condition.acceleration_limit_x)
                    or abs(accelerations[1])
                    > float(self._overload_condition.acceleration_limit_y)
                    or abs(accelerations[2])
                    > float(self._overload_condition.acceleration_limit_z)
                )
        if not math.isfinite(altitude_m) or not math.isfinite(sim_time):
            numerical = True
            self.mark_baiyepower_v2_numerical_failure(
                agent_id,
                "termination_state",
                ValueError("non-finite altitude or simulation time"),
            )
        return {
            "numerical": bool(numerical),
            "extreme": bool(extreme),
            "overload": bool(overload),
            "low_altitude": bool(low_altitude),
        }

    @staticmethod
    def _first_true_reason(flags):
        for name in ("numerical", "extreme", "overload", "low_altitude"):
            if flags.get(name, False):
                return name
        return ""

    def get_termination(self, env, agent_id, info=None):
        info = {} if info is None else info
        flags = self._collect_safety_flags(env, agent_id)
        safety_any = any(flags.values())
        route_info = {}

        # Route evaluation can advance its target.  On a safety terminal step
        # that side effect is both unnecessary and misleading, so safety wins
        # before the route lifecycle is evaluated.
        if not safety_any and not self._external_target_mode:
            route_done, _, route_info = self._route_condition.get_termination(
                self,
                env,
                agent_id,
                {"current_step": int(env.current_step)},
            )
            if route_done:
                raise RuntimeError(
                    "BaiyePower v2 route outcomes must not terminate episodes"
                )
            info.update(route_info)
        elif safety_any and hasattr(env, "heading_route_state"):
            # Preserve the already-existing lifecycle identity for metrics
            # without calling the mutating route evaluator.
            # A numerical state cannot be evaluated against tolerances.  Empty
            # flags keep all condition booleans false while retaining the
            # attempt ID and frozen command for a correctly classified unsafe
            # endpoint.
            if flags["numerical"]:
                info.update(
                    self._route_condition.passive_route_identity_diag(env)
                )
            else:
                passive_flags = self._route_condition._condition_flags(
                    env,
                    agent_id,
                )
                info.update(
                    self._route_condition._route_diag(
                        env,
                        agent_id,
                        passive_flags,
                        point_completed=False,
                    )
                )
            info["heading_route_attempt_outcome"] = "episode_terminated"
            if flags["numerical"]:
                info["heading_route_attempt_sample_valid"] = False

        admin_limit = bool(
            int(env.current_step) >= int(self._timeout_condition.max_steps)
        )
        if safety_any:
            terminated = True
            truncated = False
            reason = self._first_true_reason(flags)
            kind = "safety"
            try:
                env.agents[agent_id].crash()
            except Exception:
                # crash() is a simulator side effect, not part of the
                # termination decision.  The episode must still end.
                pass
        elif admin_limit:
            terminated = False
            truncated = True
            reason = "timeout"
            kind = "administrative_limit"
        else:
            terminated = False
            truncated = False
            reason = ""
            kind = "ongoing"

        done = bool(terminated or truncated)
        info.update(
            {
                "baiyepower_v2_flag_numerical": bool(flags["numerical"]),
                "baiyepower_v2_flag_extreme_state": bool(flags["extreme"]),
                "baiyepower_v2_flag_overload": bool(flags["overload"]),
                "baiyepower_v2_flag_low_altitude": bool(
                    flags["low_altitude"]
                ),
                # Retained as an always-false compatibility field.  A point
                # miss is an attempt outcome, not a route terminal failure.
                "baiyepower_v2_flag_route_failure": False,
                "baiyepower_v2_flag_point_attempt_miss": bool(
                    info.get("heading_route_attempt_outcome", "")
                    == "route_timeout"
                ),
                "baiyepower_v2_flag_attempt_boundary": bool(
                    info.get("heading_route_attempt_outcome", "")
                    in (
                        "point_hit",
                        "route_timeout",
                        "arc_duration_completed",
                    )
                ),
                "baiyepower_v2_flag_admin_limit": bool(admin_limit),
                "baiyepower_v2_route_evaluated": bool(
                    not safety_any and not self._external_target_mode
                ),
                "baiyepower_v2_external_target_mode": bool(
                    self._external_target_mode
                ),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "TimeLimit.truncated": bool(truncated),
                "termination_kind": kind,
            }
        )
        if done:
            info["termination_reason"] = reason
        return done, info

    def get_terminal_reward(self, env, agent_id, info):
        if not bool(info.get("terminated", False)):
            terminal_reward = 0.0
        elif bool(info.get("baiyepower_v2_flag_numerical", False)):
            terminal_reward = self.numerical_terminal_penalty
        elif str(info.get("termination_kind", "")) == "safety":
            terminal_reward = self.unsafe_terminal_penalty
        else:
            terminal_reward = self.task_terminal_penalty
        self.record_baiyepower_v2_diag(
            agent_id, {"terminal_reward": terminal_reward}
        )
        return terminal_reward

    def end_transition(self, env, info):
        self._transition_contexts.clear()
        self._transition_diag.clear()
        self._numerical_failures.clear()

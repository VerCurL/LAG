from ..core.catalog import Catalog as c
from ..utils.baiyepower_physics import speed_magnitude_mps
from .unreach_heading import UnreachHeading


class BaiyePowerUnreachHeading(UnreachHeading):
    """Heading-route lifecycle with scalar speed-magnitude targets."""

    def __init__(self, config):
        super().__init__(config)
        self.threshold_deg = float(
            getattr(
                config,
                "BaiyePowerUnreachHeading_threshold_deg",
                getattr(config, "UnreachHeading_threshold_deg", 10.0),
            )
        )

    @staticmethod
    def _current_speed_mps(sim):
        return speed_magnitude_mps(
            sim.get_property_value(c.velocities_u_mps),
            sim.get_property_value(c.velocities_v_mps),
            sim.get_property_value(c.velocities_w_mps),
        )

    @staticmethod
    def _target_speed_mps(sim):
        return float(sim.get_property_value(c.baiyepower_target_speed_mps))

    @staticmethod
    def _set_target_speed_mps(sim, target_speed_mps):
        sim.set_property_value(
            c.baiyepower_target_speed_mps,
            max(float(target_speed_mps), 0.0),
        )

    def _speed_error_mps(self, sim):
        return self._target_speed_mps(sim) - self._current_speed_mps(sim)


class BaiyePowerV2FixedAttemptRoute(BaiyePowerUnreachHeading):
    """Policy-independent point/arc attempt scheduler for BaiyePower v2.

    A point hit is latched but never advances the target early.  Point and arc
    attempts are both finalized only at a deadline selected when the attempt
    is created.  Consequently, the policy cannot delay the next target by
    deliberately staying just outside a success tolerance.

    Attempt outcomes are diagnostic lifecycle events, not episode
    terminations.  True termination remains the responsibility of the v2 task
    safety arbiter.
    """

    def _reset_route_state(
        self,
        env,
        agent_id,
        force=False,
        reset_attempt_sequence=False,
    ):
        if not force and hasattr(env, "heading_route_state"):
            return
        super()._reset_route_state(
            env,
            agent_id,
            force=force,
            reset_attempt_sequence=reset_attempt_sequence,
        )
        state = env.heading_route_state
        mode = str(state.get("mode", "point"))
        if mode == "arc":
            # The inherited route keeps a grace interval so failure can be
            # terminal.  V2 has no route terminal: an arc attempt ends exactly
            # at its exogenous planned duration.
            duration = max(float(state.get("arc_duration_sec", 0.0)), 1e-6)
            start = float(state["attempt_start_time_sec"])
            state["attempt_planned_duration_sec"] = duration
            state["attempt_timeout_sec"] = duration
            state["attempt_deadline_sec"] = start + duration
        state.update(
            {
                "attempt_ever_hit": False,
                "attempt_first_hit_time_sec": None,
                "attempt_first_hit_elapsed_sec": None,
                "attempt_first_hit_quality": {},
            }
        )

    def _route_diag(self, env, agent_id, flags, point_completed):
        info = super()._route_diag(
            env,
            agent_id,
            flags,
            point_completed,
        )
        state = env.heading_route_state
        sim_time = float(
            env.agents[agent_id].get_property_value(
                c.simulation_sim_time_sec
            )
        )
        deadline = float(state["attempt_deadline_sec"])
        first_hit_time = state.get("attempt_first_hit_time_sec")
        info.update(
            {
                "heading_route_attempt_fixed_duration": True,
                "heading_route_attempt_ever_hit": bool(
                    state.get("attempt_ever_hit", False)
                ),
                "heading_route_attempt_time_remaining_sec": max(
                    deadline - sim_time,
                    0.0,
                ),
                "heading_route_attempt_deadline_reached": bool(
                    sim_time >= deadline
                ),
            }
        )
        if first_hit_time is not None:
            info.update(
                {
                    "heading_route_attempt_first_hit_time_sec": float(
                        first_hit_time
                    ),
                    "heading_route_attempt_first_hit_elapsed_sec": float(
                        state["attempt_first_hit_elapsed_sec"]
                    ),
                }
            )
        return info

    @staticmethod
    def passive_route_identity_diag(env):
        """Return lifecycle identity without reading a failed simulator."""

        state = env.heading_route_state
        return {
            "heading_route_level": int(state["level"]),
            "heading_route_mode": str(state.get("mode", "point")),
            "heading_route_attempt_id": int(state["attempt_id"]),
            "heading_route_attempt_index": int(state["attempt_index"]),
            "heading_route_attempt_start_time_sec": float(
                state["attempt_start_time_sec"]
            ),
            "heading_route_attempt_deadline_sec": float(
                state["attempt_deadline_sec"]
            ),
            "heading_route_attempt_planned_duration_sec": float(
                state["attempt_planned_duration_sec"]
            ),
            "heading_route_attempt_timeout_sec": float(
                state["attempt_timeout_sec"]
            ),
            "heading_route_attempt_command_heading_abs_deg": float(
                state["attempt_command_heading_abs_deg"]
            ),
            "heading_route_attempt_command_altitude_abs_m": float(
                state["attempt_command_altitude_abs_m"]
            ),
            "heading_route_attempt_command_speed_abs_mps": float(
                state["attempt_command_speed_abs_mps"]
            ),
            "heading_route_attempt_initial_in_zone": state[
                "attempt_initial_in_zone"
            ],
            "heading_route_attempt_fixed_duration": True,
            "heading_route_attempt_ever_hit": bool(
                state.get("attempt_ever_hit", False)
            ),
            "heading_route_attempt_started_this_step": False,
            "heading_route_attempt_sample_valid": False,
            "heading_route_attempt_outcome": "ongoing",
        }

    def _get_route_termination(self, task, env, agent_id, info):
        initialized_on_reset = bool(
            getattr(task, "route_initialized_on_reset", False)
        )
        if not hasattr(env, "heading_route_state") or (
            env.current_step <= 1 and not initialized_on_reset
        ):
            self._reset_route_state(
                env,
                agent_id,
                force=True,
                reset_attempt_sequence=env.current_step <= 1,
            )

        state = env.heading_route_state
        sim = env.agents[agent_id]
        sim_time = float(
            sim.get_property_value(c.simulation_sim_time_sec)
        )
        mode = str(state.get("mode", "point"))
        level = state["level_config"]

        if mode == "arc":
            _, arc_progress, _ = self._update_arc_target(
                env,
                agent_id,
            )
            state["arc_progress"] = float(arc_progress)

        self._update_segment_quality_state(env, agent_id)
        flags = self._condition_flags(env, agent_id)
        conditions = list(
            level.get(
                "hold_conditions",
                ["heading", "altitude", "speed"],
            )
        )
        in_zone = all(flags.get(name, False) for name in conditions)
        dt = float(getattr(env, "time_interval", self.check_interval))
        hold_required = float(level.get("last_hold_sec", 0.0))
        if in_zone:
            state["hold_timer"] += dt
        else:
            state["hold_timer"] = max(
                0.0,
                float(state["hold_timer"])
                - max(self.hold_decay, 0.0) * dt,
            )
        state["in_target_zone"] = bool(in_zone)

        hit_latched_this_step = False
        if (
            mode == "point"
            and not bool(state.get("attempt_ever_hit", False))
            and in_zone
            and float(state["hold_timer"]) >= hold_required
        ):
            hit_latched_this_step = True
            point_time = sim_time - float(state["point_start_time"])
            state["attempt_ever_hit"] = True
            state["attempt_first_hit_time_sec"] = sim_time
            state["attempt_first_hit_elapsed_sec"] = point_time
            state["attempt_first_hit_quality"] = (
                self._completion_quality_diag(
                    env,
                    agent_id,
                    point_time,
                )
            )

        deadline = float(state["attempt_deadline_sec"])
        deadline_reached = bool(sim_time >= deadline)
        resolved_success = bool(
            deadline_reached
            and (
                mode == "arc"
                or bool(state.get("attempt_ever_hit", False))
            )
        )
        info.update(
            self._route_diag(
                env,
                agent_id,
                flags,
                point_completed=resolved_success,
            )
        )
        info.update(
            {
                "heading_route_attempt_hit_latched_this_step": bool(
                    hit_latched_this_step
                ),
                "heading_route_attempt_outcome": "ongoing",
            }
        )
        if state.get("attempt_first_hit_quality"):
            info.update(state["attempt_first_hit_quality"])
        info["heading_turn_counts"] = env.heading_turn_counts

        if not deadline_reached:
            return False, False, info

        if mode == "arc":
            outcome = "arc_duration_completed"
        elif bool(state.get("attempt_ever_hit", False)):
            outcome = "point_hit"
        else:
            outcome = "route_timeout"
        info.update(
            {
                "heading_route_attempt_outcome": outcome,
                "heading_route_attempt_end_reason": outcome,
            }
        )

        if resolved_success:
            env.heading_turn_counts += 1
        info["heading_turn_counts"] = env.heading_turn_counts
        completed_attempt_id = int(state["attempt_id"])
        completed_level = int(state["level"])
        self._reset_route_state(env, agent_id, force=True)
        self.log(
            "fixed attempt completed. id:{} level:{} outcome:{}".format(
                completed_attempt_id,
                completed_level,
                outcome,
            )
        )
        # A route miss is an attempt result, never an episode terminal.
        return False, resolved_success, info

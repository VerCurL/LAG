import math

import numpy as np

from .env_base import BaseEnv
from ..core.catalog import Catalog as c
from ..tasks.heading_task import HeadingTask
from ..utils.action_projector import ConvexActionProjector
from ..utils.action_smoother import MinimumJerkActionSmoother
from ..utils.coordinated_action import (
    ACTION_CHANNELS,
    apply_structured_feedback_action_mask,
    compose_coordinated_residual_action,
    compose_energy_exchange_fixed_gain,
    compose_energy_exchange_gain_residual,
    compose_structured_feedback_gain_residual,
    compose_structured_reference_gain_residual,
    compose_vertical_speed_command_residual,
    coordinated_reference_action,
    euler_heading_rate_deg_s,
    measured_energy_exchange_offset,
    scheduled_flight_path_tau,
    tecs_energy_allocation_commands,
    update_energy_trim_estimate,
)
from ..utils.heading_metrics import HeadingMetricsTracker
from ..utils.turn_vertical_trim import update_turn_vertical_trim
from ..utils.turn_bank_feedback import update_turn_bank_feedback
from ..utils.steady_bank_feedback import update_steady_bank_feedback

from ..utils.vs_bank_target import update_vs_bank_target

from ..utils.vs_bank_target_v5 import update_vs_bank_target as update_vs_bank_target_v5
from ..utils.baiyepower_v2_physics import compute_energy_cost_terms


class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"
        self.init_states = None
        self._heading_metrics = None
        self._last_heading_step = None
        self._trim_ema_state = None
        self._last_trim_ema_log = {}
        self._action_projector = None
        self._last_action_projector_log = {}
        self._action_smoother = None
        self._last_coordinated_residual_log = {}
        self._last_direct_student_teacher_log = {}
        self._energy_trim_states = {}
        self._turn_vertical_trim_states = {}
        self._turn_bank_feedback_states = {}
        self._steady_bank_feedback_states = {}
        self._vs_bank_target_states = {}
        self._vs_bank_target_v5_states = {}
        self._controller_measurement_rng = None
        self._controller_measurement_cache = {}
        self._last_controller_measurement_noise = {}
        self._last_controller_measurement_bias = {}
        # Metrics Plus keeps simulator-rate traces opt-in so training rollouts
        # do not pay the memory and Python-object cost of 60 Hz diagnostics.
        self._metrics_plus_substep_trace_enabled = False
        self._last_metrics_plus_substep_trace = []

    def load_task(self):
        raise NotImplementedError("SingleControlEnv is a control-context namespace in LAGLayer3")

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0
        self.task.reset(self)
        self._reset_controller_measurement_noise()
        self._reset_heading_metrics()
        self._reset_trim_ema()
        self._reset_coordinated_residual()
        self._last_direct_student_teacher_log = {}
        self._reset_action_projector()
        self._reset_action_smoother()
        self._last_metrics_plus_substep_trace = []
        obs = self.get_obs()
        return self._pack(obs)

    def _controller_measurement_noise_stds(self):
        names = (
            "course_deg",
            "altitude_m",
            "speed_mps",
            "roll_deg",
            "pitch_deg",
            "vertical_speed_mps",
            "pitch_rate_deg_s",
            "sideslip_deg",
        )
        values = {
            name: float(
                getattr(
                    self.config,
                    "ControllerMeasurementNoise_{}_std".format(name),
                    0.0,
                )
            )
            for name in names
        }
        if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("controller measurement-noise std values must be finite and non-negative")
        return values

    def _reset_controller_measurement_noise(self):
        self._controller_measurement_cache = {}
        self._last_controller_measurement_noise = {}
        self._last_controller_measurement_bias = {}
        enabled = bool(getattr(self.config, "ControllerMeasurementNoise_enable", False))
        stds = self._controller_measurement_noise_stds()
        if enabled and not any(value > 0.0 for value in stds.values()):
            raise ValueError("enabled controller measurement noise requires a positive std")
        seed = int(getattr(self.config, "ControllerMeasurementNoise_seed", 0))
        self._controller_measurement_rng = np.random.RandomState(seed)

    def _controller_measurement_biases(self):
        values = {
            name: float(
                getattr(
                    self.config,
                    "ControllerMeasurementBias_{}".format(name),
                    0.0,
                )
            )
            for name in self._controller_measurement_noise_stds()
        }
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError("controller measurement-bias values must be finite")
        return values

    def _sample_controller_measurements(self, agent_id):
        sim = self.agents[agent_id]
        north = float(sim.get_property_value(c.velocities_v_north_mps))
        east = float(sim.get_property_value(c.velocities_v_east_mps))
        true_horizontal_speed = float(np.hypot(north, east))
        true_course = float(np.rad2deg(np.arctan2(east, north)) % 360.0)
        true_speed = self._get_task_current_speed_mps(agent_id)
        true_values = {
            "course_deg": true_course,
            "altitude_m": float(sim.get_property_value(c.position_h_sl_m)),
            "speed_mps": true_speed,
            "roll_deg": float(sim.get_property_value(c.attitude_phi_deg)),
            "pitch_deg": float(sim.get_property_value(c.attitude_theta_deg)),
            "vertical_speed_mps": float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048,
            "pitch_rate_deg_s": float(np.rad2deg(sim.get_property_value(c.velocities_q_rad_sec))),
            "sideslip_deg": float(sim.get_property_value(c.aero_beta_deg)),
        }
        enabled = bool(getattr(self.config, "ControllerMeasurementNoise_enable", False))
        stds = self._controller_measurement_noise_stds()
        bias = self._controller_measurement_biases()
        noise = {
            name: (
                float(self._controller_measurement_rng.normal(0.0, std))
                if enabled and std > 0.0
                else 0.0
            )
            for name, std in stds.items()
        }
        measured = {
            name: true_values[name] + bias[name] + noise[name]
            for name in true_values
        }
        measured["course_deg"] %= 360.0
        measured["speed_mps"] = max(measured["speed_mps"], 1e-3)
        measured["horizontal_speed_mps"] = max(
            true_horizontal_speed + bias["speed_mps"] + noise["speed_mps"],
            1e-3,
        )
        target_heading = float(sim.get_property_value(c.target_heading_deg))
        target_altitude_m = float(sim.get_property_value(c.target_altitude_ft)) * 0.3048
        target_speed = float(self.task.get_target_speed_mps(sim))
        measured["course_error_deg"] = float((target_heading - measured["course_deg"] + 180.0) % 360.0 - 180.0)
        measured["altitude_error_m"] = target_altitude_m - measured["altitude_m"]
        measured["speed_error_mps"] = target_speed - measured["speed_mps"]
        measured["target_altitude_m"] = target_altitude_m
        measured["target_speed_mps"] = target_speed
        self._controller_measurement_cache[agent_id] = measured
        self._last_controller_measurement_noise[agent_id] = dict(noise)
        self._last_controller_measurement_bias[agent_id] = dict(bias)
        return measured

    def _controller_measurements(self, agent_id):
        measured = self._controller_measurement_cache.get(agent_id)
        return measured if measured is not None else self._sample_controller_measurements(agent_id)

    def _controller_energy_terms(self, agent_id, transition):
        if not bool(getattr(self.config, "ControllerMeasurementNoise_enable", False)):
            return transition.pre_energy_terms
        measured = self._controller_measurements(agent_id)
        return compute_energy_cost_terms(
            altitude_m=measured["altitude_m"],
            target_altitude_m=measured["target_altitude_m"],
            u_mps=measured["speed_mps"],
            v_mps=0.0,
            w_mps=0.0,
            target_speed_mps=measured["target_speed_mps"],
            height_cost_scale_m=self.task.height_cost_scale_m,
            speed_cost_scale_mps=self.task.speed_cost_scale_mps,
            gravity_mps2=self.task.gravity_mps2,
        )

    def get_controller_measurement_noise_log(self, agent_id):
        return dict(self._last_controller_measurement_noise.get(agent_id, {}))

    def get_controller_measurement_bias_log(self, agent_id):
        return dict(self._last_controller_measurement_bias.get(agent_id, {}))

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        init_heading = self.np_random.uniform(0., 180.)
        if bool(
            getattr(self.config, "InitialEnvelopeLattice_enable", False)
        ):
            altitude_choices_ft = np.asarray(
                getattr(
                    self.config,
                    "InitialEnvelopeLattice_altitudes_ft",
                    [],
                ),
                dtype=np.float64,
            ).reshape(-1)
            speed_choices_mps = np.asarray(
                getattr(
                    self.config,
                    "InitialEnvelopeLattice_speeds_mps",
                    [],
                ),
                dtype=np.float64,
            ).reshape(-1)
            if (
                altitude_choices_ft.size == 0
                or speed_choices_mps.size == 0
                or not np.all(np.isfinite(altitude_choices_ft))
                or not np.all(np.isfinite(speed_choices_mps))
                or np.any(altitude_choices_ft <= 0.0)
                or np.any(speed_choices_mps <= 0.0)
            ):
                raise ValueError(
                    "InitialEnvelopeLattice choices must be finite and positive"
                )
            init_altitude = float(self.np_random.choice(altitude_choices_ft))
            init_speed_mps = float(self.np_random.choice(speed_choices_mps))
            init_velocities_u = init_speed_mps / 0.3048
            self._initial_envelope_lattice_active = True
            self._initial_envelope_altitude_ft = init_altitude
            self._initial_envelope_speed_mps = init_speed_mps
        else:
            init_altitude = self.np_random.uniform(14000., 30000.)
            init_velocities_u = self.np_random.uniform(400., 1200.)
            self._initial_envelope_lattice_active = False
            self._initial_envelope_altitude_ft = init_altitude
            self._initial_envelope_speed_mps = init_velocities_u * 0.3048
        for init_state in self.init_states:
            init_state.update({
                'ic_psi_true_deg': init_heading,
                'ic_h_sl_ft': init_altitude,
                'ic_u_fps': init_velocities_u,
                'target_heading_deg': init_heading,
                'target_altitude_ft': init_altitude,
                'target_velocities_u_mps': init_velocities_u * 0.3048,
            })
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])
        self._tempsims.clear()

    def step(self, action):
        self.current_step += 1
        info = {"current_step": self.current_step}
        self._last_metrics_plus_substep_trace = []
        action = self._unpack(action)
        begin_transition = getattr(self.task, "begin_transition", None)
        if begin_transition is not None:
            begin_transition(self)
        action_targets = {}
        substep_hook = getattr(self.task, "on_simulation_substep", None)
        for agent_id in self.agents.keys():
            direct_action = self._apply_direct_student_teacher_mix(
                agent_id,
                action[agent_id],
            )
            policy_action = self._apply_coordinated_residual(
                agent_id,
                direct_action,
            )
            a_action = self.task.normalize_action(
                self,
                agent_id,
                policy_action,
            )
            a_action = self._apply_trim_bias_ema(agent_id, a_action)
            a_action = self._apply_action_projector(agent_id, a_action)
            action_targets[agent_id] = a_action
            teacher_log = self.get_direct_student_teacher_log(agent_id)
            if bool(teacher_log.get("enabled", False)):
                info["direct_student_teacher_action"] = np.asarray(
                    teacher_log["teacher_action"], dtype=np.float32
                ).copy()
                info["direct_student_teacher_valid"] = True
                info["direct_student_teacher_student_fraction"] = float(
                    teacher_log["student_fraction"]
                )
                info["direct_student_teacher_replace_mask"] = np.asarray(
                    teacher_log["replace_mask"], dtype=np.float32
                ).copy()
        if self._action_smoother_enabled():
            for agent_id, a_action in action_targets.items():
                self._action_smoother.set_target(
                    agent_id,
                    a_action,
                    context=self._get_action_smoother_context(agent_id),
                )
            for substep_idx in range(self.agent_interaction_steps):
                for agent_id, sim in self.agents.items():
                    sim.set_property_values(self.task.action_var, self._action_smoother.sample(agent_id))
                for sim in self._jsbsims.values():
                    sim.run()
                for sim in self._tempsims.values():
                    sim.run()
                if self._metrics_plus_substep_trace_enabled:
                    self._record_metrics_plus_substep(substep_idx)
                if substep_hook is not None:
                    substep_hook(self, substep_idx)
        else:
            for agent_id, a_action in action_targets.items():
                self.agents[agent_id].set_property_values(self.task.action_var, a_action)
            for substep_idx in range(self.agent_interaction_steps):
                for sim in self._jsbsims.values():
                    sim.run()
                for sim in self._tempsims.values():
                    sim.run()
                if self._metrics_plus_substep_trace_enabled:
                    self._record_metrics_plus_substep(substep_idx)
                if substep_hook is not None:
                    substep_hook(self, substep_idx)
        self.task.step(self)
        validate_post_transition = getattr(
            self.task, "validate_post_transition", None
        )
        if validate_post_transition is not None:
            validate_post_transition(self)

        agent_id = next(iter(self.agents.keys()))
        target_before = self._get_heading_target(agent_id)
        has_numerical_failure = getattr(
            self.task, "has_transition_numerical_failure", None
        )
        numerical_failure = bool(
            has_numerical_failure(agent_id)
            if has_numerical_failure is not None
            else False
        )
        if numerical_failure:
            sample = self._collect_numerical_terminal_sample(agent_id)
        else:
            sample = self._collect_heading_sample(agent_id)
        if isinstance(self.task, HeadingTask):
            # Capture safety endpoints independently of termination-condition
            # short-circuit order.  This is diagnostic only and does not alter
            # when or why the environment terminates.
            info.update(self._collect_metrics_plus_safety_flags(agent_id))

        # Credit the action against the target under which it was selected and
        # executed.  Route termination may advance point/arc targets below.
        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]
            get_terminal_reward = getattr(
                self.task, "get_terminal_reward", None
            )
            terminal_reward = float(
                get_terminal_reward(self, agent_id, info)
                if get_terminal_reward is not None
                else 0.0
            )
            if not np.isfinite(terminal_reward):
                raise RuntimeError("terminal reward must be finite")
            rewards[agent_id][0] += terminal_reward
            get_transition_diagnostics = getattr(
                self.task, "get_transition_diagnostics", None
            )
            if get_transition_diagnostics is not None:
                info.update(
                    get_transition_diagnostics(self, agent_id)
                )

        target_after = self._get_heading_target(agent_id)
        target_changed = not np.allclose(np.asarray(target_before), np.asarray(target_after), atol=1e-6)
        done = bool(next(iter(dones.values()))[0])
        termination_reason = info.get("termination_reason", "")
        if target_changed:
            self._mark_trim_ema_target_change(agent_id)
        metrics_target_changed = target_changed
        target_change_success = True
        target_change_end_reason = "next_target"
        if bool(
            getattr(
                self.task,
                "metrics_use_route_attempt_boundaries",
                False,
            )
        ):
            route_outcome = str(
                info.get("heading_route_attempt_outcome", "")
            )
            metrics_target_changed = route_outcome in (
                "point_hit",
                "route_timeout",
                "arc_duration_completed",
            )
            target_change_success = route_outcome in (
                "point_hit",
                "arc_duration_completed",
            )
            target_change_end_reason = route_outcome or "attempt_boundary"
        self._update_heading_metrics(
            sample,
            metrics_target_changed,
            target_after,
            done,
            termination_reason,
            target_change_success=target_change_success,
            target_change_end_reason=target_change_end_reason,
        )
        info["heading_target_changed"] = target_changed
        end_transition = getattr(self.task, "end_transition", None)
        if end_transition is not None:
            end_transition(self, info)

        # A numerical true terminal cannot safely build a post-state policy
        # observation.  It does not bootstrap, so return the last finite
        # pre-action observation only to let VecEnv preserve/reset cleanly.
        if done and bool(info.get("baiyepower_v2_flag_numerical", False)):
            obs = {
                current_agent_id: self.task.get_terminal_observation_fallback(
                    current_agent_id
                )
                for current_agent_id in self.agents.keys()
            }
        else:
            # The returned observation is consumed by the next policy action
            # and must reflect any route target advanced by termination.
            obs = self.get_obs()

        if done and self._heading_metrics is not None:
            info["heading_episode_metrics"] = self._heading_metrics.get_episode_metrics()
        if self._heading_train_diag_enabled():
            train_diag = self._collect_heading_train_diag(agent_id, sample)
            # BaiyePower v2 diagnostics are copied before end_transition clears
            # its private context.  Keeping them inside the established
            # heading sidecar makes subprocess training and JSONL aggregation
            # explicit without changing the VecEnv return contract.
            train_diag.update(
                {
                    key: value
                    for key, value in info.items()
                    if str(key).startswith("baiyepower_v2_")
                }
            )
            info["heading_train_step_diag"] = train_diag

        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def set_metrics_plus_substep_trace(self, enabled=True):
        """Enable opt-in simulator-rate diagnostics for fixed evaluation."""
        self._metrics_plus_substep_trace_enabled = bool(enabled)
        if not self._metrics_plus_substep_trace_enabled:
            self._last_metrics_plus_substep_trace = []

    def get_last_metrics_plus_substep_trace(self):
        """Return a defensive copy of the most recent 60 Hz trace."""
        return [dict(row) for row in self._last_metrics_plus_substep_trace]

    def _collect_metrics_plus_safety_flags(self, agent_id):
        sim = self.agents[agent_id]
        sim_time_sec = float(sim.get_property_value(c.simulation_sim_time_sec))
        low_altitude = bool(
            float(sim.get_property_value(c.position_h_sl_m))
            <= float(getattr(self.config, "altitude_limit", 2500.0))
        )
        extreme_state = bool(sim.get_property_value(c.detect_extreme_state))
        overload = False
        if sim_time_sec > 10.0:
            acceleration_limits = (
                float(getattr(self.config, "acceleration_limit_x", 10.0)),
                float(getattr(self.config, "acceleration_limit_y", 10.0)),
                float(getattr(self.config, "acceleration_limit_z", 10.0)),
            )
            accelerations = (
                abs(float(sim.get_property_value(c.accelerations_n_pilot_x_norm))),
                abs(float(sim.get_property_value(c.accelerations_n_pilot_y_norm))),
                abs(
                    float(sim.get_property_value(c.accelerations_n_pilot_z_norm))
                    + 1.0
                ),
            )
            overload = bool(
                any(
                    value > limit
                    for value, limit in zip(accelerations, acceleration_limits)
                )
            )
        return {
            "metrics_plus_safety_low_altitude": low_altitude,
            "metrics_plus_safety_extreme_state": extreme_state,
            "metrics_plus_safety_overload": overload,
            "metrics_plus_safety_any": bool(
                low_altitude or extreme_state or overload
            ),
        }

    def _record_metrics_plus_substep(self, substep_idx):
        if not self._metrics_plus_substep_trace_enabled:
            return
        for agent_id, sim in self.agents.items():
            qp_log = self._last_action_projector_log.get(agent_id, {})
            vertical_speed_mps = float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048
            roll_rad = float(
                sim.get_property_value(c.attitude_roll_rad)
            )
            pitch_rad = float(
                sim.get_property_value(c.attitude_pitch_rad)
            )
            upright_cos = float(np.cos(roll_rad) * np.cos(pitch_rad))
            body_u_mps = float(
                sim.get_property_value(c.velocities_u_mps)
            )
            body_v_mps = float(
                sim.get_property_value(c.velocities_v_mps)
            )
            body_w_mps = float(
                sim.get_property_value(c.velocities_w_mps)
            )
            body_speed_mps = float(
                np.sqrt(
                    body_u_mps * body_u_mps
                    + body_v_mps * body_v_mps
                    + body_w_mps * body_w_mps
                )
            )
            body_forward_ratio = (
                float(body_u_mps / body_speed_mps)
                if body_speed_mps > 1.0
                else 0.0
            )
            self._last_metrics_plus_substep_trace.append(
                {
                    "decision_step": int(self.current_step),
                    "substep_index": int(substep_idx),
                    "agent_id": str(agent_id),
                    "sim_time_sec": float(sim.get_property_value(c.simulation_sim_time_sec)),
                    "elevator_cmd_norm": float(sim.get_property_value(c.fcs_elevator_cmd_norm)),
                    "elevator_pos_norm": float(sim.get_property_value(c.fcs_elevator_pos_norm)),
                    "vertical_speed_mps": vertical_speed_mps,
                    "pitch_rate_deg_s": float(np.rad2deg(sim.get_property_value(c.velocities_q_rad_sec))),
                    "pitch_accel_deg_s2": float(np.rad2deg(sim.get_property_value(c.accelerations_qdot_rad_sec2))),
                    "aoa_deg": float(sim.get_property_value(c.aero_alpha_deg)),
                    "roll_deg": float(np.rad2deg(roll_rad)),
                    "pitch_deg": float(np.rad2deg(pitch_rad)),
                    "upright_cos": upright_cos,
                    "inverted_indicator": float(upright_cos < 0.0),
                    "deep_inverted_indicator": float(
                        upright_cos < -0.5
                    ),
                    "body_forward_ratio": body_forward_ratio,
                    "negative_body_forward_indicator": float(
                        body_speed_mps > 1.0 and body_forward_ratio < 0.0
                    ),
                    # These targets are held across the 12 simulator substeps;
                    # the suffix prevents treating repeated 5 Hz values as a
                    # genuine 60 Hz signal in spectral analysis.
                    "qp_input_elevator_5hz_zoh": float(
                        qp_log.get("qp_raw_elevator", np.nan)
                    ),
                    "qp_target_elevator_5hz_zoh": float(
                        qp_log.get("qp_action_elevator", np.nan)
                    ),
                }
            )

    def get_heading_metrics_report(self):
        if self._heading_metrics is None:
            return {"episode_metrics": {}, "segment_metrics": []}
        return {
            "episode_metrics": self._heading_metrics.get_episode_metrics(),
            "segment_metrics": self._heading_metrics.get_segment_metrics(),
        }

    def get_last_heading_step(self):
        return None if self._last_heading_step is None else self._last_heading_step.copy()

    def _reset_heading_metrics(self):
        if not isinstance(self.task, HeadingTask):
            self._heading_metrics = None
            self._last_heading_step = None
            return
        agent_id = next(iter(self.agents.keys()))
        self._heading_metrics = HeadingMetricsTracker()
        self._heading_metrics.reset(self._get_heading_target(agent_id))
        self._last_heading_step = None

    def _reset_trim_ema(self):
        agent_id = next(iter(self.agents.keys()))
        sim = self.agents[agent_id]
        sim_time = float(sim.get_property_value(c.simulation_sim_time_sec))
        self._trim_ema_state = {
            "elevator_trim_est": 0.0,
            "aoa_trim_ema": 0.0,
            "initialized": False,
            "aoa_initialized": False,
            "last_target_change_time": sim_time,
        }
        self._last_trim_ema_log = self._make_trim_ema_log(
            gate_on=False,
            policy_elevator=np.nan,
            elevator_out=np.nan,
            elevator_fast_part=np.nan,
            elevator_correction=0.0,
            aoa_deg=np.nan,
            target_age_sec=0.0,
        )

    def _action_smoother_enabled(self):
        return (
            isinstance(self.task, HeadingTask)
            and getattr(self.task, "action_type", "discrete") == "continuous"
            and bool(getattr(self.config, "ActionSmoother_enable", False))
        )

    def _action_projector_enabled(self):
        return (
            isinstance(self.task, HeadingTask)
            and getattr(self.task, "action_type", "discrete") == "continuous"
            and bool(getattr(self.config, "ActionProjector_enable", False))
        )

    def _coordinated_residual_enabled(self):
        return (
            isinstance(self.task, HeadingTask)
            and getattr(self.task, "action_type", "discrete") == "continuous"
            and bool(
                getattr(
                    self.config,
                    "CoordinatedResidual_enable",
                    False,
                )
            )
        )

    def _reset_coordinated_residual(self):
        self._last_coordinated_residual_log = {}
        self._energy_trim_states = {}
        self._turn_vertical_trim_states = {}
        self._turn_bank_feedback_states = {}
        self._steady_bank_feedback_states = {}
        self._vs_bank_target_states = {}
        self._vs_bank_target_v5_states = {}
        if not self._coordinated_residual_enabled():
            return
        mode = str(
            getattr(
                self.config,
                "CoordinatedResidual_mode",
                "",
            )
        ).strip().lower()
        if mode not in (
            "zero_residual",
            "residual",
            "vertical_speed_command",
            "energy_balance_probe",
            "energy_exchange_gain_residual",
            "energy_exchange_fixed_gain",
            "structured_reference_gain_residual",
            "structured_feedback_gain_residual",
            "actuator_residual_fixed_center",
        ):
            raise ValueError(
                "CoordinatedResidual_mode must be "
                "zero_residual, residual, vertical_speed_command, "
                "energy_balance_probe, energy_exchange_gain_residual, "
                "energy_exchange_fixed_gain, or "
                "structured_reference_gain_residual or "
                "structured_feedback_gain_residual or actuator_residual_fixed_center"
            )
        scale = np.asarray(
            getattr(
                self.config,
                "CoordinatedResidual_residual_scale",
                [0.0, 0.0, 0.0, 0.0],
            ),
            dtype=np.float32,
        ).reshape(-1)
        if (
            scale.shape != (4,)
            or not np.all(np.isfinite(scale))
            or np.any(scale < 0.0)
            or np.any(scale > 1.0)
        ):
            raise ValueError(
                "CoordinatedResidual_residual_scale must be a finite "
                "4D vector in [0, 1]"
            )
        if mode == "zero_residual" and np.any(scale != 0.0):
            raise ValueError(
                "zero_residual requires all residual scales to be zero"
            )
        turn_pitch_coupling_gain = float(
            getattr(
                self.config,
                "CoordinatedResidual_turn_pitch_coupling_gain",
                1.0,
            )
        )
        if (
            not np.isfinite(turn_pitch_coupling_gain)
            or turn_pitch_coupling_gain < 0.0
            or turn_pitch_coupling_gain > 1.0
        ):
            raise ValueError(
                "CoordinatedResidual_turn_pitch_coupling_gain "
                "must lie in [0, 1]"
            )
        turn_pitch_actual_rate_blend = float(
            getattr(
                self.config,
                "CoordinatedResidual_turn_pitch_actual_rate_blend",
                0.0,
            )
        )
        if (
            not np.isfinite(turn_pitch_actual_rate_blend)
            or turn_pitch_actual_rate_blend < 0.0
            or turn_pitch_actual_rate_blend > 1.0
        ):
            raise ValueError(
                "CoordinatedResidual_turn_pitch_actual_rate_blend "
                "must lie in [0, 1]"
            )
        pitch_rate_feedback_gain = float(
            getattr(
                self.config,
                "CoordinatedResidual_pitch_rate_feedback_gain",
                0.0,
            )
        )
        if (
            not np.isfinite(pitch_rate_feedback_gain)
            or pitch_rate_feedback_gain < 0.0
            or pitch_rate_feedback_gain > 1.0
        ):
            raise ValueError(
                "CoordinatedResidual_pitch_rate_feedback_gain "
                "must lie in [0, 1]"
            )
        direct_flight_path_gain = float(
            getattr(
                self.config,
                "CoordinatedResidual_direct_flight_path_feedback_gain_per_rad",
                0.0,
            )
        )
        direct_flight_path_limit = float(
            getattr(
                self.config,
                "CoordinatedResidual_direct_flight_path_feedback_max_action",
                0.0,
            )
        )
        if (
            not np.isfinite(direct_flight_path_gain)
            or direct_flight_path_gain < 0.0
        ):
            raise ValueError(
                "CoordinatedResidual_direct_flight_path_feedback_gain_per_rad "
                "must be finite and nonnegative"
            )
        if (
            not np.isfinite(direct_flight_path_limit)
            or direct_flight_path_limit < 0.0
            or direct_flight_path_limit > 1.0
        ):
            raise ValueError(
                "CoordinatedResidual_direct_flight_path_feedback_max_action "
                "must lie in [0, 1]"
            )
        if direct_flight_path_gain > 0.0 and direct_flight_path_limit <= 0.0:
            raise ValueError(
                "positive direct flight-path gain requires a positive limit"
            )
        if (
            mode in (
                "vertical_speed_command",
                "energy_balance_probe",
                "energy_exchange_gain_residual",
                "energy_exchange_fixed_gain",
                "structured_reference_gain_residual",
                "structured_feedback_gain_residual",
            )
            and np.any(scale != 0.0)
        ):
            raise ValueError(
                "{} forbids direct actuator residuals".format(mode)
            )
        allocator_mode = str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "independent",
            )
        ).strip().lower()
        if allocator_mode not in (
            "independent",
            "tecs_v1",
            "total_energy_throttle_v1",
            "adaptive_total_energy_throttle_v1",
        ):
            raise ValueError(
                "CoordinatedResidual_energy_allocator_mode must be "
                "independent, tecs_v1, total_energy_throttle_v1, "
                "or adaptive_total_energy_throttle_v1"
            )
        exchange_mode = str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_exchange_mode",
                "none",
            )
        ).strip().lower()
        if exchange_mode not in (
            "none",
            "measured_opposite_sign_v1",
        ):
            raise ValueError(
                "CoordinatedResidual_energy_exchange_mode must be "
                "none or measured_opposite_sign_v1"
            )
        if mode in (
            "energy_exchange_fixed_gain",
            "structured_reference_gain_residual",
            "structured_feedback_gain_residual",
        ):
            fixed_gain = float(
                getattr(
                    self.config,
                    "CoordinatedResidual_fixed_energy_exchange_gain",
                    np.nan,
                )
            )
            if (
                not np.isfinite(fixed_gain)
                or fixed_gain < 0.0
                or fixed_gain > 2.0
            ):
                raise ValueError(
                    "CoordinatedResidual_fixed_energy_exchange_gain "
                    "must lie in [0, 2]"
                )

    def _coordinated_nominal_action(
        self,
        agent_id,
        *,
        target_rates=None,
        target_vertical_speed_mps=None,
        desired_vertical_speed_offset_mps=0.0,
        structured_feedback_gain_multipliers=None,
    ):
        sim = self.agents[agent_id]
        if target_rates is None:
            target_rates = self.task.get_target_rate_commands(
                self,
                agent_id,
            )
        rates = np.asarray(target_rates, dtype=np.float64).reshape(-1)
        if rates.shape != (3,) or not np.all(np.isfinite(rates)):
            raise RuntimeError(
                "C1 requires a finite heading/altitude/speed target-rate "
                "vector"
            )
        if structured_feedback_gain_multipliers is None:
            feedback_multipliers = np.ones(3, dtype=np.float64)
        else:
            feedback_multipliers = np.asarray(
                structured_feedback_gain_multipliers,
                dtype=np.float64,
            ).reshape(-1)
            if (
                feedback_multipliers.shape != (3,)
                or not np.all(np.isfinite(feedback_multipliers))
                or np.any(feedback_multipliers <= 0.0)
            ):
                raise ValueError(
                    "structured feedback multipliers must be a positive "
                    "finite 3D vector"
                )
        effective_course_tau_sec = float(
            getattr(
                self.config,
                "CoordinatedResidual_course_tau_sec",
                10.0,
            )
        ) / float(feedback_multipliers[0])
        effective_altitude_tau_sec = float(
            getattr(
                self.config,
                "CoordinatedResidual_altitude_tau_sec",
                20.0,
            )
        ) / float(feedback_multipliers[1])
        effective_total_energy_tau_sec = float(
            getattr(
                self.config,
                "CoordinatedResidual_total_energy_tau_sec",
                20.0,
            )
        ) / float(feedback_multipliers[2])
        route_target_vertical_speed_mps = float(rates[1])
        if target_vertical_speed_mps is None:
            used_target_vertical_speed_mps = (
                route_target_vertical_speed_mps
            )
        else:
            used_target_vertical_speed_mps = float(
                target_vertical_speed_mps
            )
            if not np.isfinite(used_target_vertical_speed_mps):
                raise ValueError(
                    "target_vertical_speed_mps must be finite"
                )

        measured = self._controller_measurements(agent_id)
        horizontal_speed_mps = measured["horizontal_speed_mps"]
        course_error_deg = measured["course_error_deg"]
        altitude_error_m = measured["altitude_error_m"]
        vertical_speed_mps = measured["vertical_speed_mps"]
        turn_pitch_rate_source_configured = hasattr(
            self.config,
            "CoordinatedResidual_turn_pitch_actual_rate_blend",
        )
        turn_pitch_actual_rate_blend = float(
            getattr(
                self.config,
                "CoordinatedResidual_turn_pitch_actual_rate_blend",
                0.0,
            )
        )
        measured_heading_rate_deg_s = None
        if turn_pitch_rate_source_configured:
            sim = self.agents[agent_id]
            measured_heading_rate_deg_s = euler_heading_rate_deg_s(
                roll_rad=float(np.deg2rad(measured["roll_deg"])),
                pitch_rad=float(np.deg2rad(measured["pitch_deg"])),
                pitch_rate_deg_s=measured["pitch_rate_deg_s"],
                yaw_rate_deg_s=float(
                    np.rad2deg(
                        sim.get_property_value(c.velocities_r_rad_sec)
                    )
                ),
            )
        configured_flight_path_tau_sec = float(
            getattr(
                self.config,
                "CoordinatedResidual_flight_path_tau_sec",
                2.0,
            )
        )

        flight_path_tau_log = {
            "active": 0.0,
            "speed_mps": horizontal_speed_mps,
            "altitude_m": measured["altitude_m"],
            "speed_blend": 0.0,
            "altitude_blend": 0.0,
            "high_speed_tau_sec": configured_flight_path_tau_sec,
            "used_tau_sec": configured_flight_path_tau_sec,
        }
        if bool(
            getattr(
                self.config,
                "CoordinatedResidual_flight_path_tau_schedule_enable",
                False,
            )
        ):
            configured_flight_path_tau_sec, flight_path_tau_log = (
                scheduled_flight_path_tau(
                    speed_mps=horizontal_speed_mps,
                    altitude_m=flight_path_tau_log["altitude_m"],
                    low_speed_tau_sec=getattr(
                        self.config,
                        "CoordinatedResidual_flight_path_tau_low_speed_sec",
                        0.9,
                    ),
                    high_speed_low_alt_tau_sec=getattr(
                        self.config,
                        "CoordinatedResidual_flight_path_tau_high_speed_low_alt_sec",
                        0.97,
                    ),
                    high_speed_high_alt_tau_sec=getattr(
                        self.config,
                        "CoordinatedResidual_flight_path_tau_high_speed_high_alt_sec",
                        1.0,
                    ),
                    speed_blend_start_mps=getattr(
                        self.config,
                        "CoordinatedResidual_flight_path_tau_speed_blend_start_mps",
                        240.0,
                    ),
                    speed_blend_end_mps=getattr(
                        self.config,
                        "CoordinatedResidual_flight_path_tau_speed_blend_end_mps",
                        260.0,
                    ),
                    altitude_blend_start_m=getattr(
                        self.config,
                        "CoordinatedResidual_flight_path_tau_altitude_blend_start_m",
                        5500.0,
                    ),
                    altitude_blend_end_m=getattr(
                        self.config,
                        "CoordinatedResidual_flight_path_tau_altitude_blend_end_m",
                        6500.0,
                    ),
                )
            )
        sideslip_deg = measured["sideslip_deg"]
        sideslip_rudder_gain_per_deg = float(
            getattr(
                self.config,
                "CoordinatedResidual_sideslip_rudder_gain_per_deg",
                0.0,
            )
        )
        max_rudder_action = float(
            getattr(
                self.config,
                "CoordinatedResidual_max_rudder_action",
                1.0,
            )
        )
        energy_allocator_mode = str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "independent",
            )
        ).strip().lower()
        energy_allocator_log = None
        energy_exchange_log = None
        desired_vertical_speed_override_mps = None
        throttle_action_override = None
        if energy_allocator_mode in (
            "tecs_v1",
            "total_energy_throttle_v1",
            "adaptive_total_energy_throttle_v1",
        ):
            transition = (
                self.task.get_baiyepower_v2_transition_context(agent_id)
            )
            energy_terms = self._controller_energy_terms(agent_id, transition)
            configured_trim_throttle = float(
                getattr(
                    self.config,
                    "CoordinatedResidual_trim_physical_throttle",
                    0.65,
                )
            )
            adaptive_trim_log = None
            used_trim_throttle = configured_trim_throttle
            if (
                energy_allocator_mode
                == "adaptive_total_energy_throttle_v1"
            ):
                sim_time_sec = float(
                    sim.get_property_value(c.simulation_sim_time_sec)
                )
                energy_height_m = float(energy_terms.energy_height_m)
                trim_measurement_filter_tau_sec = getattr(
                    self.config,
                    (
                        "CoordinatedResidual_"
                        "energy_trim_measurement_filter_tau_sec"
                    ),
                    None,
                )
                if trim_measurement_filter_tau_sec is not None:
                    trim_measurement_filter_tau_sec = float(
                        trim_measurement_filter_tau_sec
                    )
                    if (
                        not np.isfinite(trim_measurement_filter_tau_sec)
                        or trim_measurement_filter_tau_sec <= 0.0
                    ):
                        raise ValueError(
                            "energy trim measurement-filter tau must be "
                            "finite and positive"
                        )
                trim_state = self._energy_trim_states.get(agent_id)
                if trim_state is None:
                    trim_state = {
                        "estimated_trim_throttle": (
                            configured_trim_throttle
                        ),
                        "previous_energy_height_m": energy_height_m,
                        "filtered_energy_height_m": energy_height_m,
                        "previous_time_sec": sim_time_sec,
                    }
                else:
                    dt_sec = sim_time_sec - float(
                        trim_state["previous_time_sec"]
                    )
                    if dt_sec > 1e-9:
                        previous_filtered_energy_height_m = float(
                            trim_state.get(
                                "filtered_energy_height_m",
                                trim_state["previous_energy_height_m"],
                            )
                        )
                        if trim_measurement_filter_tau_sec is None:
                            filtered_energy_height_m = energy_height_m
                            measurement_filter_alpha = 1.0
                            rate_reference_energy_height_m = float(
                                trim_state["previous_energy_height_m"]
                            )
                        else:
                            measurement_filter_alpha = (
                                1.0
                                - math.exp(
                                    -dt_sec
                                    / trim_measurement_filter_tau_sec
                                )
                            )
                            filtered_energy_height_m = (
                                previous_filtered_energy_height_m
                                + measurement_filter_alpha
                                * (
                                    energy_height_m
                                    - previous_filtered_energy_height_m
                                )
                            )
                            rate_reference_energy_height_m = (
                                previous_filtered_energy_height_m
                            )
                        observed_energy_rate_mps = (
                            filtered_energy_height_m
                            - rate_reference_energy_height_m
                        ) / dt_sec
                        (
                            trim_state["estimated_trim_throttle"],
                            adaptive_trim_log,
                        ) = update_energy_trim_estimate(
                            previous_trim_throttle=trim_state[
                                "estimated_trim_throttle"
                            ],
                            actual_throttle=float(
                                sim.get_property_value(
                                    c.fcs_throttle_cmd_norm
                                )
                            ),
                            observed_energy_rate_mps=(
                                observed_energy_rate_mps
                            ),
                            plant_gain_energy_rate_per_throttle=float(
                                getattr(
                                    self.config,
                                    (
                                        "CoordinatedResidual_"
                                        "energy_rate_plant_gain"
                                    ),
                                    313.38932658872494,
                                )
                            ),
                            dt_sec=dt_sec,
                            adaptation_tau_sec=float(
                                getattr(
                                    self.config,
                                    (
                                        "CoordinatedResidual_"
                                        "energy_trim_adaptation_tau_sec"
                                    ),
                                    5.0,
                                )
                            ),
                            min_trim_throttle=float(
                                getattr(
                                    self.config,
                                    (
                                        "CoordinatedResidual_"
                                        "min_adaptive_trim_throttle"
                                    ),
                                    0.35,
                                )
                            ),
                            max_trim_throttle=float(
                                getattr(
                                    self.config,
                                    (
                                        "CoordinatedResidual_"
                                        "max_adaptive_trim_throttle"
                                    ),
                                    0.75,
                                )
                            ),
                            max_trim_innovation=(
                                float(
                                    getattr(
                                        self.config,
                                        (
                                            "CoordinatedResidual_"
                                            "energy_trim_max_innovation"
                                        ),
                                    )
                                )
                                if hasattr(
                                    self.config,
                                    (
                                        "CoordinatedResidual_"
                                        "energy_trim_max_innovation"
                                    ),
                                )
                                else None
                            ),
                        )
                        adaptive_trim_log.update(
                            {
                                "raw_energy_height_m": energy_height_m,
                                "filtered_energy_height_m": (
                                    filtered_energy_height_m
                                ),
                                "measurement_filter_tau_sec": (
                                    trim_measurement_filter_tau_sec
                                    if trim_measurement_filter_tau_sec
                                    is not None
                                    else 0.0
                                ),
                                "measurement_filter_alpha": (
                                    measurement_filter_alpha
                                ),
                                "measurement_filter_active": float(
                                    trim_measurement_filter_tau_sec
                                    is not None
                                ),
                            }
                        )
                        trim_state["filtered_energy_height_m"] = (
                            filtered_energy_height_m
                        )
                    trim_state["previous_energy_height_m"] = (
                        energy_height_m
                    )
                    trim_state["previous_time_sec"] = sim_time_sec
                self._energy_trim_states[agent_id] = trim_state
                used_trim_throttle = float(
                    trim_state["estimated_trim_throttle"]
                )
            (
                desired_vertical_speed_override_mps,
                throttle_action_override,
                energy_allocator_log,
            ) = tecs_energy_allocation_commands(
                target_vertical_speed_mps=(
                    used_target_vertical_speed_mps
                ),
                target_speed_mps=transition.target_speed_mps,
                target_speed_rate_mps2=float(rates[2]),
                total_energy_error_m=energy_terms.e_T_m,
                energy_balance_error_m=energy_terms.e_B_m,
                total_energy_tau_sec=effective_total_energy_tau_sec,
                energy_balance_tau_sec=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_balance_tau_sec",
                        20.0,
                    )
                ),
                throttle_energy_rate_gain=float(
                    getattr(
                        self.config,
                        (
                            "CoordinatedResidual_"
                            "throttle_energy_rate_gain"
                        ),
                        0.008,
                    )
                ),
                trim_physical_throttle=used_trim_throttle,
                policy_throttle_center=float(
                    getattr(
                        self.config,
                        (
                            "CoordinatedResidual_"
                            "policy_throttle_center"
                        ),
                        0.65,
                    )
                ),
                policy_throttle_half_range=float(
                    getattr(
                        self.config,
                        (
                            "CoordinatedResidual_"
                            "policy_throttle_half_range"
                        ),
                        0.25,
                    )
                ),
                max_balance_correction_mps=float(
                    getattr(
                        self.config,
                        (
                            "CoordinatedResidual_"
                            "max_balance_correction_mps"
                        ),
                        10.0,
                    )
                ),
                gravity_mps2=float(
                    getattr(
                        self.config,
                        "BaiyePowerV2_gravity_mps2",
                        9.80665,
                    )
                ),
            )
            if energy_allocator_mode in (
                "total_energy_throttle_v1",
                "adaptive_total_energy_throttle_v1",
            ):
                desired_vertical_speed_override_mps = None
            energy_allocator_log["configured_trim_throttle"] = (
                configured_trim_throttle
            )
            energy_allocator_log["used_trim_throttle"] = (
                used_trim_throttle
            )
            if adaptive_trim_log is not None:
                for name, value in adaptive_trim_log.items():
                    energy_allocator_log[
                        "adaptive_{}".format(name)
                    ] = value
        energy_exchange_mode = str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_exchange_mode",
                "none",
            )
        ).strip().lower()
        if energy_exchange_mode == "measured_opposite_sign_v1":
            if energy_allocator_log is None:
                raise RuntimeError(
                    "measured energy exchange requires energy terms"
                )
            exchange_offset_mps, energy_exchange_log = (
                measured_energy_exchange_offset(
                    height_energy_error_m=energy_terms.e_h_m,
                    kinetic_energy_error_m=energy_terms.e_k_m,
                    closure_tau_sec=float(
                        getattr(
                            self.config,
                            (
                                "CoordinatedResidual_"
                                "energy_exchange_tau_sec"
                            ),
                            60.0,
                        )
                    ),
                    height_rate_gain=float(
                        getattr(
                            self.config,
                            (
                                "CoordinatedResidual_"
                                "energy_exchange_height_rate_gain"
                            ),
                            0.9042178500025742,
                        )
                    ),
                    kinetic_rate_gain_magnitude=float(
                        getattr(
                            self.config,
                            (
                                "CoordinatedResidual_"
                                "energy_exchange_kinetic_rate_gain"
                            ),
                            0.6609097763737128,
                        )
                    ),
                    max_offset_mps=float(
                        getattr(
                            self.config,
                            (
                                "CoordinatedResidual_"
                                "max_energy_exchange_offset_mps"
                            ),
                            1.0,
                        )
                    ),
                )
            )
            desired_vertical_speed_offset_mps = float(
                desired_vertical_speed_offset_mps
            ) + exchange_offset_mps
        nominal, direct_flight_path_log = coordinated_reference_action(
            target_course_rate_deg_s=float(rates[0]),
            course_error_deg=course_error_deg,
            horizontal_speed_mps=horizontal_speed_mps,
            roll_rad=float(np.deg2rad(measured["roll_deg"])),
            pitch_rad=float(np.deg2rad(measured["pitch_deg"])),
            vertical_speed_mps=vertical_speed_mps,
            target_vertical_speed_mps=used_target_vertical_speed_mps,
            altitude_error_m=altitude_error_m,
            speed_error_mps=measured["speed_error_mps"],
            course_tau_sec=effective_course_tau_sec,
            bank_tau_sec=float(
                getattr(
                    self.config,
                    "CoordinatedResidual_bank_tau_sec",
                    1.5,
                )
            ),
            flight_path_tau_sec=configured_flight_path_tau_sec,
            altitude_tau_sec=effective_altitude_tau_sec,
            speed_throttle_gain=float(
                getattr(
                    self.config,
                    "CoordinatedResidual_speed_throttle_gain",
                    0.01,
                )
            ),
            max_course_rate_deg_s=float(
                getattr(
                    self.config,
                    "CoordinatedResidual_max_course_rate_deg_s",
                    3.0,
                )
            ),
            max_bank_deg=float(
                getattr(
                    self.config,
                    "CoordinatedResidual_max_bank_deg",
                    45.0,
                )
            ),
            max_altitude_correction_mps=float(
                getattr(
                    self.config,
                    "CoordinatedResidual_max_altitude_correction_mps",
                    10.0,
                )
            ),
            desired_vertical_speed_override_mps=(
                desired_vertical_speed_override_mps
            ),
            desired_vertical_speed_offset_mps=(
                desired_vertical_speed_offset_mps
            ),
            turn_pitch_coupling_gain=float(
                getattr(
                    self.config,
                    "CoordinatedResidual_turn_pitch_coupling_gain",
                    1.0,
                )
            ),
            actual_heading_rate_deg_s=measured_heading_rate_deg_s,
            turn_pitch_actual_rate_blend=turn_pitch_actual_rate_blend,
            actual_pitch_rate_deg_s=measured["pitch_rate_deg_s"],
            pitch_rate_feedback_gain=float(
                getattr(
                    self.config,
                    "CoordinatedResidual_pitch_rate_feedback_gain",
                    0.0,
                )
            ),
            direct_flight_path_feedback_gain_per_rad=float(
                getattr(
                    self.config,
                    (
                        "CoordinatedResidual_direct_flight_path_"
                        "feedback_gain_per_rad"
                    ),
                    0.0,
                )
            ),
            direct_flight_path_feedback_max_action=float(
                getattr(
                    self.config,
                    (
                        "CoordinatedResidual_direct_flight_path_"
                        "feedback_max_action"
                    ),
                    0.0,
                )
            ),
            throttle_action_override=throttle_action_override,
            sideslip_deg=sideslip_deg,
            sideslip_rudder_gain_per_deg=(
                sideslip_rudder_gain_per_deg
            ),
            max_rudder_action=max_rudder_action,
            return_diagnostics=True,
            flight_path_geometry=getattr(
                self.config, "CoordinatedResidual_flight_path_geometry",
                "legacy_asin_horizontal",
            ),
        )
        context = {
            "target_heading_rate_deg_s": float(rates[0]),
            "target_altitude_rate_mps": route_target_vertical_speed_mps,
            "commanded_target_altitude_rate_mps": (
                used_target_vertical_speed_mps
            ),
            "target_altitude_rate_residual_mps": float(
                used_target_vertical_speed_mps
                - route_target_vertical_speed_mps
            ),
            "desired_vertical_speed_offset_mps": float(
                desired_vertical_speed_offset_mps
            ),
            "turn_pitch_coupling_gain": float(
                getattr(
                    self.config,
                    "CoordinatedResidual_turn_pitch_coupling_gain",
                    1.0,
                )
            ),
            "measured_heading_rate_deg_s": float(
                measured_heading_rate_deg_s
                if measured_heading_rate_deg_s is not None
                else np.nan
            ),
            "turn_pitch_actual_rate_blend": turn_pitch_actual_rate_blend,
            "pitch_rate_feedback_gain": float(
                getattr(
                    self.config,
                    "CoordinatedResidual_pitch_rate_feedback_gain",
                    0.0,
                )
            ),
            "direct_flight_path_log": direct_flight_path_log,
            "target_speed_rate_mps2": float(rates[2]),
            "structured_feedback_gain_multipliers": (
                feedback_multipliers.copy()
            ),
            "effective_course_tau_sec": effective_course_tau_sec,
            "effective_altitude_tau_sec": effective_altitude_tau_sec,
            "effective_total_energy_tau_sec": (
                effective_total_energy_tau_sec
            ),
            "course_error_deg": course_error_deg,
            "altitude_error_m": altitude_error_m,
            "speed_error_mps": measured["speed_error_mps"],
            "flight_path_tau_log": flight_path_tau_log,
            "energy_allocator_mode": energy_allocator_mode,
            "energy_allocator_log": energy_allocator_log,
            "energy_exchange_mode": energy_exchange_mode,
            "energy_exchange_log": energy_exchange_log,
            "sideslip_deg": sideslip_deg,
            "sideslip_rudder_feedback_action": float(
                np.clip(
                    sideslip_rudder_gain_per_deg * sideslip_deg,
                    -max_rudder_action,
                    max_rudder_action,
                )
            ),
        }
        if energy_allocator_log is not None:
            context["energy_errors"] = {
                "height_error_m": float(energy_terms.e_h_m),
                "kinetic_error_m": float(energy_terms.e_k_m),
                "total_error_m": float(energy_terms.e_T_m),
                "balance_error_m": float(energy_terms.e_B_m),
            }
        return nominal, context

    def _direct_student_teacher_enabled(self):
        config = getattr(self, "config", None)
        return bool(
            getattr(config, "DirectStudentTeacher_enable", False)
            or getattr(config, "DirectStudentTeacherKL_enable", False)
        )

    def _apply_direct_student_teacher_mix(self, agent_id, student_action):
        """Label and optionally mix a direct student action on the same state.

        This path is opt-in data-collection instrumentation. Normal C21
        deployment configs keep it disabled, so no analytic action is
        evaluated or applied during a direct-student audit.
        """

        student = np.asarray(student_action, dtype=np.float32).reshape(-1)
        if student.shape != (4,) or not np.all(np.isfinite(student)):
            raise ValueError("direct student action must be a finite 4D vector")
        if (
            not self._direct_student_teacher_enabled()
            and getattr(self.task, "action_type", "discrete") == "discrete"
        ):
            return student
        student = np.clip(student, -1.0, 1.0)
        if not self._direct_student_teacher_enabled():
            return student
        if self._coordinated_residual_enabled():
            raise RuntimeError(
                "direct-student teacher collection requires analytic deployment off"
            )
        student_fraction = float(
            getattr(self.config, "DirectStudentTeacher_student_fraction", np.nan)
        )
        if (
            not np.isfinite(student_fraction)
            or student_fraction < 0.0
            or student_fraction > 1.0
        ):
            raise ValueError("student fraction must lie in [0, 1]")
        replace_mask = np.asarray(
            getattr(
                self.config,
                "DirectStudentTeacher_replace_mask",
                np.zeros(4, dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        if (
            replace_mask.shape != (4,)
            or not np.all(np.isfinite(replace_mask))
            or not np.all((replace_mask == 0.0) | (replace_mask == 1.0))
        ):
            raise ValueError(
                "direct-student teacher replace mask must be four binary values"
            )

        # Reuse the frozen C20-D3 composer on this exact transition state.
        # The temporary flag is restored before the actual direct-action path
        # proceeds, so deployment action semantics remain explicit.
        original = getattr(self.config, "CoordinatedResidual_enable", False)
        self.config.CoordinatedResidual_enable = True
        try:
            teacher = self._apply_coordinated_residual(
                agent_id,
                np.zeros(4, dtype=np.float32),
            )
        finally:
            self.config.CoordinatedResidual_enable = original
        teacher = np.asarray(teacher, dtype=np.float32).reshape(4)
        mixed = (
            (1.0 - student_fraction) * teacher
            + student_fraction * student
        ).astype(np.float32)
        executed = np.where(replace_mask > 0.5, teacher, mixed).astype(np.float32)
        self._last_direct_student_teacher_log[agent_id] = {
            "enabled": 1.0,
            "student_fraction": student_fraction,
            "replace_mask": replace_mask.copy(),
            "student_action": student.copy(),
            "teacher_action": teacher.copy(),
            "executed_action": executed.copy(),
        }
        return executed

    def get_direct_student_teacher_log(self, agent_id):
        logs = getattr(self, "_last_direct_student_teacher_log", {})
        log = logs.get(agent_id, {}) if isinstance(logs, dict) else {}
        if not isinstance(log, dict):
            return {}
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in log.items()
        }

    def _apply_coordinated_residual(self, agent_id, policy_action):
        residual_input = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        if not self._coordinated_residual_enabled():
            return residual_input
        if residual_input.shape != (4,) or not np.all(
            np.isfinite(residual_input)
        ):
            raise ValueError(
                "continuous policy action must be a finite 4D vector"
            )
        residual_input = np.clip(residual_input, -1.0, 1.0)
        self._sample_controller_measurements(agent_id)

        mode = str(
            getattr(
                self.config,
                "CoordinatedResidual_mode",
                "",
            )
        ).strip().lower()
        command_log = None
        structured_reference_log = None
        structured_feedback_log = None
        composition_mode = mode
        residual_scale = getattr(
            self.config,
            "CoordinatedResidual_residual_scale",
            [0.0, 0.0, 0.0, 0.0],
        )
        if mode == "vertical_speed_command":
            target_rates = np.asarray(
                self.task.get_target_rate_commands(self, agent_id),
                dtype=np.float64,
            ).reshape(-1)
            if (
                target_rates.shape != (3,)
                or not np.all(np.isfinite(target_rates))
            ):
                raise RuntimeError(
                    "C3 requires a finite heading/altitude/speed "
                    "target-rate vector"
                )
            commanded_target, command_log = (
                compose_vertical_speed_command_residual(
                    float(target_rates[1]),
                    residual_input,
                    action_index=getattr(
                        self.config,
                        "CoordinatedResidual_command_action_index",
                        3,
                    ),
                    scale_mps=getattr(
                        self.config,
                        (
                            "CoordinatedResidual_"
                            "vertical_speed_command_scale_mps"
                        ),
                        5.0,
                    ),
                )
            )
            nominal, context = self._coordinated_nominal_action(
                agent_id,
                target_rates=target_rates,
                target_vertical_speed_mps=commanded_target,
            )
            composition_mode = "zero_residual"
            residual_scale = np.zeros(4, dtype=np.float32)
        elif mode == "energy_balance_probe":
            _, command_log = compose_vertical_speed_command_residual(
                0.0,
                residual_input,
                action_index=getattr(
                    self.config,
                    "CoordinatedResidual_command_action_index",
                    3,
                ),
                scale_mps=getattr(
                    self.config,
                    (
                        "CoordinatedResidual_"
                        "vertical_speed_command_scale_mps"
                    ),
                    2.0,
                ),
            )
            nominal, context = self._coordinated_nominal_action(
                agent_id,
                desired_vertical_speed_offset_mps=(
                    command_log["residual_mps"]
                ),
            )
            composition_mode = "zero_residual"
            residual_scale = np.zeros(4, dtype=np.float32)
        elif mode == "energy_exchange_gain_residual":
            transition = self.task.get_baiyepower_v2_transition_context(
                agent_id
            )
            energy_terms = self._controller_energy_terms(agent_id, transition)
            base_exchange_offset_mps, _ = measured_energy_exchange_offset(
                height_energy_error_m=energy_terms.e_h_m,
                kinetic_energy_error_m=energy_terms.e_k_m,
                closure_tau_sec=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_tau_sec",
                        60.0,
                    )
                ),
                height_rate_gain=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_height_rate_gain",
                        0.9042178500025742,
                    )
                ),
                kinetic_rate_gain_magnitude=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_kinetic_rate_gain",
                        0.6609097763737128,
                    )
                ),
                max_offset_mps=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_max_energy_exchange_offset_mps",
                        1.0,
                    )
                ),
            )
            learned_offset_mps, command_log = (
                compose_energy_exchange_gain_residual(
                    base_exchange_offset_mps,
                    residual_input,
                    action_index=getattr(
                        self.config,
                        "CoordinatedResidual_command_action_index",
                        3,
                    ),
                    max_fraction=getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_gain_fraction",
                        0.25,
                    ),
                )
            )
            nominal, context = self._coordinated_nominal_action(
                agent_id,
                desired_vertical_speed_offset_mps=learned_offset_mps,
            )
            composition_mode = "zero_residual"
            residual_scale = np.zeros(4, dtype=np.float32)
        elif mode == "energy_exchange_fixed_gain":
            transition = self.task.get_baiyepower_v2_transition_context(
                agent_id
            )
            energy_terms = self._controller_energy_terms(agent_id, transition)
            base_exchange_offset_mps, _ = measured_energy_exchange_offset(
                height_energy_error_m=energy_terms.e_h_m,
                kinetic_energy_error_m=energy_terms.e_k_m,
                closure_tau_sec=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_tau_sec",
                        60.0,
                    )
                ),
                height_rate_gain=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_height_rate_gain",
                        0.9042178500025742,
                    )
                ),
                kinetic_rate_gain_magnitude=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_kinetic_rate_gain",
                        0.6609097763737128,
                    )
                ),
                max_offset_mps=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_max_energy_exchange_offset_mps",
                        1.0,
                    )
                ),
            )
            fixed_offset_mps, command_log = (
                compose_energy_exchange_fixed_gain(
                    base_exchange_offset_mps,
                    getattr(
                        self.config,
                        "CoordinatedResidual_fixed_energy_exchange_gain",
                        1.0,
                    ),
                )
            )
            nominal, context = self._coordinated_nominal_action(
                agent_id,
                desired_vertical_speed_offset_mps=fixed_offset_mps,
            )
            composition_mode = "zero_residual"
            residual_scale = np.zeros(4, dtype=np.float32)
        elif mode == "structured_reference_gain_residual":
            base_target_rates = np.asarray(
                self.task.get_target_rate_commands(self, agent_id),
                dtype=np.float64,
            ).reshape(-1)
            if (
                base_target_rates.shape != (3,)
                or not np.all(np.isfinite(base_target_rates))
            ):
                raise RuntimeError(
                    "C13 requires a finite heading/altitude/speed "
                    "target-rate vector"
                )
            target_rates, structured_reference_log = (
                compose_structured_reference_gain_residual(
                    base_target_rates,
                    residual_input,
                    reference_gain_fractions=getattr(
                        self.config,
                        "CoordinatedResidual_structured_reference_gain_fractions",
                        [0.10, 0.15, 0.15],
                    ),
                    energy_exchange_gain_fraction=getattr(
                        self.config,
                        "CoordinatedResidual_structured_energy_exchange_gain_fraction",
                        0.10,
                    ),
                )
            )
            transition = self.task.get_baiyepower_v2_transition_context(
                agent_id
            )
            energy_terms = self._controller_energy_terms(agent_id, transition)
            base_exchange_offset_mps, _ = measured_energy_exchange_offset(
                height_energy_error_m=energy_terms.e_h_m,
                kinetic_energy_error_m=energy_terms.e_k_m,
                closure_tau_sec=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_tau_sec",
                        60.0,
                    )
                ),
                height_rate_gain=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_height_rate_gain",
                        0.9042178500025742,
                    )
                ),
                kinetic_rate_gain_magnitude=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_kinetic_rate_gain",
                        0.6609097763737128,
                    )
                ),
                max_offset_mps=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_max_energy_exchange_offset_mps",
                        1.0,
                    )
                ),
            )
            effective_gain = float(
                getattr(
                    self.config,
                    "CoordinatedResidual_fixed_energy_exchange_gain",
                    1.4,
                )
            ) * float(
                structured_reference_log[
                    "energy_exchange_gain_multiplier"
                ]
            )
            structured_reference_log[
                "energy_exchange_effective_gain"
            ] = effective_gain
            fixed_offset_mps, fixed_gain_log = (
                compose_energy_exchange_fixed_gain(
                    base_exchange_offset_mps,
                    effective_gain,
                )
            )
            structured_reference_log[
                "energy_exchange_base_offset_mps"
            ] = float(base_exchange_offset_mps)
            structured_reference_log[
                "energy_exchange_effective_offset_mps"
            ] = float(
                fixed_gain_log["effective_energy_exchange_offset_mps"]
            )
            structured_reference_log[
                "energy_exchange_residual_mps"
            ] = float(fixed_offset_mps)
            nominal, context = self._coordinated_nominal_action(
                agent_id,
                target_rates=target_rates,
                desired_vertical_speed_offset_mps=fixed_offset_mps,
            )
            composition_mode = "zero_residual"
            residual_scale = np.zeros(4, dtype=np.float32)
        elif mode in ("structured_feedback_gain_residual", "actuator_residual_fixed_center"):
            base_target_rates = np.asarray(
                self.task.get_target_rate_commands(self, agent_id),
                dtype=np.float64,
            ).reshape(-1)
            if (
                base_target_rates.shape != (3,)
                or not np.all(np.isfinite(base_target_rates))
            ):
                raise RuntimeError(
                    "C13-S1 requires a finite heading/altitude/speed "
                    "target-rate vector"
                )
            structured_action, structured_action_mask_log = (
                apply_structured_feedback_action_mask(
                    (np.zeros(4) if mode == "actuator_residual_fixed_center" else residual_input),
                    getattr(
                        self.config,
                        "CoordinatedResidual_structured_feedback_action_mask",
                        [1.0, 1.0, 1.0, 1.0],
                    ),
                )
            )
            feedback_multipliers, structured_feedback_log = (
                compose_structured_feedback_gain_residual(
                    structured_action,
                    feedback_gain_fractions=getattr(
                        self.config,
                        "CoordinatedResidual_structured_feedback_gain_fractions",
                        [0.10, 0.15, 0.15],
                    ),
                    energy_exchange_gain_fraction=getattr(
                        self.config,
                        "CoordinatedResidual_structured_energy_exchange_gain_fraction",
                        0.10,
                    ),
                )
            )
            structured_feedback_log.update(structured_action_mask_log)
            transition = self.task.get_baiyepower_v2_transition_context(
                agent_id
            )
            energy_terms = self._controller_energy_terms(agent_id, transition)
            base_exchange_offset_mps, _ = measured_energy_exchange_offset(
                height_energy_error_m=energy_terms.e_h_m,
                kinetic_energy_error_m=energy_terms.e_k_m,
                closure_tau_sec=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_tau_sec",
                        60.0,
                    )
                ),
                height_rate_gain=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_height_rate_gain",
                        0.9042178500025742,
                    )
                ),
                kinetic_rate_gain_magnitude=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_energy_exchange_kinetic_rate_gain",
                        0.6609097763737128,
                    )
                ),
                max_offset_mps=float(
                    getattr(
                        self.config,
                        "CoordinatedResidual_max_energy_exchange_offset_mps",
                        1.0,
                    )
                ),
            )
            effective_gain = float(
                getattr(
                    self.config,
                    "CoordinatedResidual_fixed_energy_exchange_gain",
                    1.4,
                )
            ) * float(
                structured_feedback_log[
                    "energy_exchange_gain_multiplier"
                ]
            )
            fixed_offset_mps, fixed_gain_log = (
                compose_energy_exchange_fixed_gain(
                    base_exchange_offset_mps,
                    effective_gain,
                )
            )
            structured_feedback_log.update(
                {
                    "energy_exchange_effective_gain": effective_gain,
                    "energy_exchange_base_offset_mps": float(
                        base_exchange_offset_mps
                    ),
                    "energy_exchange_effective_offset_mps": float(
                        fixed_gain_log[
                            "effective_energy_exchange_offset_mps"
                        ]
                    ),
                    "energy_exchange_residual_mps": float(
                        fixed_offset_mps
                    ),
                    "effective_course_tau_sec": float(
                        getattr(
                            self.config,
                            "CoordinatedResidual_course_tau_sec",
                            10.0,
                        )
                    ) / float(feedback_multipliers[0]),
                    "effective_altitude_tau_sec": float(
                        getattr(
                            self.config,
                            "CoordinatedResidual_altitude_tau_sec",
                            20.0,
                        )
                    ) / float(feedback_multipliers[1]),
                    "effective_total_energy_tau_sec": float(
                        getattr(
                            self.config,
                            "CoordinatedResidual_total_energy_tau_sec",
                            20.0,
                        )
                    ) / float(feedback_multipliers[2]),
                }
            )
            nominal, context = self._coordinated_nominal_action(
                agent_id,
                target_rates=base_target_rates,
                desired_vertical_speed_offset_mps=fixed_offset_mps,
                structured_feedback_gain_multipliers=(
                    feedback_multipliers
                ),
            )
            composition_mode = "zero_residual"
            residual_scale = np.zeros(4, dtype=np.float32)
        else:
            nominal, context = self._coordinated_nominal_action(agent_id)
        turn_trim_log = None
        if bool(getattr(self.config, "CoordinatedResidual_turn_vertical_trim_enable", False)):
            if (getattr(self.config, "task", None) != "baiyepower_maneuver_c31_v1"
                    or mode != "actuator_residual_fixed_center"):
                raise ValueError("turn vertical trim is isolated to C31-V1")
            measured = self._controller_measurements(agent_id)
            previous_qp = self._last_action_projector_log.get(agent_id, {})
            sim = self.agents[agent_id]
            trim_state, turn_trim_log = update_turn_vertical_trim(
                self._turn_vertical_trim_states.get(agent_id),
                dt_sec=self.time_interval,
                target_turn_rate_deg_s=context["target_heading_rate_deg_s"],
                roll_deg=measured["roll_deg"],
                vertical_speed_mps=measured["vertical_speed_mps"],
                target_vertical_speed_mps=context["commanded_target_altitude_rate_mps"],
                speed_error_mps=measured["speed_error_mps"],
                qp_elevator_limited=bool(previous_qp.get("qp_active_elevator", 0.0)),
                fcs_elevator_pos_norm=float(sim.get_property_value(c.fcs_elevator_pos_norm)),
            )
            self._turn_vertical_trim_states[agent_id] = trim_state
            nominal = np.asarray(nominal, dtype=np.float32).copy()
            base_elevator = float(nominal[1])
            nominal[1] = np.float32(np.clip(
                base_elevator + trim_state["bias_action"], -1.0, 1.0,
            ))
            turn_trim_log["base_elevator_action"] = base_elevator
            turn_trim_log["commanded_elevator_action"] = float(nominal[1])
            turn_trim_log["speed_error_mps"] = float(measured["speed_error_mps"])
            turn_trim_log["total_energy_error_m"] = float(
                context["energy_errors"]["total_error_m"]
            )
        turn_bank_log = None
        if bool(getattr(self.config, "CoordinatedResidual_turn_bank_feedback_enable", False)):
            if (getattr(self.config, "task", None) != "baiyepower_maneuver_c31_v2"
                    or mode != "actuator_residual_fixed_center" or turn_trim_log is not None):
                raise ValueError("turn bank feedback is isolated to C31-V2")
            measured = self._controller_measurements(agent_id)
            filtered_error, turn_bank_log = update_turn_bank_feedback(
                self._turn_bank_feedback_states.get(agent_id),
                dt_sec=self.time_interval,
                target_course_rate_deg_s=context["target_heading_rate_deg_s"],
                course_error_deg=context["course_error_deg"],
                course_tau_sec=context["effective_course_tau_sec"],
                horizontal_speed_mps=measured["horizontal_speed_mps"],
                roll_deg=measured["roll_deg"],
                max_course_rate_deg_s=float(getattr(
                    self.config, "CoordinatedResidual_max_course_rate_deg_s")),
                max_bank_deg=float(getattr(
                    self.config, "CoordinatedResidual_max_bank_deg")),
            )
            self._turn_bank_feedback_states[agent_id] = filtered_error
            nominal = np.asarray(nominal, dtype=np.float32).copy()
            base_aileron = float(nominal[0])
            nominal[0] = np.float32(np.clip(
                base_aileron + turn_bank_log["aileron_correction_action"], -1.0, 1.0,
            ))
            turn_bank_log["base_aileron_action"] = base_aileron
            turn_bank_log["commanded_aileron_action"] = float(nominal[0])
        steady_bank_log = None
        if bool(getattr(self.config, "CoordinatedResidual_steady_bank_feedback_enable", False)):
            if (getattr(self.config, "task", None) != "baiyepower_maneuver_c31_v3"
                    or mode != "actuator_residual_fixed_center"
                    or turn_trim_log is not None or turn_bank_log is not None):
                raise ValueError("steady bank feedback is isolated to C31-V3")
            measured = self._controller_measurements(agent_id)
            state, steady_bank_log = update_steady_bank_feedback(
                self._steady_bank_feedback_states.get(agent_id),
                dt_sec=self.time_interval,
                target_course_rate_deg_s=context["target_heading_rate_deg_s"],
                course_error_deg=context["course_error_deg"],
                course_tau_sec=context["effective_course_tau_sec"],
                horizontal_speed_mps=measured["horizontal_speed_mps"],
                roll_deg=measured["roll_deg"],
                max_course_rate_deg_s=float(getattr(
                    self.config, "CoordinatedResidual_max_course_rate_deg_s")),
                max_bank_deg=float(getattr(
                    self.config, "CoordinatedResidual_max_bank_deg")),
            )
            self._steady_bank_feedback_states[agent_id] = state
            nominal = np.asarray(nominal, dtype=np.float32).copy()
            base_aileron = float(nominal[0])
            nominal[0] = np.float32(np.clip(
                base_aileron + steady_bank_log["aileron_correction_action"], -1.0, 1.0,
            ))
            steady_bank_log["base_aileron_action"] = base_aileron
            steady_bank_log["commanded_aileron_action"] = float(nominal[0])
        vs_bank_log = None
        if bool(getattr(self.config, "CoordinatedResidual_vs_bank_target_enable", False)):
            if (getattr(self.config, "task", None) != "baiyepower_maneuver_c31_v4"
                    or mode != "actuator_residual_fixed_center"
                    or turn_trim_log is not None or turn_bank_log is not None
                    or steady_bank_log is not None):
                raise ValueError("vertical-speed bank target is isolated to C31-V4")
            measured = self._controller_measurements(agent_id)
            state, vs_bank_log = update_vs_bank_target(
                self._vs_bank_target_states.get(agent_id),
                dt_sec=self.time_interval,
                target_course_rate_deg_s=context["target_heading_rate_deg_s"],
                course_error_deg=context["course_error_deg"],
                course_tau_sec=context["effective_course_tau_sec"],
                horizontal_speed_mps=measured["horizontal_speed_mps"],
                roll_deg=measured["roll_deg"],
                vertical_speed_mps=measured["vertical_speed_mps"],
                target_vertical_speed_mps=context["commanded_target_altitude_rate_mps"],
                speed_error_mps=measured["speed_error_mps"],
                max_course_rate_deg_s=float(getattr(
                    self.config, "CoordinatedResidual_max_course_rate_deg_s")),
                max_bank_deg=float(getattr(
                    self.config, "CoordinatedResidual_max_bank_deg")),
            )
            self._vs_bank_target_states[agent_id] = state
            nominal = np.asarray(nominal, dtype=np.float32).copy()
            base_aileron = float(nominal[0])
            nominal[0] = np.float32(np.clip(
                base_aileron + vs_bank_log["aileron_correction_action"], -1.0, 1.0,
            ))
            vs_bank_log["base_aileron_action"] = base_aileron
            vs_bank_log["commanded_aileron_action"] = float(nominal[0])
        vs_bank_v5_log = None
        if bool(getattr(self.config, "CoordinatedResidual_vs_bank_target_v5_enable", False)):
            if (getattr(self.config, "task", None) != "baiyepower_maneuver_c31_v5"
                    or mode != "actuator_residual_fixed_center"
                    or turn_trim_log is not None or turn_bank_log is not None
                    or steady_bank_log is not None or vs_bank_log is not None):
                raise ValueError("V5 bank target is isolated to C31-V5")
            measured = self._controller_measurements(agent_id)
            state, vs_bank_v5_log = update_vs_bank_target_v5(
                self._vs_bank_target_v5_states.get(agent_id),
                dt_sec=self.time_interval,
                target_course_rate_deg_s=context["target_heading_rate_deg_s"],
                course_error_deg=context["course_error_deg"],
                course_tau_sec=context["effective_course_tau_sec"],
                horizontal_speed_mps=measured["horizontal_speed_mps"],
                roll_deg=measured["roll_deg"],
                vertical_speed_mps=measured["vertical_speed_mps"],
                target_vertical_speed_mps=context["commanded_target_altitude_rate_mps"],
                speed_error_mps=measured["speed_error_mps"],
                max_course_rate_deg_s=float(getattr(
                    self.config, "CoordinatedResidual_max_course_rate_deg_s")),
                max_bank_deg=float(getattr(
                    self.config, "CoordinatedResidual_max_bank_deg")),
            )
            self._vs_bank_target_v5_states[agent_id] = state
            nominal = np.asarray(nominal, dtype=np.float32).copy()
            base_aileron = float(nominal[0])
            nominal[0] = np.float32(np.clip(
                base_aileron + vs_bank_v5_log["aileron_correction_action"], -1.0, 1.0,
            ))
            vs_bank_v5_log["base_aileron_action"] = base_aileron
            vs_bank_v5_log["commanded_aileron_action"] = float(nominal[0])
        if mode == "actuator_residual_fixed_center":
            # Reuse the exact zero-action structured center, but do not let
            # the structured branch suppress C28's physical actuator residual.
            composition_mode = "residual"
            residual_scale = self.config.CoordinatedResidual_residual_scale
        composed, composition = compose_coordinated_residual_action(
            nominal,
            residual_input,
            mode=composition_mode,
            residual_scale=residual_scale,
        )
        log = {
            "baiyepower_v2_c1_enabled": 1.0,
            "baiyepower_v2_c1_zero_residual_active": float(
                mode == "zero_residual"
            ),
            "baiyepower_v2_c1_command_vertical_speed_active": float(
                mode == "vertical_speed_command"
            ),
            "baiyepower_v2_c1_energy_balance_probe_active": float(
                mode == "energy_balance_probe"
            ),
            "baiyepower_v2_c1_energy_exchange_gain_residual_active": float(
                mode == "energy_exchange_gain_residual"
            ),
            "baiyepower_v2_c1_energy_exchange_fixed_gain_active": float(
                mode == "energy_exchange_fixed_gain"
            ),
            "baiyepower_v2_c1_structured_reference_gain_active": float(
                mode == "structured_reference_gain_residual"
            ),
            "baiyepower_v2_c1_structured_feedback_gain_active": float(
                mode == "structured_feedback_gain_residual"
            ),
            "baiyepower_v2_c1_clip_count": float(
                composition["clip_count"]
            ),
            "baiyepower_v2_c1_target_heading_rate_deg_s": float(
                context["target_heading_rate_deg_s"]
            ),
            "baiyepower_v2_c1_target_altitude_rate_mps": float(
                context["target_altitude_rate_mps"]
            ),
            (
                "baiyepower_v2_c1_commanded_target_"
                "altitude_rate_mps"
            ): float(
                context["commanded_target_altitude_rate_mps"]
            ),
            (
                "baiyepower_v2_c1_target_altitude_"
                "rate_residual_mps"
            ): float(
                context["target_altitude_rate_residual_mps"]
            ),
            "baiyepower_v2_c1_desired_vertical_speed_offset_mps": float(
                context["desired_vertical_speed_offset_mps"]
            ),
            "baiyepower_v2_c1_turn_pitch_coupling_gain": float(
                context["turn_pitch_coupling_gain"]
            ),
            "baiyepower_v2_c1_measured_heading_rate_deg_s": float(
                context["measured_heading_rate_deg_s"]
            ),
            "baiyepower_v2_c1_turn_pitch_actual_rate_blend": float(
                context["turn_pitch_actual_rate_blend"]
            ),
            "baiyepower_v2_c1_pitch_rate_feedback_gain": float(
                context["pitch_rate_feedback_gain"]
            ),
            "baiyepower_v2_c1_target_speed_rate_mps2": float(
                context["target_speed_rate_mps2"]
            ),
            "baiyepower_v2_c1_course_error_deg": float(
                context["course_error_deg"]
            ),
            "baiyepower_v2_c1_altitude_error_m": float(
                context["altitude_error_m"]
            ),
            "baiyepower_v2_c1_speed_error_mps": float(
                context["speed_error_mps"]
            ),
            "baiyepower_v2_c1_energy_allocator_active": float(
                context["energy_allocator_mode"]
                in (
                    "tecs_v1",
                    "total_energy_throttle_v1",
                    "adaptive_total_energy_throttle_v1",
                )
            ),
            "baiyepower_v2_c1_energy_balance_pitch_active": float(
                context["energy_allocator_mode"] == "tecs_v1"
            ),
            "baiyepower_v2_c1_energy_exchange_active": float(
                context["energy_exchange_mode"]
                == "measured_opposite_sign_v1"
            ),
            "baiyepower_v2_c1_sideslip_deg": float(
                context["sideslip_deg"]
            ),
            "baiyepower_v2_c1_sideslip_rudder_feedback_action": float(
                context["sideslip_rudder_feedback_action"]
            ),
        }
        for name, value in context["flight_path_tau_log"].items():
            log[
                "baiyepower_v2_c1_flight_path_tau_{}".format(name)
            ] = float(value)
        for name, value in context["direct_flight_path_log"].items():
            log[
                "baiyepower_v2_c1_{}".format(name)
            ] = float(value)
        if context["energy_allocator_log"] is not None:
            for name, value in context["energy_errors"].items():
                log[
                    "baiyepower_v2_c1_energy_{}".format(name)
                ] = float(value)
            for name, value in context[
                "energy_allocator_log"
            ].items():
                log[
                    "baiyepower_v2_c1_energy_{}".format(name)
                ] = float(value)
        if context["energy_exchange_log"] is not None:
            for name, value in context["energy_exchange_log"].items():
                log[
                    "baiyepower_v2_c1_energy_exchange_{}".format(
                        name
                    )
                ] = float(value)
        if command_log is not None:
            command_index = int(command_log["action_index"])
            log.update(
                {
                    "baiyepower_v2_c1_command_action_index": float(
                        command_index
                    ),
                    "baiyepower_v2_c1_command_scale_mps": float(
                        command_log["scale_mps"]
                    ),
                    "baiyepower_v2_c1_command_input": float(
                        command_log["bounded_policy_input"][
                            command_index
                        ]
                    ),
                    "baiyepower_v2_c1_command_residual_mps": float(
                        command_log["residual_mps"]
                    ),
                }
            )
            for name in (
                "base_energy_exchange_offset_mps",
                "gain_residual_fraction",
                "gain_multiplier",
                "effective_energy_exchange_offset_mps",
            ):
                if name in command_log:
                    log[
                        "baiyepower_v2_c1_energy_exchange_gain_{}".format(name)
                    ] = float(command_log[name])
        if structured_reference_log is not None:
            for name, value in structured_reference_log.items():
                log[
                    "baiyepower_v2_c1_structured_{}".format(name)
                ] = float(value)
        if structured_feedback_log is not None:
            for name, value in structured_feedback_log.items():
                log[
                    "baiyepower_v2_c1_structured_{}".format(name)
                ] = float(value)
        if turn_trim_log is not None:
            for name, value in turn_trim_log.items():
                log["baiyepower_v2_c1_turn_vertical_trim_{}".format(name)] = float(value)
        if turn_bank_log is not None:
            for name, value in turn_bank_log.items():
                log["baiyepower_v2_c1_turn_bank_feedback_{}".format(name)] = float(value)
        if steady_bank_log is not None:
            for name, value in steady_bank_log.items():
                log["baiyepower_v2_c1_steady_bank_feedback_{}".format(name)] = float(value)
        if vs_bank_log is not None:
            for name, value in vs_bank_log.items():
                log["baiyepower_v2_c1_vs_bank_target_{}".format(name)] = float(value)
        if vs_bank_v5_log is not None:
            for name, value in vs_bank_v5_log.items():
                log["baiyepower_v2_c1_vs_bank_target_v5_{}".format(name)] = float(value)
        for index, channel in enumerate(ACTION_CHANNELS):
            log[
                "baiyepower_v2_c1_nominal_{}".format(channel)
            ] = float(composition["nominal_action"][index])
            log[
                "baiyepower_v2_c1_residual_input_{}".format(
                    channel
                )
            ] = float(composition["residual_input"][index])
            log[
                "baiyepower_v2_c1_residual_used_{}".format(
                    channel
                )
            ] = float(composition["residual_used"][index])
            log[
                "baiyepower_v2_c1_preclip_{}".format(channel)
            ] = float(composition["preclip_action"][index])
            log[
                "baiyepower_v2_c1_composed_{}".format(channel)
            ] = float(composition["composed_action"][index])
            log[
                "baiyepower_v2_c1_clipped_{}".format(channel)
            ] = float(composition["clip_mask"][index])
        self._last_coordinated_residual_log[agent_id] = log
        return composed

    def get_coordinated_residual_log(self, agent_id):
        log = self._last_coordinated_residual_log.get(agent_id, {})
        return dict(log) if isinstance(log, dict) else {}

    def _qp_state_observation_mode(self):
        mode = str(getattr(self.config, "QPStateObservation_mode", "off")).strip().lower()
        obs_version = getattr(self.task, "obs_version", "v1")
        qp_observation_versions = ("v4_qp_state", "v5_target_rates")
        if (
            obs_version in qp_observation_versions
            and mode not in ("real", "repeat_prev")
        ):
            raise ValueError(
                "QP-state observations require "
                "QPStateObservation_mode=real or repeat_prev"
            )
        if obs_version not in qp_observation_versions and mode != "off":
            raise ValueError(
                "QPStateObservation_mode requires "
                "obs_version=v4_qp_state or v5_target_rates"
            )
        return mode

    def _reset_action_projector(self):
        self._action_projector = None
        self._last_action_projector_log = {}
        qp_state_mode = self._qp_state_observation_mode()
        if qp_state_mode != "off" and not self._action_projector_enabled():
            raise ValueError(
                "QPStateObservation_mode requires ActionProjector_enable=true"
            )
        if not self._action_projector_enabled():
            return
        mode = str(getattr(self.config, "ActionProjector_mode", "diagonal_qp")).lower()
        if mode != "diagonal_qp":
            raise ValueError(f"Unsupported ActionProjector_mode: {mode}")
        self._action_projector = ConvexActionProjector(
            action_dim=len(self.task.action_var),
            action_min=getattr(self.config, "ActionProjector_action_min", [-1.0, -1.0, -1.0, 0.4]),
            action_max=getattr(self.config, "ActionProjector_action_max", [1.0, 1.0, 1.0, 0.9]),
            slew_limit=getattr(self.config, "ActionProjector_slew_limit", [0.35, 0.20, 0.30, 0.05]),
            raw_weight=getattr(self.config, "ActionProjector_raw_weight", 1.0),
            rate_weight=getattr(self.config, "ActionProjector_rate_weight", 0.25),
            jerk_weight=getattr(self.config, "ActionProjector_jerk_weight", 0.08),
            hessian_eps=float(getattr(self.config, "ActionProjector_hessian_eps", 1e-6)),
            enabled_channels=getattr(self.config, "ActionProjector_apply_to", None),
            channel_names=("aileron", "elevator", "rudder", "throttle"),
        )
        for agent_id, sim in self.agents.items():
            initial_action = np.asarray(sim.get_property_values(self.task.action_var), dtype=np.float32)
            self._action_projector.reset_agent(agent_id, initial_action)
            self._last_action_projector_log[agent_id] = self._action_projector.get_last_log(agent_id)

    def _apply_action_projector(self, agent_id, norm_act):
        if not self._action_projector_enabled() or self._action_projector is None:
            return norm_act
        projected = self._action_projector.project(agent_id, norm_act)
        self._last_action_projector_log[agent_id] = self._action_projector.get_last_log(agent_id)
        return projected

    def get_qp_state_observation(self, agent_id):
        """Return real or repeat-prev QP history in policy coordinates."""
        mode = self._qp_state_observation_mode()
        if mode == "off":
            raise RuntimeError("QP state observation requested for a non-v4 task")
        if self._action_projector is None:
            raise RuntimeError("QP state observation requested before projector reset")
        state = self._action_projector.get_observation_state(agent_id)
        if state is None:
            raise RuntimeError(f"QP state is unavailable for agent {agent_id}")
        prev = self.task.action_to_policy_coordinates(state["prev"])
        history = state["prev2"] if mode == "real" else state["prev"]
        history = self.task.action_to_policy_coordinates(history)
        return np.concatenate((prev, history)).astype(np.float32, copy=False)

    def _reset_action_smoother(self):
        self._action_smoother = None
        if not self._action_smoother_enabled():
            return
        mode = str(getattr(self.config, "ActionSmoother_mode", "min_jerk")).lower()
        if mode != "min_jerk":
            raise ValueError(f"Unsupported ActionSmoother_mode: {mode}")
        apply_to = getattr(self.config, "ActionSmoother_apply_to", None)
        duration_sec = float(getattr(self.config, "ActionSmoother_duration_sec", 0.2))
        sim_freq = float(getattr(self.config, "sim_freq", 60.0))
        base_steps = int(round(duration_sec * sim_freq))
        base_steps = int(np.clip(base_steps, 1, max(1, int(self.agent_interaction_steps))))
        self._action_smoother = MinimumJerkActionSmoother(
            action_dim=len(self.task.action_var),
            base_steps=base_steps,
            enabled_channels=apply_to,
            large_delta_threshold=float(getattr(self.config, "ActionSmoother_large_delta_threshold", 0.65)),
            large_delta_duration_scale=float(getattr(self.config, "ActionSmoother_large_delta_duration_scale", 0.5)),
            target_slew_enable=bool(getattr(self.config, "ActionSmoother_target_slew_enable", False)),
            target_deadband=getattr(self.config, "ActionSmoother_target_deadband", None),
            target_slew_base=getattr(self.config, "ActionSmoother_target_slew_base", None),
            urgency_enable=bool(getattr(self.config, "ActionSmoother_urgency_enable", False)),
            urgency_gain=float(getattr(self.config, "ActionSmoother_urgency_gain", 1.0)),
            aileron_turn_boost_enable=bool(getattr(self.config, "ActionSmoother_aileron_turn_boost_enable", False)),
            aileron_turn_boost_gain=float(getattr(self.config, "ActionSmoother_aileron_turn_boost_gain", 0.7)),
            aileron_max_gate=float(getattr(self.config, "ActionSmoother_aileron_max_gate", 2.0)),
            elevator_vertical_gate_enable=bool(getattr(self.config, "ActionSmoother_elevator_vertical_gate_enable", False)),
            elevator_vertical_damping_gain=float(getattr(self.config, "ActionSmoother_elevator_vertical_damping_gain", 0.45)),
            elevator_min_gate=float(getattr(self.config, "ActionSmoother_elevator_min_gate", 0.45)),
            elevator_max_gate=float(getattr(self.config, "ActionSmoother_elevator_max_gate", 2.0)),
            elevator_momentum_unload_enable=bool(getattr(self.config, "ActionSmoother_elevator_momentum_unload_enable", False)),
            elevator_momentum_unload_gain=float(getattr(self.config, "ActionSmoother_elevator_momentum_unload_gain", 0.25)),
            elevator_vertical_activity_blend=float(getattr(self.config, "ActionSmoother_elevator_vertical_activity_blend", 0.5)),
        )
        for agent_id, sim in self.agents.items():
            initial_action = np.asarray(sim.get_property_values(self.task.action_var), dtype=np.float32)
            self._action_smoother.reset_agent(agent_id, initial_action)

    def _get_action_smoother_context(self, agent_id):
        measured = self._controller_measurements(agent_id)
        heading_scale = max(float(getattr(self.config, "ActionSmoother_urgency_heading_scale_deg", 25.0)), 1e-6)
        altitude_scale = max(float(getattr(self.config, "ActionSmoother_urgency_altitude_scale_m", 250.0)), 1e-6)
        speed_scale = max(float(getattr(self.config, "ActionSmoother_urgency_speed_scale_mps", 25.0)), 1e-6)
        abs_delta_heading = abs(measured["course_error_deg"])
        urgency = max(
            abs_delta_heading / heading_scale,
            abs(measured["altitude_error_m"]) / altitude_scale,
            abs(measured["speed_error_mps"]) / speed_scale,
        )
        turn_scale = max(float(getattr(self.config, "ActionSmoother_aileron_turn_boost_heading_scale_deg", 20.0)), 1e-6)
        aileron_turn_need = abs_delta_heading / turn_scale

        q_scale = max(float(getattr(self.config, "ActionSmoother_elevator_q_scale_deg_s", 8.0)), 1e-6)
        vs_scale = max(float(getattr(self.config, "ActionSmoother_elevator_vs_scale_mps", 8.0)), 1e-6)
        pitch_scale = max(float(getattr(self.config, "ActionSmoother_elevator_pitch_scale_deg", 12.0)), 1e-6)
        pitch_rate_deg_s = measured["pitch_rate_deg_s"]
        vertical_speed_mps = measured["vertical_speed_mps"]
        pitch_deg = measured["pitch_deg"]
        vertical_activity = max(
            abs(pitch_rate_deg_s) / q_scale,
            abs(vertical_speed_mps) / vs_scale,
            abs(pitch_deg) / pitch_scale,
        )

        delta_altitude_m = measured["altitude_error_m"]
        vertical_momentum_excess = self._get_vertical_momentum_excess(delta_altitude_m, vertical_speed_mps)
        return {
            "urgency": float(np.clip(urgency, 0.0, 1.0)),
            "aileron_turn_need": float(np.clip(aileron_turn_need, 0.0, 1.0)),
            "vertical_activity": float(np.clip(vertical_activity, 0.0, 1.0)),
            "momentum_envelope_excess": float(vertical_momentum_excess),
        }

    def _get_vertical_momentum_excess(self, delta_altitude_m, vertical_speed_mps):
        if not bool(getattr(self.config, "ActionSmoother_elevator_momentum_unload_enable", False)):
            return 0.0
        if abs(delta_altitude_m) < 1e-6:
            direction = 0.0
        else:
            direction = float(np.sign(delta_altitude_m))
        closing_speed_mps = direction * float(vertical_speed_mps)
        if closing_speed_mps <= 0.0:
            return 0.0
        tau_delay = max(float(getattr(self.config, "ActionSmoother_elevator_momentum_tau_delay_sec", 0.35)), 0.0)
        brake_accel = max(float(getattr(self.config, "ActionSmoother_elevator_momentum_brake_accel_mps2", 2.0)), 1e-6)
        v_floor = max(float(getattr(self.config, "ActionSmoother_elevator_momentum_v_floor_mps", 2.5)), 0.0)
        excess_scale = max(float(getattr(self.config, "ActionSmoother_elevator_momentum_excess_scale_mps", 10.0)), 1e-6)

        distance_m = abs(float(delta_altitude_m))
        effective_distance_m = max(0.0, distance_m - closing_speed_mps * tau_delay)
        v_allow_mps = float(np.sqrt(2.0 * brake_accel * effective_distance_m) + v_floor)
        v_excess_mps = max(0.0, closing_speed_mps - v_allow_mps)
        return float(np.clip(v_excess_mps / excess_scale, 0.0, 1.0))

    def _heading_train_diag_enabled(self):
        return (
            isinstance(self.task, HeadingTask)
            and bool(getattr(self.config, "HeadingTrainDiag_enable", False))
        )

    def _is_heading_train_diag_gate_on(self, agent_id, sample=None):
        sim = self.agents[agent_id]
        sample = sample if isinstance(sample, dict) else {}
        vertical_speed_mps = float(
            sample.get(
                "vertical_speed_mps",
                float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048,
            )
        )
        pitch_rate_deg_s = float(
            sample.get(
                "pitch_rate_deg_s",
                np.rad2deg(sim.get_property_value(c.velocities_q_rad_sec)),
            )
        )
        return bool(
            abs(float(sample.get("delta_heading_deg", sim.get_property_value(c.delta_heading)))) <= float(getattr(self.config, "HeadingTrainDiag_heading_error_gate_deg", 8.0))
            and abs(float(sample.get("delta_altitude_m", sim.get_property_value(c.delta_altitude)))) <= float(getattr(self.config, "HeadingTrainDiag_altitude_error_gate_m", 40.0))
            and abs(float(sample.get("delta_speed_mps", self._get_task_speed_error_mps(agent_id)))) <= float(getattr(self.config, "HeadingTrainDiag_speed_error_gate_mps", 20.0))
            and abs(float(sample.get("roll_deg", sim.get_property_value(c.attitude_phi_deg)))) <= float(getattr(self.config, "HeadingTrainDiag_roll_gate_deg", 25.0))
            and abs(pitch_rate_deg_s) <= float(getattr(self.config, "HeadingTrainDiag_pitch_rate_gate_deg_s", 12.0))
            and abs(vertical_speed_mps) <= float(getattr(self.config, "HeadingTrainDiag_vertical_speed_gate_mps", 4.0))
        )

    def _get_task_current_speed_mps(self, agent_id):
        sim = self.agents[agent_id]
        task = getattr(self, "task", None)
        if hasattr(task, "get_current_speed_mps"):
            return float(task.get_current_speed_mps(sim))
        return float(sim.get_property_value(c.velocities_u_mps))

    def _get_task_target_speed_mps(self, agent_id):
        sim = self.agents[agent_id]
        task = getattr(self, "task", None)
        if hasattr(task, "get_target_speed_mps"):
            return float(task.get_target_speed_mps(sim))
        return float(sim.get_property_value(c.target_velocities_u_mps))

    def _get_task_speed_error_mps(self, agent_id):
        sim = self.agents[agent_id]
        task = getattr(self, "task", None)
        if hasattr(task, "get_speed_error_mps"):
            return float(task.get_speed_error_mps(sim))
        return float(sim.get_property_value(c.delta_velocities_u))

    def _collect_heading_train_diag(self, agent_id, sample):
        if not bool(sample.get("sample_valid", True)):
            # A numerical terminal is intentionally not converted into a
            # plausible-looking physics sample.  Preserve only transition
            # timing, frozen target identity, and the already-computed QP
            # sidecar so rollout bookkeeping can finish without touching the
            # failed simulator state.
            if hasattr(self, "agent_interaction_steps") and hasattr(self, "sim_freq"):
                decision_dt_sec = float(
                    self.agent_interaction_steps / self.sim_freq
                )
            else:
                decision_dt_sec = float(
                    getattr(self.config, "time_interval", 0.2)
                )
            diag = {
                "heading_diag_sample_valid": False,
                "steady_gate": False,
                "decision_dt_sec": decision_dt_sec,
                "decision_sample_rate_hz": float(
                    1.0 / max(decision_dt_sec, 1e-9)
                ),
                "elevator_cmd_norm": float(
                    sample.get("elevator_cmd_norm", np.nan)
                ),
                "elevator_pos_norm": float(
                    sample.get("elevator_pos_norm", np.nan)
                ),
                "aoa_deg": float(sample.get("aoa_deg", np.nan)),
                "vertical_speed_mps": float(
                    sample.get("vertical_speed_mps", np.nan)
                ),
                "roll_sin": np.nan,
                "roll_cos": np.nan,
                "roll_deg": float(sample.get("roll_deg", np.nan)),
                "pitch_deg": float(sample.get("pitch_deg", np.nan)),
                "pitch_rate_deg_s": float(
                    sample.get("pitch_rate_deg_s", np.nan)
                ),
                "delta_heading_deg": float(
                    sample.get("delta_heading_deg", np.nan)
                ),
                "delta_altitude_m": float(
                    sample.get("delta_altitude_m", np.nan)
                ),
                "delta_speed_mps": float(
                    sample.get("delta_speed_mps", np.nan)
                ),
                "target_heading_deg": float(
                    sample.get("target_heading_deg", np.nan)
                ),
                "target_altitude_ft": float(
                    sample.get("target_altitude_ft", np.nan)
                ),
                "target_speed_mps": float(
                    sample.get(
                        "target_speed_mps",
                        sample.get("target_velocities_u_mps", np.nan),
                    )
                ),
                "qp_projector_enabled": bool(
                    sample.get("qp_projector_enabled", False)
                ),
                "qp_projector_fail_streak": float(
                    sample.get("qp_projector_fail_streak", 0.0)
                ),
                "qp_projector_active_count": float(
                    sample.get("qp_projector_active_count", np.nan)
                ),
                "qp_projector_box_active_count": float(
                    sample.get("qp_projector_box_active_count", np.nan)
                ),
                "qp_projector_slew_active_count": float(
                    sample.get("qp_projector_slew_active_count", np.nan)
                ),
                "qp_projector_residual_abs_mean": float(
                    sample.get("qp_projector_residual_abs_mean", np.nan)
                ),
                "qp_projector_delta_abs_mean": float(
                    sample.get("qp_projector_delta_abs_mean", np.nan)
                ),
            }
            for channel in ("aileron", "elevator", "rudder", "throttle"):
                for key in (
                    "raw",
                    "action",
                    "residual",
                    "active",
                    "box_active",
                    "slew_active",
                ):
                    sample_key = f"qp_{key}_{channel}"
                    diag[sample_key] = float(
                        sample.get(sample_key, np.nan)
                    )
            for key, value in sample.items():
                if str(key).startswith("baiyepower_v2_c1_"):
                    diag[str(key)] = float(value)
            return diag

        sim = self.agents[agent_id]
        g = float(getattr(self.config, "PowerMomentumReward_gravity", 9.80665))
        height_scale = max(
            float(getattr(self.config, "PowerMomentumReward_energy_height_scale_m", 300.0)),
            1e-6,
        )
        split_eps = max(
            float(getattr(self.config, "PowerMomentumReward_split_eps", 0.05)),
            1e-9,
        )
        split_sigma = max(
            float(getattr(self.config, "PowerMomentumReward_split_sigma", 0.6)),
            1e-6,
        )
        velocity_gate_mps = max(
            float(getattr(self.config, "PowerMomentumReward_velocity_gate_mps", 40.0)),
            1e-6,
        )
        velocity_eps = max(
            float(getattr(self.config, "PowerMomentumReward_velocity_eps", 1e-6)),
            1e-9,
        )
        h = float(sim.get_property_value(c.position_h_sl_m))
        h_target = float(
            sample.get(
                "target_altitude_ft",
                sim.get_property_value(c.target_altitude_ft),
            )
        ) * 0.3048
        v = self._get_task_current_speed_mps(agent_id)
        v_target = float(
            sample.get(
                "target_speed_mps",
                sample.get(
                    "target_velocities_u_mps",
                    self._get_task_target_speed_mps(agent_id),
                ),
            )
        )
        energy_error_norm = (
            g * (h - h_target) + 0.5 * (v * v - v_target * v_target)
        ) / max(g * height_scale, 1e-6)
        z_h = (h - h_target) / height_scale
        z_v = (v * v - v_target * v_target) / (2.0 * g * height_scale)
        split_error = (z_h - z_v) / (abs(z_h) + abs(z_v) + split_eps)
        r_energy = float(np.exp(-(np.sqrt(1.0 + energy_error_norm * energy_error_norm) - 1.0)))
        r_split = float(np.exp(-(np.sqrt(1.0 + (split_error / split_sigma) ** 2) - 1.0)))
        v_n = float(sim.get_property_value(c.velocities_v_north_mps))
        v_e = float(sim.get_property_value(c.velocities_v_east_mps))
        v_h = float(np.sqrt(v_n * v_n + v_e * v_e))
        target_heading_rad = float(
            np.deg2rad(
                sample.get(
                    "target_heading_deg",
                    sim.get_property_value(c.target_heading_deg),
                )
            )
        )
        alignment = float(
            (v_n * np.cos(target_heading_rad) + v_e * np.sin(target_heading_rad))
            / (v_h + velocity_eps)
        )
        alignment = float(np.clip(alignment, -1.0, 1.0))
        momentum_gate = float(v_h / (v_h + velocity_gate_mps))
        r_momentum = float(0.5 + 0.5 * momentum_gate * alignment)
        if hasattr(self, "agent_interaction_steps") and hasattr(self, "sim_freq"):
            decision_dt_sec = float(self.agent_interaction_steps / self.sim_freq)
        else:
            # Keeps the diagnostic helper usable in isolated CPU unit tests;
            # real environments always define both sampling attributes.
            decision_dt_sec = float(getattr(self.config, "time_interval", 0.2))
        diag = {
            "heading_diag_sample_valid": True,
            "steady_gate": self._is_heading_train_diag_gate_on(agent_id, sample),
            "decision_dt_sec": decision_dt_sec,
            "decision_sample_rate_hz": float(1.0 / max(decision_dt_sec, 1e-9)),
            "elevator_cmd_norm": float(sample.get("elevator_cmd_norm", np.nan)),
            "elevator_pos_norm": float(sample.get("elevator_pos_norm", np.nan)),
            "aoa_deg": float(sample.get("aoa_deg", np.nan)),
            "vertical_speed_mps": float(sample.get("vertical_speed_mps", np.nan)),
            "roll_sin": float(np.sin(np.deg2rad(sample.get("roll_deg", np.nan)))),
            "roll_cos": float(np.cos(np.deg2rad(sample.get("roll_deg", np.nan)))),
            "roll_deg": float(sample.get("roll_deg", np.nan)),
            "pitch_deg": float(sample.get("pitch_deg", np.nan)),
            "pitch_rate_deg_s": float(sample.get("pitch_rate_deg_s", np.nan)),
            "delta_heading_deg": float(sample.get("delta_heading_deg", np.nan)),
            "delta_altitude_m": float(sample.get("delta_altitude_m", np.nan)),
            "delta_speed_mps": float(sample.get("delta_speed_mps", np.nan)),
            "target_heading_deg": float(sample.get("target_heading_deg", np.nan)),
            "target_altitude_ft": float(sample.get("target_altitude_ft", np.nan)),
            "target_speed_mps": float(
                sample.get(
                    "target_speed_mps",
                    sample.get("target_velocities_u_mps", np.nan),
                )
            ),
            "power_energy_r": r_energy,
            "power_split_r": r_split,
            "power_momentum_r": r_momentum,
            "power_energy_error_norm": float(energy_error_norm),
            "power_split_error": float(split_error),
            "power_momentum_alignment": alignment,
            "power_momentum_gate": momentum_gate,
            "power_energy_weight": float(getattr(self.config, "PowerMomentumReward_energy_weight", 0.05)),
            "power_split_weight": float(getattr(self.config, "PowerMomentumReward_split_weight", 0.05)),
            "power_momentum_weight": float(getattr(self.config, "PowerMomentumReward_momentum_weight", 0.08)),
            "power_reward_scale": float(getattr(self.config, "PowerMomentumReward_scale", 1.0)),
            "power_reward_enabled": float(bool(getattr(self.config, "PowerMomentumReward_enable", False))),
            "qp_projector_enabled": bool(sample.get("qp_projector_enabled", False)),
            "qp_projector_fail_streak": float(sample.get("qp_projector_fail_streak", 0.0)),
            "qp_projector_active_count": float(sample.get("qp_projector_active_count", np.nan)),
            "qp_projector_box_active_count": float(sample.get("qp_projector_box_active_count", np.nan)),
            "qp_projector_slew_active_count": float(sample.get("qp_projector_slew_active_count", np.nan)),
            "qp_projector_residual_abs_mean": float(sample.get("qp_projector_residual_abs_mean", np.nan)),
            "qp_projector_delta_abs_mean": float(sample.get("qp_projector_delta_abs_mean", np.nan)),
        }
        for key in (
            "baiyepower_speed_mps",
            "baiyepower_target_speed_mps",
            "baiyepower_energy_height_m",
            "baiyepower_target_energy_height_m",
            "baiyepower_altitude_deficit_m",
            "baiyepower_kinetic_height_deficit_m",
            "baiyepower_total_energy_height_deficit_m",
            "baiyepower_vt_audit_mps",
            "baiyepower_speed_minus_vt_mps",
        ):
            if key in sample:
                diag[key] = float(sample[key])
        align_logs = getattr(self, "_last_qp_align_reward_log", {})
        align_log = align_logs.get(agent_id, {}) if isinstance(align_logs, dict) else {}
        diag.update(
            {
                "qp_align_reward": float(align_log.get("qp_align_reward", np.nan)),
                "qp_align": float(align_log.get("qp_align", np.nan)),
                "qp_align_gate": float(align_log.get("qp_align_gate", np.nan)),
                "qp_align_residual": float(align_log.get("qp_align_residual", np.nan)),
                "qp_align_scale": float(align_log.get("qp_align_scale", np.nan)),
                "qp_align_health_gate": float(align_log.get("qp_align_health_gate", np.nan)),
                "qp_align_box_gate": float(align_log.get("qp_align_box_gate", np.nan)),
                "qp_align_residual_gate": float(align_log.get("qp_align_residual_gate", np.nan)),
                "qp_align_applicable": float(bool(align_log.get("qp_align_applicable", False))),
                "qp_align_reward_enabled": float(bool(align_log.get("qp_align_reward_enabled", False))),
            }
        )
        for channel in ("aileron", "elevator", "rudder", "throttle"):
            for key in ("raw", "action", "residual", "active", "box_active", "slew_active"):
                sample_key = f"qp_{key}_{channel}"
                diag[sample_key] = float(sample.get(sample_key, np.nan))
        for key, value in sample.items():
            if str(key).startswith("baiyepower_v2_c1_"):
                diag[str(key)] = float(value)
        route_state = getattr(self, "heading_route_state", None)
        if isinstance(route_state, dict):
            level = route_state.get("level_config", {})
            hold_required = float(level.get("last_hold_sec", 0.0))
            sim_time = float(self.agents[agent_id].get_property_value(c.simulation_sim_time_sec))
            point_start_time = float(route_state.get("point_start_time", sim_time))
            diag.update(
                {
                    "curriculum_level": int(route_state.get("level", -1)),
                    "active_point_index": int(route_state.get("active_point_index", 0)),
                    "point_count": int(route_state.get("point_count", 1)),
                    "in_target_zone": bool(route_state.get("in_target_zone", False)),
                    "hold_timer": float(route_state.get("hold_timer", 0.0)),
                    "hold_required": hold_required,
                    "hold_progress": float(route_state.get("hold_timer", 0.0) / max(hold_required, 1e-6)) if hold_required > 0 else float(route_state.get("in_target_zone", False)),
                    "time_since_point_start": float(sim_time - point_start_time),
                }
            )
        return diag

    def _trim_ema_enabled(self):
        return (
            isinstance(self.task, HeadingTask)
            and getattr(self.task, "action_type", "discrete") == "continuous"
            and bool(getattr(self.config, "TrimBiasEMA_enable", False))
        )

    def _mark_trim_ema_target_change(self, agent_id):
        if self._trim_ema_state is None:
            return
        self._trim_ema_state["last_target_change_time"] = float(
            self.agents[agent_id].get_property_value(c.simulation_sim_time_sec)
        )

    def _make_trim_ema_log(
        self,
        gate_on,
        policy_elevator,
        elevator_out,
        elevator_fast_part,
        elevator_correction,
        aoa_deg,
        target_age_sec,
    ):
        elevator_trim_est = np.nan if self._trim_ema_state is None else self._trim_ema_state["elevator_trim_est"]
        aoa_trim_ema = np.nan if self._trim_ema_state is None else self._trim_ema_state["aoa_trim_ema"]
        return {
            "trim_ema_enabled": bool(self._trim_ema_enabled()),
            "trim_ema_gate_on": bool(gate_on),
            "trim_ema_target_age_sec": float(target_age_sec),
            "trim_ema_elevator_policy": float(policy_elevator),
            "trim_ema_elevator_out": float(elevator_out),
            "trim_ema_elevator_est": float(elevator_trim_est),
            "trim_ema_elevator_fast_part": float(elevator_fast_part),
            "trim_ema_elevator_correction": float(elevator_correction),
            "trim_ema_aoa_deg": float(aoa_deg),
            "trim_ema_aoa_ema": float(aoa_trim_ema),
            "trim_ema_aoa_residual": float(aoa_deg - aoa_trim_ema) if np.isfinite(aoa_deg) else np.nan,
        }

    def _is_trim_ema_gate_on(self, agent_id):
        sim = self.agents[agent_id]
        sim_time = float(sim.get_property_value(c.simulation_sim_time_sec))
        target_age_sec = sim_time - float(self._trim_ema_state["last_target_change_time"])
        vertical_speed_mps = float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048
        gate_on = (
            target_age_sec >= float(getattr(self.config, "TrimBiasEMA_min_target_age_sec", 5.0))
            and abs(float(sim.get_property_value(c.delta_heading))) <= float(getattr(self.config, "TrimBiasEMA_heading_error_gate_deg", 8.0))
            and abs(float(sim.get_property_value(c.delta_altitude))) <= float(getattr(self.config, "TrimBiasEMA_altitude_error_gate_m", 40.0))
            and abs(self._get_task_speed_error_mps(agent_id)) <= float(getattr(self.config, "TrimBiasEMA_speed_error_gate_mps", 20.0))
            and abs(float(sim.get_property_value(c.attitude_phi_deg))) <= float(getattr(self.config, "TrimBiasEMA_roll_gate_deg", 25.0))
            and abs(float(np.rad2deg(sim.get_property_value(c.velocities_q_rad_sec)))) <= float(getattr(self.config, "TrimBiasEMA_pitch_rate_gate_deg_s", 12.0))
            and abs(vertical_speed_mps) <= float(getattr(self.config, "TrimBiasEMA_vertical_speed_gate_mps", 4.0))
        )
        return bool(gate_on), float(target_age_sec)

    def _apply_trim_bias_ema(self, agent_id, norm_act):
        if self._trim_ema_state is None:
            self._reset_trim_ema()
        if getattr(self.task, "action_type", "discrete") != "continuous":
            return norm_act

        sim = self.agents[agent_id]
        policy_elevator = float(norm_act[1])
        aoa_deg = float(sim.get_property_value(c.aero_alpha_deg))
        gate_on, target_age_sec = self._is_trim_ema_gate_on(agent_id) if self._trim_ema_enabled() else (False, 0.0)
        elevator_out = policy_elevator
        elevator_fast_part = policy_elevator - float(self._trim_ema_state["elevator_trim_est"])
        elevator_correction = 0.0

        if self._trim_ema_enabled() and gate_on:
            alpha = float(getattr(self.config, "TrimBiasEMA_alpha", 0.02))
            alpha = float(np.clip(alpha, 0.0, 1.0))
            if not self._trim_ema_state["initialized"]:
                self._trim_ema_state["elevator_trim_est"] = policy_elevator
                self._trim_ema_state["initialized"] = True
            else:
                self._trim_ema_state["elevator_trim_est"] = (
                    (1.0 - alpha) * self._trim_ema_state["elevator_trim_est"]
                    + alpha * policy_elevator
                )
            if not self._trim_ema_state["aoa_initialized"]:
                self._trim_ema_state["aoa_trim_ema"] = aoa_deg
                self._trim_ema_state["aoa_initialized"] = True
            else:
                self._trim_ema_state["aoa_trim_ema"] = (
                    (1.0 - alpha) * self._trim_ema_state["aoa_trim_ema"]
                    + alpha * aoa_deg
                )

            elevator_fast_part = policy_elevator - float(self._trim_ema_state["elevator_trim_est"])
            mode = str(getattr(self.config, "TrimBiasEMA_mode", "observe"))
            if mode == "damp_fast_part":
                fast_damping = float(getattr(self.config, "TrimBiasEMA_fast_damping", 0.90))
                fast_damping = float(np.clip(fast_damping, 0.0, 1.0))
                elevator_out = (
                    float(self._trim_ema_state["elevator_trim_est"])
                    + fast_damping * elevator_fast_part
                )
                elevator_out = float(np.clip(elevator_out, -1.0, 1.0))
                elevator_correction = elevator_out - policy_elevator
                norm_act = np.asarray(norm_act, dtype=np.float32).copy()
                norm_act[1] = elevator_out

        self._last_trim_ema_log = self._make_trim_ema_log(
            gate_on=gate_on,
            policy_elevator=policy_elevator,
            elevator_out=elevator_out,
            elevator_fast_part=elevator_fast_part,
            elevator_correction=elevator_correction,
            aoa_deg=aoa_deg,
            target_age_sec=target_age_sec,
        )
        return norm_act

    def _get_heading_target(self, agent_id):
        agent = self.agents[agent_id]
        return (
            agent.get_property_value(c.target_heading_deg),
            agent.get_property_value(c.target_altitude_ft),
            self._get_task_target_speed_mps(agent_id),
        )

    def _collect_heading_sample(self, agent_id):
        agent = self.agents[agent_id]
        target = self._get_heading_target(agent_id)
        vertical_speed_fps = float(agent.get_property_value(c.velocities_h_dot_fps))
        sample = {
            "step_idx": int(self.current_step),
            "sim_time": float(agent.get_property_value(c.simulation_sim_time_sec)),
            "delta_heading_deg": float(agent.get_property_value(c.delta_heading)),
            "delta_altitude_m": float(agent.get_property_value(c.delta_altitude)),
            "delta_speed_mps": self._get_task_speed_error_mps(agent_id),
            "speed_mps": self._get_task_current_speed_mps(agent_id),
            "roll_deg": float(agent.get_property_value(c.attitude_phi_deg)),
            "pitch_deg": float(agent.get_property_value(c.attitude_theta_deg)),
            "heading_deg": float(agent.get_property_value(c.attitude_psi_deg)),
            "roll_rate_deg_s": float(np.rad2deg(agent.get_property_value(c.velocities_p_rad_sec))),
            "pitch_rate_deg_s": float(np.rad2deg(agent.get_property_value(c.velocities_q_rad_sec))),
            "yaw_rate_deg_s": float(np.rad2deg(agent.get_property_value(c.velocities_r_rad_sec))),
            "pitch_accel_deg_s2": float(np.rad2deg(agent.get_property_value(c.accelerations_qdot_rad_sec2))),
            "aoa_deg": float(agent.get_property_value(c.aero_alpha_deg)),
            "vertical_speed_fps": vertical_speed_fps,
            "vertical_speed_mps": vertical_speed_fps * 0.3048,
            "vertical_speed_ft_min": vertical_speed_fps * 60.0,
            "aileron_cmd_norm": float(agent.get_property_value(c.fcs_aileron_cmd_norm)),
            "elevator_cmd_norm": float(agent.get_property_value(c.fcs_elevator_cmd_norm)),
            "elevator_pos_norm": float(agent.get_property_value(c.fcs_elevator_pos_norm)),
            "rudder_cmd_norm": float(agent.get_property_value(c.fcs_rudder_cmd_norm)),
            "throttle_cmd_norm": float(agent.get_property_value(c.fcs_throttle_cmd_norm)),
            "target_heading_deg": float(target[0]),
            "target_altitude_ft": float(target[1]),
            "target_velocities_u_mps": float(target[2]),
            "target_speed_mps": float(target[2]),
        }
        if hasattr(self.task, "get_baiyepower_energy_terms"):
            terms = self.task.get_baiyepower_energy_terms(agent)
            vt_audit_mps = self.task.get_vt_audit_mps(agent)
            sample.update(
                {
                    "baiyepower_speed_mps": terms.speed_mps,
                    "baiyepower_target_speed_mps": terms.target_speed_mps,
                    "baiyepower_energy_height_m": terms.energy_height_m,
                    "baiyepower_target_energy_height_m": (
                        terms.target_energy_height_m
                    ),
                    "baiyepower_altitude_deficit_m": terms.altitude_deficit_m,
                    "baiyepower_kinetic_height_deficit_m": (
                        terms.kinetic_height_deficit_m
                    ),
                    "baiyepower_total_energy_height_deficit_m": (
                        terms.total_energy_height_deficit_m
                    ),
                    "baiyepower_vt_audit_mps": vt_audit_mps,
                    "baiyepower_speed_minus_vt_mps": (
                        terms.speed_mps - vt_audit_mps
                    ),
                }
            )
        sample.update(
            {
                "baiyepower_v2_c1_initial_envelope_lattice_active": float(
                    bool(
                        getattr(
                            self,
                            "_initial_envelope_lattice_active",
                            False,
                        )
                    )
                ),
                "baiyepower_v2_c1_initial_envelope_altitude_ft": float(
                    getattr(self, "_initial_envelope_altitude_ft", np.nan)
                ),
                "baiyepower_v2_c1_initial_envelope_speed_mps": float(
                    getattr(self, "_initial_envelope_speed_mps", np.nan)
                ),
                "baiyepower_v2_c1_linear_arc_curriculum_active": float(
                    isinstance(
                        getattr(self, "heading_route_state", None),
                        dict,
                    )
                    and str(self.heading_route_state.get("mode", ""))
                    == "arc"
                    and str(
                        self.heading_route_state.get(
                            "arc_interpolation_mode",
                            "smoothstep",
                        )
                    )
                    == "linear"
                ),
            }
        )
        if getattr(self.task, "action_type", "discrete") == "continuous":
            sample.update(self._last_trim_ema_log)
            sample.update(
                getattr(
                    self,
                    "_last_coordinated_residual_log",
                    {},
                ).get(
                    agent_id,
                    {},
                )
            )
            sample.update(self._last_action_projector_log.get(agent_id, {}))
        return sample

    def _collect_numerical_terminal_sample(self, agent_id):
        """Build an explicitly invalid sample without reading failed physics."""

        context = self.task.get_baiyepower_v2_transition_context(agent_id)
        sample = {
            "sample_valid": False,
            "step_idx": int(self.current_step),
            "sim_time": float(self.current_step * self.time_interval),
            "target_heading_deg": float(
                np.rad2deg(context.target_heading_rad)
            ),
            "target_altitude_ft": float(
                context.target_altitude_m / 0.3048
            ),
            "target_velocities_u_mps": float(context.target_speed_mps),
            "target_speed_mps": float(context.target_speed_mps),
        }
        for key in (
            "delta_heading_deg",
            "delta_altitude_m",
            "delta_speed_mps",
            "speed_mps",
            "roll_deg",
            "pitch_deg",
            "heading_deg",
            "roll_rate_deg_s",
            "pitch_rate_deg_s",
            "yaw_rate_deg_s",
            "pitch_accel_deg_s2",
            "aoa_deg",
            "vertical_speed_fps",
            "vertical_speed_mps",
            "vertical_speed_ft_min",
            "aileron_cmd_norm",
            "elevator_cmd_norm",
            "elevator_pos_norm",
            "rudder_cmd_norm",
            "throttle_cmd_norm",
        ):
            sample[key] = np.nan
        if getattr(self.task, "action_type", "discrete") == "continuous":
            sample.update(
                getattr(
                    self,
                    "_last_coordinated_residual_log",
                    {},
                ).get(
                    agent_id,
                    {},
                )
            )
            sample.update(self._last_action_projector_log.get(agent_id, {}))
        return sample

    def _update_heading_metrics(
        self,
        sample,
        target_changed,
        target_after,
        done,
        termination_reason,
        *,
        target_change_success=True,
        target_change_end_reason="next_target",
    ):
        if self._heading_metrics is None:
            return
        self._heading_metrics.update(
            sample,
            target_changed=target_changed,
            new_target=target_after,
            target_change_success=target_change_success,
            target_change_end_reason=target_change_end_reason,
            done=done,
            termination_reason=termination_reason,
        )
        self._last_heading_step = {
            **sample,
            "target_changed": bool(target_changed),
            "next_target_heading_deg": float(target_after[0]),
            "next_target_altitude_ft": float(target_after[1]),
            "next_target_velocities_u_mps": float(target_after[2]),
            "done": bool(done),
            "termination_reason": termination_reason,
        }

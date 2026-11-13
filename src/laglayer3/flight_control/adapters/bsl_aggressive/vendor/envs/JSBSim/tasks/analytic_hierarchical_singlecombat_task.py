"""C18 hierarchical combat task with the frozen C17 analytic inner loop."""

import math

import numpy as np
from gymnasium import spaces

from .singlecombat_task import SingleCombatTask
from ..core.catalog import Catalog as c
from ..utils.baiyepower_v2_physics import compute_energy_cost_terms
from ..utils.action_projector import ConvexActionProjector
from ..utils.action_smoother import MinimumJerkActionSmoother
from ..utils.coordinated_action import (
    compose_energy_exchange_fixed_gain,
    coordinated_reference_action,
    measured_energy_exchange_offset,
    scheduled_flight_path_tau,
    tecs_energy_allocation_commands,
    update_energy_trim_estimate,
)


class AnalyticHierarchicalSingleCombatTask(SingleCombatTask):
    """Let PPO choose maneuver intent while C17 closes the control loops.

    PPO retains the existing tactical observation and ``3 x 5 x 3`` action
    space.  Each discrete action becomes a short-horizon altitude, course,
    and speed offset.  The deterministic C17 controller converts that intent
    to actuator commands; PPO never commands an actuator directly.
    """

    def __init__(self, config):
        super().__init__(config)
        self.altitude_offsets_m = self._finite_vector(
            "C18_altitude_offsets_m",
            getattr(config, "C18_altitude_offsets_m", [150.0, 0.0, -150.0]),
            3,
        )
        self.heading_offsets_deg = self._finite_vector(
            "C18_heading_offsets_deg",
            getattr(config, "C18_heading_offsets_deg", [-30.0, -15.0, 0.0, 15.0, 30.0]),
            5,
        )
        self.speed_offsets_mps = self._finite_vector(
            "C18_speed_offsets_mps",
            getattr(config, "C18_speed_offsets_mps", [15.0, 0.0, -15.0]),
            3,
        )
        self._validate_inner_loop_config()
        self._projector = self._make_projector()
        self._smoother = self._make_smoother()
        self._energy_trim_states = {}
        self._analytic_agents = set()
        self._last_inner_loop_log = {}

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def reset(self, env):
        self._energy_trim_states = {}
        self._analytic_agents = set()
        self._last_inner_loop_log = {}
        for agent_id, sim in env.agents.items():
            initial = np.asarray(
                sim.get_property_values(self.action_var),
                dtype=np.float32,
            )
            self._projector.reset_agent(agent_id, initial)
            self._smoother.reset_agent(agent_id, initial)
        return super().reset(env)

    def normalize_action(self, env, agent_id, action):
        if self.use_baseline and agent_id in env.enm_ids:
            self._analytic_agents.discard(agent_id)
            return self.baseline_agent.get_action(env.agents[agent_id])

        indices = np.asarray(action).reshape(-1)
        if indices.shape != (3,) or not np.all(np.isfinite(indices)):
            raise ValueError("C18 high-level action must be a finite 3D vector")
        rounded = np.rint(indices).astype(np.int64)
        if not np.allclose(indices, rounded):
            raise ValueError("C18 high-level action indices must be integers")
        if np.any(rounded < 0) or np.any(rounded >= np.array([3, 5, 3])):
            raise ValueError("C18 high-level action lies outside MultiDiscrete bounds")

        sim = env.agents[agent_id]
        altitude_offset_m = float(self.altitude_offsets_m[rounded[0]])
        heading_offset_deg = float(self.heading_offsets_deg[rounded[1]])
        speed_offset_mps = float(self.speed_offsets_mps[rounded[2]])
        (altitude_offset_m, heading_offset_deg, speed_offset_mps), governor_log = self._govern_maneuver_offsets(
            sim, altitude_offset_m, heading_offset_deg, speed_offset_mps,
        )
        command, log = self._analytic_command(
            agent_id,
            sim,
            altitude_offset_m=altitude_offset_m,
            heading_offset_deg=heading_offset_deg,
            speed_offset_mps=speed_offset_mps,
        )
        projected = self._projector.project(agent_id, command)
        context = self._smoother_context(
            sim,
            altitude_offset_m,
            heading_offset_deg,
            speed_offset_mps,
        )
        self._smoother.set_target(agent_id, projected, context=context)
        self._analytic_agents.add(agent_id)
        log.update(self._projector.get_last_log(agent_id))
        for index, channel in enumerate(
            ("aileron", "elevator", "rudder", "throttle")
        ):
            log["projected_{}".format(channel)] = float(projected[index])
        log.update({"high_level_action_{}".format(i): float(v) for i, v in enumerate(rounded)})
        log.update(context)
        log.update(governor_log)
        self._last_inner_loop_log[agent_id] = log
        return projected

    def _govern_maneuver_offsets(self, sim, altitude_m, heading_deg, speed_mps):
        """Identity for historical tasks; new protocols override before Actor input."""
        return (altitude_m, heading_deg, speed_mps), {}

    def sample_substep_action(self, env, agent_id, substep_index):
        if agent_id not in self._analytic_agents:
            return None
        return self._smoother.sample(agent_id)

    def get_inner_loop_log(self, agent_id):
        return dict(self._last_inner_loop_log.get(agent_id, {}))

    def get_reward(self, env, agent_id, info=...):
        reward, info = super().get_reward(env, agent_id, info=info)
        log = self._last_inner_loop_log.get(agent_id)
        if isinstance(log, dict):
            if not isinstance(info, dict):
                info = {}
            info["c18_inner_loop"] = dict(log)
        return reward, info

    def _analytic_command(
        self,
        agent_id,
        sim,
        *,
        altitude_offset_m,
        heading_offset_deg,
        speed_offset_mps,
    ):
        altitude_m = float(sim.get_property_value(c.position_h_sl_m))
        u_mps = float(sim.get_property_value(c.velocities_u_mps))
        v_mps = float(sim.get_property_value(c.velocities_v_mps))
        w_mps = float(sim.get_property_value(c.velocities_w_mps))
        speed_mps = float(math.sqrt(u_mps * u_mps + v_mps * v_mps + w_mps * w_mps))
        target_altitude_m = altitude_m + altitude_offset_m
        target_speed_mps = max(40.0, speed_mps + speed_offset_mps)
        energy_terms = compute_energy_cost_terms(
            altitude_m=altitude_m,
            target_altitude_m=target_altitude_m,
            u_mps=u_mps,
            v_mps=v_mps,
            w_mps=w_mps,
            target_speed_mps=target_speed_mps,
            height_cost_scale_m=float(getattr(self.config, "BaiyePowerV2_height_cost_scale_m", 90.0)),
            speed_cost_scale_mps=float(getattr(self.config, "BaiyePowerV2_speed_cost_scale_mps", 20.0)),
            gravity_mps2=float(getattr(self.config, "BaiyePowerV2_gravity_mps2", 9.80665)),
        )

        trim = self._update_trim(agent_id, sim, energy_terms.energy_height_m)
        _, throttle_action, allocator_log = tecs_energy_allocation_commands(
            target_vertical_speed_mps=0.0,
            target_speed_mps=target_speed_mps,
            target_speed_rate_mps2=0.0,
            total_energy_error_m=energy_terms.e_T_m,
            energy_balance_error_m=energy_terms.e_B_m,
            total_energy_tau_sec=float(getattr(self.config, "CoordinatedResidual_total_energy_tau_sec", 20.0)),
            energy_balance_tau_sec=float(getattr(self.config, "CoordinatedResidual_energy_balance_tau_sec", 20.0)),
            throttle_energy_rate_gain=float(getattr(self.config, "CoordinatedResidual_throttle_energy_rate_gain", 0.0031909191384566376)),
            trim_physical_throttle=trim,
            policy_throttle_center=float(getattr(self.config, "CoordinatedResidual_policy_throttle_center", 0.625)),
            policy_throttle_half_range=float(getattr(self.config, "CoordinatedResidual_policy_throttle_half_range", 0.275)),
            max_balance_correction_mps=float(getattr(self.config, "CoordinatedResidual_max_balance_correction_mps", 10.0)),
            gravity_mps2=float(getattr(self.config, "BaiyePowerV2_gravity_mps2", 9.80665)),
        )
        exchange_base, exchange_log = measured_energy_exchange_offset(
            height_energy_error_m=energy_terms.e_h_m,
            kinetic_energy_error_m=energy_terms.e_k_m,
            closure_tau_sec=float(getattr(self.config, "CoordinatedResidual_energy_exchange_tau_sec", 60.0)),
            height_rate_gain=float(getattr(self.config, "CoordinatedResidual_energy_exchange_height_rate_gain", 0.9042178500025742)),
            kinetic_rate_gain_magnitude=float(getattr(self.config, "CoordinatedResidual_energy_exchange_kinetic_rate_gain", 0.6609097763737128)),
            max_offset_mps=float(getattr(self.config, "CoordinatedResidual_max_energy_exchange_offset_mps", 1.0)),
        )
        exchange_residual, gain_log = compose_energy_exchange_fixed_gain(
            exchange_base,
            float(getattr(self.config, "CoordinatedResidual_fixed_energy_exchange_gain", 1.9739999864399433)),
        )
        exchange_effective = float(
            gain_log["effective_energy_exchange_offset_mps"]
        )

        north = float(sim.get_property_value(c.velocities_v_north_mps))
        east = float(sim.get_property_value(c.velocities_v_east_mps))
        horizontal_speed = float(math.hypot(north, east))
        flight_path_tau, tau_log = scheduled_flight_path_tau(
            speed_mps=horizontal_speed,
            altitude_m=altitude_m,
            low_speed_tau_sec=float(getattr(self.config, "CoordinatedResidual_flight_path_tau_low_speed_sec", 0.9)),
            high_speed_low_alt_tau_sec=float(getattr(self.config, "CoordinatedResidual_flight_path_tau_high_speed_low_alt_sec", 0.97)),
            high_speed_high_alt_tau_sec=float(getattr(self.config, "CoordinatedResidual_flight_path_tau_high_speed_high_alt_sec", 1.0)),
            speed_blend_start_mps=float(getattr(self.config, "CoordinatedResidual_flight_path_tau_speed_blend_start_mps", 240.0)),
            speed_blend_end_mps=float(getattr(self.config, "CoordinatedResidual_flight_path_tau_speed_blend_end_mps", 260.0)),
            altitude_blend_start_m=float(getattr(self.config, "CoordinatedResidual_flight_path_tau_altitude_blend_start_m", 5500.0)),
            altitude_blend_end_m=float(getattr(self.config, "CoordinatedResidual_flight_path_tau_altitude_blend_end_m", 6500.0)),
        )
        vertical_speed = float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048
        policy_command = coordinated_reference_action(
            target_course_rate_deg_s=0.0,
            course_error_deg=heading_offset_deg,
            horizontal_speed_mps=horizontal_speed,
            roll_rad=float(sim.get_property_value(c.attitude_phi_rad)),
            pitch_rad=float(sim.get_property_value(c.attitude_theta_rad)),
            vertical_speed_mps=vertical_speed,
            target_vertical_speed_mps=0.0,
            altitude_error_m=altitude_offset_m,
            speed_error_mps=speed_offset_mps,
            course_tau_sec=float(getattr(self.config, "CoordinatedResidual_course_tau_sec", 9.09090909090909)),
            bank_tau_sec=float(getattr(self.config, "CoordinatedResidual_bank_tau_sec", 1.5)),
            flight_path_tau_sec=flight_path_tau,
            altitude_tau_sec=float(getattr(self.config, "CoordinatedResidual_altitude_tau_sec", 20.0)),
            speed_throttle_gain=float(getattr(self.config, "CoordinatedResidual_speed_throttle_gain", 0.01)),
            max_course_rate_deg_s=float(getattr(self.config, "CoordinatedResidual_max_course_rate_deg_s", 3.0)),
            max_bank_deg=float(getattr(self.config, "CoordinatedResidual_max_bank_deg", 45.0)),
            max_altitude_correction_mps=float(getattr(self.config, "CoordinatedResidual_max_altitude_correction_mps", 10.0)),
            desired_vertical_speed_offset_mps=exchange_effective,
            throttle_action_override=throttle_action,
            sideslip_deg=float(sim.get_property_value(c.aero_beta_deg)),
            sideslip_rudder_gain_per_deg=float(getattr(self.config, "CoordinatedResidual_sideslip_rudder_gain_per_deg", -0.05)),
            max_rudder_action=float(getattr(self.config, "CoordinatedResidual_max_rudder_action", 0.2)),
        )
        throttle_center = float(getattr(self.config, "CoordinatedResidual_policy_throttle_center", 0.625))
        throttle_half_range = float(getattr(self.config, "CoordinatedResidual_policy_throttle_half_range", 0.275))
        physical_command = policy_command.copy()
        physical_command[3] = policy_command[3] * throttle_half_range + throttle_center
        log = {
            "altitude_offset_m": altitude_offset_m,
            "heading_offset_deg": heading_offset_deg,
            "speed_offset_mps": speed_offset_mps,
            "energy_height_error_m": float(energy_terms.e_h_m),
            "energy_kinetic_error_m": float(energy_terms.e_k_m),
            "energy_total_error_m": float(energy_terms.e_T_m),
            "energy_balance_error_m": float(energy_terms.e_B_m),
            "trim_physical_throttle": trim,
            "exchange_base_mps": float(exchange_base),
            "exchange_residual_mps": float(exchange_residual),
            "exchange_effective_mps": exchange_effective,
            "flight_path_tau_sec": float(flight_path_tau),
        }
        log.update({"allocator_{}".format(k): float(v) for k, v in allocator_log.items()})
        log.update({"exchange_{}".format(k): float(v) for k, v in exchange_log.items()})
        log.update(
            {
                "gain_{}".format(k): float(v)
                for k, v in gain_log.items()
                if np.asarray(v).ndim == 0
            }
        )
        log.update({"tau_{}".format(k): float(v) for k, v in tau_log.items()})
        return physical_command.astype(np.float32), log

    def _update_trim(self, agent_id, sim, energy_height_m):
        now = float(sim.get_property_value(c.simulation_sim_time_sec))
        configured = float(getattr(self.config, "CoordinatedResidual_trim_physical_throttle", 0.4491547582071449))
        state = self._energy_trim_states.get(agent_id)
        if state is None:
            state = {"trim": configured, "energy": float(energy_height_m), "time": now}
        else:
            dt = now - state["time"]
            if dt > 1e-9:
                observed_rate = (float(energy_height_m) - state["energy"]) / dt
                state["trim"], _ = update_energy_trim_estimate(
                    previous_trim_throttle=state["trim"],
                    actual_throttle=float(sim.get_property_value(c.fcs_throttle_cmd_norm)),
                    observed_energy_rate_mps=observed_rate,
                    plant_gain_energy_rate_per_throttle=float(getattr(self.config, "CoordinatedResidual_energy_rate_plant_gain", 313.38932658872494)),
                    dt_sec=dt,
                    adaptation_tau_sec=float(getattr(self.config, "CoordinatedResidual_energy_trim_adaptation_tau_sec", 3.8)),
                    min_trim_throttle=float(getattr(self.config, "CoordinatedResidual_min_adaptive_trim_throttle", 0.35)),
                    max_trim_throttle=float(getattr(self.config, "CoordinatedResidual_max_adaptive_trim_throttle", 0.75)),
                )
            state["energy"] = float(energy_height_m)
            state["time"] = now
        self._energy_trim_states[agent_id] = state
        return float(state["trim"])

    def _smoother_context(self, sim, altitude_offset_m, heading_offset_deg, speed_offset_mps):
        vertical_speed = float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048
        pitch_rate_deg_s = math.degrees(float(sim.get_property_value(c.velocities_q_rad_sec)))
        pitch_deg = float(sim.get_property_value(c.attitude_theta_deg))
        urgency = max(abs(heading_offset_deg) / 25.0, abs(altitude_offset_m) / 250.0, abs(speed_offset_mps) / 25.0)
        vertical_activity = max(abs(pitch_rate_deg_s) / 8.0, abs(vertical_speed) / 8.0, abs(pitch_deg) / 12.0)
        return {
            "urgency": float(np.clip(urgency, 0.0, 1.0)),
            "aileron_turn_need": float(np.clip(abs(heading_offset_deg) / 20.0, 0.0, 1.0)),
            "vertical_activity": float(np.clip(vertical_activity, 0.0, 1.0)),
            "momentum_envelope_excess": 0.0,
        }

    def _make_projector(self):
        return ConvexActionProjector(
            action_dim=4,
            action_min=getattr(self.config, "ActionProjector_action_min", [-1.0, -1.0, -1.0, 0.35]),
            action_max=getattr(self.config, "ActionProjector_action_max", [1.0, 1.0, 1.0, 0.9]),
            slew_limit=getattr(self.config, "ActionProjector_slew_limit", [0.55, 0.4, 0.5, 0.1]),
            raw_weight=getattr(self.config, "ActionProjector_raw_weight", 1.0),
            rate_weight=getattr(self.config, "ActionProjector_rate_weight", 0.04),
            jerk_weight=getattr(self.config, "ActionProjector_jerk_weight", 0.01),
            hessian_eps=getattr(self.config, "ActionProjector_hessian_eps", 1e-6),
            enabled_channels=getattr(self.config, "ActionProjector_apply_to", None),
            channel_names=("aileron", "elevator", "rudder", "throttle"),
        )

    def _make_smoother(self):
        duration = float(getattr(self.config, "ActionSmoother_duration_sec", 0.2))
        base_steps = int(np.clip(round(duration * float(getattr(self.config, "sim_freq", 60))), 1, int(getattr(self.config, "agent_interaction_steps", 12))))
        return MinimumJerkActionSmoother(
            action_dim=4,
            base_steps=base_steps,
            enabled_channels=getattr(self.config, "ActionSmoother_apply_to", None),
            large_delta_threshold=float(getattr(self.config, "ActionSmoother_large_delta_threshold", 0.65)),
            large_delta_duration_scale=float(getattr(self.config, "ActionSmoother_large_delta_duration_scale", 0.5)),
            target_slew_enable=bool(getattr(self.config, "ActionSmoother_target_slew_enable", False)),
            target_deadband=getattr(self.config, "ActionSmoother_target_deadband", None),
            target_slew_base=getattr(self.config, "ActionSmoother_target_slew_base", None),
            urgency_enable=bool(getattr(self.config, "ActionSmoother_urgency_enable", True)),
            urgency_gain=float(getattr(self.config, "ActionSmoother_urgency_gain", 1.0)),
            aileron_turn_boost_enable=bool(getattr(self.config, "ActionSmoother_aileron_turn_boost_enable", True)),
            aileron_turn_boost_gain=float(getattr(self.config, "ActionSmoother_aileron_turn_boost_gain", 0.7)),
            aileron_max_gate=float(getattr(self.config, "ActionSmoother_aileron_max_gate", 2.0)),
            elevator_vertical_gate_enable=bool(getattr(self.config, "ActionSmoother_elevator_vertical_gate_enable", True)),
            elevator_vertical_damping_gain=float(getattr(self.config, "ActionSmoother_elevator_vertical_damping_gain", 0.45)),
            elevator_min_gate=float(getattr(self.config, "ActionSmoother_elevator_min_gate", 0.45)),
            elevator_max_gate=float(getattr(self.config, "ActionSmoother_elevator_max_gate", 2.0)),
        )

    def _validate_inner_loop_config(self):
        # C26 is a SingleControl prototype, not a validated C23 deployment.
        # Reject unsupported overrides rather than silently ignore them.
        unsupported = {
            "CoordinatedResidual_flight_path_geometry": "legacy_asin_horizontal",
            "CoordinatedResidual_direct_flight_path_feedback_gain_per_rad": 0.0,
            "CoordinatedResidual_direct_flight_path_feedback_max_action": 0.0,
            "CoordinatedResidual_pitch_rate_feedback_gain": 0.0,
            "CoordinatedResidual_turn_pitch_coupling_gain": 1.0,
            "CoordinatedResidual_turn_pitch_actual_rate_blend": 0.0,
        }
        for field, supported in unsupported.items():
            if getattr(self.config, field, supported) != supported:
                raise ValueError("1v1 inner loop does not support override {}; requires a separately validated adapter".format(field))
        gain = float(getattr(self.config, "CoordinatedResidual_fixed_energy_exchange_gain", np.nan))
        if not np.isfinite(gain) or gain < 0.0 or gain > 2.0:
            raise ValueError("C18 fixed energy-exchange gain must lie in [0, 2]")
        if not bool(getattr(self.config, "ActionProjector_enable", False)):
            raise ValueError("C18 requires ActionProjector_enable=true")
        if not bool(getattr(self.config, "ActionSmoother_enable", False)):
            raise ValueError("C18 requires ActionSmoother_enable=true")

    @staticmethod
    def _finite_vector(name, values, size):
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError("{} must be a finite {}D vector".format(name, size))
        return vector

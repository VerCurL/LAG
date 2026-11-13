"""Independent signed actuator-residual PPO task; no tactical policy.

Targets are integrated from an episode anchor, never from current aircraft
state. The inherited V2 lifecycle supplies exact safety/truncation semantics.
"""

import math

import numpy as np

from .baiyepower_v2_task import BaiyePowerV2Task
from ..core.catalog import Catalog as c


SCENARIO = "1/baiyepower_maneuver_c28"
TASK_NAME = "baiyepower_maneuver_c28"
RESIDUAL_SCALE = (0.10, 0.05, 0.05, 0.80)
ENTRY_SEC, MANEUVER_SEC, EXIT_SEC = 12.0, 20.0, 20.0
EPISODE_STEPS = 260


def wrap_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def command_at(anchor, rates, time_sec):
    """Return reference and right-continuous rates for the upcoming action."""
    elapsed = float(np.clip(time_sec - ENTRY_SEC, 0.0, MANEUVER_SEC))
    target = np.asarray(anchor, dtype=np.float64) + elapsed * np.asarray(rates)
    target[0] %= 360.0
    active = ENTRY_SEC <= time_sec < ENTRY_SEC + MANEUVER_SEC
    phase = "entry" if time_sec < ENTRY_SEC else ("maneuver" if active else "recovery")
    return target, np.asarray(rates if active else (0.0, 0.0, 0.0)), phase


def read_state(sim):
    u, v, w = sim.get_property_values([c.velocities_u_mps, c.velocities_v_mps, c.velocities_w_mps])
    north, east = sim.get_property_values([c.velocities_v_north_mps, c.velocities_v_east_mps])
    roll, pitch = sim.get_property_values([c.attitude_roll_rad, c.attitude_pitch_rad])
    return {
        "altitude_m": float(sim.get_property_value(c.position_h_sl_m)),
        "speed_mps": float(math.sqrt(u*u + v*v + w*w)),
        "vertical_speed_mps": float(sim.get_property_value(c.velocities_h_dot_fps)) * .3048,
        "course_deg": math.degrees(math.atan2(east, north)) % 360.0,
        "roll_deg": math.degrees(roll), "upright_cos": math.cos(roll)*math.cos(pitch),
        "alpha_deg": float(sim.get_property_value(c.aero_alpha_deg)),
        "beta_deg": float(sim.get_property_value(c.aero_beta_deg)),
        "nx_g": float(sim.get_property_value(c.accelerations_n_pilot_x_norm)),
        "ny_g": float(sim.get_property_value(c.accelerations_n_pilot_y_norm)),
        "nz_g": float(sim.get_property_value(c.accelerations_n_pilot_z_norm)),
    }


def hard_violations(state):
    if not all(math.isfinite(value) for value in state.values()):
        return ["nonfinite"]
    checks = {
        "altitude": state["altitude_m"] >= 2500.0,
        "speed": 120.0 <= state["speed_mps"] <= 350.0,
        "alpha": abs(state["alpha_deg"]) <= 20.0,
        "beta": abs(state["beta_deg"]) <= 10.0,
        "roll": abs(state["roll_deg"]) <= 85.0,
        "upright": state["upright_cos"] > 0.0,
        "nx": abs(state["nx_g"]) <= 2.0,
        "ny": abs(state["ny_g"]) <= 2.0,
        "nz": abs(state["nz_g"]) <= 9.0,
    }
    return [name for name, passed in checks.items() if not passed]


def tracking_metrics(state, previous_course, target, rates, dt):
    actual_rate = wrap_degrees(state["course_deg"] - previous_course) / dt
    return {
        "course_rate_error": actual_rate - float(rates[0]),
        "actual_course_rate": actual_rate,
        "vs_error": state["vertical_speed_mps"] - float(rates[1]),
        "altitude_error": state["altitude_m"] - float(target[1]),
        "speed_error": state["speed_mps"] - float(target[2]),
        "heading_error": wrap_degrees(state["course_deg"] - target[0]),
        "roll_deg": state["roll_deg"],
        "vertical_speed_mps": state["vertical_speed_mps"],
    }


def tracking_reward(metrics, dt, turning):
    scales = {"course_rate_error": 1.5, "vs_error": 3.0,
              "altitude_error": 100.0, "speed_error": 10.0, "heading_error": 15.0}
    costs = [abs(float(metrics[key])) / scale for key, scale in scales.items()]
    if not turning:
        costs.append(abs(float(metrics["roll_deg"])) / 10.0)
    if not all(math.isfinite(value) for value in costs):
        raise ValueError("C28 reward requires finite metrics")
    return float(dt / (1.0 + max(costs)))


class BaiyePowerManeuverC28Task(BaiyePowerV2Task):
    """Retain signed V2 observations, not the absolute-valued C17 ancestry."""

    metrics_use_route_attempt_boundaries = False

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        required = {
            "task": TASK_NAME, "CoordinatedResidual_enable": True,
            "CoordinatedResidual_mode": "actuator_residual_fixed_center",
            "CoordinatedResidual_flight_path_geometry": "atan2_horizontal",
            "CoordinatedResidual_max_course_rate_deg_s": 9.0,
            "CoordinatedResidual_max_bank_deg": 80.0,
            "CoordinatedResidual_turn_pitch_coupling_gain": .75,
            "CoordinatedResidual_pitch_rate_feedback_gain": .25,
            "CoordinatedResidual_direct_flight_path_feedback_gain_per_rad": 0.0,
            "CoordinatedResidual_direct_flight_path_feedback_max_action": 0.0,
            "DirectionInvariantObservation_mode": "signed_c28",
            "sim_freq": 60, "agent_interaction_steps": 12, "max_steps": EPISODE_STEPS,
            "BaiyePowerV2_unsafe_terminal_penalty": -20.0,
        }
        for key, expected in required.items():
            if getattr(self.config, key, None) != expected:
                raise ValueError("C28 identity mismatch: {}".format(key))
        if tuple(self.config.CoordinatedResidual_residual_scale) != RESIDUAL_SCALE:
            raise ValueError("C28 residual authority is frozen")
        if getattr(self.config, "DirectStudentTeacher_enable", False):
            raise ValueError("C28 may not use teacher forcing")

    def __init__(self, config):
        super().__init__(config)
        # Replaces, rather than adds to, the inherited direction/energy reward.
        self.reward_functions = []
        self._owns_commands = False
        self._substeps = []
        self._violations = set()

    def reset(self, env):
        super().reset(env)
        self._owns_commands = True
        self._substeps = []
        self._violations = set()
        advanced = float(env.training_progress) >= .20
        rates = (
            float(env.np_random.choice((-6, -3, 0, 3, 6) if advanced else (-3, 0, 3))),
            float(env.np_random.choice((-10, 0, 10) if advanced else (-5, 0, 5))),
            float(env.np_random.choice((-.5, 0, .5) if advanced else (0.0,))),
        )
        self.set_maneuver(env, rates)

    def set_maneuver(self, env, rates):
        """Frozen audits may specify an exact case before the first action."""
        if env.current_step != 0:
            raise ValueError("C28 maneuver may only be set before the first action")
        values = np.asarray(rates, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(np.abs(values) > [9, 10, .5]):
            raise ValueError("C28 rates outside registered domain")
        sim = next(iter(env.agents.values()))
        state = read_state(sim)
        self._anchor = np.array([state["course_deg"], state["altitude_m"], state["speed_mps"]])
        self._rates = values.copy()
        self._owns_commands = True
        BaiyePowerV2Task.set_baiyepower_v2_external_target_mode(self, True)
        self._apply_command(env, 0.0)

    def set_baiyepower_v2_external_target_mode(self, enabled=True):
        # Existing fixed-route evaluator owns commands after this explicit call.
        self._owns_commands = False
        super().set_baiyepower_v2_external_target_mode(enabled)

    def _apply_command(self, env, time_sec):
        target, rates, _ = command_at(self._anchor, self._rates, time_sec)
        for sim in env.agents.values():
            sim.set_property_value(c.target_heading_deg, float(target[0]))
            sim.set_property_value(c.target_altitude_ft, float(target[1] / .3048))
            self.set_target_speed_mps(sim, target[2])
        self.set_baiyepower_v2_external_target_rates(*rates)

    def begin_transition(self, env):
        super().begin_transition(env)
        agent_id = next(iter(env.agents))
        self._pre_state = read_state(env.agents[agent_id])
        self._step_rates = self.get_target_rate_commands(env, agent_id).copy()
        self._substeps = []
        self._violations = set()

    def on_simulation_substep(self, env, substep_index):
        state = read_state(next(iter(env.agents.values())))
        state["substep_index"] = int(substep_index)
        self._substeps.append(state)
        self._violations.update(hard_violations(state))

    def _collect_safety_flags(self, env, agent_id):
        flags = super()._collect_safety_flags(env, agent_id)
        # Latch every 60 Hz violation, including ones gone by the 5 Hz endpoint.
        flags["extreme"] = flags["extreme"] or bool(self._violations)
        return flags

    def get_reward(self, env, agent_id, info=None):
        info = {} if info is None else info
        if self.has_transition_numerical_failure(agent_id):
            return 0.0, info
        ctx = self.get_baiyepower_v2_transition_context(agent_id)
        target = np.array([math.degrees(ctx.target_heading_rad), ctx.target_altitude_m, ctx.target_speed_mps])
        # This transition's immutable pre-command plus its exact held rate.
        target += self._step_rates * env.time_interval
        state = read_state(env.agents[agent_id])
        metrics = tracking_metrics(state, self._pre_state["course_deg"], target, self._step_rates, env.time_interval)
        reward = tracking_reward(metrics, env.time_interval, abs(self._step_rates[0]) > 1e-9)
        phase = command_at(self._anchor, self._rates, (env.current_step-1)*env.time_interval)[2] if self._owns_commands else "external"
        info.update({"c28_" + key: value for key, value in metrics.items()})
        info.update(c28_phase=phase, c28_substep_count=len(self._substeps),
                    c28_hard_violations=sorted(self._violations), c28_reward=reward)
        log = env.get_coordinated_residual_log(agent_id)
        for name in ("aileron", "elevator", "rudder", "throttle"):
            info["c28_residual_"+name] = log["baiyepower_v2_c1_residual_used_"+name]
        info["c28_requested_rate"] = float(self._step_rates[0])
        self.record_baiyepower_v2_diag(agent_id, {"reward_contribution_sum": reward})
        return reward, info

    def get_termination(self, env, agent_id, info=None):
        done, info = super().get_termination(env, agent_id, info)
        if self._owns_commands:
            self._apply_command(env, env.current_step * env.time_interval)
        return done, info

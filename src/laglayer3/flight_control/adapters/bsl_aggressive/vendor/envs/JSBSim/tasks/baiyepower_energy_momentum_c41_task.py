"""Isolated C41 energy/direction reward with C28-compatible FCS plumbing.

The compatibility and pull/dive references share the same 54D observation and
execution chain. Only this task owns the new reward; C28/C30 remain unchanged.
"""

import math

import numpy as np
from gymnasium import spaces

from .baiyepower_maneuver_c28_task import (
    BaiyePowerManeuverC28Task, RESIDUAL_SCALE, command_at, read_state,
    tracking_metrics, tracking_reward,
)
from .baiyepower_v2_task import BaiyePowerV2Task
from ..core.catalog import Catalog as c
from ..utils.energy_momentum_kernel import EnergyMomentumConfig, compute_energy_momentum_reward
from ..utils.flight_path_geometry import geometry_from_ned, reference_direction, DirectionRateFilter
from ..utils.pull_dive_reference import PullDiveReference


TASK_NAME = "baiyepower_energy_momentum_c41"
SCENARIO = "1/baiyepower_energy_momentum_c41_e0"
CONTROL_SCENARIO = "1/baiyepower_energy_momentum_c41_e0_control"
PULL_DIVE_SCENARIO = "1/baiyepower_energy_momentum_c41_e0_pull_dive"
EXTRA_OBSERVATION_NAMES = (
    "direction_n", "direction_e", "direction_d",
    "reference_direction_n", "reference_direction_e", "reference_direction_d",
    "direction_rate_n", "direction_rate_e", "direction_rate_d",
    "reference_direction_rate_n", "reference_direction_rate_e", "reference_direction_rate_d",
    "total_energy_error", "energy_balance_error", "direction_valid", "course_valid",
    "phase_entry", "phase_pull_or_maneuver", "phase_dive", "phase_recovery", "phase_hold",
    "episode_remaining_fraction",
)


class BaiyePowerEnergyMomentumC41Task(BaiyePowerManeuverC28Task):
    """New opt-in task identity; emits ordinary reward, never terminal penalties."""

    def _validate_isolation_config(self):
        BaiyePowerV2Task._validate_isolation_config(self)
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
            "sim_freq": 60, "agent_interaction_steps": 12,
            "BaiyePowerV2_unsafe_terminal_penalty": -20.0,
            "BaiyePowerV2_numerical_terminal_penalty": -20.0,
            "C41_observation_schema": "spherical_append_v1",
        }
        for key, value in required.items():
            if getattr(self.config, key, None) != value:
                raise ValueError("C41 isolation mismatch: {}".format(key))
        for prefix in ("CoordinatedResidual", "BaiyePowerTargetRates", "BaiyePowerRewardRouting",
                       "DirectionInvariantObservation", "StructuredResidual"):
            if getattr(self.config, prefix + "_experiment_id", None) != "C41-E0":
                raise ValueError("C41 experiment identity mismatch: " + prefix)
        if tuple(self.config.CoordinatedResidual_residual_scale) != RESIDUAL_SCALE:
            raise ValueError("C41-E0 preserves C28 residual authority")
        if getattr(self.config, "DirectStudentTeacher_enable", False):
            raise ValueError("C41 does not use teacher forcing")
        if self.config.C41_reference_mode not in ("c28_compat", "pull_dive"):
            raise ValueError("unknown C41 reference mode")
        if self.config.C41_reward_mode not in ("dual_kernel", "c28_control"):
            raise ValueError("unknown C41 reward mode")
        if self.config.C41_reward_mode == "c28_control" and self.config.C41_reference_mode != "c28_compat":
            raise ValueError("C28 reward control is only defined for the C28-compatible task")
        if self.config.C41_reference_mode == "c28_compat" and self.config.max_steps != 260:
            raise ValueError("C41 compatibility horizon must be 260 decisions")

    def __init__(self, config):
        super().__init__(config)
        fields = EnergyMomentumConfig.__dataclass_fields__
        self.kernel_config = EnergyMomentumConfig(**{
            key: getattr(config, "C41_" + key) for key in fields
        })
        self._filter = DirectionRateFilter(float(config.C41_rate_filter_tau_sec))
        self._profile = None
        self._command_time_sec = 0.0
        self._actual_rate = np.zeros(3)
        self._reference_rate = np.zeros(3)
        self._c41_substeps = []
        self._last_reference = None
        self._rate_samples = 0
        for key in ("C41_min_speed_mps", "C41_min_horizontal_speed_mps"):
            value = float(getattr(config, key))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(key + " must be finite and positive")
        if config.C41_reference_mode == "pull_dive":
            profile = self._make_profile([0., 6000., 250.])
            duration = int(config.max_steps) * config.agent_interaction_steps / config.sim_freq
            if not math.isclose(profile.total_sec, duration, abs_tol=1e-9):
                raise ValueError("C41 max_steps must exactly cover the configured reference")
            if max(abs(float(config.C41_pull_angle_deg)), abs(float(config.C41_dive_angle_deg))) > 30:
                raise ValueError("C41-E0 profile is limited to the registered <=30 degree development range")

    def load_observation_space(self):
        super().load_observation_space()
        self.observation_space = spaces.Box(
            low=-10., high=10., shape=(32 + len(EXTRA_OBSERVATION_NAMES),), dtype=np.float32,
        )

    def _make_profile(self, anchor):
        keys = ("pull_angle_deg", "dive_angle_deg", "entry_sec", "pull_sec", "dive_sec",
                "recovery_sec", "hold_sec")
        return PullDiveReference(*anchor, sim_freq=self.config.sim_freq, **{
            key: float(getattr(self.config, "C41_" + key)) for key in keys
        })

    def _geometry(self, sim):
        north, east = sim.get_property_values([c.velocities_v_north_mps, c.velocities_v_east_mps])
        down = -float(sim.get_property_value(c.velocities_h_dot_fps)) * .3048
        return geometry_from_ned(
            [north, east, down], min_speed=self.config.C41_min_speed_mps,
            min_horizontal_speed=self.config.C41_min_horizontal_speed_mps,
        )

    @staticmethod
    def _reference(target, rates, phase="external", remaining=0.):
        target, rates = np.asarray(target, dtype=float), np.asarray(rates, dtype=float)
        if not np.isfinite(target).all() or not np.isfinite(rates).all() or target[2] <= 0:
            raise ValueError("C41 reference must be finite with positive speed")
        ratio = float(rates[1] / target[2])
        if abs(ratio) > 1:
            raise ValueError("reference vertical speed exceeds its ground-speed magnitude")
        gamma = math.asin(ratio)
        phase_index = {"entry": 0, "maneuver": 1, "recovery": 3, "external": -1}[phase]
        return dict(target=target.copy(), rates=rates.copy(), gamma_rad=gamma,
                    direction=reference_direction(math.radians(target[0]), gamma),
                    phase=phase, phase_index=phase_index, remaining_sec=max(float(remaining), 0.))

    def _command_reference(self, env, agent_id):
        if self._profile is not None and self._owns_commands:
            return self._profile.at(self._command_time_sec)
        sim = env.agents[agent_id]
        target = [float(sim.get_property_value(c.target_heading_deg)),
                  float(sim.get_property_value(c.target_altitude_ft)) * .3048,
                  self.get_target_speed_mps(sim)]
        phase = command_at(self._anchor, self._rates, self._command_time_sec)[2] if self._owns_commands else "external"
        return self._reference(target, self.get_target_rate_commands(env, agent_id), phase,
                               self.config.max_steps * env.time_interval - self._command_time_sec)

    def _transition_reference(self, env, elapsed):
        if self._profile is not None and self._owns_commands:
            return self._profile.at(self._step_start_sec + elapsed)
        agent_id = next(iter(env.agents))
        ctx = self.get_baiyepower_v2_transition_context(agent_id)
        target = np.array([math.degrees(ctx.target_heading_rad), ctx.target_altitude_m, ctx.target_speed_mps])
        target += self._step_rates * elapsed
        phase = command_at(self._anchor, self._rates, self._step_start_sec)[2] if self._owns_commands else "external"
        return self._reference(target, self._step_rates, phase,
                               self.config.max_steps * env.time_interval - self._step_start_sec - elapsed)

    def _initialize_filter(self, env):
        agent_id = next(iter(env.agents))
        geometry = self._geometry(env.agents[agent_id])
        if not geometry.direction_valid:
            raise ValueError("C41 reset needs a valid initial velocity direction")
        reference = self._command_reference(env, agent_id)
        self._actual_rate, self._reference_rate = self._filter.reset(geometry.direction, reference["direction"])
        self._rate_samples = 0

    def reset(self, env):
        self._profile = None
        self._owns_commands = False
        self._command_time_sec = 0.
        self._actual_rate = np.zeros(3)
        self._reference_rate = np.zeros(3)
        self._c41_substeps = []
        self._last_reference = None
        self._rate_samples = 0
        if self.config.C41_reference_mode == "c28_compat":
            super().reset(env)
        else:
            BaiyePowerV2Task.reset(self, env)
            self._substeps, self._violations = [], set()
            state = read_state(next(iter(env.agents.values())))
            self._anchor = np.array([state["course_deg"], state["altitude_m"], state["speed_mps"]])
            self._rates = np.zeros(3)
            self._profile = self._make_profile(self._anchor)
            self._owns_commands = True
            BaiyePowerV2Task.set_baiyepower_v2_external_target_mode(self, True)
            self._apply_command(env, 0.)
            self._initialize_filter(env)

    def set_maneuver(self, env, rates):
        if self.config.C41_reference_mode != "c28_compat":
            raise ValueError("set_maneuver is only available for C28 compatibility")
        super().set_maneuver(env, rates)
        self._initialize_filter(env)

    def _apply_command(self, env, time_sec):
        self._command_time_sec = float(time_sec)
        if self._profile is None:
            return super()._apply_command(env, time_sec)
        reference = self._profile.at(time_sec)
        for sim in env.agents.values():
            sim.set_property_value(c.target_heading_deg, float(reference["target"][0]))
            sim.set_property_value(c.target_altitude_ft, float(reference["target"][1] / .3048))
            self.set_target_speed_mps(sim, reference["target"][2])
        self.set_baiyepower_v2_external_target_rates(*reference["rates"])

    def begin_transition(self, env):
        super().begin_transition(env)
        self._step_start_sec = (env.current_step - 1) * env.time_interval
        self._c41_substeps, self._last_reference = [], None

    def on_simulation_substep(self, env, substep_index):
        super().on_simulation_substep(env, substep_index)
        agent_id = next(iter(env.agents))
        if self.has_transition_numerical_failure(agent_id):
            return
        try:
            geometry = self._geometry(env.agents[agent_id])
            reference = self._transition_reference(env, (substep_index + 1) / float(env.sim_freq))
            self._last_reference = reference
            if not geometry.direction_valid:
                self._violations.add("direction_invalid")
                return
            self._actual_rate, self._reference_rate = self._filter.update(
                geometry.direction, reference["direction"], 1. / float(env.sim_freq),
            )
            self._rate_samples += 1
            self._c41_substeps.append(dict(
                time_sec=self._step_start_sec + (substep_index + 1) / float(env.sim_freq),
                direction=geometry.direction.tolist(), reference_direction=reference["direction"].tolist(),
                actual_direction_rate=self._actual_rate.tolist(), reference_direction_rate=self._reference_rate.tolist(),
                speed_mps=geometry.speed_mps, target=reference["target"].tolist(),
                rates=reference["rates"].tolist(), phase=reference["phase"],
            ))
        except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
            self.mark_baiyepower_v2_numerical_failure(agent_id, "c41_substep", error)

    def _kernel(self, env, agent_id, reference, geometry):
        return compute_energy_momentum_reward(
            float(env.agents[agent_id].get_property_value(c.position_h_sl_m)), geometry.speed_mps,
            reference["target"][1], reference["target"][2], geometry.direction, reference["direction"],
            self._actual_rate, self._reference_rate, env.time_interval, self.kernel_config,
        )

    def get_obs(self, env, agent_id):
        obs = super().get_obs(env, agent_id)
        geometry = self._geometry(env.agents[agent_id])
        reference = self._command_reference(env, agent_id)
        if geometry.direction_valid:
            terms = self._kernel(env, agent_id, reference, geometry)
            energy = [terms["e_T_m"] / terms["S_T_m"], terms["e_B_m"] / terms["S_B_m"]]
        else:
            energy = [0., 0.]  # Explicit validity bit; unsafe states receive no direction reward.
        phase = np.zeros(5)
        if reference["phase_index"] >= 0:
            phase[reference["phase_index"]] = 1.
        scale = math.radians(self.kernel_config.rate_scale_deg_s)
        extra = np.concatenate((geometry.direction, reference["direction"],
                                self._actual_rate / scale, self._reference_rate / scale, energy,
                                [geometry.direction_valid, geometry.course_valid], phase,
                                [reference["remaining_sec"] / (self.config.max_steps * env.time_interval)]))
        if extra.shape != (len(EXTRA_OBSERVATION_NAMES),) or not np.isfinite(extra).all():
            raise ValueError("C41 appended observation must be finite and schema-aligned")
        obs[32:] = np.clip(extra, -10., 10.)
        return obs.astype(np.float32, copy=False)

    def get_reward(self, env, agent_id, info=None):
        info = {} if info is None else info
        info.update(c41_substep_count=len(self._c41_substeps),
                    c41_physics_substep_count=len(self._substeps),
                    c41_rate_samples=self._rate_samples,
                    c41_hard_violations=sorted(self._violations),
                    c41_reward_mode=self.config.C41_reward_mode,
                    c41_reference_mode=self.config.C41_reference_mode)
        if self.has_transition_numerical_failure(agent_id):
            info.update(c41_reward=0., c41_reward_valid=False)
            return 0., info
        try:
            geometry = self._geometry(env.agents[agent_id])
            if not geometry.direction_valid or "direction_invalid" in self._violations:
                self._violations.add("direction_invalid")
                info.update(c41_reward=0., c41_reward_valid=False,
                            c41_hard_violations=sorted(self._violations))
                return 0., info
            if len(self._c41_substeps) != env.agent_interaction_steps:
                raise ValueError("C41 reward requires every simulation substep")
            reference = self._last_reference
            terms = self._kernel(env, agent_id, reference, geometry)
            state = read_state(env.agents[agent_id])
            metrics = tracking_metrics(state, self._pre_state["course_deg"], reference["target"],
                                       self._step_rates, env.time_interval)
            control_reward = tracking_reward(metrics, env.time_interval, abs(self._step_rates[0]) > 1e-9)
            reward = terms["running_reward"] if self.config.C41_reward_mode == "dual_kernel" else control_reward
            info.update({"c41_" + key: value for key, value in terms.items()})
            info.update(c41_reward=reward, c41_reward_valid=True, c41_control_reward=control_reward,
                        c41_reward_mode=self.config.C41_reward_mode, c41_phase=reference["phase"],
                        c41_reference_mode=self.config.C41_reference_mode,
                        c41_reference_target=reference["target"].tolist(),
                        c41_reference_rates=reference["rates"].tolist(),
                        c41_gamma_deg=math.degrees(geometry.gamma_rad),
                        c41_gamma_ref_deg=math.degrees(reference["gamma_rad"]),
                        c41_course_valid=geometry.course_valid, c41_direction_valid=geometry.direction_valid,
                        c41_substep_count=len(self._c41_substeps), c41_rate_samples=self._rate_samples,
                        c41_hard_violations=sorted(self._violations))
            self.record_baiyepower_v2_diag(agent_id, dict(
                {"c41_" + key: value for key, value in terms.items()}, reward_contribution_sum=reward,
            ))
            return float(reward), info
        except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
            self.mark_baiyepower_v2_numerical_failure(agent_id, "c41_reward", error)
            info.update(c41_reward=0., c41_reward_valid=False)
            return 0., info

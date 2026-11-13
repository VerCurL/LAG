import math
import numpy as np
from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition


class UnreachHeading(BaseTerminationCondition):
    """
    UnreachHeading [0, 1]
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        uid = list(config.aircraft_configs.keys())[0]
        aircraft_config = config.aircraft_configs[uid]
        self.max_heading_increment = aircraft_config['max_heading_increment']
        self.max_altitude_increment = aircraft_config['max_altitude_increment']
        self.max_velocities_u_increment = aircraft_config['max_velocities_u_increment']
        self.check_interval = aircraft_config['check_interval']
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10
        self.threshold_deg = float(getattr(config, f"{self.__class__.__name__}_threshold_deg", 10.0))
        self.curriculum_enable = bool(getattr(config, "HeadingCurriculum_enable", False))
        self.curriculum_progress = np.asarray(
            getattr(config, "HeadingCurriculum_progress", [0.0, 1.0]),
            dtype=np.float64,
        )
        self.curriculum_scales = np.asarray(
            getattr(config, "HeadingCurriculum_scales", [1.0, 1.0]),
            dtype=np.float64,
        )
        if self.curriculum_progress.shape != self.curriculum_scales.shape:
            raise ValueError("HeadingCurriculum_progress and HeadingCurriculum_scales must have the same length.")
        if self.curriculum_progress.size < 2:
            raise ValueError("HeadingCurriculum requires at least two progress points.")
        self.curriculum_progress = np.clip(self.curriculum_progress, 0.0, 1.0)
        self.curriculum_scales = np.clip(self.curriculum_scales, 0.0, 1.0)
        self.route_curriculum_enable = bool(getattr(config, "HeadingRouteCurriculum_enable", False))
        self.route_levels = self._load_route_levels(config)
        self.arc_curriculum_enable = bool(getattr(config, "HeadingArcCurriculum_enable", False))
        self.arc_levels = self._load_arc_levels(config)
        self.arc_interpolation_mode = str(
            getattr(
                config,
                "HeadingArcCurriculum_interpolation_mode",
                "smoothstep",
            )
        ).strip().lower()
        if self.arc_interpolation_mode not in ("smoothstep", "linear"):
            raise ValueError(
                "HeadingArcCurriculum_interpolation_mode must be "
                "smoothstep or linear"
            )
        self.arc_turn_rate_choices_deg_s = self._finite_choices(
            config,
            "HeadingArcCurriculum_turn_rate_choices_deg_s",
        )
        self.arc_altitude_rate_choices_mps = self._finite_choices(
            config,
            "HeadingArcCurriculum_altitude_rate_choices_mps",
        )
        self.arc_speed_rate_choices_mps2 = self._finite_choices(
            config,
            "HeadingArcCurriculum_speed_rate_choices_mps2",
        )
        self.mixed_curriculum_enable = bool(getattr(config, "HeadingMixedCurriculum_enable", False))
        self.mixed_schedule = self._load_mixed_schedule(config)
        self.hold_heading_tol_deg = float(getattr(config, "HeadingRouteHold_heading_tol_deg", 5.0))
        self.hold_altitude_tol_m = float(getattr(config, "HeadingRouteHold_altitude_tol_m", 100.0))
        self.hold_speed_tol_mps = float(getattr(config, "HeadingRouteHold_speed_tol_mps", 20.0))
        self.hold_roll_tol_deg = float(getattr(config, "HeadingRouteHold_roll_tol_deg", 20.0))
        self.hold_pitch_rate_tol_deg_s = float(getattr(config, "HeadingRouteHold_pitch_rate_tol_deg_s", 8.0))
        self.hold_vertical_speed_tol_mps = float(getattr(config, "HeadingRouteHold_vertical_speed_tol_mps", 5.0))
        self.hold_decay = float(getattr(config, "HeadingRouteHold_decay", 2.0))

    @staticmethod
    def _finite_choices(config, name):
        if not hasattr(config, name):
            return None
        values = np.asarray(getattr(config, name), dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("{} must contain finite values".format(name))
        return values

    def _load_route_levels(self, config):
        default_levels = [
            {
                "level": 0,
                "progress_min": 0.0,
                "progress_max": 0.15,
                "point_count": 1,
                "max_heading_delta_deg": 10.0,
                "max_altitude_delta_ft": 300.0,
                "max_speed_delta_mps": 10.0,
                "last_hold_sec": 0.0,
                "max_point_time_sec": 40.0,
                "hold_conditions": ["heading", "altitude"],
            },
            {
                "level": 1,
                "progress_min": 0.15,
                "progress_max": 0.30,
                "point_count": 1,
                "max_heading_delta_deg": 30.0,
                "max_altitude_delta_ft": 1000.0,
                "max_speed_delta_mps": 20.0,
                "last_hold_sec": 3.0,
                "max_point_time_sec": 40.0,
                "hold_conditions": ["heading", "altitude", "speed"],
            },
            {
                "level": 2,
                "progress_min": 0.30,
                "progress_max": 1.0,
                "point_count": 1,
                "max_heading_delta_deg": 60.0,
                "max_altitude_delta_ft": 2000.0,
                "max_speed_delta_mps": 35.0,
                "last_hold_sec": 5.0,
                "max_point_time_sec": 50.0,
                "hold_conditions": ["heading", "altitude", "speed", "roll", "vertical_speed"],
            },
        ]
        levels = getattr(config, "HeadingRouteCurriculum_levels", default_levels)
        cleaned = []
        for idx, level in enumerate(levels):
            item = dict(level)
            item.setdefault("level", idx)
            item.setdefault("progress_min", 0.0)
            item.setdefault("progress_max", 1.0)
            item.setdefault("point_count", 1)
            item.setdefault("max_heading_delta_deg", 30.0)
            item.setdefault("max_altitude_delta_ft", 1000.0)
            item.setdefault("max_speed_delta_mps", 20.0)
            item.setdefault("last_hold_sec", 0.0)
            item.setdefault("mid_hold_sec", item["last_hold_sec"])
            item.setdefault("max_point_time_sec", self.check_interval)
            item.setdefault("hold_conditions", ["heading", "altitude", "speed"])
            cleaned.append(item)
        return cleaned

    def _load_arc_levels(self, config):
        default_levels = [
            {
                "level": 0,
                "progress_min": 0.0,
                "progress_max": 1.0,
                "turn_total_deg": 45.0,
                "duration_sec": 45.0,
                "altitude_delta_ft": 0.0,
                "speed_delta_mps": 0.0,
                "max_point_time_sec": 55.0,
            },
        ]
        levels = getattr(config, "HeadingArcCurriculum_levels", default_levels)
        cleaned = []
        for idx, level in enumerate(levels):
            item = dict(level)
            item.setdefault("level", idx)
            item.setdefault("progress_min", 0.0)
            item.setdefault("progress_max", 1.0)
            item.setdefault("turn_total_deg", 45.0)
            item.setdefault("duration_sec", 45.0)
            item.setdefault("altitude_delta_ft", 0.0)
            item.setdefault("speed_delta_mps", 0.0)
            item.setdefault("max_point_time_sec", float(item["duration_sec"]) + 10.0)
            item.setdefault("hold_conditions", ["heading", "altitude", "speed"])
            cleaned.append(item)
        return cleaned

    def _load_mixed_schedule(self, config):
        default_schedule = [
            {
                "progress_min": 0.0,
                "progress_max": 1.0,
                "point_prob": 1.0,
                "arc_prob": 0.0,
            },
        ]
        schedule = getattr(config, "HeadingMixedCurriculum_schedule", default_schedule)
        cleaned = []
        for idx, item in enumerate(schedule):
            if not isinstance(item, dict):
                raise ValueError(f"HeadingMixedCurriculum_schedule entry {idx} must be a mapping.")
            missing = [key for key in ("progress_min", "progress_max") if key not in item]
            if missing:
                raise ValueError(
                    f"HeadingMixedCurriculum_schedule entry {idx} missing required keys: {missing}"
                )
            entry = dict(item)
            progress_min = float(entry["progress_min"])
            progress_max = float(entry["progress_max"])
            if progress_max <= progress_min:
                raise ValueError(
                    f"HeadingMixedCurriculum_schedule entry {idx} must satisfy progress_max > progress_min."
                )
            entry["progress_min"] = progress_min
            entry["progress_max"] = progress_max
            entry.setdefault("point_prob", 1.0)
            entry.setdefault("arc_prob", 0.0)
            entry.setdefault("point_level_min", 0)
            entry.setdefault("point_level_max", max(0, len(self.route_levels) - 1))
            entry.setdefault("arc_level_min", 0)
            entry.setdefault("arc_level_max", max(0, len(self.arc_levels) - 1))
            cleaned.append(entry)
        return sorted(cleaned, key=lambda entry: float(entry.get("progress_min", 0.0)))

    def _get_curriculum_scale(self, env):
        if not self.curriculum_enable:
            return 1.0
        progress = float(getattr(env, "training_progress", 1.0))
        return float(np.interp(progress, self.curriculum_progress, self.curriculum_scales))

    def _get_route_level(self, env):
        progress = float(getattr(env, "training_progress", 1.0))
        selected = self.route_levels[-1]
        for level in self.route_levels:
            if progress >= float(level.get("progress_min", 0.0)) and progress < float(level.get("progress_max", 1.0)):
                selected = level
                break
        return selected

    def _get_arc_level(self, env, min_level=None, max_level=None):
        progress = float(getattr(env, "training_progress", 1.0))
        levels = self.arc_levels
        if min_level is not None or max_level is not None:
            lo = 0 if min_level is None else int(min_level)
            hi = len(levels) - 1 if max_level is None else int(max_level)
            levels = [level for level in levels if lo <= int(level.get("level", 0)) <= hi] or self.arc_levels
        selected = levels[-1]
        for level in levels:
            if progress >= float(level.get("progress_min", 0.0)) and progress < float(level.get("progress_max", 1.0)):
                selected = level
                break
        return selected

    def _choose_route_mode(self, env):
        if not (self.mixed_curriculum_enable and self.arc_curriculum_enable):
            return "point", self._get_route_level(env)
        progress = float(getattr(env, "training_progress", 1.0))
        schedule = self.mixed_schedule[-1]
        for entry in self.mixed_schedule:
            if progress >= float(entry.get("progress_min", 0.0)) and progress < float(entry.get("progress_max", 1.0)):
                schedule = entry
                break
        point_prob = max(float(schedule.get("point_prob", 1.0)), 0.0)
        arc_prob = max(float(schedule.get("arc_prob", 0.0)), 0.0)
        total = point_prob + arc_prob
        if total <= 1e-8 or env.np_random.random() < point_prob / total:
            level = self._get_route_level(env)
            lo = int(schedule.get("point_level_min", 0))
            hi = int(schedule.get("point_level_max", len(self.route_levels) - 1))
            if int(level.get("level", 0)) < lo or int(level.get("level", 0)) > hi:
                candidates = [x for x in self.route_levels if lo <= int(x.get("level", 0)) <= hi]
                level = candidates[-1] if candidates else level
            return "point", level
        return "arc", self._get_arc_level(
            env,
            min_level=schedule.get("arc_level_min", 0),
            max_level=schedule.get("arc_level_max", len(self.arc_levels) - 1),
        )

    @staticmethod
    def _wrap_heading_deg(value):
        return float((value + 360.0) % 360.0)

    def _current_speed_mps(self, sim):
        """Speed semantics used by this target generator.

        The base heading task intentionally retains its historical body-axis
        ``u`` definition.  Opt-in tasks may override these four small hooks
        without duplicating route lifecycle logic.
        """

        return float(sim.get_property_value(c.velocities_u_mps))

    def _target_speed_mps(self, sim):
        return float(sim.get_property_value(c.target_velocities_u_mps))

    def _set_target_speed_mps(self, sim, target_speed_mps):
        sim.set_property_value(c.target_velocities_u_mps, float(target_speed_mps))

    def _speed_error_mps(self, sim):
        return float(sim.get_property_value(c.delta_velocities_u))

    def _make_target(self, env, agent_id, level):
        sim = env.agents[agent_id]
        cur_heading = float(sim.get_property_value(c.attitude_psi_deg))
        cur_altitude = float(sim.get_property_value(c.position_h_sl_ft))
        cur_speed = self._current_speed_mps(sim)
        heading_delta = env.np_random.uniform(
            -float(level["max_heading_delta_deg"]),
            float(level["max_heading_delta_deg"]),
        )
        altitude_delta = env.np_random.uniform(
            -float(level["max_altitude_delta_ft"]),
            float(level["max_altitude_delta_ft"]),
        )
        speed_delta = env.np_random.uniform(
            -float(level["max_speed_delta_mps"]),
            float(level["max_speed_delta_mps"]),
        )
        return (
            self._wrap_heading_deg(cur_heading + heading_delta),
            max(cur_altitude + altitude_delta, 15000.0),
            max(cur_speed + speed_delta, 120.0),
        )

    def _set_target(self, env, agent_id, target):
        sim = env.agents[agent_id]
        sim.set_property_value(c.target_heading_deg, float(target[0]))
        sim.set_property_value(c.target_altitude_ft, float(target[1]))
        self._set_target_speed_mps(sim, target[2])

    @staticmethod
    def _smoothstep01(x):
        t = float(np.clip(x, 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    def _make_arc_state(self, env, agent_id, level):
        sim = env.agents[agent_id]
        start_heading = float(sim.get_property_value(c.attitude_psi_deg))
        start_altitude = float(sim.get_property_value(c.position_h_sl_ft))
        start_speed = self._current_speed_mps(sim)
        duration = max(float(level.get("duration_sec", 45.0)), 1e-6)
        if self.arc_turn_rate_choices_deg_s is None:
            direction = -1.0 if env.np_random.random() < 0.5 else 1.0
            turn_total = direction * float(level.get("turn_total_deg", 45.0))
        else:
            turn_rate = float(env.np_random.choice(self.arc_turn_rate_choices_deg_s))
            turn_total = turn_rate * duration
        if self.arc_altitude_rate_choices_mps is None:
            altitude_delta = env.np_random.uniform(-1.0, 1.0) * abs(float(level.get("altitude_delta_ft", 0.0)))
        else:
            altitude_rate = float(env.np_random.choice(self.arc_altitude_rate_choices_mps))
            altitude_delta = altitude_rate * duration / 0.3048
        if self.arc_speed_rate_choices_mps2 is None:
            speed_delta = env.np_random.uniform(-1.0, 1.0) * abs(float(level.get("speed_delta_mps", 0.0)))
        else:
            speed_rate = float(env.np_random.choice(self.arc_speed_rate_choices_mps2))
            speed_delta = speed_rate * duration
        return {
            "arc_start_heading_deg": start_heading,
            "arc_start_altitude_ft": start_altitude,
            "arc_start_speed_mps": start_speed,
            "arc_turn_total_deg": turn_total,
            "arc_altitude_delta_ft": altitude_delta,
            "arc_speed_delta_mps": speed_delta,
            "arc_duration_sec": duration,
            "arc_interpolation_mode": self.arc_interpolation_mode,
        }

    def _update_arc_target(self, env, agent_id):
        state = env.heading_route_state
        sim_time = float(env.agents[agent_id].get_property_value(c.simulation_sim_time_sec))
        elapsed = sim_time - float(state["point_start_time"])
        phase = elapsed / max(float(state.get("arc_duration_sec", 1.0)), 1e-6)
        if str(state.get("arc_interpolation_mode", "smoothstep")) == "linear":
            tau = float(np.clip(phase, 0.0, 1.0))
        else:
            tau = self._smoothstep01(phase)
        target = (
            self._wrap_heading_deg(float(state["arc_start_heading_deg"]) + float(state["arc_turn_total_deg"]) * tau),
            max(float(state["arc_start_altitude_ft"]) + float(state["arc_altitude_delta_ft"]) * tau, 15000.0),
            max(float(state["arc_start_speed_mps"]) + float(state["arc_speed_delta_mps"]) * tau, 120.0),
        )
        state["target"] = target
        self._set_target(env, agent_id, target)
        return elapsed, tau, target

    def _make_segment_quality_state(self, env, agent_id):
        sim = env.agents[agent_id]
        abs_heading = abs(float(sim.get_property_value(c.delta_heading)))
        abs_altitude = abs(float(sim.get_property_value(c.delta_altitude)))
        abs_speed = abs(self._speed_error_mps(sim))
        return {
            "command_heading_abs_deg": abs_heading,
            "command_altitude_abs_m": abs_altitude,
            "command_speed_abs_mps": abs_speed,
            "norm_heading_denom_deg": max(abs_heading, self.hold_heading_tol_deg, 1e-6),
            "norm_altitude_denom_m": max(abs_altitude, self.hold_altitude_tol_m, 1e-6),
            "norm_speed_denom_mps": max(abs_speed, self.hold_speed_tol_mps, 1e-6),
            "max_heading_abs_deg": abs_heading,
            "max_altitude_abs_m": abs_altitude,
            "max_speed_abs_mps": abs_speed,
        }

    def _update_segment_quality_state(self, env, agent_id):
        state = env.heading_route_state
        quality = state.setdefault("quality", self._make_segment_quality_state(env, agent_id))
        sim = env.agents[agent_id]
        quality["max_heading_abs_deg"] = max(
            float(quality.get("max_heading_abs_deg", 0.0)),
            abs(float(sim.get_property_value(c.delta_heading))),
        )
        quality["max_altitude_abs_m"] = max(
            float(quality.get("max_altitude_abs_m", 0.0)),
            abs(float(sim.get_property_value(c.delta_altitude))),
        )
        quality["max_speed_abs_mps"] = max(
            float(quality.get("max_speed_abs_mps", 0.0)),
            abs(self._speed_error_mps(sim)),
        )

    def _completion_quality_diag(self, env, agent_id, point_time):
        state = env.heading_route_state
        quality = state.setdefault("quality", self._make_segment_quality_state(env, agent_id))
        sim = env.agents[agent_id]
        hit_heading = abs(float(sim.get_property_value(c.delta_heading)))
        hit_altitude = abs(float(sim.get_property_value(c.delta_altitude)))
        hit_speed = abs(self._speed_error_mps(sim))
        denom_heading = max(float(quality.get("norm_heading_denom_deg", self.hold_heading_tol_deg)), 1e-6)
        denom_altitude = max(float(quality.get("norm_altitude_denom_m", self.hold_altitude_tol_m)), 1e-6)
        denom_speed = max(float(quality.get("norm_speed_denom_mps", self.hold_speed_tol_mps)), 1e-6)
        nr_heading = hit_heading / denom_heading
        nr_altitude = hit_altitude / denom_altitude
        nr_speed = hit_speed / denom_speed
        overshoot_heading = max(0.0, float(quality.get("max_heading_abs_deg", hit_heading)) - float(quality.get("command_heading_abs_deg", denom_heading))) / denom_heading
        overshoot_altitude = max(0.0, float(quality.get("max_altitude_abs_m", hit_altitude)) - float(quality.get("command_altitude_abs_m", denom_altitude))) / denom_altitude
        overshoot_speed = max(0.0, float(quality.get("max_speed_abs_mps", hit_speed)) - float(quality.get("command_speed_abs_mps", denom_speed))) / denom_speed
        return {
            "heading_route_completion_time_sec": float(point_time),
            "heading_route_completion_heading_error_deg": hit_heading,
            "heading_route_completion_altitude_error_m": hit_altitude,
            "heading_route_completion_speed_error_mps": hit_speed,
            "heading_route_norm_residual_heading": nr_heading,
            "heading_route_norm_residual_altitude": nr_altitude,
            "heading_route_norm_residual_speed": nr_speed,
            "heading_route_norm_residual_combined": float((nr_heading + nr_altitude + nr_speed) / 3.0),
            "heading_route_overshoot_heading": overshoot_heading,
            "heading_route_overshoot_altitude": overshoot_altitude,
            "heading_route_overshoot_speed": overshoot_speed,
            "heading_route_overshoot_combined": float((overshoot_heading + overshoot_altitude + overshoot_speed) / 3.0),
        }

    @staticmethod
    def _next_attempt_index(env, reset_sequence=False):
        if reset_sequence or not hasattr(env, "heading_route_attempt_index"):
            attempt_index = 0
        else:
            attempt_index = int(env.heading_route_attempt_index) + 1
        env.heading_route_attempt_index = attempt_index
        return attempt_index

    def _reset_route_state(self, env, agent_id, force=False, reset_attempt_sequence=False):
        if not force and hasattr(env, "heading_route_state"):
            return
        mode, level = self._choose_route_mode(env)
        arc_state = {}
        if mode == "arc":
            arc_state = self._make_arc_state(env, agent_id, level)
            target = (
                arc_state["arc_start_heading_deg"],
                arc_state["arc_start_altitude_ft"],
                arc_state["arc_start_speed_mps"],
            )
        else:
            target = self._make_target(env, agent_id, level)
        self._set_target(env, agent_id, target)
        sim_time = float(env.agents[agent_id].get_property_value(c.simulation_sim_time_sec))
        max_point_time = float(level.get("max_point_time_sec", self.check_interval))
        planned_duration_sec = max_point_time
        if mode == "arc":
            planned_duration_sec = float(arc_state.get("arc_duration_sec", 0.0))
            max_point_time = max(
                max_point_time,
                planned_duration_sec + 10.0,
            )
        attempt_index = self._next_attempt_index(env, reset_sequence=reset_attempt_sequence)
        quality = self._make_segment_quality_state(env, agent_id)
        if mode == "arc":
            attempt_command_heading_abs_deg = abs(float(arc_state["arc_turn_total_deg"]))
            arc_start_altitude_ft = float(arc_state["arc_start_altitude_ft"])
            arc_end_altitude_ft = max(
                arc_start_altitude_ft
                + float(arc_state["arc_altitude_delta_ft"]),
                15000.0,
            )
            attempt_command_altitude_abs_m = (
                abs(arc_end_altitude_ft - arc_start_altitude_ft) * 0.3048
            )
            arc_start_speed_mps = float(arc_state["arc_start_speed_mps"])
            arc_end_speed_mps = max(
                arc_start_speed_mps + float(arc_state["arc_speed_delta_mps"]),
                120.0,
            )
            attempt_command_speed_abs_mps = abs(
                arc_end_speed_mps - arc_start_speed_mps
            )
        else:
            attempt_command_heading_abs_deg = float(quality["command_heading_abs_deg"])
            attempt_command_altitude_abs_m = float(quality["command_altitude_abs_m"])
            attempt_command_speed_abs_mps = float(quality["command_speed_abs_mps"])
        initial_flags = self._condition_flags(env, agent_id)
        initial_conditions = list(
            level.get("hold_conditions", ["heading", "altitude", "speed"])
        )
        attempt_initial_in_zone = (
            None
            if mode == "arc"
            else bool(
                all(initial_flags.get(name, False) for name in initial_conditions)
            )
        )
        env.heading_route_state = {
            "mode": mode,
            "level": int(level.get("level", 0)),
            "level_config": level,
            "active_point_index": 0,
            "point_count": int(level.get("point_count", 1)),
            "hold_timer": 0.0,
            "point_start_time": sim_time,
            "target": target,
            "point_completed": False,
            "in_target_zone": False,
            "quality": quality,
            "attempt_id": attempt_index,
            "attempt_index": attempt_index,
            "attempt_start_time_sec": sim_time,
            "attempt_deadline_sec": sim_time + max_point_time,
            "attempt_planned_duration_sec": planned_duration_sec,
            "attempt_timeout_sec": max_point_time,
            "attempt_created_step": int(env.current_step),
            "attempt_command_heading_abs_deg": attempt_command_heading_abs_deg,
            "attempt_command_altitude_abs_m": attempt_command_altitude_abs_m,
            "attempt_command_speed_abs_mps": attempt_command_speed_abs_mps,
            "attempt_initial_in_zone": attempt_initial_in_zone,
        }
        env.heading_route_state.update(arc_state)

    def _condition_flags(self, env, agent_id):
        sim = env.agents[agent_id]
        vertical_speed_mps = float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048
        pitch_rate_deg_s = float(np.rad2deg(sim.get_property_value(c.velocities_q_rad_sec)))
        return {
            "heading": abs(float(sim.get_property_value(c.delta_heading))) <= self.hold_heading_tol_deg,
            "altitude": abs(float(sim.get_property_value(c.delta_altitude))) <= self.hold_altitude_tol_m,
            "speed": abs(self._speed_error_mps(sim)) <= self.hold_speed_tol_mps,
            "roll": abs(float(sim.get_property_value(c.attitude_phi_deg))) <= self.hold_roll_tol_deg,
            "pitch_rate": abs(pitch_rate_deg_s) <= self.hold_pitch_rate_tol_deg_s,
            "vertical_speed": abs(vertical_speed_mps) <= self.hold_vertical_speed_tol_mps,
        }

    def _route_diag(self, env, agent_id, flags, point_completed):
        state = env.heading_route_state
        level = state["level_config"]
        hold_required = float(level.get("last_hold_sec", 0.0))
        point_time = float(env.agents[agent_id].get_property_value(c.simulation_sim_time_sec)) - float(state["point_start_time"])
        return {
            "heading_route_level": int(state["level"]),
            "heading_route_mode": str(state.get("mode", "point")),
            "heading_route_attempt_id": int(state["attempt_id"]),
            "heading_route_attempt_index": int(state["attempt_index"]),
            "heading_route_attempt_start_time_sec": float(state["attempt_start_time_sec"]),
            "heading_route_attempt_deadline_sec": float(state["attempt_deadline_sec"]),
            "heading_route_attempt_planned_duration_sec": float(state["attempt_planned_duration_sec"]),
            "heading_route_attempt_timeout_sec": float(state["attempt_timeout_sec"]),
            "heading_route_attempt_started_this_step": bool(
                int(env.current_step) == int(state["attempt_created_step"])
            ),
            "heading_route_attempt_sample_valid": bool(
                int(env.current_step) > int(state["attempt_created_step"])
            ),
            "heading_route_attempt_command_heading_abs_deg": float(state["attempt_command_heading_abs_deg"]),
            "heading_route_attempt_command_altitude_abs_m": float(state["attempt_command_altitude_abs_m"]),
            "heading_route_attempt_command_speed_abs_mps": float(state["attempt_command_speed_abs_mps"]),
            "heading_route_attempt_initial_in_zone": state["attempt_initial_in_zone"],
            "heading_route_heading_tolerance_deg": float(self.hold_heading_tol_deg),
            "heading_route_altitude_tolerance_m": float(self.hold_altitude_tol_m),
            "heading_route_speed_tolerance_mps": float(self.hold_speed_tol_mps),
            "heading_route_attempt_outcome": "ongoing",
            "heading_route_active_point_index": int(state["active_point_index"]),
            "heading_route_point_count": int(state["point_count"]),
            "heading_route_in_target_zone": bool(state["in_target_zone"]),
            "heading_route_hold_timer": float(state["hold_timer"]),
            "heading_route_hold_required": hold_required,
            "heading_route_hold_progress": float(state["hold_timer"] / max(hold_required, 1e-6)) if hold_required > 0 else float(point_completed),
            "heading_route_time_since_point_start": point_time,
            "heading_route_point_completed": bool(point_completed),
            "heading_route_heading_ok": bool(flags.get("heading", False)),
            "heading_route_altitude_ok": bool(flags.get("altitude", False)),
            "heading_route_speed_ok": bool(flags.get("speed", False)),
            "heading_route_roll_ok": bool(flags.get("roll", False)),
            "heading_route_pitch_rate_ok": bool(flags.get("pitch_rate", False)),
            "heading_route_vertical_speed_ok": bool(flags.get("vertical_speed", False)),
            "heading_route_arc_progress": float(state.get("arc_progress", 0.0)),
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
        # Freeze mode, level and target for the lifetime of an attempt.  The
        # runner may advance training_progress at a PPO-update boundary while
        # an attempt is still open; changing level here would discard the old
        # target after it had already generated the current transition.  A
        # newly selected curriculum level therefore takes effect only when
        # completion/reset creates the next attempt.

        sim = env.agents[agent_id]
        sim_time = float(sim.get_property_value(c.simulation_sim_time_sec))
        level = state["level_config"]
        hold_required = float(level.get("last_hold_sec", 0.0))
        max_point_time = float(level.get("max_point_time_sec", self.check_interval))
        if str(state.get("mode", "point")) == "arc":
            elapsed, arc_progress, _ = self._update_arc_target(env, agent_id)
            state["arc_progress"] = float(arc_progress)
            max_point_time = max(max_point_time, float(state.get("arc_duration_sec", 0.0)) + 10.0)
        self._update_segment_quality_state(env, agent_id)
        flags = self._condition_flags(env, agent_id)
        conditions = list(level.get("hold_conditions", ["heading", "altitude", "speed"]))
        in_zone = all(flags.get(name, False) for name in conditions)
        dt = float(getattr(env, "time_interval", self.check_interval))
        if in_zone:
            state["hold_timer"] += dt
        else:
            state["hold_timer"] = max(0.0, float(state["hold_timer"]) - max(self.hold_decay, 0.0) * dt)
        state["in_target_zone"] = bool(in_zone)

        if str(state.get("mode", "point")) == "arc":
            point_completed = bool((sim_time - float(state["point_start_time"])) >= float(state.get("arc_duration_sec", max_point_time)))
        else:
            point_completed = bool(in_zone and state["hold_timer"] >= hold_required)
        info.update(self._route_diag(env, agent_id, flags, point_completed))
        if point_completed:
            point_time = sim_time - float(state["point_start_time"])
            info.update(self._completion_quality_diag(env, agent_id, point_time))
            if str(state.get("mode", "point")) == "arc":
                info["heading_route_attempt_outcome"] = "arc_duration_completed"
            else:
                info["heading_route_attempt_outcome"] = "point_hit"
        info["heading_turn_counts"] = env.heading_turn_counts
        if point_completed:
            env.heading_turn_counts += 1
            info["heading_turn_counts"] = env.heading_turn_counts
            self._reset_route_state(env, agent_id, force=True)
            self.log(f'route point completed. level:{state["level"]} turns:{env.heading_turn_counts}')
            return False, False, info

        if sim_time - float(state["point_start_time"]) > max_point_time:
            info["heading_turn_counts"] = env.heading_turn_counts
            info["heading_route_attempt_outcome"] = "route_timeout"
            info["termination_reason"] = "unreach_heading_route_timeout"
            self.log(f'agent[{agent_id}] route point timeout. Total Steps={env.current_step}')
            return True, False, info

        return False, False, info

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target heading in limited time.

        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        if self.route_curriculum_enable:
            return self._get_route_termination(task, env, agent_id, info)

        done = False
        success = False
        cur_step = info['current_step']
        check_time = env.agents[agent_id].get_property_value(c.heading_check_time)
        # check heading when simulation_time exceed check_time
        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if math.fabs(env.agents[agent_id].get_property_value(c.delta_heading)) > self.threshold_deg:
                done = True
            # if current target heading is reached, random generate a new target heading
            else:
                delta_index = min(env.heading_turn_counts, len(self.increment_size) - 1)
                delta = self.increment_size[delta_index] * self._get_curriculum_scale(env)
                delta_heading = env.np_random.uniform(-delta, delta) * self.max_heading_increment
                delta_altitude = env.np_random.uniform(-delta, delta) * self.max_altitude_increment
                delta_velocities_u = env.np_random.uniform(-delta, delta) * self.max_velocities_u_increment
                new_heading = env.agents[agent_id].get_property_value(c.target_heading_deg) + delta_heading
                new_heading = (new_heading + 360) % 360
                new_altitude = env.agents[agent_id].get_property_value(c.target_altitude_ft) + delta_altitude
                new_altitude = max(new_altitude, 15000) # assert the value in safe region
                new_velocities_u = self._target_speed_mps(env.agents[agent_id]) + delta_velocities_u
                new_velocities_u = max(new_velocities_u, 120.) # assert the value in safe region
                env.agents[agent_id].set_property_value(c.target_heading_deg, new_heading)
                env.agents[agent_id].set_property_value(c.target_altitude_ft, new_altitude)
                self._set_target_speed_mps(env.agents[agent_id], new_velocities_u)
                env.agents[agent_id].set_property_value(c.heading_check_time, check_time + self.check_interval)
                env.heading_turn_counts += 1
                self.log(f'current_step:{cur_step} target_heading:{new_heading} '
                         f'target_altitude_ft:{new_altitude} target_velocities_u_mps:{new_velocities_u}')
        if done:
            self.log(f'agent[{agent_id}] unreached heading. Total Steps={env.current_step}')
            info['heading_turn_counts'] = env.heading_turn_counts
            info['termination_reason'] = 'unreach_heading'
        success = False
        return done, success, info

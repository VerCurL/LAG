import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class HeadingMetricsTracker:
    """Track per-segment and per-episode metrics for heading control episodes."""

    def __init__(
        self,
        heading_tol_deg: float = 5.0,
        altitude_tol_m: float = 15.24,
        speed_tol_mps: float = 24.0,
        hold_steps: int = 10,
        steady_window: int = 10,
        large_heading_delta_deg: float = 90.0,
        large_altitude_delta_ft: float = 3000.0,
        large_speed_delta_mps: float = 50.0,
        pitch_reversal_deadband_deg: float = 0.2,
    ) -> None:
        self.heading_tol_deg = float(heading_tol_deg)
        self.altitude_tol_m = float(altitude_tol_m)
        self.speed_tol_mps = float(speed_tol_mps)
        self.hold_steps = int(hold_steps)
        self.steady_window = int(steady_window)
        self.large_heading_delta_deg = float(large_heading_delta_deg)
        self.large_altitude_delta_ft = float(large_altitude_delta_ft)
        self.large_speed_delta_mps = float(large_speed_delta_mps)
        self.pitch_reversal_deadband_deg = float(pitch_reversal_deadband_deg)
        self.reset((0.0, 0.0, 0.0))

    def reset(self, initial_target: Tuple[float, float, float]) -> None:
        self._active_target = tuple(float(v) for v in initial_target)
        self._previous_target: Optional[Tuple[float, float, float]] = None
        self._target_change_step_idx = 0
        self._target_change_time = 0.0
        self._tracking_started = False
        self._current_segment = None
        self._segments: List[Dict] = []
        self._episode_heading_errors: List[float] = []
        self._episode_altitude_errors: List[float] = []
        self._episode_speed_errors: List[float] = []
        self._episode_heading_in_tol: List[float] = []
        self._episode_altitude_in_tol: List[float] = []
        self._episode_speed_in_tol: List[float] = []
        self._last_step_idx = 0
        self._last_sim_time = 0.0
        self._termination_reason = ""

    def update(
        self,
        sample: Dict[str, float],
        target_changed: bool = False,
        new_target: Optional[Tuple[float, float, float]] = None,
        done: bool = False,
        termination_reason: str = "",
        target_change_success: bool = True,
        target_change_end_reason: str = "next_target",
    ) -> None:
        self._last_step_idx = int(sample["step_idx"])
        self._last_sim_time = float(sample["sim_time"])
        if not bool(sample.get("sample_valid", True)):
            # A failed numerical state is an episode outcome, not a tracking
            # measurement.  Do not append NaNs or fabricate a zero-error
            # sample, but still close the currently open segment.
            if done:
                self._termination_reason = termination_reason or ""
                if self._current_segment is not None:
                    self._finalize_current_segment(
                        success=False,
                        end_reason=self._termination_reason or "terminated",
                    )
            return
        started_segment = False
        if self._tracking_started and self._current_segment is None:
            self._start_segment(sample)
            started_segment = True

        if self._current_segment is not None and not started_segment:
            self._append_sample(sample)

        if target_changed:
            if self._current_segment is not None:
                self._finalize_current_segment(
                    success=bool(target_change_success),
                    end_reason=(
                        str(target_change_end_reason)
                        if target_change_end_reason
                        else "next_target"
                    ),
                )
            self._tracking_started = True
            self._previous_target = self._active_target
            self._target_change_step_idx = int(sample["step_idx"])
            self._target_change_time = float(sample["sim_time"])
            self._active_target = tuple(float(v) for v in new_target)

        if done:
            self._termination_reason = termination_reason or ""
            if self._current_segment is not None:
                self._finalize_current_segment(success=False, end_reason=self._termination_reason or "terminated")

    def get_segment_metrics(self) -> List[Dict]:
        return [segment.copy() for segment in self._segments]

    def get_episode_metrics(self) -> Dict[str, float]:
        total_segments = len(self._segments)
        completed_turns = sum(1 for segment in self._segments if segment["success"])
        settled_segments = [segment["settling_time_sec"] for segment in self._segments if segment["settled"]]
        large_segments = [segment for segment in self._segments if segment["is_large_maneuver"]]
        metrics = {
            "episode_length_steps": int(self._last_step_idx),
            "episode_length_sec": float(self._last_sim_time),
            "tracked_steps": len(self._episode_heading_errors),
            "total_segments": total_segments,
            "completed_turns": completed_turns,
            "reach_rate": self._safe_div(completed_turns, total_segments),
            "mean_abs_heading_error_deg": self._safe_mean(self._episode_heading_errors),
            "mean_abs_altitude_error_m": self._safe_mean(self._episode_altitude_errors),
            "mean_abs_speed_error_mps": self._safe_mean(self._episode_speed_errors),
            "norm_mean_heading_error": self._safe_ratio_mean(self._episode_heading_errors, self.heading_tol_deg),
            "norm_mean_altitude_error": self._safe_ratio_mean(self._episode_altitude_errors, self.altitude_tol_m),
            "norm_mean_speed_error": self._safe_ratio_mean(self._episode_speed_errors, self.speed_tol_mps),
            "heading_in_tol_ratio": self._safe_mean(self._episode_heading_in_tol),
            "altitude_in_tol_ratio": self._safe_mean(self._episode_altitude_in_tol),
            "speed_in_tol_ratio": self._safe_mean(self._episode_speed_in_tol),
            "mean_settling_time_sec": self._safe_mean(settled_segments),
            "settled_segment_ratio": self._safe_div(len(settled_segments), total_segments),
            "mean_heading_overshoot_deg": self._safe_mean(
                [segment["heading_overshoot_deg"] for segment in self._segments]
            ),
            "mean_norm_heading_overshoot": self._safe_mean(
                [segment["norm_heading_overshoot"] for segment in self._segments]
            ),
            "mean_sse_heading_deg": self._safe_mean(
                [segment["sse_heading_deg"] for segment in self._segments]
            ),
            "mean_norm_sse_heading": self._safe_mean(
                [segment["norm_sse_heading"] for segment in self._segments]
            ),
            "large_maneuver_count": len(large_segments),
            "large_maneuver_success_rate": self._safe_div(
                sum(1 for segment in large_segments if segment["success"]), len(large_segments)
            ),
            "large_maneuver_settled_ratio": self._safe_div(
                sum(1 for segment in large_segments if segment["settled"]), len(large_segments)
            ),
            "mean_segment_pitch_std_deg": self._safe_mean(
                [segment["pitch_std_deg"] for segment in self._segments]
            ),
            "mean_segment_pitch_reversal_ratio": self._safe_mean(
                [segment["pitch_reversal_ratio"] for segment in self._segments]
            ),
            "mean_segment_abs_pitch_rate_deg_s": self._safe_mean(
                [segment["mean_abs_pitch_rate_deg_s"] for segment in self._segments]
            ),
            "mean_segment_abs_elevator_delta": self._safe_mean(
                [segment["mean_abs_elevator_delta"] for segment in self._segments]
            ),
        }
        return metrics

    def _start_segment(self, sample: Dict[str, float]) -> None:
        commanded_heading_delta = 0.0
        commanded_altitude_delta = 0.0
        commanded_speed_delta = 0.0
        if self._previous_target is not None:
            commanded_heading_delta = self._signed_angle_deg(self._active_target[0] - self._previous_target[0])
            commanded_altitude_delta = self._active_target[1] - self._previous_target[1]
            commanded_speed_delta = self._active_target[2] - self._previous_target[2]
        self._current_segment = {
            "segment_id": len(self._segments),
            "target_heading_deg": self._active_target[0],
            "target_altitude_ft": self._active_target[1],
            "target_velocities_u_mps": self._active_target[2],
            "target_change_step": self._target_change_step_idx,
            "target_change_time_sec": self._target_change_time,
            "commanded_heading_delta_deg": commanded_heading_delta,
            "commanded_altitude_delta_ft": commanded_altitude_delta,
            "commanded_speed_delta_mps": commanded_speed_delta,
            "step_idx": [],
            "sim_time": [],
            "delta_heading_deg": [],
            "delta_altitude_m": [],
            "delta_speed_mps": [],
            "pitch_deg": [],
            "pitch_rate_deg_s": [],
            "elevator_cmd_norm": [],
        }
        self._append_sample(sample)

    def _append_sample(self, sample: Dict[str, float]) -> None:
        heading_error = float(sample["delta_heading_deg"])
        altitude_error = float(sample["delta_altitude_m"])
        speed_error = float(sample["delta_speed_mps"])
        self._current_segment["step_idx"].append(int(sample["step_idx"]))
        self._current_segment["sim_time"].append(float(sample["sim_time"]))
        self._current_segment["delta_heading_deg"].append(heading_error)
        self._current_segment["delta_altitude_m"].append(altitude_error)
        self._current_segment["delta_speed_mps"].append(speed_error)
        self._current_segment["pitch_deg"].append(float(sample.get("pitch_deg", math.nan)))
        self._current_segment["pitch_rate_deg_s"].append(float(sample.get("pitch_rate_deg_s", math.nan)))
        self._current_segment["elevator_cmd_norm"].append(float(sample.get("elevator_cmd_norm", math.nan)))
        self._episode_heading_errors.append(abs(heading_error))
        self._episode_altitude_errors.append(abs(altitude_error))
        self._episode_speed_errors.append(abs(speed_error))
        self._episode_heading_in_tol.append(float(abs(heading_error) <= self.heading_tol_deg))
        self._episode_altitude_in_tol.append(float(abs(altitude_error) <= self.altitude_tol_m))
        self._episode_speed_in_tol.append(float(abs(speed_error) <= self.speed_tol_mps))

    def _finalize_current_segment(self, success: bool, end_reason: str) -> None:
        heading_errors = np.asarray(self._current_segment["delta_heading_deg"], dtype=np.float64)
        altitude_errors = np.asarray(self._current_segment["delta_altitude_m"], dtype=np.float64)
        speed_errors = np.asarray(self._current_segment["delta_speed_mps"], dtype=np.float64)
        sim_time = np.asarray(self._current_segment["sim_time"], dtype=np.float64)
        step_idx = np.asarray(self._current_segment["step_idx"], dtype=np.int64)
        pitch_deg = np.asarray(self._current_segment["pitch_deg"], dtype=np.float64)
        pitch_rate_deg_s = np.asarray(self._current_segment["pitch_rate_deg_s"], dtype=np.float64)
        elevator_cmd = np.asarray(self._current_segment["elevator_cmd_norm"], dtype=np.float64)

        settling_index = self._find_settling_index(heading_errors, altitude_errors, speed_errors)
        settled = settling_index is not None
        settling_time = float(sim_time[settling_index] - sim_time[0]) if settled else math.nan
        sse_window = min(self.steady_window, len(heading_errors))
        sse_heading = float(np.mean(np.abs(heading_errors[-sse_window:])))
        pitch_reversals, pitch_reversal_ratio = self._compute_pitch_reversal(pitch_deg)
        elevator_delta = np.abs(np.diff(elevator_cmd))
        elevator_delta = elevator_delta[np.isfinite(elevator_delta)]
        commanded_heading_delta = float(self._current_segment["commanded_heading_delta_deg"])
        commanded_altitude_delta = float(self._current_segment["commanded_altitude_delta_ft"])
        commanded_speed_delta = float(self._current_segment["commanded_speed_delta_mps"])
        is_large_maneuver = (
            abs(commanded_heading_delta) >= self.large_heading_delta_deg
            or abs(commanded_altitude_delta) >= self.large_altitude_delta_ft
            or abs(commanded_speed_delta) >= self.large_speed_delta_mps
        )
        segment = {
            "segment_id": int(self._current_segment["segment_id"]),
            "target_change_step": int(self._current_segment["target_change_step"]),
            "target_change_time_sec": float(self._current_segment["target_change_time_sec"]),
            "start_step": int(step_idx[0]),
            "end_step": int(step_idx[-1]),
            "start_time_sec": float(sim_time[0]),
            "end_time_sec": float(sim_time[-1]),
            "duration_steps": int(step_idx[-1] - step_idx[0] + 1),
            "duration_sec": float(sim_time[-1] - sim_time[0]),
            "target_heading_deg": float(self._current_segment["target_heading_deg"]),
            "target_altitude_ft": float(self._current_segment["target_altitude_ft"]),
            "target_velocities_u_mps": float(self._current_segment["target_velocities_u_mps"]),
            "commanded_heading_delta_deg": commanded_heading_delta,
            "commanded_altitude_delta_ft": commanded_altitude_delta,
            "commanded_speed_delta_mps": commanded_speed_delta,
            "is_large_maneuver": bool(is_large_maneuver),
            "success": bool(success),
            "end_reason": end_reason,
            "mean_abs_heading_error_deg": float(np.mean(np.abs(heading_errors))),
            "mean_abs_altitude_error_m": float(np.mean(np.abs(altitude_errors))),
            "mean_abs_speed_error_mps": float(np.mean(np.abs(speed_errors))),
            "norm_mean_heading_error": float(np.mean(np.abs(heading_errors) / self.heading_tol_deg)),
            "norm_mean_altitude_error": float(np.mean(np.abs(altitude_errors) / self.altitude_tol_m)),
            "norm_mean_speed_error": float(np.mean(np.abs(speed_errors) / self.speed_tol_mps)),
            "heading_in_tol_ratio": float(np.mean(np.abs(heading_errors) <= self.heading_tol_deg)),
            "altitude_in_tol_ratio": float(np.mean(np.abs(altitude_errors) <= self.altitude_tol_m)),
            "speed_in_tol_ratio": float(np.mean(np.abs(speed_errors) <= self.speed_tol_mps)),
            "settled": bool(settled),
            "settling_time_sec": settling_time,
            "heading_overshoot_deg": self._compute_heading_overshoot(heading_errors),
            "sse_heading_deg": sse_heading,
            "pitch_std_deg": self._safe_np_std(pitch_deg),
            "max_abs_pitch_deg": self._safe_np_max_abs(pitch_deg),
            "mean_abs_pitch_step_delta_deg": self._safe_mean_array(np.abs(np.diff(pitch_deg))),
            "mean_abs_pitch_rate_deg_s": self._safe_mean_array(np.abs(pitch_rate_deg_s)),
            "max_abs_pitch_rate_deg_s": self._safe_np_max_abs(pitch_rate_deg_s),
            "pitch_reversal_count": int(pitch_reversals),
            "pitch_reversal_ratio": float(pitch_reversal_ratio),
            "mean_abs_elevator_delta": self._safe_mean_array(elevator_delta),
            "max_abs_elevator_delta": self._safe_np_max_abs(elevator_delta),
        }
        segment["norm_heading_overshoot"] = segment["heading_overshoot_deg"] / self.heading_tol_deg
        segment["norm_sse_heading"] = segment["sse_heading_deg"] / self.heading_tol_deg
        self._segments.append(segment)
        self._current_segment = None

    def _find_settling_index(
        self,
        heading_errors: np.ndarray,
        altitude_errors: np.ndarray,
        speed_errors: np.ndarray,
    ) -> Optional[int]:
        if heading_errors.size < self.hold_steps:
            return None
        within_tol = (
            (np.abs(heading_errors) <= self.heading_tol_deg)
            & (np.abs(altitude_errors) <= self.altitude_tol_m)
            & (np.abs(speed_errors) <= self.speed_tol_mps)
        )
        for idx in range(0, within_tol.size - self.hold_steps + 1):
            if np.all(within_tol[idx : idx + self.hold_steps]):
                return idx
        return None

    @staticmethod
    def _compute_heading_overshoot(heading_errors: np.ndarray) -> float:
        if heading_errors.size == 0:
            return 0.0
        sign_ref = 0.0
        for value in heading_errors:
            if abs(value) > 1e-6:
                sign_ref = math.copysign(1.0, value)
                break
        if sign_ref == 0.0:
            return 0.0
        opposite = heading_errors[np.sign(heading_errors) == -sign_ref]
        if opposite.size == 0:
            return 0.0
        return float(np.max(np.abs(opposite)))

    def _compute_pitch_reversal(self, pitch_deg: np.ndarray) -> Tuple[int, float]:
        pitch_deg = pitch_deg[np.isfinite(pitch_deg)]
        if pitch_deg.size < 3:
            return 0, 0.0
        delta = np.diff(pitch_deg)
        sign = np.sign(delta)
        sign[np.abs(delta) <= self.pitch_reversal_deadband_deg] = 0.0
        valid = sign != 0.0
        sign = sign[valid]
        if sign.size < 2:
            return 0, 0.0
        reversal_count = int(np.sum(sign[1:] * sign[:-1] < 0.0))
        return reversal_count, self._safe_div(reversal_count, sign.size - 1)

    @staticmethod
    def _signed_angle_deg(delta: float) -> float:
        return float(((delta + 180.0) % 360.0) - 180.0)

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        if not values:
            return math.nan
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    @staticmethod
    def _safe_mean_array(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return math.nan
        return float(np.mean(values))

    @staticmethod
    def _safe_np_std(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return math.nan
        return float(np.std(values))

    @staticmethod
    def _safe_np_max_abs(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return math.nan
        return float(np.max(np.abs(values)))

    @staticmethod
    def _safe_ratio_mean(values: List[float], scale: float) -> float:
        if not values:
            return math.nan
        return float(np.mean(np.asarray(values, dtype=np.float64) / scale))

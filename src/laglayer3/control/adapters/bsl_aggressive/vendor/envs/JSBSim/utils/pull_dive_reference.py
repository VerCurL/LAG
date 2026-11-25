"""Episode-anchored pull-up, dive and recovery reference for isolated C41.

This is a kinematic reference generator, not an aircraft reachability or safety
assessment. Speed and course are constant; altitude is the integral of vertical
speed and is never recentered on the measured aircraft state.
"""

import math

import numpy as np


class PullDiveReference:
    """Generate a continuous five-phase reference without mutable time state.

    Angles are in degrees at construction and radians in returned gamma fields.
    Direction vectors use NED coordinates, so positive climb has negative D.
    Altitude is integrated by trapezoids on a ``sim_freq`` grid augmented with
    exact phase boundaries. Between grid points the integral of the linearly
    interpolated vertical speed is used. The returned gamma/rates are analytic;
    their mismatch with the altitude interpolant derivative is O(1/sim_freq**2).
    """

    PHASES = ("entry", "pull", "dive", "recovery", "hold")

    def __init__(
        self,
        anchor_course_deg,
        anchor_altitude_m,
        speed_mps,
        pull_angle_deg=5,
        dive_angle_deg=-5,
        entry_sec=12,
        pull_sec=8,
        dive_sec=16,
        recovery_sec=8,
        hold_sec=8,
        sim_freq=60,
    ):
        values = {
            "anchor_course_deg": anchor_course_deg,
            "anchor_altitude_m": anchor_altitude_m,
            "speed_mps": speed_mps,
            "pull_angle_deg": pull_angle_deg,
            "dive_angle_deg": dive_angle_deg,
            "entry_sec": entry_sec,
            "pull_sec": pull_sec,
            "dive_sec": dive_sec,
            "recovery_sec": recovery_sec,
            "hold_sec": hold_sec,
            "sim_freq": sim_freq,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError("{} must be finite".format(name))
        if float(speed_mps) <= 0 or float(sim_freq) <= 0:
            raise ValueError("speed_mps and sim_freq must be positive")
        if not 0 <= float(pull_angle_deg) < 90:
            raise ValueError("pull_angle_deg must be in [0, 90)")
        if not -90 < float(dive_angle_deg) <= 0:
            raise ValueError("dive_angle_deg must be in (-90, 0]")
        if float(entry_sec) < 0 or float(hold_sec) < 0:
            raise ValueError("entry_sec and hold_sec must be nonnegative")
        if min(float(pull_sec), float(dive_sec), float(recovery_sec)) <= 0:
            raise ValueError("pull/dive/recovery durations must be positive")

        self.anchor_course_deg = float(anchor_course_deg) % 360.0
        self.anchor_altitude_m = float(anchor_altitude_m)
        self.speed_mps = float(speed_mps)
        self.sim_freq = float(sim_freq)
        self.pull_angle_rad = math.radians(float(pull_angle_deg))
        self.dive_angle_rad = math.radians(float(dive_angle_deg))
        self._durations = np.array(
            [entry_sec, pull_sec, dive_sec, recovery_sec, hold_sec], dtype=float
        )
        self._ends = np.cumsum(self._durations)
        self.total_sec = float(self._ends[-1])
        if not math.isfinite(self.total_sec):
            raise ValueError("total duration must be finite")
        self._transition_starts = self._ends[:3].copy()
        self._transition_durations = self._durations[1:4].copy()
        self._angle_changes = np.array(
            [self.pull_angle_rad,
             self.dive_angle_rad - self.pull_angle_rad,
             -self.dive_angle_rad]
        )

        ticks = int(math.floor(self.total_sec * self.sim_freq))
        self._times = np.unique(np.concatenate((
            np.arange(ticks + 1, dtype=float) / self.sim_freq,
            self._ends,
            [0.0, self.total_sec],
        )))
        self._vertical_speeds = self.speed_mps * np.sin(self._gamma(self._times)[0])
        increments = np.diff(self._times) * (
            self._vertical_speeds[:-1] + self._vertical_speeds[1:]
        ) * 0.5
        self._height_offsets = np.concatenate(([0.0], np.cumsum(increments)))

    def _gamma(self, times):
        times = np.asarray(times, dtype=float)
        gamma = np.zeros_like(times)
        gamma_rate = np.zeros_like(times)
        for start, duration, delta in zip(
            self._transition_starts, self._transition_durations, self._angle_changes
        ):
            tau = np.clip((times - start) / duration, 0.0, 1.0)
            smooth = tau ** 3 * (10.0 + tau * (-15.0 + 6.0 * tau))
            derivative = 30.0 * tau ** 2 * (1.0 - tau) ** 2 / duration
            gamma += delta * smooth
            gamma_rate += delta * derivative
        return gamma, gamma_rate

    def at(self, t):
        """Return a fresh reference dict at elapsed seconds in [0, total_sec].

        ``target`` is [course_deg, altitude_m, speed_mps], ``rates`` is
        [course_rate_deg_s, vertical_speed_mps, speed_rate_mps2].
        ``remaining_sec`` refers to the whole episode, while
        ``phase_remaining_sec`` refers to the current reference phase.
        Queries do not advance a cursor and can be made in any order.
        """
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("reference time must be finite")
        tolerance = 1e-9 * max(1.0, self.total_sec)
        if t < -tolerance or t > self.total_sec + tolerance:
            raise ValueError("reference time must be inside [0, total_sec]")
        t = min(max(t, 0.0), self.total_sec)
        gamma, gamma_rate = (float(value) for value in self._gamma(t))
        interval = min(
            max(int(np.searchsorted(self._times, t, side="right")) - 1, 0),
            len(self._times) - 2,
        )
        offset = t - self._times[interval]
        width = self._times[interval + 1] - self._times[interval]
        vertical_slope = (
            self._vertical_speeds[interval + 1] - self._vertical_speeds[interval]
        ) / width
        altitude = self.anchor_altitude_m + self._height_offsets[interval] + (
            offset * self._vertical_speeds[interval] + 0.5 * offset ** 2 * vertical_slope
        )
        course = math.radians(self.anchor_course_deg)
        phase_index = min(int(np.searchsorted(self._ends[:4], t, side="right")), 4)
        return {
            "target": np.array([self.anchor_course_deg, altitude, self.speed_mps]),
            "rates": np.array([0.0, self.speed_mps * math.sin(gamma), 0.0]),
            "gamma_rad": gamma,
            "gamma_rate_radps": gamma_rate,
            "direction": np.array([
                math.cos(gamma) * math.cos(course),
                math.cos(gamma) * math.sin(course),
                -math.sin(gamma),
            ]),
            "phase": self.PHASES[phase_index],
            "phase_index": phase_index,
            "remaining_sec": self.total_sec - t,
            "phase_remaining_sec": max(float(self._ends[phase_index]) - t, 0.0),
        }

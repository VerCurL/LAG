"""Analytic target-rate semantics for the isolated O1 observation."""

import math

import numpy as np


TARGET_RATE_KEYS = (
    "target_heading_rate_deg_s",
    "target_altitude_rate_mps",
    "target_speed_rate_mps2",
)


def smoothstep01_rate_per_sec(elapsed_sec, duration_sec):
    """Return d/dt of smoothstep(clamp(elapsed / duration, 0, 1))."""

    elapsed = float(elapsed_sec)
    duration = float(duration_sec)
    if not math.isfinite(elapsed):
        raise ValueError("elapsed_sec must be finite")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive")
    phase = elapsed / duration
    if phase <= 0.0 or phase >= 1.0:
        return 0.0
    return float(6.0 * phase * (1.0 - phase) / duration)


def compute_route_target_rates(route_state, sim_time_sec):
    """Return heading [deg/s], altitude [m/s], and speed [m/s^2] rates.

    Point attempts are piecewise-constant commands, so their rate is zero
    between exogenous target changes.  Arc attempts use the exact derivative
    of the route generator's smoothstep interpolation.
    """

    if not isinstance(route_state, dict):
        return np.zeros(3, dtype=np.float32)
    if str(route_state.get("mode", "point")) != "arc":
        return np.zeros(3, dtype=np.float32)

    required = (
        "point_start_time",
        "arc_duration_sec",
        "arc_turn_total_deg",
        "arc_altitude_delta_ft",
        "arc_speed_delta_mps",
    )
    missing = [key for key in required if key not in route_state]
    if missing:
        raise ValueError(
            "arc route state is missing target-rate fields: {}".format(
                ", ".join(missing)
            )
        )
    sim_time = float(sim_time_sec)
    if not math.isfinite(sim_time):
        raise ValueError("sim_time_sec must be finite")
    elapsed = sim_time - float(route_state["point_start_time"])
    interpolation_mode = str(
        route_state.get("arc_interpolation_mode", "smoothstep")
    ).strip().lower()
    if interpolation_mode == "linear":
        duration = float(route_state["arc_duration_sec"])
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("arc_duration_sec must be finite and positive")
        interpolation_rate = (
            1.0 / duration if 0.0 <= elapsed < duration else 0.0
        )
    elif interpolation_mode == "smoothstep":
        interpolation_rate = smoothstep01_rate_per_sec(
            elapsed,
            route_state["arc_duration_sec"],
        )
    else:
        raise ValueError(
            "unsupported arc interpolation mode: {}".format(
                interpolation_mode
            )
        )
    rates = np.asarray(
        [
            float(route_state["arc_turn_total_deg"])
            * interpolation_rate,
            float(route_state["arc_altitude_delta_ft"])
            * 0.3048
            * interpolation_rate,
            float(route_state["arc_speed_delta_mps"])
            * interpolation_rate,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(rates)):
        raise ValueError("analytic target rates must be finite")
    return rates

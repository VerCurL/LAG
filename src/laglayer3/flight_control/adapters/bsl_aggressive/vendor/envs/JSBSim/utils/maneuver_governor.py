"""C27 deterministic maneuver-intent governor (heuristic, not a safety proof).

Speed uses sqrt(u^2+v^2+w^2) in m/s, the same quantity as C24's gate.
It is NOT calibrated airspeed. No actuator commands or network weights change.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GovernorConfig:
    speed_soft_min_mps: float = 150.0
    speed_soft_max_mps: float = 320.0
    speed_hard_min_mps: float = 120.0
    speed_hard_max_mps: float = 350.0
    altitude_soft_min_m: float = 3000.0
    altitude_hard_min_m: float = 2500.0
    descent_lookahead_sec: float = 3.0
    speed_offset_tau: float = 2.0
    altitude_offset_max_m: float = 150.0
    heading_offset_max_deg: float = 30.0
    speed_offset_max_mps: float = 15.0

    def __post_init__(self):
        if not all(math.isfinite(value) and value > 0.0 for value in vars(self).values()):
            raise ValueError("governor parameters must be finite and positive")
        if not self.speed_hard_min_mps < self.speed_soft_min_mps < self.speed_soft_max_mps < self.speed_hard_max_mps:
            raise ValueError("governor speed boundaries are unordered")
        if self.altitude_soft_min_m <= self.altitude_hard_min_m:
            raise ValueError("governor altitude boundaries are unordered")


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def govern_maneuver(altitude_offset_m, heading_offset_deg, speed_offset_mps, *,
                    altitude_m, speed_mps, vertical_speed_mps, enabled=True,
                    config=GovernorConfig()):
    proposed = (float(altitude_offset_m), float(heading_offset_deg), float(speed_offset_mps))
    if not all(math.isfinite(value) for value in proposed + (altitude_m, speed_mps, vertical_speed_mps)):
        raise ValueError("non-finite maneuver governor input")
    if speed_mps < 0:
        raise ValueError("negative governor speed")
    bounds = (config.altitude_offset_max_m, config.heading_offset_max_deg, config.speed_offset_max_mps)
    if any(abs(value) > bound + 1e-8 for value, bound in zip(proposed, bounds)):
        raise ValueError("maneuver exceeds frozen governor interface")
    if not isinstance(enabled, bool):
        raise ValueError("governor enabled must be boolean")

    altitude, heading, speed = proposed
    predicted_altitude = altitude_m + min(vertical_speed_mps, 0.0) * config.descent_lookahead_sec
    low_factor = _clip((speed_mps - config.speed_hard_min_mps) /
                       (config.speed_soft_min_mps - config.speed_hard_min_mps), 0.0, 1.0)
    high_factor = _clip((config.speed_hard_max_mps - speed_mps) /
                        (config.speed_hard_max_mps - config.speed_soft_max_mps), 0.0, 1.0)
    altitude_lower = _clip(config.altitude_soft_min_m - predicted_altitude,
                           -config.altitude_offset_max_m, config.altitude_offset_max_m)
    low_altitude_conflict = altitude_lower > 0.0 and low_factor < 1.0
    if enabled:
        speed_lower = _clip((config.speed_soft_min_mps - speed_mps) / config.speed_offset_tau,
                            -config.speed_offset_max_mps, config.speed_offset_max_mps)
        speed_upper = _clip((config.speed_soft_max_mps - speed_mps) / config.speed_offset_tau,
                            -config.speed_offset_max_mps, config.speed_offset_max_mps)
        speed = _clip(speed, speed_lower, speed_upper)
        # Reduce energy-consuming climb/turn at low speed; reduce dives at high speed.
        altitude = altitude * (low_factor if altitude > 0.0 else high_factor)
        heading *= low_factor
        # Ground-clearance intent has priority if speed and altitude demands conflict.
        # This is a declared heuristic; no guaranteed recovery is implied.
        altitude = max(altitude, altitude_lower)
        if low_altitude_conflict:
            speed = config.speed_offset_max_mps
            heading = 0.0

    effective = (altitude, heading, speed)
    changed = [abs(a - b) > 1e-9 for a, b in zip(proposed, effective)]
    log = {
        "governor_enabled": float(enabled),
        "governor_intervened": float(any(changed)),
        "governor_speed_mps": float(speed_mps),
        "governor_altitude_m": float(altitude_m),
        "governor_vertical_speed_mps": float(vertical_speed_mps),
        "governor_predicted_altitude_m": predicted_altitude,
        "governor_low_speed_factor": low_factor,
        "governor_high_speed_factor": high_factor,
        "governor_altitude_lower_offset_m": altitude_lower,
        "governor_low_altitude_speed_conflict": float(low_altitude_conflict),
    }
    for i, channel in enumerate(("altitude_m", "heading_deg", "speed_mps")):
        log["governor_proposed_" + channel] = proposed[i]
        log["governor_effective_" + channel] = effective[i]
        log["governor_changed_" + channel] = float(changed[i])
    return effective, log

"""Ground-velocity geometry and paired direction-rate filtering in NED.

Angles are radians. North/east/down velocities are metres per second; a
positive flight-path angle is a climb. These helpers never substitute pitch
for flight-path angle or airspeed for ground speed.
"""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("{} must be a finite real scalar".format(name))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


def _positive_scalar(value, name):
    value = _finite_scalar(value, name)
    if value <= 0.0:
        raise ValueError("{} must be positive".format(name))
    return value


def _finite_vector(value, name):
    vector = np.asarray(value)
    if vector.shape != (3,) or vector.dtype.kind not in "iuf":
        raise ValueError("{} must be a real three-vector".format(name))
    vector = np.array(vector, dtype=np.float64, copy=True)
    if not np.isfinite(vector).all():
        raise ValueError("{} must be finite".format(name))
    return vector


def _unit_vector(value, name):
    vector = _finite_vector(value, name)
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
        raise ValueError("{} must be a unit vector".format(name))
    # Remove round-off only after rejecting non-unit and undefined directions.
    return vector / norm


@dataclass(frozen=True)
class FlightPathGeometry:
    speed_mps: float
    horizontal_speed_mps: float
    course_rad: float
    gamma_rad: float
    direction: np.ndarray
    direction_valid: bool
    course_valid: bool


def geometry_from_ned(velocity_ned, min_speed=1.0, min_horizontal_speed=1.0):
    """Describe a NED ground-velocity vector, explicitly marking singularities.

    Below ``min_speed``, direction/gamma are unavailable and have zero
    placeholders. Near vertical, gamma and direction remain valid while course
    is unavailable. Callers must inspect the flags; the zero direction cannot
    be passed to the reward kernel as a valid unit vector.
    """
    velocity = _finite_vector(velocity_ned, "velocity_ned")
    minimum = _positive_scalar(min_speed, "min_speed")
    minimum_horizontal = _positive_scalar(min_horizontal_speed, "min_horizontal_speed")
    speed = math.hypot(*velocity)
    horizontal = math.hypot(velocity[0], velocity[1])
    if not math.isfinite(speed):
        raise ValueError("ground speed must be finite")
    direction_valid = speed >= minimum
    course_valid = direction_valid and horizontal >= minimum_horizontal
    direction = velocity / speed if direction_valid else np.zeros(3, dtype=np.float64)
    direction.setflags(write=False)
    return FlightPathGeometry(
        speed_mps=speed,
        horizontal_speed_mps=horizontal,
        course_rad=math.atan2(velocity[1], velocity[0]) if course_valid else 0.0,
        gamma_rad=math.atan2(-velocity[2], horizontal) if direction_valid else 0.0,
        direction=direction,
        direction_valid=direction_valid,
        course_valid=course_valid,
    )


def reference_direction(course_rad, gamma_rad):
    """Return the unit NED direction for course and signed flight-path angle.

    Gamma follows the spherical-coordinate convention [-pi/2, pi/2]. Passing
    through vertical requires a three-vector reference rather than extending
    this angle beyond its chart.
    """
    course = _finite_scalar(course_rad, "course_rad")
    gamma = _finite_scalar(gamma_rad, "gamma_rad")
    if abs(gamma) > math.pi / 2.0:
        raise ValueError("gamma_rad must lie in [-pi/2, pi/2]")
    cos_gamma = math.cos(gamma)
    return np.array([
        cos_gamma * math.cos(course),
        cos_gamma * math.sin(course),
        -math.sin(gamma),
    ], dtype=np.float64)


class DirectionRateFilter:
    """Identical causal differences and low-pass filters for both directions.

    ``reset`` seeds previous directions and zero filtered rates. ``update`` is
    called once per sample (normally 60 Hz), not just once per PPO decision.
    Returned rates have units 1/s. A zero time constant disables the low-pass
    filter, while retaining the same backward difference on both sides.
    """

    def __init__(self, tau_sec):
        self.tau_sec = _finite_scalar(tau_sec, "tau_sec")
        if self.tau_sec < 0.0:
            raise ValueError("tau_sec cannot be negative")
        self._previous_actual = None
        self._previous_reference = None
        self._actual_rate = np.zeros(3, dtype=np.float64)
        self._reference_rate = np.zeros(3, dtype=np.float64)

    def reset(self, actual, reference):
        actual = _unit_vector(actual, "actual direction")
        reference = _unit_vector(reference, "reference direction")
        self._previous_actual = actual
        self._previous_reference = reference
        self._actual_rate = np.zeros(3, dtype=np.float64)
        self._reference_rate = np.zeros(3, dtype=np.float64)
        return self._actual_rate.copy(), self._reference_rate.copy()

    def update(self, actual, reference, dt):
        if self._previous_actual is None:
            raise RuntimeError("DirectionRateFilter requires reset before update")
        actual = _unit_vector(actual, "actual direction")
        reference = _unit_vector(reference, "reference direction")
        dt = _positive_scalar(dt, "dt")
        alpha = 1.0 if self.tau_sec == 0.0 else -math.expm1(-dt / self.tau_sec)
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                actual_raw = (actual - self._previous_actual) / dt
                reference_raw = (reference - self._previous_reference) / dt
                actual_rate = (1.0 - alpha) * self._actual_rate + alpha * actual_raw
                reference_rate = (1.0 - alpha) * self._reference_rate + alpha * reference_raw
            except FloatingPointError as error:
                raise ValueError("direction-rate computation is non-finite") from error
        if not np.isfinite(actual_rate).all() or not np.isfinite(reference_rate).all():
            raise ValueError("direction-rate computation is non-finite")
        # Commit only after both sides validate, preserving state after a bad sample.
        self._previous_actual = actual
        self._previous_reference = reference
        self._actual_rate = actual_rate
        self._reference_rate = reference_rate
        return actual_rate.copy(), reference_rate.copy()

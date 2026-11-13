"""Pure energy and ground-momentum-direction reward for an isolated experiment.

All actual/reference inputs must refer to the same post-action instant. The
task owns direction validity, safety, numerical termination and terminal reward;
this module returns only the finite ordinary running reward and its components.
"""

import math
from dataclasses import dataclass

import numpy as np

from .baiyepower_v2_physics import kinetic_height_scale_m, rational_cost
from .flight_path_geometry import (
    _finite_scalar,
    _finite_vector,
    _positive_scalar,
    _unit_vector,
)


@dataclass(frozen=True)
class EnergyMomentumConfig:
    height_scale_m: float = 100.0
    speed_scale_mps: float = 10.0
    direction_scale_deg: float = 5.0
    rate_scale_deg_s: float = 3.0
    rate_weight: float = 0.25
    energy_weight: float = 1.0
    momentum_weight: float = 1.0
    gravity: float = 9.80665

    def __post_init__(self):
        for name in (
            "height_scale_m", "speed_scale_mps", "direction_scale_deg",
            "rate_scale_deg_s", "energy_weight", "momentum_weight", "gravity",
        ):
            _positive_scalar(getattr(self, name), name)
        if self.direction_scale_deg > 180.0:
            raise ValueError("direction_scale_deg must lie in (0, 180]")
        if _finite_scalar(self.rate_weight, "rate_weight") < 0.0:
            raise ValueError("rate_weight cannot be negative")
        # Reject scales that underflow in the actual normalization, as well as zero.
        if 2.0 * math.sin(math.radians(self.direction_scale_deg) / 2.0) ** 2 <= 0.0:
            raise ValueError("direction scale is too small for finite normalization")
        if math.radians(self.rate_scale_deg_s) <= 0.0:
            raise ValueError("rate scale is too small for finite normalization")


def compute_energy_momentum_reward(
    altitude_m,
    speed_mps,
    target_altitude_m,
    target_speed_mps,
    direction,
    reference_direction,
    actual_direction_rate,
    reference_direction_rate,
    dt,
    config,
):
    """Return scalar diagnostics for dt / (1 + wE*CE + wM*CM).

    Speeds are ground-speed magnitudes, angles use unit NED directions, and
    rates are paired filtered direction differences in 1/s. Energy errors and
    normalization scales use metres of energy height. No terminal reward is
    injected here.
    """
    if not isinstance(config, EnergyMomentumConfig):
        raise TypeError("config must be EnergyMomentumConfig")
    altitude = _finite_scalar(altitude_m, "altitude_m")
    target_altitude = _finite_scalar(target_altitude_m, "target_altitude_m")
    speed = _positive_scalar(speed_mps, "speed_mps")
    target_speed = _positive_scalar(target_speed_mps, "target_speed_mps")
    dt = _positive_scalar(dt, "dt")
    actual = _unit_vector(direction, "direction")
    target = _unit_vector(reference_direction, "reference_direction")
    actual_rate = _finite_vector(actual_direction_rate, "actual_direction_rate")
    target_rate = _finite_vector(reference_direction_rate, "reference_direction_rate")

    e_h = _finite_scalar(target_altitude - altitude, "height error")
    e_k = _finite_scalar(
        (target_speed - speed) * (target_speed + speed) / (2.0 * config.gravity),
        "kinetic height error",
    )
    e_total = _finite_scalar(e_h + e_k, "total energy error")
    e_balance = _finite_scalar(e_h - e_k, "energy balance error")
    kinetic_scale = kinetic_height_scale_m(
        target_speed,
        speed_cost_scale_mps=config.speed_scale_mps,
        gravity_mps2=config.gravity,
    )
    scale = _positive_scalar(config.height_scale_m + kinetic_scale, "energy scale")
    cost_total = rational_cost(e_total / scale)
    cost_balance = rational_cost(e_balance / scale)
    cost_energy = _finite_scalar(0.5 * cost_total + 0.5 * cost_balance, "energy cost")

    dot = float(np.clip(np.dot(actual, target), -1.0, 1.0))
    direction_cost = 1.0 - dot
    direction_scale = 2.0 * math.sin(math.radians(config.direction_scale_deg) / 2.0) ** 2
    cost_direction = _finite_scalar(direction_cost / direction_scale, "direction cost")
    direction_error = math.atan2(math.hypot(*np.cross(actual, target)), dot)
    with np.errstate(over="raise", invalid="raise"):
        try:
            rate_error = math.hypot(*(actual_rate - target_rate))
        except FloatingPointError as error:
            raise ValueError("direction-rate error is non-finite") from error
    rate_error = _finite_scalar(rate_error, "direction-rate error")
    cost_rate = rational_cost(rate_error / math.radians(config.rate_scale_deg_s))
    rate_fraction = config.rate_weight / (1.0 + config.rate_weight)
    cost_momentum = _finite_scalar(
        cost_direction / (1.0 + config.rate_weight) + rate_fraction * cost_rate,
        "momentum cost",
    )
    weighted_cost = _finite_scalar(
        config.energy_weight * cost_energy + config.momentum_weight * cost_momentum,
        "weighted cost",
    )
    denominator = _positive_scalar(1.0 + weighted_cost, "reward denominator")
    reward = _finite_scalar(dt / denominator, "running_reward")
    return {
        "running_reward": reward,
        "C_E": cost_energy,
        "C_M": cost_momentum,
        "C_dir": cost_direction,
        "C_rate": cost_rate,
        "C_total_energy": cost_total,
        "C_energy_balance": cost_balance,
        "e_h_m": e_h,
        "e_k_m": e_k,
        "e_T_m": e_total,
        "e_B_m": e_balance,
        "S_T_m": scale,
        "S_B_m": scale,
        "S_k_m": kinetic_scale,
        "direction_dot": dot,
        "direction_cost": direction_cost,
        "direction_error_deg": math.degrees(direction_error),
        "direction_rate_error_per_sec": rate_error,
        "weighted_cost": weighted_cost,
        "dt": dt,
    }

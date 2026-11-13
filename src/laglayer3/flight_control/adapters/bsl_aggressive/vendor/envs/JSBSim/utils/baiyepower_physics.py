"""Pure physics helpers for the opt-in ``baiyepower`` task.

The first baiyepower revision is intentionally limited to a no-wind baseline.
It uses the magnitude of the JSBSim body-axis velocity vector as the scalar
speed in specific mechanical-energy calculations.
"""

import math
from dataclasses import dataclass


STANDARD_GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True)
class BaiyePowerEnergyTerms:
    """Specific-energy state and target-minus-current decomposition.

    All energy quantities are expressed as equivalent height in metres:

        H_E = h + V**2 / (2 g)

    Positive error terms therefore mean that the aircraft is below the
    requested target in the corresponding component.
    """

    speed_mps: float
    target_speed_mps: float
    energy_height_m: float
    target_energy_height_m: float
    altitude_deficit_m: float
    kinetic_height_deficit_m: float
    total_energy_height_deficit_m: float


def speed_magnitude_mps(u_mps, v_mps, w_mps):
    """Return ``sqrt(u**2 + v**2 + w**2)`` in m/s."""

    values = tuple(float(value) for value in (u_mps, v_mps, w_mps))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("baiyepower velocity components must be finite")
    return math.sqrt(sum(value * value for value in values))


def compute_baiyepower_energy_terms(
    altitude_m,
    target_altitude_m,
    u_mps,
    v_mps,
    w_mps,
    target_speed_mps,
    gravity_mps2=STANDARD_GRAVITY_MPS2,
):
    """Compute no-wind specific-energy terms using one scalar speed semantics."""

    altitude_m = float(altitude_m)
    target_altitude_m = float(target_altitude_m)
    target_speed_mps = float(target_speed_mps)
    gravity_mps2 = float(gravity_mps2)
    if not all(
        math.isfinite(value)
        for value in (
            altitude_m,
            target_altitude_m,
            target_speed_mps,
            gravity_mps2,
        )
    ):
        raise ValueError("baiyepower energy inputs must be finite")
    if target_speed_mps < 0.0:
        raise ValueError("baiyepower target speed magnitude cannot be negative")
    if gravity_mps2 <= 0.0:
        raise ValueError("baiyepower gravity must be positive")

    speed_mps = speed_magnitude_mps(u_mps, v_mps, w_mps)
    kinetic_height_m = speed_mps * speed_mps / (2.0 * gravity_mps2)
    target_kinetic_height_m = (
        target_speed_mps * target_speed_mps / (2.0 * gravity_mps2)
    )
    energy_height_m = altitude_m + kinetic_height_m
    target_energy_height_m = target_altitude_m + target_kinetic_height_m
    altitude_deficit_m = target_altitude_m - altitude_m
    kinetic_height_deficit_m = target_kinetic_height_m - kinetic_height_m

    return BaiyePowerEnergyTerms(
        speed_mps=speed_mps,
        target_speed_mps=target_speed_mps,
        energy_height_m=energy_height_m,
        target_energy_height_m=target_energy_height_m,
        altitude_deficit_m=altitude_deficit_m,
        kinetic_height_deficit_m=kinetic_height_deficit_m,
        total_energy_height_deficit_m=(
            altitude_deficit_m + kinetic_height_deficit_m
        ),
    )

"""Opt-in reference-weighted speed recovery cost for the C58 experiment.

This pure wrapper leaves the frozen energy-height kernel unchanged.  It adds
an independent, symmetric ground-speed cost to the running-reward denominator.
The reference geometry controls its smooth weight; actual tracking errors,
altitude, episode time and terminal conditions never control that weight.
"""

import math

from .baiyepower_v2_physics import rational_cost
from .energy_height_transition_kernel import compute_energy_height_transition_reward
from .flight_path_geometry import (
    _finite_scalar,
    _finite_vector,
    _positive_scalar,
    _unit_vector,
)


def compute_reference_weighted_speed_reward(
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
    *,
    transition_eta,
    speed_weight,
    speed_scale_mps,
    reference_gamma_scale_deg,
    reference_rate_scale_deg_s,
):
    """Return C57 diagnostics with an explicitly selected speed cost.

    ``w = 1 / (1 + (gamma_ref / gamma_scale)**2
                    + (norm(filtered_reference_rate) / rate_scale)**2)``
    ``extra = speed_weight * w * rho((V - V_ref) / speed_scale)``
    ``reward = dt / (1 + C57_weighted_cost + extra)``

    Speeds are ground-speed magnitudes in m/s.  Gamma is extracted from the
    unit NED reference direction, and rates are the already filtered direction
    differences in 1/s.  Angle and rate scales are explicitly supplied in
    degrees and degrees/s and are converted consistently for normalization.
    The rate norm represents angular speed of the unit flight-path direction,
    not the body yaw/pitch/roll rate.  The weight applies during any reference
    flight state, including initial level flight; it is not an exit-time gate.

    Every new parameter is required.  Weights must be finite and nonnegative;
    scales must be finite and strictly positive.  Booleans, silent clipping,
    non-finite derived values and numerically vanishing reference weights are
    rejected.  At speed_weight=0 every C57 return field is bit-for-bit retained.

    At nonzero speed_weight only ``weighted_cost`` and ``running_reward`` may
    change among the C57 fields.  In particular ``C_E`` and ``original_*`` keep
    their C57 meanings.  ``reference_speed_base_*`` records the active C57 eta
    baseline, whereas ``original_*`` still records the eta=0 energy baseline.
    The extra cost is not multiplied by config.energy_weight.  Terminal
    rewards, safety constraints and task admission gates remain external.
    """
    strength = _finite_scalar(speed_weight, "speed_weight")
    if strength < 0.0:
        raise ValueError("speed_weight cannot be negative")
    speed_scale = _positive_scalar(speed_scale_mps, "speed_scale_mps")
    gamma_scale = _positive_scalar(reference_gamma_scale_deg, "reference_gamma_scale_deg")
    rate_scale_deg = _positive_scalar(
        reference_rate_scale_deg_s, "reference_rate_scale_deg_s",
    )
    rate_scale = _positive_scalar(
        math.radians(rate_scale_deg), "reference rate scale in radians/s",
    )

    # All pre-existing physical validation and eta arithmetic stays owned by
    # the frozen kernel.  No observations, policy inputs or simulator state
    # are introduced by this wrapper.
    result = compute_energy_height_transition_reward(
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
        transition_eta=transition_eta,
    )
    target_direction = _unit_vector(reference_direction, "reference_direction")
    target_rate = _finite_vector(reference_direction_rate, "reference_direction_rate")
    gamma_deg = math.degrees(math.atan2(
        -target_direction[2], math.hypot(target_direction[0], target_direction[1]),
    ))
    rate_norm = _finite_scalar(math.hypot(*target_rate), "reference direction-rate norm")
    rate_deg_s = _finite_scalar(math.degrees(rate_norm), "reference direction-rate degrees/s")
    gamma_ratio = _finite_scalar(gamma_deg / gamma_scale, "normalized reference gamma")
    rate_ratio = _finite_scalar(rate_norm / rate_scale, "normalized reference direction rate")
    # hypot avoids overflowing the squared denominator when a positive weight
    # is still representable.  Reject underflow to zero rather than silently
    # turning recovery off for an extreme input or an unusably small scale.
    weight_norm = _positive_scalar(
        math.hypot(1.0, gamma_ratio, rate_ratio), "reference speed weight norm",
    )
    reference_weight = _positive_scalar(
        (1.0 / weight_norm) ** 2, "reference speed reference weight",
    )
    speed_error = _finite_scalar(
        float(speed_mps) - float(target_speed_mps), "reference speed error",
    )
    normalized_error = _finite_scalar(speed_error / speed_scale, "normalized reference speed error")
    cost = rational_cost(normalized_error)
    extra = _finite_scalar(strength * reference_weight * cost, "reference speed extra cost")
    base_weighted_cost = result["weighted_cost"]
    base_reward = result["running_reward"]
    if strength != 0.0 and extra != 0.0:
        result["weighted_cost"] = _finite_scalar(
            base_weighted_cost + extra, "reference speed weighted cost",
        )
        denominator = _positive_scalar(
            1.0 + result["weighted_cost"], "reference speed reward denominator",
        )
        result["running_reward"] = _finite_scalar(
            result["dt"] / denominator, "reference speed running_reward",
        )
    result.update(
        reference_speed_weight=strength,
        reference_speed_scale_mps=speed_scale,
        reference_speed_gamma_scale_deg=gamma_scale,
        reference_speed_rate_scale_deg_s=rate_scale_deg,
        reference_speed_reference_gamma_deg=gamma_deg,
        reference_speed_reference_rate_deg_s=rate_deg_s,
        reference_speed_reference_weight=reference_weight,
        reference_speed_error_mps=speed_error,
        reference_speed_normalized_error=normalized_error,
        reference_speed_cost=cost,
        reference_speed_extra_cost=extra,
        reference_speed_base_weighted_cost=base_weighted_cost,
        reference_speed_base_running_reward=base_reward,
        reference_speed_running_reward=result["running_reward"],
    )
    return result

"""Analytic coordinated-flight action law and residual composition.

All actions in this module use PPO policy coordinates:

* aileron/elevator/rudder: ``[-1, 1]``;
* throttle: ``[-1, 1]``, mapped by :class:`HeadingTask` to ``[0.4, 0.9]``.

Keeping the law in policy coordinates lets the composed action pass through
the repository's unchanged normalization, QP projector, minimum-jerk
smoother, and F-16 inner loops.
"""

import math

import numpy as np


ACTION_CHANNELS = ("aileron", "elevator", "rudder", "throttle")


def euler_heading_rate_deg_s(
    *, roll_rad, pitch_rad, pitch_rate_deg_s, yaw_rate_deg_s
):
    """Recover Euler heading rate from measured body ``q`` and ``r`` rates."""

    values = {
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "pitch_rate_deg_s": pitch_rate_deg_s,
        "yaw_rate_deg_s": yaw_rate_deg_s,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError("{} must be finite".format(name))
    cos_pitch = math.cos(float(pitch_rad))
    if abs(cos_pitch) < 1e-3:
        raise ValueError("Euler heading rate is singular near 90 degree pitch")
    return (
        float(pitch_rate_deg_s) * math.sin(float(roll_rad))
        + float(yaw_rate_deg_s) * math.cos(float(roll_rad))
    ) / cos_pitch


def scheduled_flight_path_tau(
    *,
    speed_mps,
    altitude_m,
    low_speed_tau_sec,
    high_speed_low_alt_tau_sec,
    high_speed_high_alt_tau_sec,
    speed_blend_start_mps,
    speed_blend_end_mps,
    altitude_blend_start_m,
    altitude_blend_end_m,
):
    """Return a continuous speed/altitude-scheduled flight-path tau.

    Low speed always uses ``low_speed_tau_sec``.  At high speed, altitude
    blends between the low- and high-altitude values.  Cubic smoothstep
    avoids derivative discontinuities at both scheduling boundaries.
    """

    finite_values = {
        "speed_mps": speed_mps,
        "altitude_m": altitude_m,
        "low_speed_tau_sec": low_speed_tau_sec,
        "high_speed_low_alt_tau_sec": high_speed_low_alt_tau_sec,
        "high_speed_high_alt_tau_sec": high_speed_high_alt_tau_sec,
        "speed_blend_start_mps": speed_blend_start_mps,
        "speed_blend_end_mps": speed_blend_end_mps,
        "altitude_blend_start_m": altitude_blend_start_m,
        "altitude_blend_end_m": altitude_blend_end_m,
    }
    for name, value in finite_values.items():
        if not math.isfinite(float(value)):
            raise ValueError("{} must be finite".format(name))
    for name in (
        "low_speed_tau_sec",
        "high_speed_low_alt_tau_sec",
        "high_speed_high_alt_tau_sec",
    ):
        if float(finite_values[name]) <= 0.0:
            raise ValueError("{} must be positive".format(name))
    if float(speed_blend_end_mps) <= float(speed_blend_start_mps):
        raise ValueError("speed blend end must exceed start")
    if float(altitude_blend_end_m) <= float(altitude_blend_start_m):
        raise ValueError("altitude blend end must exceed start")

    def smoothstep(value, start, end):
        unit = float(np.clip((float(value) - float(start)) / (float(end) - float(start)), 0.0, 1.0))
        return unit * unit * (3.0 - 2.0 * unit)

    speed_blend = smoothstep(
        speed_mps,
        speed_blend_start_mps,
        speed_blend_end_mps,
    )
    altitude_blend = smoothstep(
        altitude_m,
        altitude_blend_start_m,
        altitude_blend_end_m,
    )
    high_speed_tau_sec = float(high_speed_low_alt_tau_sec) + altitude_blend * (
        float(high_speed_high_alt_tau_sec)
        - float(high_speed_low_alt_tau_sec)
    )
    used_tau_sec = float(low_speed_tau_sec) + speed_blend * (
        high_speed_tau_sec - float(low_speed_tau_sec)
    )
    return used_tau_sec, {
        "active": 1.0,
        "speed_mps": float(speed_mps),
        "altitude_m": float(altitude_m),
        "speed_blend": speed_blend,
        "altitude_blend": altitude_blend,
        "high_speed_tau_sec": high_speed_tau_sec,
        "used_tau_sec": used_tau_sec,
    }


def _finite_vector(name, value, size):
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (int(size),) or not np.all(np.isfinite(vector)):
        raise ValueError(
            "{} must be a finite {}D vector".format(name, int(size))
        )
    return vector


def apply_structured_feedback_action_mask(policy_action, action_mask):
    """Mask structured PPO authorities before analytic gain composition.

    The policy interface intentionally remains four-dimensional for the
    C15 engineering smoke.  A binary mask controls which semantic gain axes
    can affect the analytic controller.  Logging both raw and used actions
    makes a disabled authority auditable without changing the actor or the
    direct-actuator action path.
    """

    raw_action = np.clip(
        _finite_vector("policy_action", policy_action, 4),
        -1.0,
        1.0,
    )
    mask = _finite_vector("action_mask", action_mask, 4)
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("action_mask must be a binary 4D vector")
    used_action = raw_action * mask
    log = {
        "action_mask_active": float(np.any(mask == 0.0)),
        "action_mask_enabled_count": float(np.sum(mask)),
    }
    for index in range(4):
        log["raw_policy_action_{}".format(index)] = float(
            raw_action[index]
        )
        log["action_mask_{}".format(index)] = float(mask[index])
    return used_action.astype(np.float64), log


def compose_structured_reference_gain_residual(
    base_target_rates,
    policy_action,
    *,
    reference_gain_fractions,
    energy_exchange_gain_fraction,
):
    """Apply bounded, sign-preserving PPO gains to analytic references.

    The first three actions modulate course-, vertical-, and speed-rate
    references.  The fourth modulates the already sign-safe analytic energy
    exchange gain.  A zero policy action is exact analytic-baseline parity;
    zero references remain zero and no channel can reverse sign.
    """

    rates = _finite_vector("base_target_rates", base_target_rates, 3)
    action = np.clip(
        _finite_vector("policy_action", policy_action, 4),
        -1.0,
        1.0,
    )
    fractions = _finite_vector(
        "reference_gain_fractions",
        reference_gain_fractions,
        3,
    )
    if np.any(fractions <= 0.0) or np.any(fractions > 1.0):
        raise ValueError(
            "reference_gain_fractions must lie in (0, 1]"
        )
    exchange_fraction = float(energy_exchange_gain_fraction)
    if (
        not math.isfinite(exchange_fraction)
        or exchange_fraction <= 0.0
        or exchange_fraction > 1.0
    ):
        raise ValueError(
            "energy_exchange_gain_fraction must lie in (0, 1]"
        )

    rate_multipliers = 1.0 + fractions * action[:3]
    modified_rates = rates * rate_multipliers
    exchange_multiplier = 1.0 + exchange_fraction * float(action[3])
    if np.any(rate_multipliers < 0.0) or exchange_multiplier < 0.0:
        raise RuntimeError("structured gain residual reversed authority")
    return modified_rates.astype(np.float64), {
        "course_rate_base": float(rates[0]),
        "vertical_rate_base": float(rates[1]),
        "speed_rate_base": float(rates[2]),
        "course_rate_multiplier": float(rate_multipliers[0]),
        "vertical_rate_multiplier": float(rate_multipliers[1]),
        "speed_rate_multiplier": float(rate_multipliers[2]),
        "course_rate_modified": float(modified_rates[0]),
        "vertical_rate_modified": float(modified_rates[1]),
        "speed_rate_modified": float(modified_rates[2]),
        "energy_exchange_gain_multiplier": exchange_multiplier,
        "policy_action_0": float(action[0]),
        "policy_action_1": float(action[1]),
        "policy_action_2": float(action[2]),
        "policy_action_3": float(action[3]),
    }


def compose_structured_feedback_gain_residual(
    policy_action,
    *,
    feedback_gain_fractions,
    energy_exchange_gain_fraction,
):
    """Return bounded positive gains for feedback laws used on point routes.

    Point-route targets are piecewise constant, so their exogenous target
    derivatives are identically zero.  These gains instead modulate the
    analytic course-error, altitude-error, and total-energy feedback terms.
    Every multiplier remains positive, preserving the corrective sign chosen
    by the analytic controller.  The fourth action modulates the already
    sign-safe measured energy-exchange gain.
    """

    action = np.clip(
        _finite_vector("policy_action", policy_action, 4),
        -1.0,
        1.0,
    )
    fractions = _finite_vector(
        "feedback_gain_fractions",
        feedback_gain_fractions,
        3,
    )
    if np.any(fractions <= 0.0) or np.any(fractions > 1.0):
        raise ValueError("feedback_gain_fractions must lie in (0, 1]")
    exchange_fraction = float(energy_exchange_gain_fraction)
    if (
        not math.isfinite(exchange_fraction)
        or exchange_fraction <= 0.0
        or exchange_fraction > 1.0
    ):
        raise ValueError(
            "energy_exchange_gain_fraction must lie in (0, 1]"
        )

    feedback_multipliers = 1.0 + fractions * action[:3]
    exchange_multiplier = 1.0 + exchange_fraction * float(action[3])
    if (
        np.any(feedback_multipliers <= 0.0)
        or exchange_multiplier <= 0.0
    ):
        raise RuntimeError("structured feedback gain reversed authority")
    return feedback_multipliers.astype(np.float64), {
        "course_feedback_multiplier": float(feedback_multipliers[0]),
        "altitude_feedback_multiplier": float(feedback_multipliers[1]),
        "total_energy_feedback_multiplier": float(
            feedback_multipliers[2]
        ),
        "energy_exchange_gain_multiplier": exchange_multiplier,
        "policy_action_0": float(action[0]),
        "policy_action_1": float(action[1]),
        "policy_action_2": float(action[2]),
        "policy_action_3": float(action[3]),
    }


def measured_energy_exchange_offset(
    *,
    height_energy_error_m,
    kinetic_energy_error_m,
    closure_tau_sec,
    height_rate_gain,
    kinetic_rate_gain_magnitude,
    max_offset_mps,
):
    """Return a bounded, opposite-sign-only energy exchange command.

    The smaller of the height and kinetic errors limits transferable energy.
    Same-sign errors require total-energy correction and therefore produce
    exactly zero flight-path offset.
    """

    positive_constants = {
        "closure_tau_sec": closure_tau_sec,
        "height_rate_gain": height_rate_gain,
        "kinetic_rate_gain_magnitude": kinetic_rate_gain_magnitude,
        "max_offset_mps": max_offset_mps,
    }
    for name, value in positive_constants.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
    height_error = float(height_energy_error_m)
    kinetic_error = float(kinetic_energy_error_m)
    if not math.isfinite(height_error) or not math.isfinite(
        kinetic_error
    ):
        raise ValueError("energy errors must be finite")

    opposite_sign = height_error * kinetic_error < 0.0
    transferable = (
        min(abs(height_error), abs(kinetic_error))
        if opposite_sign
        else 0.0
    )
    height_limited = abs(height_error) <= abs(kinetic_error)
    limiting_gain = (
        float(height_rate_gain)
        if height_limited
        else float(kinetic_rate_gain_magnitude)
    )
    raw_offset = (
        math.copysign(
            transferable
            / (float(closure_tau_sec) * limiting_gain),
            height_error,
        )
        if opposite_sign
        else 0.0
    )
    offset = float(
        np.clip(
            raw_offset,
            -float(max_offset_mps),
            float(max_offset_mps),
        )
    )
    return offset, {
        "height_error_m": height_error,
        "kinetic_error_m": kinetic_error,
        "opposite_sign_active": float(opposite_sign),
        "transferable_error_m": transferable,
        "height_limited": float(height_limited),
        "limiting_gain": limiting_gain,
        "raw_offset_mps": raw_offset,
        "offset_mps": offset,
        "offset_clipped": float(offset != raw_offset),
    }


def tecs_energy_allocation_commands(
    *,
    target_vertical_speed_mps,
    target_speed_mps,
    target_speed_rate_mps2,
    total_energy_error_m,
    energy_balance_error_m,
    total_energy_tau_sec=20.0,
    energy_balance_tau_sec=20.0,
    throttle_energy_rate_gain=0.008,
    trim_physical_throttle=0.65,
    policy_throttle_center=0.65,
    policy_throttle_half_range=0.25,
    max_balance_correction_mps=10.0,
    gravity_mps2=9.80665,
):
    """Allocate BaiyePower energy errors to throttle and vertical speed.

    The law follows the TECS separation:

    * total specific-energy error controls throttle;
    * specific-energy balance error controls vertical speed, and therefore
      the downstream pitch loop.

    All energy quantities are specific-energy heights in metres, matching
    :func:`compute_energy_cost_terms`.
    """

    positive_constants = {
        "total_energy_tau_sec": total_energy_tau_sec,
        "energy_balance_tau_sec": energy_balance_tau_sec,
        "policy_throttle_half_range": policy_throttle_half_range,
        "max_balance_correction_mps": max_balance_correction_mps,
        "gravity_mps2": gravity_mps2,
    }
    for name, value in positive_constants.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
    finite_inputs = {
        "target_vertical_speed_mps": target_vertical_speed_mps,
        "target_speed_mps": target_speed_mps,
        "target_speed_rate_mps2": target_speed_rate_mps2,
        "total_energy_error_m": total_energy_error_m,
        "energy_balance_error_m": energy_balance_error_m,
        "throttle_energy_rate_gain": throttle_energy_rate_gain,
        "trim_physical_throttle": trim_physical_throttle,
        "policy_throttle_center": policy_throttle_center,
    }
    for name, value in finite_inputs.items():
        if not math.isfinite(float(value)):
            raise ValueError("{} must be finite".format(name))
    if float(target_speed_mps) <= 0.0:
        raise ValueError("target_speed_mps must be positive")

    target_kinetic_height_rate_mps = (
        float(target_speed_mps)
        * float(target_speed_rate_mps2)
        / float(gravity_mps2)
    )
    target_total_energy_rate_mps = (
        float(target_vertical_speed_mps)
        + target_kinetic_height_rate_mps
    )
    total_energy_feedback_mps = (
        float(total_energy_error_m) / float(total_energy_tau_sec)
    )
    commanded_total_energy_rate_mps = (
        target_total_energy_rate_mps + total_energy_feedback_mps
    )
    raw_physical_throttle = (
        float(trim_physical_throttle)
        + float(throttle_energy_rate_gain)
        * commanded_total_energy_rate_mps
    )
    raw_policy_throttle = (
        raw_physical_throttle - float(policy_throttle_center)
    ) / float(policy_throttle_half_range)
    policy_throttle = float(np.clip(raw_policy_throttle, -1.0, 1.0))

    target_energy_balance_rate_mps = (
        float(target_vertical_speed_mps)
        - target_kinetic_height_rate_mps
    )
    raw_balance_feedback_mps = (
        float(energy_balance_error_m)
        / float(energy_balance_tau_sec)
    )
    balance_feedback_mps = float(
        np.clip(
            raw_balance_feedback_mps,
            -float(max_balance_correction_mps),
            float(max_balance_correction_mps),
        )
    )
    desired_vertical_speed_mps = (
        target_energy_balance_rate_mps + balance_feedback_mps
    )

    diagnostics = {
        "target_kinetic_height_rate_mps": (
            target_kinetic_height_rate_mps
        ),
        "target_total_energy_rate_mps": target_total_energy_rate_mps,
        "total_energy_feedback_mps": total_energy_feedback_mps,
        "commanded_total_energy_rate_mps": (
            commanded_total_energy_rate_mps
        ),
        "raw_physical_throttle": raw_physical_throttle,
        "raw_policy_throttle": raw_policy_throttle,
        "policy_throttle": policy_throttle,
        "target_energy_balance_rate_mps": (
            target_energy_balance_rate_mps
        ),
        "raw_balance_feedback_mps": raw_balance_feedback_mps,
        "balance_feedback_mps": balance_feedback_mps,
        "desired_vertical_speed_mps": desired_vertical_speed_mps,
    }
    if not all(math.isfinite(float(value)) for value in diagnostics.values()):
        raise ValueError("TECS energy allocation produced a non-finite result")
    return desired_vertical_speed_mps, policy_throttle, diagnostics


def update_energy_trim_estimate(
    *,
    previous_trim_throttle,
    actual_throttle,
    observed_energy_rate_mps,
    plant_gain_energy_rate_per_throttle,
    dt_sec,
    adaptation_tau_sec=5.0,
    min_trim_throttle=0.35,
    max_trim_throttle=0.75,
    max_trim_innovation=None,
):
    """Update a direction-agnostic throttle trim estimate.

    The local C5-SID plant is ``energy_rate = K * (throttle - trim)``.
    Rearranging each observation gives an instantaneous trim sample.  A
    physical-time EMA rejects decision-rate derivative noise.
    """

    finite_inputs = {
        "previous_trim_throttle": previous_trim_throttle,
        "actual_throttle": actual_throttle,
        "observed_energy_rate_mps": observed_energy_rate_mps,
        "plant_gain_energy_rate_per_throttle": (
            plant_gain_energy_rate_per_throttle
        ),
        "dt_sec": dt_sec,
        "adaptation_tau_sec": adaptation_tau_sec,
        "min_trim_throttle": min_trim_throttle,
        "max_trim_throttle": max_trim_throttle,
    }
    for name, value in finite_inputs.items():
        if not math.isfinite(float(value)):
            raise ValueError("{} must be finite".format(name))
    if float(plant_gain_energy_rate_per_throttle) <= 0.0:
        raise ValueError(
            "plant_gain_energy_rate_per_throttle must be positive"
        )
    if float(dt_sec) <= 0.0 or float(adaptation_tau_sec) <= 0.0:
        raise ValueError("dt_sec and adaptation_tau_sec must be positive")
    if float(min_trim_throttle) >= float(max_trim_throttle):
        raise ValueError(
            "min_trim_throttle must be below max_trim_throttle"
        )
    innovation_limit_enabled = max_trim_innovation is not None
    if innovation_limit_enabled and (
        not math.isfinite(float(max_trim_innovation))
        or float(max_trim_innovation) <= 0.0
    ):
        raise ValueError("max_trim_innovation must be finite and positive")

    raw_sample = (
        float(actual_throttle)
        - float(observed_energy_rate_mps)
        / float(plant_gain_energy_rate_per_throttle)
    )
    raw_innovation = raw_sample - float(previous_trim_throttle)
    bounded_innovation = (
        float(
            np.clip(
                raw_innovation,
                -float(max_trim_innovation),
                float(max_trim_innovation),
            )
        )
        if innovation_limit_enabled
        else raw_innovation
    )
    innovation_limited_sample = (
        float(previous_trim_throttle) + bounded_innovation
    )
    bounded_sample = float(
        np.clip(
            innovation_limited_sample,
            float(min_trim_throttle),
            float(max_trim_throttle),
        )
    )
    alpha = 1.0 - math.exp(
        -float(dt_sec) / float(adaptation_tau_sec)
    )
    updated = (
        float(previous_trim_throttle)
        + alpha * (bounded_sample - float(previous_trim_throttle))
    )
    updated = float(
        np.clip(
            updated,
            float(min_trim_throttle),
            float(max_trim_throttle),
        )
    )
    return updated, {
        "observed_energy_rate_mps": float(observed_energy_rate_mps),
        "actual_throttle": float(actual_throttle),
        "raw_trim_sample": raw_sample,
        "raw_trim_innovation": raw_innovation,
        "bounded_trim_innovation": bounded_innovation,
        "trim_innovation_limit": (
            float(max_trim_innovation)
            if innovation_limit_enabled
            else 0.0
        ),
        "trim_innovation_limit_active": float(
            innovation_limit_enabled
        ),
        "trim_innovation_clipped": float(
            bounded_innovation != raw_innovation
        ),
        "bounded_trim_sample": bounded_sample,
        "adaptation_alpha": alpha,
        "estimated_trim_throttle": updated,
    }


def coordinated_reference_action(
    *,
    target_course_rate_deg_s,
    course_error_deg,
    horizontal_speed_mps,
    roll_rad,
    pitch_rad,
    vertical_speed_mps,
    target_vertical_speed_mps,
    altitude_error_m,
    speed_error_mps,
    course_tau_sec=10.0,
    bank_tau_sec=1.5,
    flight_path_tau_sec=2.0,
    altitude_tau_sec=20.0,
    speed_throttle_gain=0.01,
    max_course_rate_deg_s=3.0,
    max_bank_deg=45.0,
    max_altitude_correction_mps=10.0,
    desired_vertical_speed_override_mps=None,
    desired_vertical_speed_offset_mps=0.0,
    turn_pitch_coupling_gain=1.0,
    actual_heading_rate_deg_s=None,
    turn_pitch_actual_rate_blend=0.0,
    actual_pitch_rate_deg_s=None,
    pitch_rate_feedback_gain=0.0,
    direct_flight_path_feedback_gain_per_rad=0.0,
    direct_flight_path_feedback_max_action=0.0,
    throttle_action_override=None,
    sideslip_deg=0.0,
    sideslip_rudder_gain_per_deg=0.0,
    max_rudder_action=1.0,
    return_diagnostics=False,
    flight_path_geometry="legacy_asin_horizontal",
):
    """Return a four-channel coordinated nominal action.

    Feedback gains are expressed as physical closure times rather than fitted
    policy coefficients.  The pitch command includes the Euler-rate to
    body-rate kinematic term required during a steady banked turn.
    """

    positive_constants = {
        "course_tau_sec": course_tau_sec,
        "bank_tau_sec": bank_tau_sec,
        "flight_path_tau_sec": flight_path_tau_sec,
        "altitude_tau_sec": altitude_tau_sec,
        "max_course_rate_deg_s": max_course_rate_deg_s,
        "max_bank_deg": max_bank_deg,
        "max_altitude_correction_mps": max_altitude_correction_mps,
        "max_rudder_action": max_rudder_action,
    }
    for name, value in positive_constants.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
    if not math.isfinite(float(speed_throttle_gain)):
        raise ValueError("speed_throttle_gain must be finite")

    finite_inputs = {
        "target_course_rate_deg_s": target_course_rate_deg_s,
        "course_error_deg": course_error_deg,
        "horizontal_speed_mps": horizontal_speed_mps,
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "vertical_speed_mps": vertical_speed_mps,
        "target_vertical_speed_mps": target_vertical_speed_mps,
        "altitude_error_m": altitude_error_m,
        "speed_error_mps": speed_error_mps,
        "sideslip_deg": sideslip_deg,
        "sideslip_rudder_gain_per_deg": (
            sideslip_rudder_gain_per_deg
        ),
        "desired_vertical_speed_offset_mps": (
            desired_vertical_speed_offset_mps
        ),
    }
    for name, value in finite_inputs.items():
        if not math.isfinite(float(value)):
            raise ValueError("{} must be finite".format(name))
    if (
        not math.isfinite(float(turn_pitch_coupling_gain))
        or float(turn_pitch_coupling_gain) < 0.0
        or float(turn_pitch_coupling_gain) > 1.0
    ):
        raise ValueError("turn_pitch_coupling_gain must lie in [0, 1]")
    if (
        not math.isfinite(float(turn_pitch_actual_rate_blend))
        or float(turn_pitch_actual_rate_blend) < 0.0
        or float(turn_pitch_actual_rate_blend) > 1.0
    ):
        raise ValueError("turn_pitch_actual_rate_blend must lie in [0, 1]")
    if actual_heading_rate_deg_s is not None and not math.isfinite(
        float(actual_heading_rate_deg_s)
    ):
        raise ValueError("actual_heading_rate_deg_s must be finite")
    if (
        not math.isfinite(float(pitch_rate_feedback_gain))
        or float(pitch_rate_feedback_gain) < 0.0
        or float(pitch_rate_feedback_gain) > 1.0
    ):
        raise ValueError("pitch_rate_feedback_gain must lie in [0, 1]")
    if actual_pitch_rate_deg_s is not None and not math.isfinite(
        float(actual_pitch_rate_deg_s)
    ):
        raise ValueError("actual_pitch_rate_deg_s must be finite")
    if (
        float(turn_pitch_actual_rate_blend) > 0.0
        and actual_heading_rate_deg_s is None
    ):
        raise ValueError(
            "actual_heading_rate_deg_s is required when actual-rate blend is positive"
        )
    if float(pitch_rate_feedback_gain) > 0.0 and actual_pitch_rate_deg_s is None:
        raise ValueError(
            "actual_pitch_rate_deg_s is required when pitch-rate feedback is positive"
        )
    direct_gain = float(direct_flight_path_feedback_gain_per_rad)
    direct_limit = float(direct_flight_path_feedback_max_action)
    if not math.isfinite(direct_gain) or direct_gain < 0.0:
        raise ValueError(
            "direct_flight_path_feedback_gain_per_rad must be finite and nonnegative"
        )
    if (
        not math.isfinite(direct_limit)
        or direct_limit < 0.0
        or direct_limit > 1.0
    ):
        raise ValueError(
            "direct_flight_path_feedback_max_action must lie in [0, 1]"
        )
    if direct_gain > 0.0 and direct_limit <= 0.0:
        raise ValueError(
            "positive direct flight-path gain requires a positive action limit"
        )

    speed_horizontal = max(float(horizontal_speed_mps), 1.0)
    desired_course_rate_deg_s = float(
        np.clip(
            float(target_course_rate_deg_s)
            + float(course_error_deg) / float(course_tau_sec),
            -float(max_course_rate_deg_s),
            float(max_course_rate_deg_s),
        )
    )
    desired_course_rate_rad_s = math.radians(
        desired_course_rate_deg_s
    )
    actual_heading_rate_rad_s = (
        desired_course_rate_rad_s
        if actual_heading_rate_deg_s is None
        else math.radians(float(actual_heading_rate_deg_s))
    )
    turn_pitch_rate_rad_s = (
        (1.0 - float(turn_pitch_actual_rate_blend))
        * desired_course_rate_rad_s
        + float(turn_pitch_actual_rate_blend)
        * actual_heading_rate_rad_s
    )
    desired_bank_rad = math.atan(
        speed_horizontal * desired_course_rate_rad_s / 9.80665
    )
    desired_bank_rad = float(
        np.clip(
            desired_bank_rad,
            -math.radians(float(max_bank_deg)),
            math.radians(float(max_bank_deg)),
        )
    )
    bank_error_rad = math.remainder(
        desired_bank_rad - float(roll_rad),
        2.0 * math.pi,
    )
    desired_bank_rate_rad_s = bank_error_rad / float(bank_tau_sec)
    desired_roll_rate_rad_s = (
        desired_bank_rate_rad_s
        - desired_course_rate_rad_s * math.sin(float(pitch_rad))
    )
    # f16.xml: roll-rate-norm = 0.31821 * p[rad/s].
    aileron_action = 0.31821 * desired_roll_rate_rad_s

    if desired_vertical_speed_override_mps is None:
        altitude_correction_mps = float(
            np.clip(
                float(altitude_error_m) / float(altitude_tau_sec),
                -float(max_altitude_correction_mps),
                float(max_altitude_correction_mps),
            )
        )
        desired_vertical_speed_mps = (
            float(target_vertical_speed_mps) + altitude_correction_mps
        )
    else:
        if not math.isfinite(float(desired_vertical_speed_override_mps)):
            raise ValueError(
                "desired_vertical_speed_override_mps must be finite"
            )
        desired_vertical_speed_mps = float(
            desired_vertical_speed_override_mps
        )
    desired_vertical_speed_mps += float(
        desired_vertical_speed_offset_mps
    )
    desired_vertical_speed_mps = float(
        np.clip(
            desired_vertical_speed_mps,
            -0.8 * speed_horizontal,
            0.8 * speed_horizontal,
        )
    )
    # Explicit compatibility mode: never silently change the plant seen by
    # frozen checkpoints. New experiments use consistent horizontal geometry.
    if flight_path_geometry == "atan2_horizontal":
        desired_flight_path_rad = math.atan2(
            desired_vertical_speed_mps, speed_horizontal
        )
    elif flight_path_geometry == "legacy_asin_horizontal":
        desired_flight_path_rad = math.asin(
            desired_vertical_speed_mps / speed_horizontal
        )
    else:
        raise ValueError("unknown flight_path_geometry: {}".format(flight_path_geometry))
    current_flight_path_rad = math.atan2(
        float(vertical_speed_mps),
        speed_horizontal,
    )
    desired_flight_path_rate_rad_s = (
        desired_flight_path_rad - current_flight_path_rad
    ) / float(flight_path_tau_sec)
    # In a coordinated turn,
    # q = theta_dot*cos(phi) + psi_dot*sin(phi)*cos(theta).
    desired_pitch_rate_rad_s = (
        desired_flight_path_rate_rad_s * math.cos(float(roll_rad))
        + float(turn_pitch_coupling_gain)
        * turn_pitch_rate_rad_s
        * math.sin(float(roll_rad))
        * math.cos(float(pitch_rad))
    )
    effective_pitch_rate_command_rad_s = desired_pitch_rate_rad_s
    if actual_pitch_rate_deg_s is not None:
        actual_pitch_rate_rad_s = math.radians(
            float(actual_pitch_rate_deg_s)
        )
        effective_pitch_rate_command_rad_s += float(
            pitch_rate_feedback_gain
        ) * (
            desired_pitch_rate_rad_s - actual_pitch_rate_rad_s
        )
    # The F-16 model uses negative elevator command for positive pitch rate;
    # f16.xml: pitch-rate-norm = 6.2 * q[rad/s].
    base_elevator_action = -6.2 * effective_pitch_rate_command_rad_s
    flight_path_error_rad = desired_flight_path_rad - current_flight_path_rad
    direct_flight_path_raw_action = -direct_gain * flight_path_error_rad
    direct_flight_path_action = float(
        np.clip(
            direct_flight_path_raw_action,
            -direct_limit,
            direct_limit,
        )
    )
    elevator_action = base_elevator_action + direct_flight_path_action

    rudder_action = float(
        np.clip(
            float(sideslip_rudder_gain_per_deg)
            * float(sideslip_deg),
            -float(max_rudder_action),
            float(max_rudder_action),
        )
    )
    if throttle_action_override is None:
        physical_throttle = 0.65 + (
            float(speed_throttle_gain) * float(speed_error_mps)
        )
        throttle_action = (physical_throttle - 0.65) / 0.25
    else:
        if not math.isfinite(float(throttle_action_override)):
            raise ValueError("throttle_action_override must be finite")
        throttle_action = float(throttle_action_override)

    action = np.clip(
        np.asarray(
            [
                aileron_action,
                elevator_action,
                rudder_action,
                throttle_action,
            ],
            dtype=np.float32,
        ),
        -1.0,
        1.0,
    )
    if not return_diagnostics:
        return action
    return action, {
        "flight_path_geometry_atan2": float(flight_path_geometry == "atan2_horizontal"),
        "direct_flight_path_feedback_gain_per_rad": direct_gain,
        "direct_flight_path_feedback_max_action": direct_limit,
        "direct_flight_path_desired_angle_deg": math.degrees(
            desired_flight_path_rad
        ),
        "direct_flight_path_measured_angle_deg": math.degrees(
            current_flight_path_rad
        ),
        "direct_flight_path_error_deg": math.degrees(
            flight_path_error_rad
        ),
        "direct_flight_path_base_elevator_action": base_elevator_action,
        "direct_flight_path_raw_action": direct_flight_path_raw_action,
        "direct_flight_path_action": direct_flight_path_action,
        "direct_flight_path_action_clipped": float(
            not math.isclose(
                direct_flight_path_raw_action,
                direct_flight_path_action,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "direct_flight_path_preclip_elevator_action": elevator_action,
    }


def compose_vertical_speed_command_residual(
    base_target_vertical_speed_mps,
    policy_action,
    *,
    action_index,
    scale_mps,
):
    """Map one policy coordinate into an analytic vertical-speed reference.

    The learned value changes a command, never a control surface or throttle
    directly.  Unused policy coordinates are retained only to preserve the
    repository's frozen four-dimensional PPO interface.
    """

    if not math.isfinite(float(base_target_vertical_speed_mps)):
        raise ValueError(
            "base_target_vertical_speed_mps must be finite"
        )
    residual_input = _finite_vector("policy_action", policy_action, 4)
    if isinstance(action_index, bool):
        raise ValueError("action_index must be an integer in [0, 3]")
    try:
        normalized_index = int(action_index)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "action_index must be an integer in [0, 3]"
        ) from None
    if (
        normalized_index != action_index
        or normalized_index < 0
        or normalized_index >= residual_input.size
    ):
        raise ValueError("action_index must be an integer in [0, 3]")
    if (
        not math.isfinite(float(scale_mps))
        or float(scale_mps) <= 0.0
    ):
        raise ValueError("scale_mps must be finite and positive")

    bounded_input = np.clip(residual_input, -1.0, 1.0)
    residual_mps = float(
        bounded_input[normalized_index] * float(scale_mps)
    )
    commanded_target_mps = float(
        float(base_target_vertical_speed_mps) + residual_mps
    )
    return (
        commanded_target_mps,
        {
            "action_index": normalized_index,
            "scale_mps": float(scale_mps),
            "policy_input": residual_input.astype(np.float32),
            "bounded_policy_input": bounded_input.astype(np.float32),
            "base_target_vertical_speed_mps": float(
                base_target_vertical_speed_mps
            ),
            "residual_mps": residual_mps,
            "commanded_target_vertical_speed_mps": commanded_target_mps,
        },
    )


def compose_energy_exchange_gain_residual(
    base_offset_mps,
    policy_action,
    *,
    action_index,
    max_fraction,
):
    """Modulate the analytic C7 exchange magnitude without changing its sign.

    The policy controls only a fractional gain around the analytic offset.  A
    zero policy value gives exact C7 parity; a bounded value in ``[-1, 1]``
    maps to ``1 +/- max_fraction``.  With ``max_fraction <= 1`` the learned
    branch can at most reduce the command to zero; it can neither reverse the
    command nor create authority when C7's opposite-sign gate is inactive.
    """

    base_offset = float(base_offset_mps)
    if not math.isfinite(base_offset):
        raise ValueError("base_offset_mps must be finite")
    residual_input = _finite_vector("policy_action", policy_action, 4)
    if isinstance(action_index, bool):
        raise ValueError("action_index must be an integer in [0, 3]")
    try:
        normalized_index = int(action_index)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("action_index must be an integer in [0, 3]") from None
    if (
        normalized_index != action_index
        or normalized_index < 0
        or normalized_index >= residual_input.size
    ):
        raise ValueError("action_index must be an integer in [0, 3]")
    fraction = float(max_fraction)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError("max_fraction must be finite and lie in (0, 1]")

    bounded_input = np.clip(residual_input, -1.0, 1.0)
    gain_residual = float(bounded_input[normalized_index] * fraction)
    gain_multiplier = 1.0 + gain_residual
    effective_offset = base_offset * gain_multiplier
    residual_mps = effective_offset - base_offset
    if base_offset != 0.0 and math.copysign(1.0, effective_offset) != math.copysign(
        1.0, base_offset
    ):
        raise RuntimeError("energy exchange gain residual reversed analytic sign")
    return residual_mps, {
        "action_index": normalized_index,
        "scale_mps": fraction,
        "policy_input": residual_input.astype(np.float32),
        "bounded_policy_input": bounded_input.astype(np.float32),
        "base_energy_exchange_offset_mps": base_offset,
        "gain_residual_fraction": gain_residual,
        "gain_multiplier": gain_multiplier,
        "residual_mps": residual_mps,
        "effective_energy_exchange_offset_mps": effective_offset,
    }


def compose_energy_exchange_fixed_gain(base_offset_mps, gain_multiplier):
    """Apply a frozen non-negative gain to the analytic C7 offset."""

    base_offset = float(base_offset_mps)
    gain = float(gain_multiplier)
    if not math.isfinite(base_offset):
        raise ValueError("base_offset_mps must be finite")
    if not math.isfinite(gain) or gain < 0.0 or gain > 2.0:
        raise ValueError("gain_multiplier must be finite and lie in [0, 2]")
    residual_mps = base_offset * (gain - 1.0)
    effective_offset = base_offset + residual_mps
    if base_offset != 0.0 and effective_offset != 0.0:
        if math.copysign(1.0, effective_offset) != math.copysign(
            1.0, base_offset
        ):
            raise RuntimeError("fixed gain reversed the analytic C7 sign")
    return residual_mps, {
        "action_index": 3,
        "scale_mps": 0.0,
        "policy_input": np.zeros(4, dtype=np.float32),
        "bounded_policy_input": np.zeros(4, dtype=np.float32),
        "base_energy_exchange_offset_mps": base_offset,
        "gain_residual_fraction": gain - 1.0,
        "gain_multiplier": gain,
        "residual_mps": residual_mps,
        "effective_energy_exchange_offset_mps": effective_offset,
    }


def compose_coordinated_residual_action(
    nominal_action,
    policy_residual,
    *,
    mode,
    residual_scale,
):
    """Compose nominal and residual actions with auditable saturation.

    ``zero_residual`` ignores the supplied policy value and is the mandatory
    feasibility gate before learning.  ``residual`` applies a fixed
    per-channel scale and is reserved for the later C1 training phase.
    """

    nominal = _finite_vector("nominal_action", nominal_action, 4)
    residual_input = _finite_vector(
        "policy_residual", policy_residual, 4
    )
    scale = _finite_vector("residual_scale", residual_scale, 4)
    if np.any(scale < 0.0) or np.any(scale > 1.0):
        raise ValueError("residual_scale must lie in [0, 1]")

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "zero_residual":
        residual_used = np.zeros(4, dtype=np.float64)
    elif normalized_mode == "residual":
        residual_used = residual_input * scale
    else:
        raise ValueError(
            "Coordinated residual mode must be zero_residual or residual"
        )

    preclip = nominal + residual_used
    composed = np.clip(preclip, -1.0, 1.0)
    clipped = np.abs(preclip - composed) > 1e-7
    return (
        composed.astype(np.float32),
        {
            "mode": normalized_mode,
            "nominal_action": nominal.astype(np.float32),
            "residual_input": residual_input.astype(np.float32),
            "residual_used": residual_used.astype(np.float32),
            "preclip_action": preclip.astype(np.float32),
            "composed_action": composed.astype(np.float32),
            "clip_mask": clipped.astype(np.float32),
            "clip_count": float(np.count_nonzero(clipped)),
        },
    )

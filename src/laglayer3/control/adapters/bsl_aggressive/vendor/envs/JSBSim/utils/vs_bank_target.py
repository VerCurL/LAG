"""C31-V4 steady-turn bank target shaped by signed vertical-speed error.

This experimental outer loop uses current measurements only. Its correction
is added before PPO residual and QP; safety remains an independent monitor.
"""

import math


VS_FILTER_TAU_SEC = 2.0
BANK_FILTER_TAU_SEC = 0.5
VS_TO_BANK_DEG_PER_MPS = 0.5
MAX_BANK_OFFSET_DEG = 10.0
GAIN_ACTION_PER_DEG = 0.003
MAX_ACTION = 0.06
MIN_TURN_RATE_DEG_S = 2.0
MIN_TURN_DWELL_SEC = 5.0
ENGAGE_ROLL_RATE_DEG_S = 3.0
DISENGAGE_ROLL_RATE_DEG_S = 8.0
FADE_IN_SEC = 2.0
FADE_OUT_SEC = 0.5
OUTWARD_BANK_GUARD_DEG = 75.0
SPEED_DEFICIT_START_MPS = 8.0
SPEED_DEFICIT_FULL_MPS = 15.0
GRAVITY_MPS2 = 9.80665


def initial_vs_bank_state():
    return dict(filtered_vs_error_mps=0.0, filtered_bank_error_deg=0.0,
                previous_roll_deg=None, turn_elapsed_sec=0.0, turn_sign=0,
                engaged=False, fade=0.0)


def update_vs_bank_target(
    state, *, dt_sec, target_course_rate_deg_s, course_error_deg,
    course_tau_sec, horizontal_speed_mps, roll_deg,
    vertical_speed_mps, target_vertical_speed_mps, speed_error_mps,
    max_course_rate_deg_s, max_bank_deg,
):
    values = (dt_sec, target_course_rate_deg_s, course_error_deg,
              course_tau_sec, horizontal_speed_mps, roll_deg,
              vertical_speed_mps, target_vertical_speed_mps, speed_error_mps,
              max_course_rate_deg_s, max_bank_deg)
    if (not all(math.isfinite(float(value)) for value in values)
            or dt_sec <= 0 or course_tau_sec <= 0 or horizontal_speed_mps <= 0
            or max_course_rate_deg_s <= 0 or not 0 < max_bank_deg < 90):
        raise ValueError("invalid C31-V4 vertical-speed bank input")
    previous = initial_vs_bank_state() if state is None else dict(state)
    prior_roll = previous["previous_roll_deg"]
    numeric = (previous["filtered_vs_error_mps"],
               previous["filtered_bank_error_deg"],
               previous["turn_elapsed_sec"], previous["fade"])
    if (not all(math.isfinite(float(value)) for value in numeric)
            or (prior_roll is not None and not math.isfinite(float(prior_roll)))
            or int(previous["turn_sign"]) not in (-1, 0, 1)):
        raise ValueError("nonfinite C31-V4 state")

    turn_sign = 1 if target_course_rate_deg_s >= MIN_TURN_RATE_DEG_S else (
        -1 if target_course_rate_deg_s <= -MIN_TURN_RATE_DEG_S else 0)
    continuing = turn_sign != 0 and previous["turn_sign"] == turn_sign
    elapsed = float(previous["turn_elapsed_sec"]) + dt_sec if continuing else (
        dt_sec if turn_sign else 0.0)
    roll_rate = 0.0 if prior_roll is None else math.degrees(math.remainder(
        math.radians(roll_deg - prior_roll), 2.0 * math.pi)) / dt_sec
    raw_vs_error = float(vertical_speed_mps - target_vertical_speed_mps)
    filtered_vs_error = float(previous["filtered_vs_error_mps"]) if continuing else 0.0
    if turn_sign:
        filtered_vs_error += (1.0 - math.exp(-dt_sec / VS_FILTER_TAU_SEC)) * (
            raw_vs_error - filtered_vs_error)

    commanded_rate = max(-max_course_rate_deg_s, min(max_course_rate_deg_s,
        target_course_rate_deg_s + course_error_deg / course_tau_sec))
    geometric_bank = math.degrees(math.atan(
        horizontal_speed_mps * math.radians(commanded_rate) / GRAVITY_MPS2))
    geometric_bank = max(-max_bank_deg, min(max_bank_deg, geometric_bank))
    signed_offset = max(-MAX_BANK_OFFSET_DEG, min(MAX_BANK_OFFSET_DEG,
        VS_TO_BANK_DEG_PER_MPS * filtered_vs_error)) * turn_sign
    adjusted_bank = max(-max_bank_deg, min(max_bank_deg,
        geometric_bank + signed_offset))
    raw_bank_error = math.degrees(math.remainder(
        math.radians(adjusted_bank - roll_deg), 2.0 * math.pi))
    filtered_bank_error = float(previous["filtered_bank_error_deg"])
    if not continuing:
        filtered_bank_error = raw_bank_error
    else:
        filtered_bank_error += (1.0 - math.exp(-dt_sec / BANK_FILTER_TAU_SEC)) * (
            raw_bank_error - filtered_bank_error)

    engaged = bool(previous["engaged"]) if continuing else False
    if turn_sign and elapsed >= MIN_TURN_DWELL_SEC:
        if engaged and abs(roll_rate) > DISENGAGE_ROLL_RATE_DEG_S:
            engaged = False
        elif not engaged and abs(roll_rate) < ENGAGE_ROLL_RATE_DEG_S:
            engaged = True
    else:
        engaged = False
    fade = float(previous["fade"]) if continuing else 0.0
    if engaged:
        fade = min(1.0, fade + dt_sec / FADE_IN_SEC)
    else:
        fade = max(0.0, fade - dt_sec / FADE_OUT_SEC)
    correction = max(-MAX_ACTION, min(MAX_ACTION,
        GAIN_ACTION_PER_DEG * filtered_bank_error * fade))
    outward = correction * roll_deg > 0.0
    speed_guard = 1.0
    if outward and speed_error_mps > SPEED_DEFICIT_START_MPS:
        speed_guard = max(0.0, min(1.0,
            (SPEED_DEFICIT_FULL_MPS - speed_error_mps)
            / (SPEED_DEFICIT_FULL_MPS - SPEED_DEFICIT_START_MPS)))
        correction *= speed_guard
    outward_suppressed = abs(roll_deg) >= OUTWARD_BANK_GUARD_DEG and outward
    if outward_suppressed:
        correction = 0.0

    updated = dict(filtered_vs_error_mps=filtered_vs_error,
                   filtered_bank_error_deg=filtered_bank_error,
                   previous_roll_deg=float(roll_deg), turn_elapsed_sec=elapsed,
                   turn_sign=turn_sign, engaged=engaged, fade=fade)
    log = dict(desired_course_rate_deg_s=commanded_rate,
               geometric_bank_deg=geometric_bank, bank_offset_deg=signed_offset,
               adjusted_bank_deg=adjusted_bank, measured_bank_deg=float(roll_deg),
               raw_vs_error_mps=raw_vs_error, filtered_vs_error_mps=filtered_vs_error,
               bank_error_deg=raw_bank_error,
               filtered_bank_error_deg=filtered_bank_error,
               measured_roll_rate_deg_s=roll_rate, turn_elapsed_sec=elapsed,
               engaged=float(engaged), fade=fade, speed_guard=speed_guard,
               outward_suppressed=float(outward_suppressed),
               aileron_correction_action=correction)
    return updated, log

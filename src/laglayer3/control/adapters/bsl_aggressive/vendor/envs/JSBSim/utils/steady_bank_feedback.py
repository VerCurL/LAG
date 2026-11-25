"""C31-V3 bank correction only after the turn has settled.

Unlike C31-V2, entry roll is left entirely to the frozen C28 nominal loop.
The candidate is still experimental and does not replace the safety monitor.
"""

import math


GAIN_ACTION_PER_DEG = 0.002
MAX_ACTION = 0.04
ERROR_FILTER_TAU_SEC = 0.5
MIN_TURN_RATE_DEG_S = 2.0
MIN_TURN_DWELL_SEC = 5.0
ENGAGE_ROLL_RATE_DEG_S = 3.0
DISENGAGE_ROLL_RATE_DEG_S = 8.0
FADE_IN_SEC = 2.0
FADE_OUT_SEC = 0.5
OUTWARD_BANK_GUARD_DEG = 75.0
GRAVITY_MPS2 = 9.80665


def initial_steady_bank_state():
    return dict(filtered_error_deg=0.0, previous_roll_deg=None,
                turn_elapsed_sec=0.0, turn_sign=0, engaged=False, fade=0.0)


def update_steady_bank_feedback(
    state, *, dt_sec, target_course_rate_deg_s, course_error_deg,
    course_tau_sec, horizontal_speed_mps, roll_deg,
    max_course_rate_deg_s, max_bank_deg,
):
    values = (dt_sec, target_course_rate_deg_s, course_error_deg,
              course_tau_sec, horizontal_speed_mps, roll_deg,
              max_course_rate_deg_s, max_bank_deg)
    if (not all(math.isfinite(float(value)) for value in values)
            or dt_sec <= 0 or course_tau_sec <= 0
            or horizontal_speed_mps <= 0 or max_course_rate_deg_s <= 0
            or not 0 < max_bank_deg < 90):
        raise ValueError("invalid C31-V3 steady bank feedback input")
    previous = initial_steady_bank_state() if state is None else dict(state)
    prior_roll = previous["previous_roll_deg"]
    numeric = (previous["filtered_error_deg"], previous["turn_elapsed_sec"], previous["fade"])
    if (not all(math.isfinite(float(value)) for value in numeric)
            or (prior_roll is not None and not math.isfinite(float(prior_roll)))
            or int(previous["turn_sign"]) not in (-1, 0, 1)):
        raise ValueError("nonfinite C31-V3 state")

    commanded_rate = max(-max_course_rate_deg_s, min(max_course_rate_deg_s,
        target_course_rate_deg_s + course_error_deg / course_tau_sec))
    desired_bank = math.degrees(math.atan(
        horizontal_speed_mps * math.radians(commanded_rate) / GRAVITY_MPS2))
    desired_bank = max(-max_bank_deg, min(max_bank_deg, desired_bank))
    raw_error = math.degrees(math.remainder(
        math.radians(desired_bank - roll_deg), 2.0 * math.pi))
    roll_rate = 0.0 if prior_roll is None else math.degrees(math.remainder(
        math.radians(roll_deg - prior_roll), 2.0 * math.pi)) / dt_sec
    turn_sign = 1 if target_course_rate_deg_s >= MIN_TURN_RATE_DEG_S else (
        -1 if target_course_rate_deg_s <= -MIN_TURN_RATE_DEG_S else 0)
    continuing = turn_sign != 0 and previous["turn_sign"] == turn_sign
    elapsed = float(previous["turn_elapsed_sec"]) + dt_sec if continuing else (
        dt_sec if turn_sign else 0.0)
    filtered = float(previous["filtered_error_deg"])
    if not continuing:
        filtered = raw_error
    else:
        filtered += (1.0 - math.exp(-dt_sec / ERROR_FILTER_TAU_SEC)) * (raw_error - filtered)

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
        GAIN_ACTION_PER_DEG * filtered * fade))
    outward_suppressed = (
        abs(roll_deg) >= OUTWARD_BANK_GUARD_DEG
        and correction * roll_deg > 0.0
    )
    if outward_suppressed:
        correction = 0.0

    updated = dict(filtered_error_deg=filtered, previous_roll_deg=float(roll_deg),
                   turn_elapsed_sec=elapsed, turn_sign=turn_sign,
                   engaged=engaged, fade=fade)
    log = dict(desired_course_rate_deg_s=commanded_rate,
               desired_bank_deg=desired_bank, measured_bank_deg=float(roll_deg),
               bank_error_deg=raw_error, filtered_bank_error_deg=filtered,
               measured_roll_rate_deg_s=roll_rate, turn_elapsed_sec=elapsed,
               engaged=float(engaged), fade=fade,
               outward_suppressed=float(outward_suppressed),
               aileron_correction_action=correction)
    return updated, log

"""C31-V2 bounded bank-error feedback added before PPO and QP.

The reference bank is recomputed from the same geometry and limits as the
existing coordinated nominal controller. This module is an isolated candidate.
"""

import math


GAIN_ACTION_PER_DEG = 0.004
MAX_ACTION = 0.08
ERROR_FILTER_TAU_SEC = 0.5
GRAVITY_MPS2 = 9.80665


def update_turn_bank_feedback(
    previous_error_deg, *, dt_sec, target_course_rate_deg_s,
    course_error_deg, course_tau_sec, horizontal_speed_mps,
    roll_deg, max_course_rate_deg_s, max_bank_deg,
):
    values = (dt_sec, target_course_rate_deg_s, course_error_deg,
              course_tau_sec, horizontal_speed_mps, roll_deg,
              max_course_rate_deg_s, max_bank_deg)
    if (not all(math.isfinite(float(value)) for value in values)
            or dt_sec <= 0 or course_tau_sec <= 0
            or horizontal_speed_mps <= 0 or max_course_rate_deg_s <= 0
            or not 0 < max_bank_deg < 90):
        raise ValueError("invalid C31-V2 bank feedback input")
    if previous_error_deg is not None and not math.isfinite(float(previous_error_deg)):
        raise ValueError("nonfinite previous bank error")
    commanded_rate = max(-max_course_rate_deg_s, min(max_course_rate_deg_s,
        target_course_rate_deg_s + course_error_deg / course_tau_sec))
    desired_bank = math.degrees(math.atan(
        horizontal_speed_mps * math.radians(commanded_rate) / GRAVITY_MPS2))
    desired_bank = max(-max_bank_deg, min(max_bank_deg, desired_bank))
    raw_error = math.degrees(math.remainder(
        math.radians(desired_bank - roll_deg), 2.0 * math.pi))
    if previous_error_deg is None:
        filtered_error = raw_error
    else:
        alpha = 1.0 - math.exp(-dt_sec / ERROR_FILTER_TAU_SEC)
        filtered_error = float(previous_error_deg) + alpha * (raw_error - previous_error_deg)
    action = max(-MAX_ACTION, min(MAX_ACTION,
        GAIN_ACTION_PER_DEG * filtered_error))
    log = {"desired_course_rate_deg_s": commanded_rate,
           "desired_bank_deg": desired_bank, "measured_bank_deg": float(roll_deg),
           "bank_error_deg": raw_error, "filtered_bank_error_deg": filtered_error,
           "aileron_correction_action": action}
    return filtered_error, log

"""C31-V1 bounded, episode-local turn vertical-speed bias observer.

The output is an additive *nominal elevator action* before PPO residual and QP.
It does not alter the native FCS or estimate an aerodynamic control derivative.
"""

import math


# Frozen engineering-candidate settings. The cap is the C28 elevator residual
# authority; the slower update avoids repeating the rejected instantaneous P.
GAIN_PER_M = 0.0012
MAX_ACTION = 0.05
ERROR_FILTER_TAU_SEC = 2.0
ENGAGE_SEC = 2.0
RELEASE_TAU_SEC = 1.5
MIN_TURN_RATE_DEG_S = 2.0
MIN_BANK_DEG = 30.0
SPEED_DEFICIT_START_MPS = 8.0
SPEED_DEFICIT_FULL_MPS = 15.0
FCS_ELEVATOR_LIMIT = 0.95


def initial_turn_vertical_trim_state():
    return {"bias_action": 0.0, "filtered_error_mps": 0.0,
            "engaged_sec": 0.0, "turn_sign": 0}


def update_turn_vertical_trim(
    state, *, dt_sec, target_turn_rate_deg_s, roll_deg,
    vertical_speed_mps, target_vertical_speed_mps, speed_error_mps,
    qp_elevator_limited=False, fcs_elevator_pos_norm=0.0,
):
    """Update a slow signed task-error correction using current measurements.

    Positive elevator action reduced climb rate in the local C28 perturbation
    tests. This is a bounded closed-loop candidate, not a global plant model.
    """
    values = (dt_sec, target_turn_rate_deg_s, roll_deg, vertical_speed_mps,
              target_vertical_speed_mps, speed_error_mps, fcs_elevator_pos_norm)
    if not all(math.isfinite(float(value)) for value in values) or dt_sec <= 0:
        raise ValueError("turn vertical trim inputs must be finite with positive dt")
    previous = initial_turn_vertical_trim_state() if state is None else dict(state)
    if not all(math.isfinite(float(previous[name])) for name in
               ("bias_action", "filtered_error_mps", "engaged_sec")):
        raise ValueError("turn vertical trim state must be finite")
    if int(previous["turn_sign"]) not in (-1, 0, 1):
        raise ValueError("invalid turn sign in vertical trim state")

    turn_sign = 1 if target_turn_rate_deg_s >= MIN_TURN_RATE_DEG_S else (
        -1 if target_turn_rate_deg_s <= -MIN_TURN_RATE_DEG_S else 0)
    active = turn_sign != 0 and abs(roll_deg) >= MIN_BANK_DEG
    reversed_turn = turn_sign != 0 and previous["turn_sign"] not in (0, turn_sign)
    bias = float(previous["bias_action"])
    filtered = float(previous["filtered_error_mps"])
    engaged = float(previous["engaged_sec"])
    if reversed_turn:
        # A prior turn's trim must not be carried into the opposite turn.
        bias, filtered, engaged = 0.0, 0.0, 0.0

    raw_error = float(vertical_speed_mps - target_vertical_speed_mps)
    limited = bool(qp_elevator_limited) or abs(fcs_elevator_pos_norm) >= FCS_ELEVATOR_LIMIT
    speed_guard = 1.0
    if active:
        filtered += (1.0 - math.exp(-dt_sec / ERROR_FILTER_TAU_SEC)) * (raw_error - filtered)
        engaged += dt_sec
        if engaged >= ENGAGE_SEC and not limited:
            increment = GAIN_PER_M * filtered * dt_sec
            if increment < 0.0 and speed_error_mps > SPEED_DEFICIT_START_MPS:
                speed_guard = max(0.0, min(1.0,
                    (SPEED_DEFICIT_FULL_MPS - speed_error_mps)
                    / (SPEED_DEFICIT_FULL_MPS - SPEED_DEFICIT_START_MPS)))
                increment *= speed_guard
            bias = max(-MAX_ACTION, min(MAX_ACTION, bias + increment))
        if bias < 0.0 and speed_error_mps >= SPEED_DEFICIT_FULL_MPS:
            bias *= math.exp(-dt_sec / RELEASE_TAU_SEC)
    else:
        filtered = 0.0
        engaged = 0.0
        bias *= math.exp(-dt_sec / RELEASE_TAU_SEC)
        if abs(bias) < 1e-7:
            bias = 0.0

    updated = {"bias_action": bias, "filtered_error_mps": filtered,
               "engaged_sec": engaged, "turn_sign": turn_sign}
    log = {"active": float(active), "raw_error_mps": raw_error,
           "filtered_error_mps": filtered, "engaged_sec": engaged,
           "bias_action": bias, "qp_or_fcs_limited": float(limited),
           "speed_guard": speed_guard, "reversed_turn": float(reversed_turn)}
    return updated, log

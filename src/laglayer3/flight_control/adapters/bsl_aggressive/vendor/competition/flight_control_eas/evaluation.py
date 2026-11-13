"""Strict observed-domain scoring for the isolated 5 Hz EAS hold task.

These checks describe the available platform observations. They do not infer
CG load factor, TAS, sustained-turn capability, or aerodynamic stall speed.
"""

import math


SIDES = ("red", "blue")
REQUIRED_FIELDS = (
    "simulation/sim-time-sec", "simulation/dt", "position/h-sl-ft",
    "attitude/psi-deg", "attitude/roll-rad", "attitude/pitch-rad",
    "aero/alpha-deg", "aero/beta-deg", "velocities/ve-fps",
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",
    "LifeCurrent", "DeathEvent",
)


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("missing or nonfinite numeric observation: " + name)
    return float(value)


def observed_aircraft(observation):
    if not isinstance(observation, dict) or set(observation) != set(SIDES):
        raise ValueError("expected actual red/red_0 and blue/blue_0 frame")
    result = {}
    for side in SIDES:
        uid = side + "_0"
        team = observation[side]
        if not isinstance(team, dict) or set(team) != {uid} or not isinstance(team[uid], dict):
            raise ValueError("missing/extra aircraft in hold task: " + side)
        aircraft = team[uid]
        for field in REQUIRED_FIELDS:
            if field not in aircraft:
                raise ValueError("required observation absent: " + field)
            finite(aircraft[field], field)
        result[side] = aircraft
    return result


def wrap_degrees(value):
    return (value + 180.) % 360. - 180.


def check_frame(observation, index, protocol, previous=None):
    """Raise before emitting another action on a bad clock or unsafe frame."""
    states = observed_aircraft(observation)
    domain = protocol["observed_domain"]
    nominal = protocol["control_dt_sec"]
    tolerance = protocol["clock_tolerance_sec"]
    for side, state in states.items():
        time_sec = state["simulation/sim-time-sec"]
        if abs(time_sec - index * nominal) > tolerance:
            raise ValueError("missing, duplicate, reset or drifted timestamp: " + side)
        if previous is not None:
            previous_state = observed_aircraft(previous)[side]
            delta = time_sec - previous_state["simulation/sim-time-sec"]
            if delta <= 0. or abs(delta - nominal) > tolerance:
                raise ValueError("control interval differs from registered 5 Hz: " + side)
        if abs(state["simulation/dt"] - protocol["physics_dt_sec"]) > 1e-8:
            raise ValueError("native physics dt differs from registered observation")
        if state["LifeCurrent"] <= 0. or state["DeathEvent"] != 99.:
            raise ValueError("aircraft death/lifecycle event: " + side)
        values = {
            "height_m": state["position/h-sl-ft"] * .3048,
            "eas_mps": state["velocities/ve-fps"] * .3048,
            "alpha_deg": state["aero/alpha-deg"],
            "beta_deg": state["aero/beta-deg"],
            "roll_deg": math.degrees(state["attitude/roll-rad"]),
            "pitch_deg": math.degrees(state["attitude/pitch-rad"]),
        }
        for name, value in values.items():
            minimum, maximum = domain[name]
            if not minimum <= value <= maximum:
                raise ValueError("observed-domain exit %s %s=%r" % (side, name, value))
    if abs(states["red"]["simulation/sim-time-sec"] - states["blue"]["simulation/sim-time-sec"]) > tolerance:
        raise ValueError("red/blue clocks differ")
    return states


def complete_action(output):
    """Create the entire official mode-0 no-weapon action, without clipping."""
    vector = output.commands if hasattr(output, "commands") else output
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if not isinstance(vector, (list, tuple)) or len(vector) != 4:
        raise ValueError("controller output must provide A/E/R/T vector")
    vector = [finite(value, "control") for value in vector]
    if not all(-1. <= value <= 1. for value in vector[:3]) or not 0. <= vector[3] <= 1.:
        raise ValueError("controller command outside registered actuator box")
    return dict(zip(("fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm", "fcs/rudder-cmd-norm",
                     "fcs/throttle-cmd-norm"), vector), mode=0,
                **{"fcs/weapon-launch": 0, "switch-missile": 0, "change-target": 9})


def score_frames(frames, protocol):
    """Score the 100 post-action samples in (40, 60] seconds, both aircraft."""
    expected = protocol["actions"] + 1
    if len(frames) != expected:
        raise ValueError("complete init plus all action responses required for hold score")
    by_side = {side: [] for side in SIDES}
    previous = None
    for index, frame in enumerate(frames):
        if frame["index"] != index or frame["kind"] != ("init" if index == 0 else "hold_step"):
            raise ValueError("frame count/order/kind differs")
        states = check_frame(frame["observation"], index, protocol, previous)
        previous = frame["observation"]
        if index * protocol["control_dt_sec"] <= protocol["score_start_sec"]:
            continue
        for side, state in states.items():
            target = protocol["references"][side]
            by_side[side].append((abs(wrap_degrees(state["attitude/psi-deg"] - target["heading_deg"])),
                abs(state["position/h-sl-ft"] * .3048 - target["height_m"]),
                abs(state["velocities/ve-fps"] * .3048 - target["eas_mps"])))
    result = {}
    for side, errors in by_side.items():
        if len(errors) != protocol["score_samples"]:
            raise ValueError("registered scoring sample count differs")
        metrics = dict(zip(("heading_mae_deg", "height_mae_m", "eas_mae_mps"),
            [sum(row[column] for row in errors) / len(errors) for column in range(3)]))
        limits = protocol["acceptance"]
        passed = all(metrics[name] <= limits[name] for name in metrics)
        result[side] = dict(samples=len(errors), metrics=metrics, task_pass=passed)
    return dict(aircraft=result, red_primary_pass=result["red"]["task_pass"],
        both_aircraft_pass=all(value["task_pass"] for value in result.values()),
        observed_domain_pass=True, complete=True,
        score_window="post-action sample indices 201..300; (40,60] seconds",
        metric_speed_kind="EAS", TAS_validated=False, high_maneuver_validated=False)

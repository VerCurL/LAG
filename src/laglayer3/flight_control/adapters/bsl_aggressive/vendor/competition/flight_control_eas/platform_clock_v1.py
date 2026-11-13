"""V1-only binary32 timestamp contract and explicit controller-time mapping.

Only the time metadata passed to the frozen controller changes. All raw wire
data and physical observations are preserved. This is a compatible observed
clock model, not proof of Unity's internal implementation or sub-bin timing.
"""
from dataclasses import replace
import math
import struct


def binary32(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric platform clock required")
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("platform time does not fit binary32") from exc


NATIVE_DT_BINARY32 = binary32(1/60.)
CONTROL_DT_SEC = .2
NATIVE_STEPS_PER_ACTION = 12


def native_time_bin(index):
    if type(index) is not int or index < 0:
        raise ValueError("nonnegative exact action endpoint index required")
    return binary32(index * NATIVE_STEPS_PER_ACTION * NATIVE_DT_BINARY32)


class Binary32ControlClock:
    """A new instance belongs to one explicit init/reset episode only."""
    def __init__(self):
        self._next = 0
        self._previous = {}
        self._faulted = False

    def observe(self, observation, index):
        if self._faulted:
            raise ValueError("clock faulted; create a new clock only for explicit episode reset")
        try:
            if type(index) is not int or index != self._next:
                raise ValueError("missing, duplicate or reordered endpoint index")
            result = dict(index=index, logical_control_time_sec=index * CONTROL_DT_SEC,
                expected_native_time_binary32=native_time_bin(index), native_steps_per_action=NATIVE_STEPS_PER_ACTION,
                controller_dt_sec=CONTROL_DT_SEC, physical_observations_changed=False, aircraft={})
            for side in ("red", "blue"):
                raw = observation[side][side+"_0"]
                time_sec, dt = raw["simulation/sim-time-sec"], raw["simulation/dt"]
                actual_bin = binary32(time_sec)
                if binary32(dt) != NATIVE_DT_BINARY32:
                    raise ValueError("native-dt binary32 observation differs")
                if actual_bin != result["expected_native_time_binary32"]:
                    raise ValueError("native timestamp is not the registered binary32 action grid")
                if side in self._previous and time_sec <= self._previous[side]:
                    raise ValueError("native timestamps must be strictly increasing within episode")
                result["aircraft"][side] = dict(raw_time_sec=time_sec, raw_dt_sec=dt,
                    raw_time_binary32=actual_bin, logical_control_time_sec=result["logical_control_time_sec"])
            if result["aircraft"]["red"]["raw_time_binary32"] != result["aircraft"]["blue"]["raw_time_binary32"]:
                raise ValueError("aircraft native clock bins are not synchronized")
            self._previous = {side: value["raw_time_sec"] for side, value in result["aircraft"].items()}
            self._next += 1
            return result
        except Exception:
            self._faulted = True
            raise


def controller_state(raw_state, clock_receipt, side):
    observed = clock_receipt["aircraft"][side]
    if raw_state.time_sec != observed["raw_time_sec"]:
        raise ValueError("parsed state time differs from validated raw clock")
    logical = replace(raw_state, time_sec=clock_receipt["logical_control_time_sec"])
    raw_fields, logical_fields = raw_state.as_dict(), logical.as_dict()
    if any(value != logical_fields[key] for key, value in raw_fields.items() if key != "time_sec"):
        raise ValueError("clock mapping may not change any physical observation")
    return logical

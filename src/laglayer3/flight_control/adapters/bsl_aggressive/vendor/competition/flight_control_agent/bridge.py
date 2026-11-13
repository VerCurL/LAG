"""Own-aircraft 5 Hz bridge for the frozen PID/LQR project-1 reference.

No opponent observation, socket, simulator, online model or training is used.
Stopping on inactive or malformed own data is a conservative engineering
policy. It is not evidence that a particular death-action protocol is legal
on the real platform. Only an explicit, checked init/reset clears a fault.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
import math

from ...competition.flight_control_eas.contracts import (
    COMMAND_PATHS, EASReference, StrictEASNativeParser, command_output, finite,
)
from ...competition.flight_control_eas.platform_clock_v1 import (
    NATIVE_DT_BINARY32, binary32, native_time_bin,
)
from ...competition.flight_control_eas.platform_reset_v1 import INITIAL_FIELDS, DEFAULT_TOLERANCES

CONTROL_DT_SEC = .2
DEFAULT_HEIGHT_M = 8534.4
DEFAULT_EAS_MPS = 101.590961664
# Same limited observable bounds as the frozen V1/V2 protocols. They are not
# a structural/CG-load/stall envelope or a claim about unobserved 60 Hz extrema.
OBSERVED_DOMAIN = dict(height_m=(2500., 15000.), eas_mps=(70., 150.),
    alpha_deg=(-20., 20.), beta_deg=(-10., 10.), roll_deg=(-75., 75.), pitch_deg=(-45., 45.))


class BridgeError(ValueError):
    """A latched stop with a serializable no-action receipt."""
    def __init__(self, reason, result):
        super().__init__(reason)
        self.reason, self.result = reason, deepcopy(result)


class OwnSideBinary32Clock:
    """Verify only one aircraft's raw cadence; never synthesize a peer clock."""
    def __init__(self, side):
        if side not in ("red", "blue"):
            raise ValueError("explicit own red/blue side required")
        self.side, self.uid = side, side+"_0"
        self._next, self._previous, self._faulted = 0, None, False

    def observe(self, raw, index):
        if self._faulted:
            raise ValueError("own-side clock faulted; explicit episode reset required")
        try:
            if type(index) is not int or index != self._next:
                raise ValueError("own-side endpoint missing, duplicate or reordered")
            time_sec, dt = finite(raw["simulation/sim-time-sec"], "own simulation time"), finite(raw["simulation/dt"], "own native dt")
            expected = native_time_bin(index)
            if binary32(dt) != NATIVE_DT_BINARY32 or binary32(time_sec) != expected:
                raise ValueError("own-side raw binary32 cadence differs")
            if self._previous is not None and time_sec <= self._previous:
                raise ValueError("own-side raw time must increase strictly")
            self._previous, self._next = time_sec, self._next+1
            return dict(schema="competition-own-side-binary32-clock-v1", side=self.side, aircraft=self.uid,
                index=index, raw_time_sec=time_sec, raw_dt_sec=dt, raw_time_binary32=binary32(time_sec),
                expected_native_time_binary32=expected, logical_control_time_sec=index*CONTROL_DT_SEC,
                native_steps_per_action=12, controller_dt_sec=CONTROL_DT_SEC,
                physical_observations_changed=False, opponent_fields_read=False)
        except Exception:
            self._faulted = True
            raise


def _own(observation, side, layout):
    if not isinstance(observation, Mapping):
        raise ValueError("explicit own-aircraft observation object required")
    uid = side+"_0"
    if layout == "nested":
        team = observation.get(side)
        if not isinstance(team, Mapping):
            raise ValueError("own nested side container missing")
        raw = team.get(uid)
    elif layout == "top_level":
        raw = observation.get(uid)
    else:
        raise ValueError("observation layout must be explicitly nested or top_level")
    if not isinstance(raw, Mapping):
        raise ValueError("own aircraft "+uid+" missing")
    return raw


def own_initial_signature(raw, reference, side):
    """Use the frozen 39 fields/tolerances on own data without a fake opponent."""
    fields, failed = {}, []
    for name in INITIAL_FIELDS:
        expected = finite(reference[name], "archived own E6 "+name)
        absolute, relative = DEFAULT_TOLERANCES["overrides"].get(name,
            (DEFAULT_TOLERANCES["default_absolute"], DEFAULT_TOLERANCES["default_relative"]))
        tolerance = absolute+relative*abs(expected)
        try:
            actual = finite(raw.get(name), "own initial "+name)
        except (ValueError, TypeError):
            fields[name] = dict(actual=None, expected=expected, tolerance=tolerance,
                passed=False, error="missing_or_nonfinite_numeric_field")
            failed.append(name)
            continue
        difference = abs((actual-expected+180.)%360.-180.) if name == "attitude/psi-deg" else abs(actual-expected)
        passed = difference <= tolerance
        fields[name] = dict(actual=actual, expected=expected, tolerance=tolerance,
            abs_difference=difference, passed=passed)
        if not passed:
            failed.append(name)
    return dict(side=side, aircraft=side+"_0", passed=not failed, fields=fields,
        fields_checked=len(INITIAL_FIELDS), failed_fields=failed,
        source="C49-E6 own-aircraft observable initial signature",
        opponent_fields_read=False, hidden_trim_observed=False, unobserved_state_reset_proven=False)


def _domain(state):
    values = dict(height_m=state.altitude_m, eas_mps=state.eas_mps,
        alpha_deg=math.degrees(state.alpha_rad), beta_deg=math.degrees(state.beta_rad),
        roll_deg=math.degrees(state.roll_rad), pitch_deg=math.degrees(state.pitch_rad))
    if any(not low <= values[name] <= high for name, (low, high) in OBSERVED_DOMAIN.items()):
        raise ValueError("own state outside frozen limited observed domain")


class Bridge:
    """One project-1 aircraft, explicit layout and episode lifecycle.

    initial_observation is the SHA-bound E6 *nested* reference, independently
    of the selected live layout. Defaults are fixed own-side heading/height/EAS
    hold targets. An optional EASReference supports the sealed response replay.
    begin emits endpoint0's action; end validates the final observation only.
    """
    def __init__(self, kind, model, initial_observation, *, side="red", layout="nested"):
        if kind not in ("pid", "lqr") or side not in ("red", "blue") or layout not in ("nested", "top_level"):
            raise ValueError("explicit pure PID/LQR, own side and observation layout required")
        self.kind, self.side, self.uid, self.layout = kind, side, side+"_0", layout
        basis = model.get("command_basis", {})
        if (basis.get("schema") != "competition-native-aert-zero-hidden-trim-v1" or
            basis.get("sendable_commands") is not True or basis.get("hidden_trim_zero") is not True or
            model.get("provenance", {}).get("command_basis") != basis or
            basis.get("model_center_source") != "fresh_mapped_trim_t0_no_explicit_integration" or
            basis.get("model_center_native_time_sec") != 0 or basis.get("explicit_integration_steps_before_identification") != 0):
            raise ValueError("only the audited fresh0s sendable E2 model basis is supported")
        source = _own(initial_observation, side, "nested")
        self._reference_initial = {name: finite(source.get(name), "archived E6 "+name) for name in INITIAL_FIELDS}
        target_heading = 90. if side == "red" else 270.
        # This engineering adapter supports only the original project-1 center.
        if (abs(self._reference_initial["position/h-sl-ft"]-28000.) > .01 or
            abs(self._reference_initial["velocities/u-fps"]-525.) > .01 or
            abs(self._reference_initial["velocities/ve-fps"]*.3048-DEFAULT_EAS_MPS) > .001 or
            abs((self._reference_initial["attitude/psi-deg"]-target_heading+180.)%360.-180.) > .0001 or
            self._reference_initial["simulation/sim-time-sec"] != 0.):
            raise ValueError("only the archived project-1 center initial reference is supported")
        if kind == "pid":
            from ...competition.flight_control_eas_development.pid_speed_proportional import make_controller
        else:
            from ...competition.flight_control_eas.runtime import make_controller
        self._controller = make_controller(kind, deepcopy(model))
        self._parser = StrictEASNativeParser(side, (self.uid,))
        self._default_reference = EASReference(math.radians(target_heading), DEFAULT_HEIGHT_M, DEFAULT_EAS_MPS)
        self._clock = None
        self._init_seen, self._active, self._faulted = False, False, False
        self._episode, self._calls, self._attempted = 0, 0, 0
        self.last_result = None

    @property
    def faulted(self):
        return self._faulted

    @property
    def controller_calls(self):
        return self._calls

    def _base_result(self, *, terminal=False):
        return dict(action=None, clock_receipt=None, raw_state_si=None, controller_state_si=None,
            terminal=terminal, stopped_reason=None, side=self.side, aircraft=self.uid,
            kind=self.kind, episode=self._episode, controller_calls=self._calls,
            controller_calls_attempted=self._attempted,
            opponent_fields_read=False, own_only_initial_signature=True,
            inactive_policy="conservative_stop_not_validated_platform_death_action",
            platform_internal_physics_calls=None, hidden_trim_observed=False,
            complete_competition_agent=False)

    def _stop(self, reason, result=None):
        self._active, self._faulted = False, True
        result = self._base_result() if result is None else result
        result["action"], result["stopped_reason"] = None, reason
        result["controller_calls"] = self._calls
        result["controller_calls_attempted"] = self._attempted
        self.last_result = deepcopy(result)
        raise BridgeError(reason, result)

    def _parse_own(self, observation):
        raw = _own(observation, self.side, self.layout)
        parsed = self._parser.parse({self.side: {self.uid: raw}}, episode_event="normal")
        if parsed.inactive:
            self._stop("own_aircraft_inactive")
        return raw, parsed.states[self.uid]

    def _observe(self, raw, state, index, reference, *, terminal=False, signature=None):
        result = self._base_result(terminal=terminal)
        if signature is not None:
            result["initial_signature"] = signature
        try:
            receipt = self._clock.observe(raw, index)
            if state.time_sec != receipt["raw_time_sec"]:
                raise ValueError("parsed own state time differs from own clock")
            logical = replace(state, time_sec=receipt["logical_control_time_sec"])
            _domain(logical)
            result.update(clock_receipt=receipt, raw_state_si=state.as_dict(), controller_state_si=logical.as_dict())
            if not terminal:
                goal = self._default_reference if reference is None else reference
                if not isinstance(goal, EASReference):
                    raise ValueError("optional reference must be an explicit EASReference")
                self._attempted += 1
                output, inner = self._controller.step(logical, goal, CONTROL_DT_SEC)
                self._calls += 1
                checked = command_output(output.commands)
                if checked.commands != tuple(output.commands):
                    raise ValueError("controller emitted an out-of-range command")
                action = dict(zip(COMMAND_PATHS, checked.commands), mode=0,
                    **{"fcs/weapon-launch": 0, "switch-missile": 0, "change-target": 9})
                result.update(action={self.side: {self.uid: action}}, reference_si=asdict(goal),
                    inner_reference_si=asdict(inner), output=asdict(output), controller_calls=self._calls,
                    controller_calls_attempted=self._attempted)
            self.last_result = deepcopy(result)
            return result
        except BridgeError:
            raise
        except Exception as exc:
            self._stop("own_observation_or_control_error: "+str(exc), result)

    def begin(self, observation, event="init", reference=None):
        """Start only on explicit init/reset with a valid own E6 initial state."""
        try:
            if event not in ("init", "reset") or (event == "init" and self._init_seen):
                self._stop("illegal_init_reset_event")
            if self._faulted and event != "reset":
                self._stop("explicit_reset_required_after_fault")
            if event == "reset" and not self._init_seen and not self._faulted:
                self._stop("illegal_init_reset_event")
            self._init_seen = True
            raw, state = self._parse_own(observation)
            signature = own_initial_signature(raw, self._reference_initial, self.side)
            if signature["passed"] is not True:
                result = self._base_result(); result["initial_signature"] = signature
                self._stop("own_initial_signature_mismatch", result)
            self._controller.reset()
            self._clock = OwnSideBinary32Clock(self.side)
            self._episode += 1
            self._calls, self._attempted, self._active, self._faulted = 0, 0, True, False
            return self._observe(raw, state, 0, reference, signature=signature)
        except BridgeError:
            raise
        except Exception as exc:
            self._stop("own_begin_error: "+str(exc))

    def step(self, observation, index, reference=None):
        return self._next(observation, index, reference, terminal=False)

    def end(self, observation, index):
        """Validate the next final response, producing no additional action."""
        return self._next(observation, index, None, terminal=True)

    def _next(self, observation, index, reference, *, terminal):
        try:
            if not self._active or self._faulted:
                self._stop("explicit_init_or_reset_required")
            raw, state = self._parse_own(observation)
            result = self._observe(raw, state, index, reference, terminal=terminal)
            if terminal:
                self._active = False
            return result
        except BridgeError:
            raise
        except Exception as exc:
            self._stop("own_frame_error: "+str(exc))

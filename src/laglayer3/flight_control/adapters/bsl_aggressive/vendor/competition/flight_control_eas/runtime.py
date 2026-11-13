"""Independent EAS/5 Hz controller composition, without online identification."""
from dataclasses import asdict
import math

from .contracts import (CLOCK_TOLERANCE_SEC, EASReference, EASState, EASTrim,
                        SharedGuidance, StrictEASNativeParser, validate_dt)


def parse_aircraft(properties):
    """Use the stateless strict unit parser for one live native aircraft map."""
    parsed = StrictEASNativeParser().parse({"red": {"red_0": properties}}, episode_event="normal")
    if parsed.inactive or "red_0" not in parsed.states:
        raise ValueError("no live flight-control state for inactive aircraft")
    return parsed.states["red_0"]


def make_reference(value):
    if set(value) != {"heading_deg", "height_m", "eas_mps"}:
        raise ValueError("explicit heading/height/EAS reference required")
    return EASReference(math.radians(value["heading_deg"]), value["height_m"], value["eas_mps"])


class EASFlightController:
    def __init__(self, kind, model):
        if model.get("control_dt_sec") != .2:
            raise ValueError("new independently identified 5 Hz EAS model required")
        basis = model.get("command_basis", {})
        if (basis.get("schema") != "competition-native-aert-zero-hidden-trim-v1"
                or basis.get("sendable_commands") is not True or basis.get("hidden_trim_zero") is not True
                or model.get("provenance", {}).get("command_basis") != basis):
            raise ValueError("audited sendable A/E/R/T trim basis required; hidden-trim models are refused")
        self.trim = EASTrim.from_dict(model["trim"])
        if kind == "pid":
            from .pid import PIDController
            self.inner = PIDController(self.trim)
        elif kind == "lqr":
            from .lqr import LQRController
            self.inner = LQRController(model)
        else:
            raise ValueError("pure PID or pure LQR required")
        self.kind = kind
        self.guidance = SharedGuidance(self.trim)
        self.reset()

    def reset(self):
        self.inner.reset()
        self.guidance.reset()
        self._previous_time = self._origin = None
        self._step_index, self._faulted = 0, False

    def step(self, state, reference, dt=.2):
        if self._faulted:
            raise ValueError("controller faulted; explicit reset required")
        try:
            if not isinstance(state, EASState) or not isinstance(reference, EASReference):
                raise ValueError("new EAS typed state/reference required")
            dt = validate_dt(dt)
            if self._previous_time is not None:
                interval = state.time_sec - self._previous_time
                elapsed = state.time_sec - self._origin
                if interval <= 0 or abs(interval - dt) > CLOCK_TOLERANCE_SEC or abs(elapsed - self._step_index * dt) > CLOCK_TOLERANCE_SEC:
                    raise ValueError("control clock interval/grid differs; no interpolation")
            inner_reference = self.guidance.step(state, reference, dt)
            action = self.inner.step(state, inner_reference, dt)
            action.diagnostics["common_reference"] = asdict(inner_reference)
            if self._origin is None:
                self._origin = state.time_sec
            self._previous_time = state.time_sec
            self._step_index += 1
            return action, inner_reference
        except Exception:
            self._faulted = True
            raise


def make_controller(kind, model):
    return EASFlightController(kind, model)

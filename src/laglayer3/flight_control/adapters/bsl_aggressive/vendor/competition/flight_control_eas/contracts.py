"""Explicit E6 frame semantics and a fail-closed, simulation-clock-only fleet.

No FDM, sockets, TAS inference, interpolation, online trim, or PPO is used.
The capture establishes normal frames; sparse death semantics are exercised
synthetically, not claimed to have been observed on the running platform.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math
from numbers import Real

CONTROL_DT_SEC = .2
CLOCK_TOLERANCE_SEC = 5e-6
COMMAND_NAMES = ("aileron", "elevator", "rudder", "throttle")
COMMAND_PATHS = tuple("fcs/" + name + "-cmd-norm" for name in COMMAND_NAMES)
COMMAND_MIN = (-1., -1., -1., 0.)
COMMAND_MAX = (1., 1., 1., 1.)
STATE_NAMES = ("eas_mps", "alpha_rad", "pitch_rad", "q_radps", "beta_rad",
               "roll_rad", "p_radps", "r_radps")


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(name + " must be a finite real number")
    return float(value)


def validate_dt(value):
    value = finite(value, "control dt")
    if abs(value-CONTROL_DT_SEC) > CLOCK_TOLERANCE_SEC:
        raise ValueError("new EAS controllers require fixed 0.2 second control dt")
    # The observation timestamp is checked separately; numerical clock rounding
    # never changes the registered integration/synthesis step.
    return CONTROL_DT_SEC


def wrap_angle(value):
    return math.remainder(finite(value, "angle"), 2*math.pi)


def _finite_fields(instance):
    for name in instance.__dataclass_fields__:
        if name != "speed_kind":
            object.__setattr__(instance, name, finite(getattr(instance, name), name))


@dataclass(frozen=True)
class EASState:
    time_sec: float
    altitude_m: float
    eas_mps: float
    roll_rad: float
    pitch_rad: float
    heading_rad: float
    alpha_rad: float
    beta_rad: float
    p_radps: float
    q_radps: float
    r_radps: float
    v_north_mps: float
    v_east_mps: float
    v_down_mps: float
    speed_kind: str = "EAS"

    def __post_init__(self):
        _finite_fields(self)
        if self.time_sec < 0 or self.eas_mps <= 0 or self.speed_kind != "EAS":
            raise ValueError("nonnegative simulation time and positive explicitly EAS speed required")

    @property
    def climb_mps(self):
        return -self.v_down_mps

    def vector(self):
        return tuple(getattr(self, name) for name in STATE_NAMES)

    def as_dict(self):
        return asdict(self)

    def ground_motion(self, minimum_horizontal_mps=1e-6):
        minimum = finite(minimum_horizontal_mps, "minimum ground horizontal speed")
        if minimum <= 0:
            raise ValueError("positive minimum ground speed required")
        horizontal = math.hypot(self.v_north_mps, self.v_east_mps)
        if horizontal < minimum:
            return {"horizontal_mps": horizontal, "course_rad": None,
                    "flightpath_rad": None, "defined": False}
        return {"horizontal_mps": horizontal,
                "course_rad": math.atan2(self.v_east_mps, self.v_north_mps),
                "flightpath_rad": math.atan2(self.climb_mps, horizontal), "defined": True}


@dataclass(frozen=True)
class EASReference:
    heading_rad: float
    altitude_m: float
    eas_mps: float

    def __post_init__(self):
        _finite_fields(self)
        if self.eas_mps <= 0:
            raise ValueError("positive reference EAS required")


@dataclass(frozen=True)
class EASInnerReference:
    bank_rad: float
    pitch_rad: float
    eas_mps: float
    p_radps: float = 0.
    q_radps: float = 0.
    r_radps: float = 0.
    beta_rad: float = 0.

    def __post_init__(self):
        _finite_fields(self)
        if self.eas_mps <= 0:
            raise ValueError("positive reference EAS required")

    def vector(self, alpha_trim):
        return (self.eas_mps, finite(alpha_trim, "alpha trim"), self.pitch_rad,
                self.q_radps, self.beta_rad, self.bank_rad, self.p_radps, self.r_radps)


@dataclass(frozen=True)
class EASTrim:
    state: EASState
    commands: tuple

    def __post_init__(self):
        if not isinstance(self.state, EASState):
            raise ValueError("new EAS trim requires EASState, never historical TAS state")
        values = tuple(finite(value, "trim command") for value in self.commands)
        if len(values) != 4 or any(not low <= value <= high for value, low, high in
                                   zip(values, COMMAND_MIN, COMMAND_MAX)):
            raise ValueError("trim requires four in-range A/E/R/T commands")
        object.__setattr__(self, "commands", values)

    def as_dict(self):
        return {"state": self.state.as_dict(), "commands": list(self.commands)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {"state", "commands"}:
            raise ValueError("EAS trim object requires state and commands")
        return cls(EASState(**value["state"]), tuple(value["commands"]))


@dataclass(frozen=True)
class EASControlOutput:
    commands: tuple
    raw_commands: tuple
    saturated: tuple
    diagnostics: dict = field(default_factory=dict)


def command_output(raw_commands, diagnostics=None):
    raw = tuple(finite(value, "raw command") for value in raw_commands)
    if len(raw) != 4:
        raise ValueError("four A/E/R/T commands required")
    bounded = tuple(min(high, max(low, value)) for value, low, high in
                    zip(raw, COMMAND_MIN, COMMAND_MAX))
    return EASControlOutput(bounded, raw, tuple(a != b for a, b in zip(raw, bounded)),
                            diagnostics if diagnostics is not None else {})


@dataclass(frozen=True)
class GuidanceConfig:
    altitude_gain_per_sec: float = .08
    climb_error_gain_rad_per_mps: float = .015
    heading_bank_gain: float = 2.
    heading_rate_gain_per_sec: float = .1
    bank_limit_rad: float = math.radians(30.)
    pitch_limit_rad: float = math.radians(20.)
    climb_limit_mps: float = 10.
    heading_rate_limit_radps: float = .05

    def __post_init__(self):
        _finite_fields(self)
        if any(getattr(self, name) <= 0 for name in self.__dataclass_fields__):
            raise ValueError("positive guidance gains and limits required")
        if max(self.bank_limit_rad, self.pitch_limit_rad) >= math.pi/2:
            raise ValueError("guidance is limited to upright conventional flight")


class SharedGuidance:
    """Common observable outer loops; turn rates express kinematic demands.

    They do not use EAS in a TAS coordinated-turn formula. Consistency of the
    independent bank and heading-rate demands needs small-turn flight evidence;
    this candidate is not certified coordinated-turn or high-maneuver control.
    """
    def __init__(self, trim, config=None):
        if not isinstance(trim, EASTrim):
            raise ValueError("new EAS trim required")
        self.trim = trim
        self.config = config or GuidanceConfig()
        if not isinstance(self.config, GuidanceConfig):
            raise ValueError("GuidanceConfig required")

    def reset(self):
        pass

    def step(self, state, reference, dt=CONTROL_DT_SEC):
        validate_dt(dt)
        if not isinstance(state, EASState) or not isinstance(reference, EASReference):
            raise ValueError("new typed EAS state and reference required")
        cfg = self.config
        error = wrap_angle(reference.heading_rad-state.heading_rad)
        bank = max(-cfg.bank_limit_rad, min(cfg.bank_limit_rad, cfg.heading_bank_gain*error))
        climb = max(-cfg.climb_limit_mps, min(cfg.climb_limit_mps,
                    cfg.altitude_gain_per_sec*(reference.altitude_m-state.altitude_m)))
        pitch = self.trim.state.pitch_rad + cfg.climb_error_gain_rad_per_mps*(climb-state.climb_mps)
        pitch = max(-cfg.pitch_limit_rad, min(cfg.pitch_limit_rad, pitch))
        yaw_rate = max(-cfg.heading_rate_limit_radps, min(cfg.heading_rate_limit_radps,
                       cfg.heading_rate_gain_per_sec*error))
        return EASInnerReference(bank, pitch, reference.eas_mps,
            -math.sin(pitch)*yaw_rate,
            math.sin(bank)*math.cos(pitch)*yaw_rate,
            math.cos(bank)*math.cos(pitch)*yaw_rate)


class FrameContractError(ValueError):
    """The observation cannot produce a validated flight-control payload."""


@dataclass(frozen=True)
class ParsedFrame:
    states: dict
    inactive: dict
    event: str


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise FrameContractError(label + " must be an object")
    return value


class StrictEASNativeParser:
    """Parse the observed ``frame[side][aircraft]`` layout, without guessing.

    Episode events come from the transport, not a fictitious observation field.
    Both life and death code are required. A death/zero-life frame is inactive
    before clocks/flight properties are examined, allowing documented sparse
    dead-aircraft frames without inventing state. No action is issued for it.
    """
    def __init__(self, side="red", aircraft_ids=("red_0",), *,
                 clock_tolerance_sec=CLOCK_TOLERANCE_SEC):
        if side not in ("red", "blue"):
            raise ValueError("explicit red/blue side required")
        self.side = side
        self.aircraft_ids = tuple(aircraft_ids)
        if not self.aircraft_ids or len(set(self.aircraft_ids)) != len(self.aircraft_ids):
            raise ValueError("nonempty distinct controlled aircraft ids required")
        if any(not isinstance(key, str) or not key.startswith(side+"_") or
               not key[len(side)+1:].isdigit() for key in self.aircraft_ids):
            raise ValueError("controlled ids must belong to the explicit side")
        self.clock_tolerance_sec = finite(clock_tolerance_sec, "clock tolerance")
        if not CLOCK_TOLERANCE_SEC <= self.clock_tolerance_sec <= 1e-4:
            raise ValueError("clock tolerance must be between 5e-6 and 1e-4 seconds")

    def parse(self, frame, *, episode_event="normal"):
        if episode_event not in ("init", "reset", "normal"):
            raise FrameContractError("explicit init/reset/normal transport event required")
        own = _mapping(_mapping(frame, "frame").get(self.side), "explicit side container")
        states, inactive = {}, {}
        try:
            for key in self.aircraft_ids:
                props = _mapping(own.get(key), key)
                life = finite(props.get("LifeCurrent"), key+".LifeCurrent")
                death = finite(props.get("DeathEvent"), key+".DeathEvent")
                if life < 0 or death < 0 or not death.is_integer():
                    raise FrameContractError("nonnegative life and integer death event required")
                if life == 0 or death != 99:
                    inactive[key] = {"reason": "life_zero" if life == 0 else "death_event",
                                     "life": life, "death_event": int(death)}
                    continue
                def value(path):
                    if path not in props:
                        raise FrameContractError("missing live flight property: "+path)
                    return finite(props[path], path)
                # E6 canonical fields are authoritative; old phi/theta aliases
                # must not silently override or conflict with these fields.
                for canonical, alias in (("attitude/roll-rad", "attitude/phi-rad"),
                                         ("attitude/pitch-rad", "attitude/theta-rad")):
                    if alias in props and abs(value(canonical)-value(alias)) > 1e-9:
                        raise FrameContractError("conflicting attitude alias: "+alias)
                down = value("velocities/v-down-fps")*.3048
                if "velocities/h-dot-fps" in props:
                    climb = value("velocities/h-dot-fps")*.3048
                    if abs(climb+down) > 2e-5*max(1., abs(climb), abs(down)):
                        raise FrameContractError("inconsistent h-dot and NED down velocity")
                states[key] = EASState(value("simulation/sim-time-sec"),
                    value("position/h-sl-ft")*.3048, value("velocities/ve-fps")*.3048,
                    value("attitude/roll-rad"), value("attitude/pitch-rad"),
                    math.radians(value("attitude/psi-deg")),
                    math.radians(value("aero/alpha-deg")), math.radians(value("aero/beta-deg")),
                    value("velocities/p-rad_sec"), value("velocities/q-rad_sec"),
                    value("velocities/r-rad_sec"), value("velocities/v-north-fps")*.3048,
                    value("velocities/v-east-fps")*.3048, down)
        except FrameContractError:
            raise
        except (ValueError, TypeError, OverflowError) as exc:
            raise FrameContractError(str(exc)) from exc
        return ParsedFrame(states, inactive, episode_event)


class EASControllerFleet:
    """Independent inner controller and outer guidance history per aircraft.

    Bad observations, duplicate/gapped live clocks, and control errors latch a
    fault until explicit transport init/reset. An inactive aircraft never gets
    a command. No partially populated payload escapes after a control failure.
    The result is a flight-control fragment, not a full competition action.
    """
    def __init__(self, parser, controller_factory, *, guidance_factory=None):
        if not isinstance(parser, StrictEASNativeParser):
            raise ValueError("StrictEASNativeParser required")
        self.parser = parser
        self.controllers = {key: controller_factory(key) for key in parser.aircraft_ids}
        if len({id(value) for value in self.controllers.values()}) != len(self.controllers):
            raise ValueError("one independent inner controller per aircraft required")
        self.guidance = {key: (guidance_factory(key, ctrl.trim) if guidance_factory else
                              SharedGuidance(ctrl.trim)) for key, ctrl in self.controllers.items()}
        if len({id(value) for value in self.guidance.values()}) != len(self.guidance):
            raise ValueError("one independent guidance object per aircraft required")
        self._started = self._faulted = self._last_was_reset = False
        self._last_signature = None
        self._last_time = {}
        self._clock_origin = {}
        self._frame_count = {}
        self._dead = set()
        self.round_number = 0

    @property
    def faulted(self):
        return self._faulted

    def step(self, frame, references, *, episode_event="normal"):
        try:
            return self._step(frame, references, episode_event)
        except Exception as exc:
            self._faulted = True
            if isinstance(exc, FrameContractError):
                raise
            raise FrameContractError(str(exc)) from exc

    def _step(self, frame, references, episode_event):
        parsed = self.parser.parse(frame, episode_event=episode_event)
        reset = parsed.event in ("init", "reset")
        if (not self._started or self._faulted) and not reset:
            raise FrameContractError("explicit transport init/reset required")
        references = _mapping(references, "references")
        if set(references) != set(parsed.states) or any(not isinstance(value, EASReference)
                                                       for value in references.values()):
            raise FrameContractError("typed EAS references required for exactly the live own aircraft")
        times = [state.time_sec for state in parsed.states.values()]
        tolerance = self.parser.clock_tolerance_sec
        if times and max(times)-min(times) > tolerance:
            raise FrameContractError("live controlled aircraft clocks are not synchronous")
        signature = (tuple((key, tuple(state.as_dict().items())) for key, state in parsed.states.items()),
                     tuple((key, tuple(value.items())) for key, value in parsed.inactive.items()))
        if reset and not self._faulted and self._last_was_reset and signature == self._last_signature:
            raise FrameContractError("duplicate init/reset observation")
        if not reset:
            for key, state in parsed.states.items():
                if key in self._dead:
                    raise FrameContractError("aircraft resurrected without explicit reset: "+key)
                elapsed = state.time_sec-self._last_time[key]
                if elapsed <= 0:
                    raise FrameContractError("duplicate/out-of-order live clock: "+key)
                if abs(elapsed-CONTROL_DT_SEC) > tolerance:
                    raise FrameContractError("missing live frame or invalid 5 Hz cadence: "+key)
                expected = self._clock_origin[key]+(self._frame_count[key]+1)*CONTROL_DT_SEC
                if abs(state.time_sec-expected) > tolerance:
                    raise FrameContractError("live clock drifted from the 0.2 second frame grid: "+key)
        if reset:
            for ctrl in self.controllers.values():
                ctrl.reset()
            for guidance in self.guidance.values():
                guidance.reset()
            self._last_time.clear()
            self._clock_origin = {key: state.time_sec for key, state in parsed.states.items()}
            self._frame_count = {key: 0 for key in parsed.states}
            self._dead.clear()
            self._started, self._faulted = True, False
            self.round_number += 1
        controls, diagnostics = {}, {}
        for key, state in parsed.states.items():
            target = self.guidance[key].step(state, references[key], CONTROL_DT_SEC)
            output = self.controllers[key].step(state, target, CONTROL_DT_SEC)
            checked = command_output(output.commands)
            if checked.commands != tuple(output.commands):
                raise FrameContractError("inner controller returned out-of-range commands")
            controls[key] = dict(zip(COMMAND_PATHS, checked.commands))
            diagnostics[key] = {"state_si": state.as_dict(),
                                "inner_reference_si": asdict(target),
                                "controller": output.diagnostics}
        for key in parsed.inactive:
            if key not in self._dead:
                self.controllers[key].reset()
                self.guidance[key].reset()
            self._dead.add(key)
        self._last_time.update({key: state.time_sec for key, state in parsed.states.items()})
        if not reset:
            for key in parsed.states:
                self._frame_count[key] += 1
        self._last_was_reset, self._last_signature = reset, signature
        return {"round_number": self.round_number, "controls": controls,
                "inactive": parsed.inactive, "diagnostics": diagnostics,
                "control_dt_sec": CONTROL_DT_SEC, "speed_kind": "EAS",
                "complete_platform_action": False, "competition_runtime_verified": False}

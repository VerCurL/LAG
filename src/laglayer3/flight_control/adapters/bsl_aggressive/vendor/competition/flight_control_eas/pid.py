"""Independent EAS/5 Hz cascade PID around a supplied offline EAS trim.

This is a new unvalidated candidate, not a relabelled historical TAS controller.
The EAS speed gains are scaled from the initial TAS PI using the center-point
ratio 160.02/101.590961664. This preserves the local fixed-density command
sensitivity only; density/height coupling and 5 Hz behavior require flight
validation. Angle/rate initial gains are copied as declared starting choices.
The old 0.1-second derivative filter remains a candidate setting, not a proven
5 Hz noise or delay design.

Bank/pitch P loops generate body-rate demands on top of the common reference
feedforward. Roll/pitch rate loops use PI and a filtered derivative of the
measurement, so a reference jump causes no derivative kick. Yaw combines beta
PI with signed feedback toward the supplied coordinated-turn r feedforward;
speed PI acts on throttle. Native aircraft FCS/actuator dynamics are retained.

The local model suggests A->+p, E->-q, R->-r near trim. Under the conventional
beta=asin(v/V) convention, negative r initially increases body v/beta, hence
positive beta requires negative rudder: +K_beta*(beta_ref-beta). This is a
local sign rationale, not measured competition-platform certification.

All integrators store *physical normalized command contributions*. Conditional
integration prevents further outward windup at the final shared command box;
opposite increments can unwind. No wall-clock dt, simulator, QP, PPO, legacy
controller, or live gain adaptation is used. Defaults have not been flight-tuned.
"""

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path

from .contracts import (COMMAND_MAX, COMMAND_MIN, COMMAND_NAMES, EASState,
                      EASInnerReference, EASTrim, command_output, finite,
                      validate_dt, wrap_angle)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("pid_config.json")
_GROUPS = ("roll", "pitch", "yaw", "speed")


def _configuration(overrides):
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        defaults = json.load(stream)
    result = deepcopy(defaults)
    if overrides is not None:
        if not isinstance(overrides, Mapping) or set(overrides) - set(defaults):
            raise ValueError("PID config must be a mapping of known settings")
        for key, value in overrides.items():
            if key in _GROUPS:
                if not isinstance(value, Mapping) or set(value) - set(defaults[key]):
                    raise ValueError("unknown or invalid PID gain group: " + key)
                result[key].update(value)
            elif key == "derivative_filter_tau_sec":
                result[key] = value
            elif value != defaults[key]:
                raise ValueError("PID schema/sign/status metadata is fixed: " + key)
    tau = finite(result["derivative_filter_tau_sec"], "derivative filter tau")
    if tau < 0:
        raise ValueError("derivative filter tau must be nonnegative")
    result["derivative_filter_tau_sec"] = tau
    for group in _GROUPS:
        for name, value in result[group].items():
            value = finite(value, group + "." + name)
            if value < 0 or (name == "rate_limit_radps" and value == 0):
                raise ValueError("PID gains/limits must be nonnegative and rate limits positive")
            result[group][name] = value
    return result


def _sum(values, label):
    try:
        return finite(math.fsum(values), label)
    except OverflowError as exc:
        raise ValueError(label + " overflowed") from exc


def _bounded(value, limit):
    return max(-limit, min(limit, value))


class PIDController:
    """One aircraft's mutable PID history; construct/reset separately per aircraft.

    ``config`` is None or a partial/full mapping of the published JSON settings.
    It is copied and checked at construction; ``config`` returns a fresh snapshot.
    Explicit simulation dt is validated by the shared contract. The caller owns
    frame clocks, observation validity, task references and episode resets.
    """

    def __init__(self, trim: EASTrim, config=None):
        if not isinstance(trim, EASTrim):
            raise ValueError("PID trim must be a EASTrim")
        self.trim = trim
        self._config = _configuration(config)
        self.reset()

    @property
    def config(self):
        return deepcopy(self._config)

    def reset(self):
        self._integrals = (0., 0., 0., 0.)
        self._previous_rates = None
        self._filtered_derivatives = (0., 0.)
        self._steps = 0

    def step(self, state: EASState, reference: EASInnerReference, dt):
        if not isinstance(state, EASState) or not isinstance(reference, EASInnerReference):
            raise ValueError("PID requires EASState and EASInnerReference")
        dt = validate_dt(dt)
        # Recheck used measurements even if a caller bypassed frozen-dataclass
        # construction. An invalid step must not mutate controller history.
        names = ("roll_rad", "pitch_rad", "beta_rad", "p_radps", "q_radps", "r_radps", "eas_mps")
        measured = {name: finite(getattr(state, name), name) for name in names}
        if state.speed_kind != "EAS" or measured["eas_mps"] <= 0:
            raise ValueError("verified positive EAS required")
        desired = {name: finite(getattr(reference, name), "reference." + name)
                   for name in reference.__dataclass_fields__}
        if desired["eas_mps"] <= 0:
            raise ValueError("positive reference EAS required")
        cfg = self._config
        bank_error = wrap_angle(finite(desired["bank_rad"] - measured["roll_rad"], "bank error"))
        pitch_error = wrap_angle(finite(desired["pitch_rad"] - measured["pitch_rad"], "pitch error"))
        p_raw = _sum((desired["p_radps"], cfg["roll"]["attitude_kp"] * bank_error), "requested p")
        q_raw = _sum((desired["q_radps"], cfg["pitch"]["attitude_kp"] * pitch_error), "requested q")
        p_goal = _bounded(p_raw, cfg["roll"]["rate_limit_radps"])
        q_goal = _bounded(q_raw, cfg["pitch"]["rate_limit_radps"])
        errors = dict(bank_rad=bank_error, pitch_rad=pitch_error,
            p_radps=finite(p_goal - measured["p_radps"], "p error"),
            q_radps=finite(q_goal - measured["q_radps"], "q error"),
            r_radps=finite(desired["r_radps"] - measured["r_radps"], "r error"),
            beta_rad=wrap_angle(finite(desired["beta_rad"] - measured["beta_rad"], "beta error")),
            eas_mps=finite(desired["eas_mps"] - measured["eas_mps"], "EAS error"))

        rates = (measured["p_radps"], measured["q_radps"])
        if self._previous_rates is None:
            raw_derivatives = filtered = (0., 0.)
        else:
            raw_derivatives = tuple(finite((value - previous) / dt, "body-rate derivative")
                                    for value, previous in zip(rates, self._previous_rates))
            alpha = dt / (cfg["derivative_filter_tau_sec"] + dt)
            filtered = tuple(_sum(((1. - alpha) * old, alpha * new), "filtered body-rate derivative")
                             for old, new in zip(self._filtered_derivatives, raw_derivatives))

        proportional = (cfg["roll"]["rate_kp"] * errors["p_radps"],
                        -cfg["pitch"]["rate_kp"] * errors["q_radps"],
                        cfg["yaw"]["beta_kp"] * errors["beta_rad"] - cfg["yaw"]["rate_kp"] * errors["r_radps"],
                        cfg["speed"]["kp"] * errors["eas_mps"])
        damping = (-cfg["roll"]["rate_kd"] * filtered[0], cfg["pitch"]["rate_kd"] * filtered[1], 0., 0.)
        non_integral = tuple(_sum((trim, p, d), "PID nonintegral command")
                             for trim, p, d in zip(self.trim.commands, proportional, damping))
        increments = (cfg["roll"]["rate_ki"] * errors["p_radps"] * dt,
                      -cfg["pitch"]["rate_ki"] * errors["q_radps"] * dt,
                      cfg["yaw"]["beta_ki"] * errors["beta_rad"] * dt,
                      cfg["speed"]["ki"] * errors["eas_mps"] * dt)
        limits = tuple(cfg[group]["integral_limit_command"] for group in _GROUPS)
        unlimited = tuple(_sum((old, change), "PID integral") for old, change in zip(self._integrals, increments))
        candidates = tuple(_bounded(value, limit) for value, limit in zip(unlimited, limits))
        candidate_raw = tuple(_sum((base, integral), "candidate command")
                              for base, integral in zip(non_integral, candidates))
        # The shared projection supplies the actual command box; no independent
        # clipping or hidden output range is introduced by this controller.
        candidate_output = command_output(candidate_raw)
        blocked = tuple(saturated and ((raw > hi and candidate > previous)
                                      or (raw < lo and candidate < previous))
                        for raw, lo, hi, saturated, candidate, previous in zip(
                            candidate_raw, COMMAND_MIN, COMMAND_MAX, candidate_output.saturated,
                            candidates, self._integrals))
        integrals = tuple(old if stop else new for old, new, stop in zip(self._integrals, candidates, blocked))
        raw = tuple(_sum((base, integral), "PID raw command") for base, integral in zip(non_integral, integrals))
        diagnostic = dict(schema="competition-eas-5hz-cascade-pid-diagnostics-v1", step_index=self._steps + 1,
            dt_sec=dt, gain_schema=cfg["schema"], validity_scope=cfg["scope"], errors=errors,
            rate_reference=dict(p_unlimited_radps=p_raw, q_unlimited_radps=q_raw,
                p_radps=p_goal, q_radps=q_goal, r_feedforward_radps=desired["r_radps"],
                p_limited=p_raw != p_goal, q_limited=q_raw != q_goal),
            measured_rate_derivative=dict(initialized=self._previous_rates is not None,
                raw_radps2=list(raw_derivatives), filtered_radps2=list(filtered)),
            terms={name: dict(trim=trim, proportional=p, integral=integral, derivative=d)
                   for name, trim, p, integral, d in zip(COMMAND_NAMES, self.trim.commands, proportional, integrals, damping)},
            yaw_components=dict(beta_proportional=cfg["yaw"]["beta_kp"] * errors["beta_rad"],
                rate_feedback=-cfg["yaw"]["rate_kp"] * errors["r_radps"]),
            integrators={name: dict(before=old, unlimited=value, candidate=new, used=used,
                limit_command=limit, bound_active=value != new, anti_windup_blocked=stop)
                for name, old, value, new, used, limit, stop in zip(COMMAND_NAMES, self._integrals,
                    unlimited, candidates, integrals, limits, blocked)})
        output = command_output(raw, diagnostic)
        diagnostic["command_saturation"] = dict(zip(COMMAND_NAMES, output.saturated))
        # Commit only after every arithmetic/output check succeeds.
        self._integrals = integrals
        self._previous_rates = rates
        self._filtered_derivatives = filtered
        self._steps += 1
        return output

"""Independent 5Hz EAS LQR with measured-height affine reference allocation.

Eight inner states are regulated. Height remains an explicit parameter in the
nine-state model; it is not quietly added as another inner tracking error.
The allocator enforces four physical outputs and exact constant-attitude Euler
kinematics. It does not force eight independently invented desired states.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from ...competition.flight_control_baselines.lqr.controller import (
    discretize_zero_order_hold, solve_discrete_lqr,
)
from .contracts import EASState, EASTrim, EASInnerReference, command_output, finite, wrap_angle, validate_dt
from .contracts import COMMAND_NAMES
from .model import MODEL_SCHEMA, STATE_NAMES, _array


class UnrealizableReference(ValueError):
    """A declared model, kinematic, locality or command constraint failed."""


@dataclass(frozen=True)
class LQRConfig:
    state_scales: tuple = (12., math.radians(5.), math.radians(10.), math.radians(20.),
        math.radians(5.), math.radians(30.), math.radians(30.), math.radians(20.))
    state_weights: tuple = (1.,) * 8
    command_scales: tuple = (.25, .25, .25, .2)
    command_weights: tuple = (1.,) * 4
    maximum_reference_scaled_state: float = 1.
    maximum_reference_altitude_offset_m: float = 300.
    maximum_allocation_condition: float = 1e8
    allocation_residual: float = 1e-9
    allocation_iterations: int = 20

    def __post_init__(self):
        for name, size in (("state_scales", 8), ("state_weights", 8), ("command_scales", 4), ("command_weights", 4)):
            values = tuple(finite(v, name) for v in getattr(self, name))
            if len(values) != size or any(v <= 0 for v in values):
                raise ValueError("positive vector of prescribed length required: " + name)
            object.__setattr__(self, name, values)
        for name in ("maximum_reference_scaled_state", "maximum_reference_altitude_offset_m", "maximum_allocation_condition", "allocation_residual"):
            if finite(getattr(self, name), name) <= 0:
                raise ValueError("positive allocation threshold required")
        if type(self.allocation_iterations) is not int or not 1 <= self.allocation_iterations <= 100:
            raise ValueError("bounded positive allocation iteration budget required")


def exact_euler_rates(x):
    """x follows EAS/alpha/theta/q/beta/phi/p/r order."""
    theta, phi, p, q, r = x[2], x[5], x[6], x[3], x[7]
    if abs(math.cos(theta)) < 1e-6:
        raise UnrealizableReference("Euler kinematics singular near vertical pitch")
    return np.array([p + math.sin(phi) * math.tan(theta) * q + math.cos(phi) * math.tan(theta) * r,
                     math.cos(phi) * q - math.sin(phi) * r,
                     (math.sin(phi) * q + math.cos(phi) * r) / math.cos(theta)])


def exact_air_gamma(x):
    alpha, theta, beta, phi = x[1], x[2], x[4], x[5]
    sine = (math.sin(theta) * math.cos(alpha) * math.cos(beta)
        - math.sin(phi) * math.cos(theta) * math.sin(beta)
        - math.cos(phi) * math.cos(theta) * math.sin(alpha) * math.cos(beta))
    return math.asin(max(-1., min(1., sine)))


class ReferenceAllocator:
    """Small Newton solve in a declared local model, with no silent fallback.

    The local EAS/alpha/q/beta/p/r dynamic rows are retained. The two local
    Euler rows are replaced by exact roll-dot=pitch-dot=0 kinematics, so a
    finite-bank turning equilibrium may have nonzero body rates. This hybrid
    reference approximation is not a newly identified nonlinear model.
    """
    def __init__(self, model, config):
        self.config, self.trim = config, EASTrim.from_dict(model["trim"])
        A = _array(model["A_continuous"], (9, 9), "EAS nine-state A")
        self.A, self.Eh = A[:8, :8].copy(), A[:8, 8].copy()
        self.B = _array(model["B_continuous"], (9, 4), "EAS nine-state B")[:8].copy()
        self.x0 = np.array(self.trim.state.vector())
        self.u0 = np.array(self.trim.commands)
        self.xscale = np.array(config.state_scales)
        self.uscale = np.array(config.command_scales)
        # Dynamics normalized to a one-second state-change scale; output
        # equations use their actual engineering state scales.
        self.unknown_scale = np.r_[self.xscale, self.uscale]

    def _equations(self, normalized, delta_h, outputs):
        physical = normalized * self.unknown_scale
        z, du = physical[:8], physical[8:]
        x = self.x0 + z
        dynamic = self.A @ z + self.B @ du + self.Eh * delta_h
        rates = exact_euler_rates(x)
        # STATE_NAMES index2 theta and index5 phi, not heading.
        dynamic[2], dynamic[5] = rates[1], rates[0]
        output_error = x[[0, 2, 5, 4]] - outputs
        return np.r_[dynamic / self.xscale, output_error / self.xscale[[0, 2, 5, 4]]]

    def allocate(self, reference, altitude_m):
        if not isinstance(reference, EASInnerReference):
            raise ValueError("EASInnerReference required")
        dh = finite(altitude_m, "measured height") - self.trim.state.altitude_m
        if abs(dh) > self.config.maximum_reference_altitude_offset_m:
            raise UnrealizableReference("reference height outside declared local-model range")
        outputs = np.array([reference.eas_mps, reference.pitch_rad, reference.bank_rad, reference.beta_rad])
        normalized = np.zeros(12)
        # These are only initial guesses. Actual alpha/rates/commands are solved.
        normalized[[0, 2, 5, 4]] = (outputs - self.x0[[0, 2, 5, 4]]) / self.xscale[[0, 2, 5, 4]]
        history = []
        for iteration in range(self.config.allocation_iterations + 1):
            error = self._equations(normalized, dh, outputs)
            residual = float(np.max(np.abs(error)))
            history.append(residual)
            # Always inspect rank, even when trim has zero residual.
            J = np.empty((12, 12))
            epsilon = 1e-5
            for column in range(12):
                offset = np.zeros(12); offset[column] = epsilon
                J[:, column] = (self._equations(normalized + offset, dh, outputs)
                    - self._equations(normalized - offset, dh, outputs)) / (2 * epsilon)
            singular = np.linalg.svd(J, compute_uv=False)
            condition = float(singular[0] / singular[-1]) if singular[-1] else math.inf
            if not math.isfinite(condition) or condition > self.config.maximum_allocation_condition or singular[-1] <= singular[0] * 1e-10:
                raise UnrealizableReference("reference equation rank/conditioning failed")
            if residual <= self.config.allocation_residual:
                break
            if iteration == self.config.allocation_iterations:
                raise UnrealizableReference("reference Newton equation residual did not converge")
            update = np.linalg.solve(J, -error)
            # Deterministic residual-decreasing damping, not clipping a target.
            for power in range(12):
                proposal = normalized + update * .5 ** power
                if float(np.max(np.abs(self._equations(proposal, dh, outputs)))) < residual:
                    normalized = proposal
                    break
            else:
                raise UnrealizableReference("reference Newton direction failed decrease test")
        physical = normalized * self.unknown_scale
        z, du = physical[:8], physical[8:]
        if float(np.max(np.abs(z / self.xscale))) > self.config.maximum_reference_scaled_state:
            raise UnrealizableReference("reference outside declared scaled-state locality")
        commands = self.u0 + du
        if any(not lo <= value <= hi for value, lo, hi in zip(commands, (-1., -1., -1., 0.), (1., 1., 1., 1.))):
            raise UnrealizableReference("allocated steady command outside native command box")
        x = self.x0 + z
        rates = exact_euler_rates(x)
        return dict(state_reference=x.tolist(), state_perturbation=z.tolist(),
            feedforward_commands=commands.tolist(), feedforward_delta_commands=du.tolist(),
            measured_altitude_offset_m=dh, height_forcing=(self.Eh * dh).tolist(),
            output_order=["EAS", "pitch", "bank", "beta"], output_demand=outputs.tolist(),
            exact_euler_rates_radps=rates.tolist(), exact_air_gamma_rad=exact_air_gamma(x),
            inherited_rate_reference=[reference.p_radps, reference.q_radps, reference.r_radps],
            body_rates_are_allocated_not_independent_equalities=True,
            scaled_equation_residual=residual, scaled_condition_number=condition,
            scaled_singular_values=singular.tolist(), iterations=iteration,
            residual_history=history, local_hybrid_reference=True,
            reference_scope="six linear dynamic rows plus exact constant-attitude Euler kinematics; native response unproven")


class LQRController:
    def __init__(self, model, config=None):
        if not isinstance(model, dict) or model.get("schema") != MODEL_SCHEMA or model.get("state_names") != list(STATE_NAMES):
            raise ValueError("declared new nine-state EAS model required; old TAS models are refused")
        if model.get("command_names") != list(COMMAND_NAMES):
            raise ValueError("model must use declared native A/E/R/T command order")
        if abs(finite(model["control_dt_sec"], "control period") - .2) > 1e-12:
            raise ValueError("LQR model must be synthesized for verified5Hz period")
        self.control_dt_sec = .2
        self.trim = EASTrim.from_dict(model["trim"])
        self.config = config or LQRConfig()
        if isinstance(self.config, dict):
            self.config = LQRConfig(**self.config)
        if not isinstance(self.config, LQRConfig):
            raise ValueError("LQRConfig required")
        A = _array(model["A_continuous"], (9, 9), "EAS continuous A")
        B = _array(model["B_continuous"], (9, 4), "EAS continuous B")
        self.Ad9, self.Bd9 = discretize_zero_order_hold(A, B, .2)
        self.Ad, self.Bd = discretize_zero_order_hold(A[:8, :8], B[:8], .2)
        # Exact conditional model with dh held over a decision, not merely
        # dt*Eh. Keep this value explicit in the design artifact.
        _, augmented_B = discretize_zero_order_hold(A[:8, :8], np.column_stack([B[:8], A[:8, 8]]), .2)
        self.Ehd = augmented_B[:, 4]
        xs, us = np.array(self.config.state_scales), np.array(self.config.command_scales)
        an = self.Ad * xs[None, :] / xs[:, None]
        bn = self.Bd * us[None, :] / xs[:, None]
        kn, pn, numerical = solve_discrete_lqr(an, bn,
            np.diag(self.config.state_weights), np.diag(self.config.command_weights))
        self._gain = us[:, None] * kn / xs[None, :]
        self.allocator = ReferenceAllocator(model, self.config)
        self._diagnostics = dict(schema="competition-eas-inner8-lqr-v1", model_schema=MODEL_SCHEMA,
            control_dt_sec=.2, state_names=list(STATE_NAMES[:8]), gain=self._gain.tolist(),
            state_scales=xs.tolist(), command_scales=us.tolist(), numerical=numerical,
            conditional_Ad=self.Ad.tolist(), conditional_Bd=self.Bd.tolist(),
            conditional_height_Ed=self.Ehd.tolist(), full_Ad9=self.Ad9.tolist(), full_Bd9=self.Bd9.tolist(),
            P_normalized=pn.tolist(), altitude_is_measured_affine_parameter=True,
            altitude_error_not_an_inner_feedback_state=True,
            complete_outer_loop_stability_checked=False,
            complete_outer_loop_scope="must analyze specific shared outer loop and validate nonlinear native response separately",
            integral_states_added=0, online_tas_input=False,
            cost_scope="new EAS engineering scales; not a coordinate-invariant reuse of a historical TAS cost")
        self.reset()

    @property
    def gain(self):
        return self._gain.copy()

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)

    def reset(self):
        self._steps = 0

    def step(self, state, reference, dt):
        if not isinstance(state, EASState) or not isinstance(reference, EASInnerReference):
            raise ValueError("new EASState/EASInnerReference required")
        if abs(validate_dt(dt) - self.control_dt_sec) > 1e-12:
            raise ValueError("LQR fixed5Hz period differs")
        allocation = self.allocator.allocate(reference, state.altitude_m)
        error = np.array(state.vector()) - np.array(allocation["state_reference"])
        for index in (1, 2, 4, 5):
            error[index] = wrap_angle(error[index])
        feedback = -self._gain @ error
        self._steps += 1
        return command_output(np.array(allocation["feedforward_commands"]) + feedback,
            dict(controller="eas_lqr", step=self._steps, reference_allocation=allocation,
                state_error=error.tolist(), feedback_commands=feedback.tolist(),
                native_nonlinear_stability_validated=False, online_tas_input=False))

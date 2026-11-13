"""Offline complete local LQR/outer-loop check including EAS, height, heading.

The ten-state plant is extracted afresh from archived full12 matrices. Only
latitude/longitude are held fixed. Measured native density's local height slope
is used to reconstruct offline NED kinematics for the *actual* shared guidance;
this is not an online TAS channel or a nonlinear stability certificate.
"""
from dataclasses import asdict
import math

import numpy as np

from ...competition.flight_control_baselines.lqr.controller import discretize_zero_order_hold
from .contracts import EASState, EASReference, SharedGuidance, GuidanceConfig, finite
from .lqr import LQRController
from .model import MODEL_SCHEMA, STATE_NAMES, NATIVE_NAMES, _array, FT_TO_M


def extract_ten_state_plant(model):
    if model.get("schema") != MODEL_SCHEMA:
        raise ValueError("new EAS nine-state model with archived full12 required")
    full = model["full_linearization"]
    names, inputs = full["state_names"], full["input_names"]
    retained = list(NATIVE_NAMES) + ["Psi"]
    indices = [names.index(name) for name in retained]
    omitted = [names.index(name) for name in ("Latitude", "Longitude")]
    a = _array(full["A_continuous"], (12, 12), "full native A")
    b = _array(full["B_continuous"], (12, 4), "full native B")
    order = [inputs.index(name) for name in ("DaCmd", "DeCmd", "DrCmd", "ThtlCmd")]
    scales = np.array([FT_TO_M] + [1.] * 7 + [FT_TO_M, 1.])
    at = a[np.ix_(indices, indices)] * scales[:, None] / scales[None, :]
    bt = b[np.ix_(indices, order)] * scales[:, None]
    J = np.eye(10); J[:9, :9] = _array(model["coordinate_transform"]["J"], (9, 9), "J9")
    inverse = np.eye(10); inverse[:9, :9] = _array(model["coordinate_transform"]["J_inverse"], (9, 9), "J9 inverse")
    if not np.allclose(J @ inverse, np.eye(10), atol=1e-12, rtol=0):
        raise ValueError("EAS coordinate inverse failed")
    ae, be = J @ at @ inverse, J @ bt
    ad, bd = discretize_zero_order_hold(ae, be, .2)
    omitted_native = a[np.ix_(indices, omitted)]
    return dict(A_continuous=ae, B_continuous=be, Ad=ad, Bd=bd,
        state_names=list(STATE_NAMES) + ["heading_rad"],
        omitted_state_names=["Latitude", "Longitude"],
        omitted_coupling_native=omitted_native.tolist(),
        omitted_drive_abs_max=float(np.max(np.abs(omitted_native))),
        conditional_scope="short local analysis at fixed latitude/longitude; heading and altitude remain dynamic")


def offline_state_from_perturbation(model, trim, perturbation):
    """Air-relative Euler/NED geometry in registered offline zero wind only."""
    z = _array(perturbation, (10,), "ten-state perturbation")
    center = np.r_[trim.state.vector(), trim.state.altitude_m, trim.state.heading_rad]
    x = center + z
    rho = finite(model["atmosphere"]["rho"], "density")
    rho0 = finite(model["atmosphere"]["rho0"], "sea-level native density")
    slope = finite(model["atmosphere"]["rho_gradient_per_m"], "density slope")
    local_rho = rho + slope * z[8]
    if min(local_rho, rho0) <= 0:
        raise ValueError("offline local density must remain positive")
    tas = x[0] / math.sqrt(local_rho / rho0)
    alpha, theta, beta, phi, psi = x[1], x[2], x[4], x[5], x[9]
    body = np.array([tas * math.cos(alpha) * math.cos(beta),
                     tas * math.sin(beta), tas * math.sin(alpha) * math.cos(beta)])
    cp, sp, ct, st, cy, sy = math.cos(phi), math.sin(phi), math.cos(theta), math.sin(theta), math.cos(psi), math.sin(psi)
    rotation = np.array([[ct*cy, sp*st*cy-cp*sy, cp*st*cy+sp*sy],
                         [ct*sy, sp*st*sy+cp*cy, cp*st*sy-sp*cy],
                         [-st, sp*ct, cp*ct]])
    north, east, down = rotation @ body
    return EASState(time_sec=trim.state.time_sec, altitude_m=x[8], eas_mps=x[0],
        roll_rad=x[5], pitch_rad=x[2], heading_rad=x[9], alpha_rad=x[1], beta_rad=x[4],
        p_radps=x[6], q_radps=x[3], r_radps=x[7],
        v_north_mps=north, v_east_mps=east, v_down_mps=down)


def check_lqr_outer_loop(model, guidance_config=None, lqr_config=None, *,
                         finite_difference_step=1e-3, difference_tolerance=1e-4,
                         equilibrium_command_tolerance=1e-6, schur_margin=1e-9):
    """Two-scale Jacobian of actual shared guidance and LQR at fixed trim goals.

    The normalized local state contains EAS/alpha/pitch/q/beta/roll/p/r/h/psi.
    A Schur result applies to the supplied10-state local sampled-data model,
    not hidden FCS states or nonlinear flight. Refused references become an
    explicit failed admission, never an assumed stable fallback.
    """
    for value, name in ((finite_difference_step, "difference step"), (difference_tolerance, "difference tolerance"),
                         (equilibrium_command_tolerance, "equilibrium tolerance"), (schur_margin, "Schur margin")):
        if finite(value, name) <= 0:
            raise ValueError("positive numerical thresholds required")
    plant = extract_ten_state_plant(model)
    controller = LQRController(model, lqr_config)
    trim = controller.trim
    guidance_config = guidance_config or GuidanceConfig()
    if isinstance(guidance_config, dict):
        guidance_config = GuidanceConfig(**guidance_config)
    guidance = SharedGuidance(trim, guidance_config)
    reference = EASReference(trim.state.heading_rad, trim.state.altitude_m, trim.state.eas_mps)
    scales = np.r_[controller.config.state_scales, 100., math.radians(10.)]
    result = dict(schema="competition-eas-ten-state-outer-lqr-check-v1", analysis_pass=False,
        control_dt_sec=.2, state_names=plant["state_names"], normalized_state_scales=scales.tolist(),
        guidance_config=asdict(guidance_config), omitted_state_names=plant["omitted_state_names"],
        omitted_coupling_native=plant["omitted_coupling_native"], omitted_drive_abs_max=plant["omitted_drive_abs_max"],
        conditional_scope=plant["conditional_scope"], actual_guidance_and_controller_called=True,
        offline_native_density_geometry=True, online_tas_channel_added=False,
        complete_nonlinear_stability_proven=False, native_FDM_constructed=False,
        thresholds=dict(normalized_difference_step=finite_difference_step,
            normalized_jacobian_abs_difference=difference_tolerance,
            equilibrium_command_abs_difference=equilibrium_command_tolerance, strict_schur_margin=schur_margin))
    calls = 0

    def commands(normalized):
        nonlocal calls
        state = offline_state_from_perturbation(model, trim, normalized * scales)
        controller.reset(); guidance.reset()
        inner = guidance.step(state, reference, .2)
        output = controller.step(state, inner, .2)
        calls += 1
        if any(output.saturated):
            raise ValueError("local linear analysis point touches native command clipping")
        return np.array(output.commands) - np.array(trim.commands)

    try:
        baseline = commands(np.zeros(10))
        result["center_delta_commands"] = baseline.tolist()
        result["center_command_equilibrium_pass"] = bool(np.max(np.abs(baseline)) <= equilibrium_command_tolerance)
        matrices, feedbacks, radii = [], [], []
        for step in (finite_difference_step, finite_difference_step / 2.):
            fn = np.empty((4, 10))
            for column in range(10):
                direction = np.zeros(10); direction[column] = step
                fn[:, column] = (commands(direction) - commands(-direction)) / (2. * step)
            # fn maps normalized state to physical control. Preserve all h/psi
            # plant columns, and both the guidance and affine-height effects.
            acl = plant["Ad"] + plant["Bd"] @ (fn / scales[None, :])
            normalized_acl = acl * scales[None, :] / scales[:, None]
            eigenvalues = np.linalg.eigvals(normalized_acl)
            matrices.append(normalized_acl)
            feedbacks.append(fn)
            radii.append(float(np.max(np.abs(eigenvalues))))
        difference = float(np.max(np.abs(matrices[0] - matrices[1])))
        result.update(controller_evaluations=calls, normalized_closed_loop_matrices=[a.tolist() for a in matrices],
            normalized_state_to_command_jacobians=[f.tolist() for f in feedbacks],
            closed_loop_spectral_radii=radii, normalized_jacobian_abs_difference=difference,
            finite_difference_consistency_pass=difference <= difference_tolerance,
            strict_schur_pass=all(math.isfinite(r) and r < 1. - schur_margin for r in radii),
            full_height_and_heading_feedback_retained=True)
        result["analysis_pass"] = bool(result["center_command_equilibrium_pass"] and result["finite_difference_consistency_pass"] and result["strict_schur_pass"])
        if not result["analysis_pass"]:
            result["rejection_reasons"] = [name for name in ("center_command_equilibrium_pass", "finite_difference_consistency_pass", "strict_schur_pass") if not result[name]]
    except (ValueError, np.linalg.LinAlgError) as exc:
        result.update(controller_evaluations=calls, rejection_reasons=["actual_controller_or_reference_evaluation_refused"],
            error_type=type(exc).__name__, error_message=str(exc))
    return result

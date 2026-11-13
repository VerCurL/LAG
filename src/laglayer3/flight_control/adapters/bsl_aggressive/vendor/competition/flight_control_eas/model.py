"""Offline EAS/altitude model and native backend; construction never starts FDM.

The nine-state model is a new full-state extraction, not a renamed TAS model.
Its EAS coordinate explicitly includes measured native density's height slope.
Only offline identification may inspect TAS/density; runtime state is EAS.
"""
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from ...competition.flight_control_baselines.common import COMMAND_NAMES, COMMAND_PATHS, finite
from ...competition.flight_control_validation import local_backend_v2 as native
from .contracts import EASState, EASTrim

MODEL_SCHEMA = "competition-f16-eas-nine-state-model-v1"
STATE_NAMES = ("eas_mps", "alpha_rad", "pitch_rad", "q_radps", "beta_rad",
               "roll_rad", "p_radps", "r_radps", "altitude_m")
NATIVE_NAMES = ("Vt", "Alpha", "Theta", "Q", "Beta", "Phi", "P", "R", "Alt")
FT_TO_M = .3048
CONTROL_DT = .2
DENSITY_PATH = "atmosphere/rho-slugs_ft3"
EAS_PATH = "velocities/ve-fps"
MASS_PATHS = ("inertia/mass-slugs", "inertia/ixx-slugs_ft2", "inertia/iyy-slugs_ft2",
              "inertia/izz-slugs_ft2", "inertia/ixy-slugs_ft2", "inertia/ixz-slugs_ft2",
              "inertia/iyz-slugs_ft2", "propulsion/tank/contents-lbs")
EXPECTED_INITIAL = dict(zip(MASS_PATHS, (764.6535, 15963.7676, 57380.332,
    70914.43, 0., -1167.68066, 0., 3486.)))
RAW_PATHS = tuple(dict.fromkeys(native.RAW_PATHS + (EAS_PATH, DENSITY_PATH) + MASS_PATHS))


def _array(value, shape, name):
    original = np.asarray(value, dtype=object)
    if original.shape != shape or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in original.flat):
        raise ValueError(name + " has invalid shape or boolean values")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " requires real numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(name + " requires finite numbers")
    return result.copy()


def native_eas_state(raw):
    """Direct native EAS SI mapping. No density/TAS reconstruction at runtime."""
    return EASState(time_sec=raw["simulation/sim-time-sec"],
        altitude_m=raw["position/h-sl-ft"] * FT_TO_M,
        eas_mps=raw[EAS_PATH] * FT_TO_M,
        roll_rad=raw["attitude/phi-rad"], pitch_rad=raw["attitude/theta-rad"],
        heading_rad=raw["attitude/psi-rad"], alpha_rad=raw["aero/alpha-rad"],
        beta_rad=raw["aero/beta-rad"], p_radps=raw["velocities/p-rad_sec"],
        q_radps=raw["velocities/q-rad_sec"], r_radps=raw["velocities/r-rad_sec"],
        v_north_mps=raw["velocities/v-north-fps"] * FT_TO_M,
        v_east_mps=raw["velocities/v-east-fps"] * FT_TO_M,
        v_down_mps=raw["velocities/v-down-fps"] * FT_TO_M)


def initial_mass_checks(raw):
    """First-frame quantization tolerances, not online mass adaptation."""
    checks = {}
    for path, expected in EXPECTED_INITIAL.items():
        actual = finite(raw[path], path)
        tolerance = .01 if "contents" in path else max(1e-6, abs(expected) * 1e-6)
        checks[path] = dict(actual=actual, expected=expected, tolerance=tolerance,
                            passed=abs(actual - expected) <= tolerance)
    return dict(passed=all(v["passed"] for v in checks.values()), fields=checks,
        scope="native default before trim versus C49-E6 first frame; no property override")


def transform_full_model(full, trim, atmosphere):
    """Native full12 -> SI9 TAS/h -> SI9 EAS/h, retaining the h column.

    Heading/latitude/longitude are conditional fixed parameters of this reduced
    model. Their omitted coupling is archived and never claimed to be zero.
    The full twelve-state artifact supports a separate complete outer-loop
    analysis which restores heading. A measured density slope fixes a stationary
    local Jacobian; this is not an atmosphere estimator.
    """
    if not isinstance(trim, EASTrim):
        raise ValueError("new EAS trim required")
    names, inputs = full["state_names"], full["input_names"]
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(native.LINEAR_STATE_NAMES):
        raise ValueError("full named twelve-state native linearization required")
    if set(inputs) != set(native.LINEAR_INPUT_NAMES) or len(inputs) != 4:
        raise ValueError("named four-command native linearization required")
    units, input_units = full["state_units"], full["input_units"]
    if len(units) != 12 or len(input_units) != 4:
        raise ValueError("native unit metadata required")
    for name in NATIVE_NAMES:
        accepted = ("ft/s", "fps", "feet/s") if name == "Vt" else ("ft", "feet") if name == "Alt" else ("rad/s",) if name in ("P", "Q", "R") else ("rad",)
        if native._unit(units[names.index(name)]) not in accepted:
            raise ValueError("native unit differs for " + name)
    if any(native._unit(u) not in ("norm", "normalized", "dimensionless", "1") for u in input_units):
        raise ValueError("normalized native input units required")
    a = _array(full["A_continuous"], (12, 12), "full A")
    b = _array(full["B_continuous"], (12, 4), "full B")
    x0 = _array(full["x0"], (12,), "full x0")
    u0 = _array(full["u0"], (4,), "full u0")
    indices = [names.index(name) for name in NATIVE_NAMES]
    omitted = [i for i in range(12) if i not in indices]
    omitted_abs = float(np.max(np.abs(a[np.ix_(indices, omitted)])))
    order = [inputs.index(name) for name in native.REORDERED_NATIVE_INPUTS]
    scale = np.array([FT_TO_M] + [1.] * 7 + [FT_TO_M])
    at = a[np.ix_(indices, indices)] * scale[:, None] / scale[None, :]
    bt = b[np.ix_(indices, order)] * scale[:, None]
    rho = finite(atmosphere["rho"], "native rho")
    rho0 = finite(atmosphere["rho0"], "native sea-level rho")
    gradient = finite(atmosphere["rho_gradient_per_m"], "native density slope")
    if min(rho, rho0) <= 0:
        raise ValueError("positive native densities required")
    ratio = math.sqrt(rho / rho0)
    eas = ratio * x0[names.index("Vt")] * FT_TO_M
    ch = eas * gradient / (2. * rho)
    if abs(eas - trim.state.eas_mps) > 1e-5:
        raise ValueError("native EAS and density coordinate center disagree")
    J = np.eye(9); J[0, 0] = ratio; J[0, 8] = ch
    inverse = np.eye(9); inverse[0, 0] = 1. / ratio; inverse[0, 8] = -ch / ratio
    ae, be = J @ at @ inverse, J @ bt
    center = x0[indices] * scale; center[0] = eas
    if not np.allclose(center[:8], np.array(trim.state.vector()), atol=1e-7, rtol=1e-9) or abs(center[8] - trim.state.altitude_m) > 1e-6:
        raise ValueError("linearization center differs from measured EAS trim")
    if not np.allclose(u0[order], trim.commands, atol=1e-10, rtol=0):
        raise ValueError("linearization commands differ from measured trim")
    return dict(schema=MODEL_SCHEMA, state_names=list(STATE_NAMES), command_names=list(COMMAND_NAMES),
        state_units=["m/s", "rad", "rad", "rad/s", "rad", "rad", "rad/s", "rad/s", "m"],
        control_dt_sec=CONTROL_DT, A_continuous=ae.tolist(), B_continuous=be.tolist(),
        trim=trim.as_dict(), x0=center.tolist(), u0=u0[order].tolist(),
        atmosphere=dict(atmosphere, density_ratio_sqrt=ratio, eas_altitude_gradient_per_m=ch),
        coordinate_transform=dict(J=J.tolist(), J_inverse=inverse.tolist(),
            A_TAS_SI=at.tolist(), B_TAS_SI=bt.tolist(),
            formula="z_E=J z_T; A_E=J A_T J^-1; B_E=J B_T; stationary measured native atmosphere"),
        full_linearization=deepcopy(full), extraction=dict(native_state_indices=indices,
            native_input_indices=order, omitted_drive_abs_max=omitted_abs,
            omitted_state_names=[names[i] for i in omitted],
            omitted_coupling_native=a[np.ix_(indices, omitted)].tolist(),
            conditional_fixed_parameters=["Psi", "Latitude", "Longitude"],
            omitted_coupling_assumed_zero=False,
            fdm_disposition="retired after identification; never compare controllers on this FDM"),
        limitations=["Local fixed-atmosphere model, not globally valid or wind/adaptive-density model.",
            "Reduced9 model is conditional on fixed heading/latitude/longitude; omitted couplings are recorded, not zeroed physical facts.",
            "Nine physical states omit actuator/FCS integral and engine spool internal states.",
            "EAS/height coupling retained; inner8 regulator is conditional on measured height.",
            "No nonlinear or competition capability conclusion follows from local eigenvalues."])


class OfficialEASBackend:
    """Fresh native EAS backend with a default-mass gate before optional trim.

    Reset/step/linearize are authorized offline experiment operations. No caller
    can identify and then reuse the same FDM for comparison.
    """
    def __init__(self, model_root=None, fdm_factory=None, trim_before_handover=True):
        if type(trim_before_handover) is not bool:
            raise ValueError("explicit bool trim policy required")
        self.model_root, self.fdm_factory = model_root, fdm_factory
        self.trim_before_handover = trim_before_handover
        self.fdm = self.trim = None
        self.initialization = {}
        self.trim_calls = self.physics_steps = self.linearization_calls = 0
        self._ready = self._retired = False

    def _active(self):
        if self.fdm is None or not self._ready or self._retired:
            raise RuntimeError("fresh successful non-retired reset required")

    def _sample(self):
        raw = {path: finite(self.fdm.get_property_value(path), path) for path in RAW_PATHS}
        if raw["fcs/fbw-override"] != 0:
            raise ValueError("native FCS override must remain off")
        return native_eas_state(raw), raw

    def _clock(self):
        if abs(self.fdm.get_delta_t() - native.NATIVE_DT) > 1e-15 or self.fdm.holding() or self.fdm.integration_suspended() or self.fdm.get_trim_status():
            raise ValueError("native running60Hz clock required")

    def reset(self):
        self.close()
        root = native.resolve_official_root() if self.model_root is None else native.validate_model_root(self.model_root)
        if self.fdm_factory is None:
            import jsbsim
            factory = jsbsim.FGFDMExec
        else:
            factory = self.fdm_factory
        self.initialization = dict(stage="constructing", model_root=str(root), native_dt_sec=native.NATIVE_DT,
            trim_before_handover=self.trim_before_handover, online_trim=False,
            mass_override=False, inertia_override=False, trim_internal_evaluations=None,
            native_run_ic_calls=0, native_trim_calls=0, native_linearization_calls=0)
        self.trim_calls = self.physics_steps = self.linearization_calls = 0
        try:
            self.fdm = factory(str(root))
            self.fdm.set_debug_level(0)
            self.fdm.set_dt(native.NATIVE_DT)
            for api, name in (("set_aircraft_path", "aircraft"), ("set_engine_path", "engine"), ("set_systems_path", "systems")):
                getattr(self.fdm, api)(str(root / name))
            if self.fdm.load_model("f16") is not True:
                raise RuntimeError("official native F16 load failed")
            catalog = native._catalog(self.fdm)
            native._require_properties(catalog, RAW_PATHS, "R")
            conditions = dict(native.INITIAL_CONDITIONS)
            conditions["ic/long-gc-deg"] = -.25
            native._require_properties(catalog, tuple(conditions) + tuple(native.INITIAL_SWITCHES) + COMMAND_PATHS, "W")
            for path, value in dict(conditions, **native.INITIAL_SWITCHES).items():
                self.fdm.set_property_value(path, value)
            # Match captured E6 command handover, not a claimed trim solution.
            native.write_commands(self.fdm, (0., 0., 0., .8))
            self._run_ic()
            self.fdm.get_propulsion().init_running(-1)
            state, raw = self._sample()
            self.initialization.update(stage="initial_mass_gate", initial_raw=raw, initial_state=state.as_dict(),
                                       initial_mass_checks=initial_mass_checks(raw))
            if not self.initialization["initial_mass_checks"]["passed"]:
                raise ValueError("native default mass/inertia/fuel does not match E6; no override permitted")
            self._clock()
            if self.trim_before_handover:
                self.initialization["stage"] = "trimming"
                self.trim_calls += 1
                result = self.fdm.do_trim(1)
                if result is not None:
                    raise ValueError("native trim success must return None")
            state, raw = self._sample()
            self.initialization.update(handover_raw=raw, handover_state=state.as_dict())
            if abs(state.time_sec) > 1e-10:
                raise ValueError("initial/trim simulation time must remain zero")
            if raw["gear/gear-pos-norm"] != 0 or raw["gear/gear-cmd-norm"] != 0:
                raise ValueError("retracted gear required")
            if any(abs(raw["atmosphere/total-wind-" + axis + "-fps"]) > 1e-9 for axis in ("north", "east", "down")):
                raise ValueError("registered initial atmosphere requires zero total wind")
            self.trim = EASTrim(state, native.read_commands(self.fdm))
            self.initialization.update(stage="ready", native_trim_calls=self.trim_calls,
                handover_kind="offline_trim" if self.trim_before_handover else "untrimmed_initial_state_not_trim_solution",
                trim=self.trim.as_dict())
            self._ready = True
            return self.trim
        except Exception as exc:
            self.initialization.update(error_type=type(exc).__name__, error_message=str(exc), native_trim_calls=self.trim_calls)
            self._ready = False
            raise

    def _run_ic(self):
        self.initialization["native_run_ic_calls"] += 1
        if self.fdm.run_ic() is not True:
            raise RuntimeError("native initial-condition run failed")

    def sample(self):
        self._active()
        return self._sample()

    def write_commands(self, commands):
        self._active()
        return native.write_commands(self.fdm, commands)

    def step_commands(self, commands, substeps=12):
        self._active()
        if type(substeps) is not int or substeps < 1:
            raise ValueError("positive native step count required")
        written = self.write_commands(commands)
        result = []
        for index in range(substeps):
            self._clock()
            _, before = self._sample()
            run_result = self.fdm.run()
            self.physics_steps += 1
            if run_result is not True:
                raise RuntimeError("native integration failed; last explicit run counted, post-state unobserved")
            state, after = self._sample()
            if abs(after["simulation/sim-time-sec"] - before["simulation/sim-time-sec"] - native.NATIVE_DT) > 1e-10:
                raise ValueError("native step period differs")
            if tuple(after[p] for p in COMMAND_PATHS) != written:
                raise ValueError("native command changed within zero-order hold")
            result.append(dict(substep=index, before=before, after=after, state=state.as_dict(), command_written=written))
        return result

    def linearize(self, linearization_factory=None):
        """One full12 extraction, followed by nonintegrating density probes.

        RunIC density probes intentionally mutate this disposable FDM after
        linearization; no extra trim and no explicit native integration occurs.
        """
        self._active()
        if not self.trim_before_handover:
            raise ValueError("offline linearization requires an actual trim solution")
        try:
            _, center_raw = self._sample()
            if linearization_factory is None:
                import jsbsim
                linearization_factory = jsbsim.FGLinearization
            self.linearization_calls += 1
            self.initialization["native_linearization_calls"] = self.linearization_calls
            model = linearization_factory(self.fdm)
            full = dict(state_names=list(model.x_names), input_names=list(model.u_names),
                state_units=list(model.x_units), input_units=list(model.u_units),
                A_continuous=native._matrix(model.system_matrix, 12, 12, "fullA"),
                B_continuous=native._matrix(model.input_matrix, 12, 4, "fullB"),
                x0=native._vector(model.x0, 12, "fullx0"), u0=native._vector(model.u0, 4, "fullu0"))
            # Density0 comes from native atmosphere itself at sea level, not ISA.
            h = self.trim.state.altitude_m
            probes = []
            for name, altitude in (("minus", h - 1.), ("plus", h + 1.), ("sea_level", 0.), ("center", h)):
                self.fdm.set_property_value("ic/h-sl-ft", altitude / FT_TO_M)
                self._run_ic()
                rho = finite(self.fdm.get_property_value(DENSITY_PATH), DENSITY_PATH)
                probes.append(dict(name=name, altitude_m=altitude, rho=rho,
                    native_eas_fps=finite(self.fdm.get_property_value(EAS_PATH), EAS_PATH),
                    native_tas_fps=finite(self.fdm.get_property_value("velocities/vt-fps"), "offline native TAS")))
            by_name = {p["name"]: p for p in probes}
            atmosphere = dict(rho=center_raw[DENSITY_PATH], rho0=by_name["sea_level"]["rho"],
                rho_gradient_per_m=(by_name["plus"]["rho"] - by_name["minus"]["rho"]) / 2.,
                density_unit="slug/ft^3", slope_height_unit="m", native_density_path=DENSITY_PATH,
                probes=probes, gradient_method="central difference at +/-1m using native nonintegrating RunIC",
                density_probe_explicit_integration_steps=0)
            extracted = transform_full_model(full, self.trim, atmosphere)
            extracted["provenance"] = dict(native_api="jsbsim.FGLinearization", backend_initialization=deepcopy(self.initialization),
                offline_atmosphere_only=True, online_tas_required=False, density_probes_mutate_retired_fdm=True)
            return extracted
        finally:
            self._ready, self._retired = False, True

    def close(self):
        self.fdm = self.trim = None
        self._ready = self._retired = False

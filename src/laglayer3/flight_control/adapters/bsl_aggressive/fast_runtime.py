"""Training-only diagnostic-light application of the frozen BSL control chain.

The vendored sources and their hashes stay intact. Arithmetic follows the
delivered controller, including its original QP and complete feedback logs.
Local helper bindings avoid scalar NumPy dispatch and redundant raw-state
copying; post-QP presentation dictionaries/copies are omitted.
"""

import math
from types import FunctionType, SimpleNamespace

import numpy as np

from .vendor.scripts.delivery.ppo_windows.guarded_controller import (
    GuardedController, EFFECTIVE_RESIDUAL_MULTIPLIERS, ELEVATOR_TRIM_FEEDFORWARD, UID,
)
from .vendor.competition.flight_control_ppo_delivery.controller import (
    RawStateFacade, _NominalContext, _Task,
)
from .vendor.envs.JSBSim.utils import coordinated_action
from .vendor.envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from .vendor.envs.JSBSim.core.catalog import Catalog as c

_MULTIPLIERS = np.asarray(EFFECTIVE_RESIDUAL_MULTIPLIERS, dtype=np.float32)


def _clip(value, lower, upper, *args, **kwargs):
    # Scalar np.clip spends most of its time on array protocol dispatch. Keep
    # its float64 value/type (including NaN and signed-zero endpoint choices).
    # Array operations keep the original implementation and dtype behavior.
    if (not args and not kwargs and type(value) in (float, np.float64)
            and type(lower) in (int, float) and type(upper) in (int, float)
            and lower == lower and upper == upper):
        if value != value:
            return np.float64(value)
        if value >= upper or lower >= upper:
            return np.float64(upper)
        if value <= lower:
            return np.float64(lower)
        return np.float64(value)
    return np.clip(value, lower, upper, *args, **kwargs)


def _function(function, namespace):
    """Reuse original bytecode with lean-local helpers; never patch its module."""
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__qualname__ = function.__qualname__
    result.__module__ = function.__module__
    return result


# Each copied function has private globals. Full-diagnostic controllers and
# other imports continue to see the untouched vendor functions and NumPy.
_numpy = SimpleNamespace(**{**vars(np), "clip": _clip})
_math_globals = {**vars(coordinated_action), "np": _numpy}
for _name, _value in tuple(_math_globals.items()):
    if isinstance(_value, FunctionType) and _value.__module__ == coordinated_action.__name__:
        _math_globals[_name] = _function(_value, _math_globals)
_context_globals = {**SingleControlEnv._apply_coordinated_residual.__globals__, "np": _numpy}
for _name, _value in tuple(_context_globals.items()):
    if isinstance(_value, FunctionType) and _value.__module__ == coordinated_action.__name__:
        _context_globals[_name] = _math_globals[_value.__name__]


class _LeanRawStateFacade(RawStateFacade):
    # validate_raw has already made a fresh dictionary containing only Python
    # floats. A shallow copy therefore provides the same ownership as deepcopy.
    _units = {
        "position/h-sl-m": ("position/h-sl-ft", .3048),
        "attitude/phi-rad": ("attitude/roll-rad", 1.),
        "attitude/theta-rad": ("attitude/pitch-rad", 1.),
        "attitude/phi-deg": ("attitude/roll-rad", 180./math.pi),
        "attitude/theta-deg": ("attitude/pitch-rad", 180./math.pi),
        "attitude/psi-rad": ("attitude/psi-deg", math.pi/180.),
        "velocities/h-dot-fps": ("velocities/v-down-fps", -1.),
        **{"velocities/"+axis+"-mps": ("velocities/"+axis+"-fps", .3048)
           for axis in ("u", "v", "w", "v-north", "v-east", "v-down")},
    }

    def update(self, raw, ref, logical_time_sec):
        self.raw = raw.copy()
        self.reference = ref
        self.logical_time_sec = logical_time_sec
        # Property names come from the delivered catalog; do not assume aliases.
        self.local = {c.target_heading_deg.name_jsbsim: math.degrees(ref["heading_rad"]),
                      c.target_altitude_ft.name_jsbsim: ref["height_m"]/.3048,
                      c.baiyepower_target_speed_mps.name_jsbsim: ref["ground_speed_mps"]}

    def get_property_value(self, prop):
        name = prop if isinstance(prop, str) else prop.name_jsbsim
        if name == "simulation/sim-time-sec":
            return self.logical_time_sec
        if name in self.local:
            return self.local[name]
        if name in self.raw:
            return self.raw[name]
        if name in self._units:
            path, scale = self._units[name]
            return self.raw[path]*scale
        raise KeyError("unavailable platform property; no fabricated native value: "+name)


class _LeanNominalContext(_NominalContext):
    _apply_coordinated_residual = _function(SingleControlEnv._apply_coordinated_residual, _context_globals)
    _coordinated_nominal_action = _function(SingleControlEnv._coordinated_nominal_action, _context_globals)


class _LeanTask(_Task):
    _globals = {**_Task.normalize_action.__globals__, "np": _numpy}
    normalize_action = _function(_Task.normalize_action, _globals)
    action_to_policy_coordinates = _function(_Task.action_to_policy_coordinates, _globals)


class LeanGuardedController(GuardedController):
    def reset(self, raw, public_reference, logical_time_sec=0.):
        result = super().reset(raw, public_reference, logical_time_sec)
        # Switch only these new per-aircraft objects. Initialization, validation
        # and initial observation were performed by the delivered reset.
        self._context.__class__ = _LeanNominalContext
        self._context.agents[UID].__class__ = _LeanRawStateFacade
        self._context.task.__class__ = _LeanTask
        # Keep the complete original projector, including its feedback log:
        # qp_active_elevator participates in optional nominal trim feedback.
        return result

    def apply(self, residual):
        if self._context is None or self._faulted or not self._pending:
            raise RuntimeError("one fresh observation required per control action")
        try:
            raw = np.asarray(residual, dtype=np.float32).reshape(-1)
            if raw.shape != (4,) or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1.):
                raise ValueError("four bounded finite actual PPO residuals required; no clipping or fallback")
            effective = (raw*_MULTIPLIERS).astype(np.float32)
            env = self._context
            composed = env._apply_coordinated_residual(UID, effective)
            pre = np.asarray(env.task.normalize_action(env, UID, composed), dtype=np.float32).copy()
            if pre.shape != (4,) or not np.isfinite(pre).all():
                raise ValueError("original normalized proposal must have four finite physical channels")
            physical = pre.copy()
            physical[1] = np.float32(float(pre[1])+ELEVATOR_TRIM_FEEDFORWARD)
            if bool(getattr(self, "_assault_power", False)):
                physical[3] = np.float32(1.0)
            elif bool(getattr(self, "_aggressive_accelerate", False)):
                physical[3] = np.float32(.9)
            physical[3]=np.float32(min(float(physical[3]),getattr(self,'_speed_throttle_ceiling',1.)))
            command = np.asarray(env._apply_action_projector(UID, physical), dtype=np.float32)
            qp = env._action_projector.get_last_log(UID)
            if command.shape != (4,) or not np.isfinite(command).all() or qp["qp_projector_fail_streak"] != 0:
                raise ValueError("invalid original QP output; no fallback control permitted")
            result = {"command": command, "reference": self._reference}
            self._pending = False
            self._actions += 1
            self._last_result = result
            return result
        except Exception:
            self._faulted = True
            raise

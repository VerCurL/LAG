"""Opt-in C45 PPO observation facade; never changes the C41 control chain.

The internal task keeps its historical 54D schema and terminal cache.  This
facade emits a separate 83D schema, requiring a fresh actor and critic.  It does
not instantiate JSBSim, install hooks, query PID functions, or train a model.
Future references are available only after explicit publication at reset time.
"""

import copy
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real

import numpy as np
from gymnasium import spaces

from ...envs.JSBSim.utils.pull_dive_reference import PullDiveReference


SCHEMA = "c45-c41-policy-observation-v1"
BASE_DIM, OBSERVATION_DIM = 54, 83
PREVIEW_OFFSETS_SEC = (0.2, 0.6)
GAMMA_RATE_SCALE_RADPS = math.radians(10.0)
SURFACE_PROPERTIES = (
    "fcs/left-aileron-pos-norm", "fcs/right-aileron-pos-norm",
    "fcs/elevator-pos-norm", "fcs/rudder-pos-norm", "fcs/throttle-pos-norm",
)
CG_PROPERTY = "accelerations/Nz"
EXECUTION_NAMES = (
    "qp_prev2_aileron", "qp_prev2_elevator", "qp_prev2_rudder", "qp_prev2_throttle",
    "left_aileron_position", "right_aileron_position", "elevator_position",
    "rudder_position", "throttle_position", "cg_nz_over_5g",
    "qp_valid", "surface_valid", "cg_valid",
)
POINT_NAMES = (
    "direction_delta_n", "direction_delta_e", "direction_delta_d",
    "height_delta_over_100m", "speed_delta_over_10mps",
    "gamma_rate_over_10degps", "point_valid",
)
PREVIEW_NAMES = ("phase_remaining_over_10s", "published_plan_available") + tuple(
    "preview_{}s_{}".format(offset, name)
    for offset in PREVIEW_OFFSETS_SEC for name in POINT_NAMES
)
_CATALOG_ENTRY = re.compile(r"^([^\s()]+)\s*\(([RW]+)\)$")


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _vector(value, size, name):
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "fiu":
        raise ValueError(name + " must be a real {}D vector".format(size))
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    return np.array(array, dtype=np.float64, copy=True)


def _catalog(engine):
    getter = getattr(engine, "get_property_catalog", None)
    rows = getter() if callable(getter) else engine.query_property_catalog("")
    if isinstance(rows, str):
        rows = rows.splitlines()
    result = {}
    for row in rows:
        match = _CATALOG_ENTRY.fullmatch(str(row).strip())
        if match and "R" in match.group(2):
            result[match.group(1)] = match.group(2)
    return result


@dataclass(frozen=True)
class ObservationBlocks:
    execution: bool = True
    preview: bool = True
    qp_history: bool = True
    surfaces: bool = True
    cg_load: bool = True

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not bool:
                raise ValueError(name + " must be a boolean")


class C45PolicyObservationEnv:
    """Composition facade implementing this repository's four-return API.

    ``base_env`` is an already constructed, single-aircraft C41 pull/dive env.
    Publication is opt-in: ``plan_published_at_s=0.0`` means its entire pure
    reference has been legally published at episode reset.  ``None`` means no
    plan is published.  Later publication is deliberately unsupported here.
    Disabled blocks retain all 83 dimensions and are exactly zero.

    Repeated ``get_obs`` calls return copies of the last emitted observation;
    they do not read native state, re-evaluate a reference, or advance anything.
    """

    def __init__(self, base_env, *, execution=True, preview=True, qp_history=True,
                 surfaces=True, cg_load=True, plan_published_at_s=None):
        self.base_env = base_env
        self.blocks = ObservationBlocks(execution, preview, qp_history, surfaces, cg_load)
        if plan_published_at_s is not None and _finite(plan_published_at_s, "publication time") != 0.0:
            raise ValueError("C45 only supports publication at episode reset, t=0")
        self.plan_published_at_s = None if plan_published_at_s is None else 0.0
        self._validate_base()
        self._space = spaces.Box(low=-10.0, high=10.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self._last_observation = None
        self._plan = None
        self._plan_source = None
        self._engine = None
        self._permissions = {}

    def __getattr__(self, name):
        base = self.__dict__.get("base_env")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    @property
    def observation_space(self):
        return self._space

    @property
    def action_space(self):
        return self.base_env.action_space

    def _validate_base(self):
        base = self.base_env
        if (getattr(base.config, "task", None) != "baiyepower_energy_momentum_c41"
                or getattr(base.config, "C41_reference_mode", None) != "pull_dive"
                or base.observation_space.shape != (BASE_DIM,) or base.num_agents != 1
                or len(base.agents) != 1):
            raise ValueError("C45 requires one C41 54D pull/dive aircraft")

    def _time(self):
        step = self.base_env.current_step
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral) or step < 0:
            raise ValueError("decision step must be a nonnegative integer")
        interval = _finite(self.base_env.time_interval, "decision interval")
        if interval <= 0:
            raise ValueError("decision interval must be positive")
        t = int(step) * interval
        if bool(getattr(self.base_env.task, "_owns_commands", False)):
            command_time = _finite(self.base_env.task._command_time_sec, "task command time")
            if abs(command_time - t) > 1e-9:
                raise ValueError("task command time disagrees with decision boundary")
        return t

    def _refresh_permissions(self):
        sim = next(iter(self.base_env.agents.values()))
        engine = sim.jsbsim_exec
        if engine is not self._engine:
            self._permissions = _catalog(engine)
            self._engine = engine
        return engine

    def _execution(self, agent):
        out = np.zeros(13, dtype=np.float64)
        if not self.blocks.execution:
            return out
        if self.blocks.qp_history:
            projector = self.base_env._action_projector
            if projector is None:
                raise ValueError("C45 execution block requires an initialized QP")
            state = projector.get_observation_state(agent)
            if state is None or "prev2" not in state:
                raise ValueError("C45 requires real QP prev2 history")
            physical = _vector(state["prev2"], 4, "QP prev2")
            out[:4] = _vector(self.base_env.task.action_to_policy_coordinates(physical), 4,
                              "policy-coordinate QP prev2")
            out[10] = 1.0
        if self.blocks.surfaces or self.blocks.cg_load:
            engine = self._refresh_permissions()
            if self.blocks.surfaces:
                missing = set(SURFACE_PROPERTIES) - set(self._permissions)
                if missing:
                    raise ValueError("missing/unreadable surface nodes: " + ", ".join(sorted(missing)))
                out[4:9] = [_finite(engine.get_property_value(path), path)
                            for path in SURFACE_PROPERTIES]
                out[11] = 1.0
            if self.blocks.cg_load and CG_PROPERTY in self._permissions:
                try:
                    measured = engine.get_property_value(CG_PROPERTY)
                except (KeyError, RuntimeError, ValueError):
                    measured = None
                if measured is not None:
                    out[9] = _finite(measured, CG_PROPERTY) / 5.0
                    out[12] = 1.0
        return out

    def _preview(self, t):
        out = np.zeros(16, dtype=np.float64)
        if not self.blocks.preview:
            return out
        task = self.base_env.task
        permitted = (self._plan is not None and self.plan_published_at_s == 0.0
                     and bool(getattr(task, "_owns_commands", False))
                     and getattr(task, "_profile", None) is self._plan_source
                     and self.base_env.config.C41_reference_mode == "pull_dive")
        if not permitted:
            return out
        if t < 0.0 or t > self._plan.total_sec:
            raise ValueError("current reference time outside the published plan")
        current = self._plan.at(t)
        direction = _vector(current["direction"], 3, "current reference direction")
        target = _vector(current["target"], 3, "current reference target")
        out[0] = _finite(current["phase_remaining_sec"], "phase remaining time") / 10.0
        if out[0] < 0:
            raise ValueError("negative phase remaining time")
        out[1] = 1.0
        for point_index, offset in enumerate(PREVIEW_OFFSETS_SEC):
            # Decimal-second arithmetic prevents 259*.2 + .2 becoming
            # 52.00000000000001 and falsely invalidating the final legal point.
            # This constructs the scheduled query; it never clamps to the end.
            future_t = float(Fraction(str(self.base_env.time_interval)) * int(self.base_env.current_step)
                             + Fraction(str(offset)))
            if future_t > self._plan.total_sec:
                continue  # Never clamp a future query to the terminal reference.
            future = self._plan.at(future_t)
            point = np.zeros(7, dtype=np.float64)
            point[:3] = _vector(future["direction"], 3, "future reference direction") - direction
            future_target = _vector(future["target"], 3, "future reference target")
            point[3] = (future_target[1] - target[1]) / 100.0
            point[4] = (future_target[2] - target[2]) / 10.0
            point[5] = _finite(future["gamma_rate_radps"], "future gamma rate") / GAMMA_RATE_SCALE_RADPS
            point[6] = 1.0
            start = 2 + 7 * point_index
            out[start:start + 7] = point
        return out

    def _augment(self, base_observation):
        prefix = np.asarray(base_observation)
        if prefix.shape != (1, BASE_DIM) or prefix.dtype != np.float32 or not np.isfinite(prefix).all():
            raise ValueError("C45 prefix must be finite float32 [1,54]")
        t = self._time()
        agent = next(iter(self.base_env.agents))
        extra = np.concatenate((self._execution(agent), self._preview(t)))
        if extra.shape != (29,) or not np.isfinite(extra).all():
            raise ValueError("C45 appended observation is not finite/schema-aligned")
        extra = np.clip(extra, -10.0, 10.0).astype(np.float32)
        result = np.concatenate((prefix.copy(), extra[None, :]), axis=1)
        self._last_observation = result.copy()
        return result

    def reset(self, *args, **kwargs):
        self._last_observation = None
        self._plan = self._plan_source = None
        self._engine = None
        self._permissions = {}
        prefix = self.base_env.reset(*args, **kwargs)
        self._validate_base()
        task = self.base_env.task
        if (self.plan_published_at_s == 0.0 and bool(getattr(task, "_owns_commands", False))
                and type(getattr(task, "_profile", None)) is PullDiveReference):
            self._plan_source = task._profile
            self._plan = copy.deepcopy(task._profile)
        return self._augment(prefix)

    def step(self, action):
        if self._last_observation is None:
            raise RuntimeError("reset C45 before stepping")
        previous = self._last_observation.copy()
        prefix, reward, done, info = self.base_env.step(action)
        numerical = isinstance(info, dict) and bool(info.get("baiyepower_v2_flag_numerical", False))
        if numerical:
            if not np.asarray(done).all():
                raise ValueError("numerical failure must be terminal")
            observation = previous  # Do not read the failed native/post-reference state.
            self._last_observation = observation.copy()
        else:
            observation = self._augment(prefix)
        if isinstance(info, dict) and "terminal_observation" in info:
            terminal = np.asarray(info["terminal_observation"])
            if terminal.shape != (1, BASE_DIM) or terminal.dtype != np.float32 or not np.isfinite(terminal).all():
                raise ValueError("unexpected internal terminal_observation schema")
            info = dict(info)
            info["terminal_observation"] = (previous.copy() if numerical else
                np.concatenate((terminal.copy(), observation[:, BASE_DIM:].copy()), axis=1))
        return observation, reward, done, info

    def get_obs(self):
        if self._last_observation is None:
            raise RuntimeError("reset C45 before reading its observation")
        return self._last_observation.copy()

    def metadata(self):
        return dict(schema=SCHEMA, base_schema="c41-spherical-append-v1", base_dim=BASE_DIM,
                    observation_dim=OBSERVATION_DIM, requires_fresh_actor_and_critic=True,
                    blocks=dict(vars(self.blocks)), block_ranges=dict(base=[0, 54], execution=[54, 67], preview=[67, 83]),
                    appended_feature_names=list(EXECUTION_NAMES + PREVIEW_NAMES),
                    appended_feature_indices={name: BASE_DIM + index for index, name in
                                              enumerate(EXECUTION_NAMES + PREVIEW_NAMES)},
                    preview_offsets_sec=list(PREVIEW_OFFSETS_SEC),
                    scales=dict(phase_remaining_sec=10.0, height_delta_m=100.0, speed_delta_mps=10.0,
                                gamma_rate_radps=GAMMA_RATE_SCALE_RADPS, cg_nz_g=5.0),
                    surface_properties=list(SURFACE_PROPERTIES), cg_property=CG_PROPERTY,
                    cg_definition="native support-positive CG body-z Nz; not pilot load",
                    cg_native_force_lag_sim_substeps=1,
                    cg_use="observed native channel; not an ideal load target or stability guarantee",
                    qp_source="get_observation_state()['prev2'] mapped via task.action_to_policy_coordinates",
                    publication_time_sec=self.plan_published_at_s,
                    plan_copied_at_reset=self._plan is not None,
                    property_permissions={path: self._permissions.get(path) for path in SURFACE_PROPERTIES + (CG_PROPERTY,)},
                    new_features_clip=[-10.0, 10.0], old_prefix_preserved=True,
                    control_chain_modified=False, time_source="current_step * time_interval",
                    preview_out_of_horizon="seven zero values, point_valid=0; no terminal clamp")

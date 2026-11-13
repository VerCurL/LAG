"""C46 task-specific, lossy 83D -> 40D observation selection.

The candidate is isolated from the frozen C45/C41 implementations.  Its scope
is the zero-wind, upright C41 pull/dive task with fixed reference course and
speed.  It is not a full-attitude schema, a lossless transform, or a proof that
the closed-loop observation is Markov.  No control, reward, simulation, or
training operation is performed here.
"""

import math
from numbers import Real

import numpy as np
from gymnasium import spaces

from ...scripts.experiments.c45_policy_observation import (
    C45PolicyObservationEnv, SCHEMA as SOURCE_SCHEMA,
)


SCHEMA = "c46-c41-compact-observation-v1"
SOURCE_DIM, OBSERVATION_DIM = 83, 40
FEATURE_NAMES = (
    "height_error_over_100m", "ground_speed_error_over_10mps",
    "ground_course_error_over_pi", "gamma_error_over_10deg",
    "actual_gamma_over_90deg", "altitude_over_10000m",
    "ground_speed_over_340mps", "calibrated_airspeed_over_340mps",
    "roll_sin", "roll_cos", "pitch_sin", "pitch_cos",
    "p_over_1radps", "q_over_1radps", "r_over_1radps", "alpha_over_15deg",
    "direction_valid", "course_valid",
    "qp_prev_aileron", "qp_prev_elevator", "qp_prev_rudder", "qp_prev_throttle",
    "qp_prev2_aileron", "qp_prev2_elevator", "qp_prev2_rudder", "qp_prev2_throttle",
    "left_aileron_position", "right_aileron_position", "elevator_position",
    "rudder_position", "throttle_position", "cg_nz_over_5g",
    "qp_history_valid", "surface_valid", "cg_valid",
    "longitudinal_direction_rate_error_projection_over_3degps",
    "preview_0_2s_gamma_rate_over_10degps", "preview_0_2s_valid",
    "preview_0_6s_gamma_rate_over_10degps", "preview_0_6s_valid",
)
DIRECT_INDICES = (0, 1, 2, 3, 12, 13, 14, 16, 17, 32, 33, 34, 37, 39)
CONTINUOUS_INDICES = tuple(index for index in range(OBSERVATION_DIM)
                           if index not in DIRECT_INDICES)
FLAG_SOURCE_INDICES = (46, 47, 64, 65, 66, 68, 75, 82)
DISCARDED_INFORMATION = (
    "absolute ground course and the individual full NED direction vectors",
    "individual body u/v/w components; only their ground-speed norm is kept",
    "complete actual/reference filtered direction-rate vectors; one signed projection is kept",
    "phase one-hot, episode remaining fraction and phase remaining seconds",
    "explicit total-energy and energy-balance errors",
    "legacy FCS command slots and repeat-prev QP duplicate slots",
    "current target-rate slots; this schema is limited to fixed-course/speed pull/dive",
    "preview direction/height/speed deltas and the separate plan-publication flag",
)


def _positive_scale(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + " must be a finite positive real")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(name + " must be a finite positive real")
    return result


def _require_zero(values, mask, name):
    if np.any(values[mask] != 0.0):
        raise ValueError(name + " requires an exact zero placeholder when invalid")


def compact_observation(obs83, speed_error_scale_mps=30.0, rate_scale_deg_s=3.0):
    """Project finite float32 [...,83] into copied float32 [...,40].

    ``speed_error_scale_mps`` and ``rate_scale_deg_s`` describe the SOURCE
    normalization.  Compact output scales are fixed by this versioned schema.
    Invalid source blocks must have their documented zero placeholders.  Bad
    values are rejected even when a validity flag is false; missing values are
    not silently repaired.  No additional clipping is applied.
    """
    speed_scale = _positive_scale(speed_error_scale_mps, "source speed-error scale")
    rate_scale = _positive_scale(rate_scale_deg_s, "source direction-rate scale")
    source = np.asarray(obs83)
    if source.ndim < 1 or source.shape[-1] != SOURCE_DIM:
        raise ValueError("C46 requires a source observation with last dimension 83")
    if source.dtype != np.float32:
        raise ValueError("C46 source observation must have float32 dtype")
    if not np.isfinite(source).all():
        raise ValueError("C46 source observation must be finite")
    shape = source.shape[:-1]
    x = source.reshape(-1, SOURCE_DIM).astype(np.float64)
    flags = x[:, FLAG_SOURCE_INDICES]
    if np.any((flags != 0.0) & (flags != 1.0)):
        raise ValueError("C46 source validity flags must be exactly zero or one")
    direction_valid = x[:, 46] == 1.0
    course_valid = x[:, 47] == 1.0
    if np.any(course_valid & ~direction_valid):
        raise ValueError("a valid ground course requires a valid direction")
    actual, reference = x[:, 32:35], x[:, 35:38]
    _require_zero(actual, ~direction_valid, "actual direction")
    actual_norm = np.sqrt(np.sum(actual * actual, axis=1))
    reference_norm = np.sqrt(np.sum(reference * reference, axis=1))
    if np.any(np.abs(actual_norm[direction_valid] - 1.0) > 1e-6):
        raise ValueError("valid actual direction must be a unit NED vector")
    if np.any(np.abs(reference_norm - 1.0) > 1e-6):
        raise ValueError("reference direction must be a unit NED vector")
    _require_zero(x[:, 54:58], x[:, 64] == 0.0, "unavailable real QP prev2")
    _require_zero(x[:, 58:63], x[:, 65] == 0.0, "unavailable surface positions")
    _require_zero(x[:, 63:64], x[:, 66] == 0.0, "unavailable CG load")
    _require_zero(x[:, 67:83], x[:, 68] == 0.0, "unpublished reference preview")
    for start, valid_index in ((69, 75), (76, 82)):
        _require_zero(x[:, start:start + 7], x[:, valid_index] == 0.0,
                      "unavailable future reference point")
        if np.any((x[:, valid_index] == 1.0) & (x[:, 68] == 0.0)):
            raise ValueError("a valid future point requires a published plan")

    actual_horizontal = np.hypot(actual[:, 0], actual[:, 1])
    reference_horizontal = np.hypot(reference[:, 0], reference[:, 1])
    if np.any(course_valid & (actual_horizontal <= 1e-12)):
        raise ValueError("a valid ground course needs a nonzero horizontal direction")
    if np.any(course_valid & (reference_horizontal <= 1e-12)):
        raise ValueError("a valid course comparison needs a nonvertical reference")
    gamma = np.arctan2(-actual[:, 2], actual_horizontal)
    gamma_ref = np.arctan2(-reference[:, 2], reference_horizontal)
    course = np.arctan2(actual[:, 1], actual[:, 0])
    course_ref = np.arctan2(reference[:, 1], reference[:, 0])
    course_error = (course_ref - course + math.pi) % (2.0 * math.pi) - math.pi
    tangent_gamma = np.column_stack((
        -np.sin(gamma_ref) * np.cos(course_ref),
        -np.sin(gamma_ref) * np.sin(course_ref), -np.cos(gamma_ref),
    ))
    rate_error = np.sum((x[:, 38:41] - x[:, 41:44]) * tangent_gamma, axis=1)
    rate_error *= rate_scale / 3.0

    out = np.zeros((len(x), OBSERVATION_DIM), dtype=np.float64)
    out[:, 0] = x[:, 0] * 10.0
    out[:, 1] = x[:, 2] * (speed_scale / 10.0)
    out[:, 2] = np.where(course_valid, course_error / math.pi, 0.0)
    out[:, 3] = np.where(direction_valid, (gamma_ref - gamma) / math.radians(10.0), 0.0)
    out[:, 4] = np.where(direction_valid, gamma / (math.pi / 2.0), 0.0)
    out[:, 5] = x[:, 3] * 0.5
    out[:, 6] = np.sqrt(np.sum(x[:, 8:11] ** 2, axis=1))
    out[:, 7] = x[:, 11]
    out[:, 8:12] = x[:, 4:8]
    out[:, 12:15] = x[:, 12:15] * (2.0 * math.pi)
    out[:, 15] = x[:, 15]
    out[:, 16:18] = x[:, 46:48]
    out[:, 18:22] = x[:, 21:25]
    out[:, 22:26] = x[:, 54:58]
    out[:, 26:35] = x[:, 58:67]
    out[:, 35] = np.where(direction_valid, rate_error, 0.0)
    out[:, 36:40] = x[:, (74, 75, 81, 82)]
    if not np.isfinite(out).all() or np.any(np.abs(out) > np.finfo(np.float32).max):
        raise ValueError("C46 projection is not representable as finite float32")
    return out.astype(np.float32).reshape(shape + (OBSERVATION_DIM,))


class C46CompactObservationEnv:
    """Composition wrapper around the frozen C45 facade's four-return API.

    The last raw 83D observation and raw terminal observation are copied into
    diagnostic caches.  Callers can record them with the source getters; this
    module does not write files or add observations to the reward/info path.
    Native failure recovery remains the responsibility of the inner facade.
    """

    def __init__(self, base_env):
        if not isinstance(base_env, C45PolicyObservationEnv):
            raise ValueError("C46 requires a C45PolicyObservationEnv source facade")
        self.base_env = base_env
        self.speed_error_scale_mps, self.rate_scale_deg_s = self._source_scales()
        limit = np.finfo(np.float32).max
        self._space = spaces.Box(-limit, limit, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self._last_observation = None
        self._last_source_observation = None
        self._last_source_terminal_observation = None

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

    def _source_scales(self):
        base = self.base_env
        if base.observation_space.shape != (SOURCE_DIM,):
            raise ValueError("C46 source facade must expose 83 dimensions")
        task = base.task
        try:
            speed = task.speed_error_scale_mps
            rate = task.kernel_config.rate_scale_deg_s
        except AttributeError as error:
            raise ValueError("C46 source task must expose its actual normalization scales") from error
        return (_positive_scale(speed, "task speed-error scale"),
                _positive_scale(rate, "task direction-rate scale"))

    def _check_source_scales(self):
        if self._source_scales() != (self.speed_error_scale_mps, self.rate_scale_deg_s):
            raise ValueError("C46 source normalization scales changed after construction")

    def _project(self, source):
        return compact_observation(source, self.speed_error_scale_mps, self.rate_scale_deg_s)

    def reset(self, *args, **kwargs):
        self._last_observation = None
        self._last_source_observation = None
        self._last_source_terminal_observation = None
        self._check_source_scales()
        source = self.base_env.reset(*args, **kwargs)
        self._check_source_scales()
        observation = self._project(source)
        self._last_source_observation = np.array(source, copy=True)
        self._last_observation = observation.copy()
        return observation

    def step(self, action):
        if self._last_observation is None:
            raise RuntimeError("reset C46 before stepping")
        self._check_source_scales()
        source, reward, done, info = self.base_env.step(action)
        self._check_source_scales()
        observation = self._project(source)
        source_terminal = None
        if isinstance(info, dict) and "terminal_observation" in info:
            source_terminal = np.asarray(info["terminal_observation"])
            if source_terminal.shape != np.asarray(source).shape:
                raise ValueError("C46 terminal observation must retain the full source schema")
            terminal = self._project(source_terminal)
            info = dict(info)
            info["terminal_observation"] = terminal
        self._last_source_observation = np.array(source, copy=True)
        self._last_source_terminal_observation = (None if source_terminal is None
                                                  else source_terminal.copy())
        self._last_observation = observation.copy()
        return observation, reward, done, info

    def get_obs(self):
        if self._last_observation is None:
            raise RuntimeError("reset C46 before reading its observation")
        return self._last_observation.copy()

    def get_source_observation(self):
        if self._last_source_observation is None:
            raise RuntimeError("reset C46 before reading its source observation")
        return self._last_source_observation.copy()

    def get_source_terminal_observation(self):
        if self._last_observation is None:
            raise RuntimeError("reset C46 before reading its terminal diagnostic")
        return (None if self._last_source_terminal_observation is None
                else self._last_source_terminal_observation.copy())

    def metadata(self):
        self._check_source_scales()
        return dict(
            schema=SCHEMA, source_schema=SOURCE_SCHEMA, source_dim=SOURCE_DIM,
            observation_dim=OBSERVATION_DIM, feature_names=list(FEATURE_NAMES),
            feature_indices={name: index for index, name in enumerate(FEATURE_NAMES)},
            direct_indices=list(DIRECT_INDICES), continuous_indices=list(CONTINUOUS_INDICES),
            source_scales=dict(speed_error_mps=self.speed_error_scale_mps,
                               direction_rate_deg_s=self.rate_scale_deg_s),
            output_scales=dict(height_error_m=100.0, ground_speed_error_mps=10.0,
                               course_error_rad=math.pi, gamma_error_deg=10.0,
                               actual_gamma_deg=90.0, altitude_m=10000.0,
                               speed_mps=340.0, angular_rate_radps=1.0, alpha_deg=15.0,
                               cg_nz_g=5.0, direction_rate_error_projection_deg_s=3.0,
                               future_gamma_rate_deg_s=10.0),
            direction_rate_projection="(actual_filtered_direction_rate - reference_filtered_direction_rate) dot reference_gamma_tangent; not exact gamma_dot",
            qp_history_valid_definition="C45 appended real prev2 validity; original QP prev remains available when the appended history block is disabled",
            speed_definitions=dict(ground_speed="norm of body-coordinate inertial u/v/w; not true airspeed",
                                   calibrated_airspeed="native vc; not ground speed or true airspeed"),
            no_additional_clipping=True, source_clipping_preserved=True,
            observation_space="all finite float32 values; normalization bounds are not flight safety gates",
            invalid_values="explicit validity plus exact zero source placeholders; NaN/Inf/malformed flags raise",
            discarded_information=list(DISCARDED_INFORMATION),
            applicability="zero-wind, fixed-reference-course/speed, upright C41 pull/dive only",
            lossless=False, markov_guarantee=False, beta_observed=False,
            requires_fresh_actor_and_critic=True, control_chain_modified=False,
            raw_diagnostics="copied 83D caches available via get_source_observation and get_source_terminal_observation; caller persists files",
            scales_identity="source scale changes require new metadata/schema experiment identity; runtime changes raise",
            source_metadata=self.base_env.metadata(),
        )

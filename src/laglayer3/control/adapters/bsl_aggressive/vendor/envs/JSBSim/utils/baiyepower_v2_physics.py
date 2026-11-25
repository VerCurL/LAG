"""Pure SI-unit physics and cost functions for the isolated BaiyePower v2 task.

This module deliberately has no dependency on JSBSim, Gym, NumPy, task
objects, or configuration loaders.  It is therefore safe to use in reward
code, offline audits, and lightweight unit tests without starting a simulator.

The v2 baseline is explicitly a no-wind model:

* scalar speed is the magnitude of the JSBSim body-axis ``u/v/w`` vector;
* energy errors use target-minus-current signs;
* target speed must be strictly positive (the current task generator uses
  target speeds of at least 120 m/s);
* every public calculation rejects non-finite inputs and results.
"""

import math
from dataclasses import dataclass


STANDARD_GRAVITY_MPS2 = 9.80665
DEFAULT_HEIGHT_COST_SCALE_M = 90.0
DEFAULT_SPEED_COST_SCALE_MPS = 20.0
DEFAULT_YAW_GAUSSIAN_SCALE_DEG = 20.0
DEFAULT_YAW_NARROW_WEIGHT = 0.55
DEFAULT_COURSE_HALF_COST_ANGLE_DEG = 15.0
DEFAULT_COURSE_MIN_SPEED_MPS = 40.0
DEFAULT_LOW_ALTITUDE_WARNING_M = 4000.0
DEFAULT_LOW_ALTITUDE_TERMINAL_M = 2500.0
DEFAULT_SINK_SPEED_SCALE_MPS = 20.0
DEFAULT_SINK_ETA = 0.5
DEFAULT_UPRIGHT_FREE_TILT_DEG = 90.0


@dataclass(frozen=True)
class BaiyePowerV2EnergyTerms:
    """Auditable energy decomposition and its three v2 costs.

    Energy and kinetic quantities are specific-energy heights in metres.
    Positive ``e_*`` values mean that the corresponding quantity is below its
    target.  ``C_T`` is symmetric, whereas ``C_h`` and ``C_k`` are one-sided
    shortage costs.
    """

    altitude_m: float
    target_altitude_m: float
    speed_mps: float
    target_speed_mps: float
    delta_speed_mps: float
    kinetic_height_m: float
    target_kinetic_height_m: float
    energy_height_m: float
    target_energy_height_m: float
    e_h_m: float
    e_k_m: float
    e_T_m: float
    e_B_m: float
    S_h_m: float
    S_k_m: float
    S_T_m: float
    z_T: float
    z_h_shortage: float
    z_k_shortage: float
    C_T: float
    C_h: float
    C_k: float


@dataclass(frozen=True)
class BaiyePowerV2YawTerms:
    """Wrapped yaw error and the narrow-plus-wide direction kernel."""

    current_yaw_rad: float
    target_yaw_rad: float
    error_rad: float
    gaussian_scale_rad: float
    narrow_weight: float
    narrow_component: float
    wide_component: float
    kernel: float


@dataclass(frozen=True)
class BaiyePowerV2CourseTerms:
    """Horizontal course error, raw periodic cost, and reliability weighting."""

    north_velocity_mps: float
    east_velocity_mps: float
    horizontal_speed_mps: float
    course_rad: float
    target_course_rad: float
    error_rad: float
    half_cost_angle_rad: float
    raw_cost: float
    reliability_mode: str
    reliability: float
    cost: float
    valid: bool


@dataclass(frozen=True)
class BaiyePowerV2LowAltitudeTerms:
    """Smooth low-altitude warning diagnostics.

    ``altitude_rate_mps`` is positive while climbing.  Only a negative rate
    creates a positive ``sink_speed_mps``.
    """

    altitude_m: float
    altitude_rate_mps: float
    warning_altitude_m: float
    terminal_altitude_m: float
    normalized_depth: float
    altitude_barrier: float
    sink_speed_mps: float
    sink_factor: float
    sink_eta: float
    cost: float


@dataclass(frozen=True)
class BaiyePowerUprightTerms:
    """Body-up orientation, inversion depth, and yaw-reliability gate.

    ``upright_cos`` is the dot product between body down and local down for
    the heading-task Euler convention:

    ``upright_cos = cos(roll) * cos(pitch)``.

    It is +1 while upright, 0 while side-on/vertical, and -1 while fully
    inverted.  U1 leaves the complete upright hemisphere free.  Once the
    body-up axis enters the inverted hemisphere, ``cost`` grows quadratically
    with angular depth while ``yaw_gate`` smoothly removes the positive yaw
    reward.  U1 uses the gate only; the cost is retained as a diagnostic and
    a possible later U2 ablation.
    """

    roll_rad: float
    pitch_rad: float
    upright_cos: float
    tilt_angle_deg: float
    free_tilt_deg: float
    free_cos: float
    normalized_depth: float
    cost: float
    yaw_gate: float
    active: bool
    inverted: bool
    deep_inverted: bool


def _finite_float(name, value):
    """Convert one scalar to float and reject booleans/non-finite values."""

    if isinstance(value, bool):
        raise TypeError("{} must be a real scalar, not bool".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("{} must be a real scalar".format(name)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _positive_float(name, value):
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError("{} must be strictly positive".format(name))
    return result


def _finite_result(name, value):
    if not math.isfinite(value):
        raise ValueError("{} produced a non-finite result".format(name))
    return float(value)


def _clip01(value):
    return min(max(value, 0.0), 1.0)


def smoothstep01(value):
    """Return cubic smoothstep after clipping a finite scalar to ``[0, 1]``."""

    x = _clip01(_finite_float("smoothstep value", value))
    return x * x * (3.0 - 2.0 * x)


def compute_upright_envelope_terms(
    roll_rad,
    pitch_rad,
    free_tilt_deg=DEFAULT_UPRIGHT_FREE_TILT_DEG,
):
    """Compute the isolated U1 upright-envelope state and yaw gate.

    The free-tilt angle must lie in ``(0, 90]`` degrees.  For the frozen U1
    value of 90 degrees, every attitude in the upright hemisphere is free.
    Angular inversion depth and its diagnostic cost are:

    ``alpha = acos(upright_cos)``

    ``x = clip((alpha - free_tilt) / (180 - free_tilt), 0, 1)``

    ``cost = x**2``.

    The positive yaw-reward reliability remains one through side-on flight.
    In the inverted hemisphere, with ``y = 1 + upright_cos``, it is
    ``smoothstep(y)``.  This preserves slight excursions but removes the
    incentive to earn a full yaw reward while deeply inverted.
    """

    roll = _finite_float("roll_rad", roll_rad)
    pitch = _finite_float("pitch_rad", pitch_rad)
    free_tilt = _finite_float("free_tilt_deg", free_tilt_deg)
    if not 0.0 < free_tilt <= 90.0:
        raise ValueError("free_tilt_deg must lie in (0, 90]")

    free_cos = _finite_result(
        "upright free cosine",
        math.cos(math.radians(free_tilt)),
    )
    upright_cos = _finite_result(
        "upright cosine",
        math.cos(roll) * math.cos(pitch),
    )
    # Guard harmless round-off outside the analytic cosine range.
    upright_cos = min(max(upright_cos, -1.0), 1.0)
    tilt_angle_deg = _finite_result(
        "upright tilt angle",
        math.degrees(math.acos(upright_cos)),
    )
    normalized_depth = _clip01(
        _finite_result(
            "upright normalized depth",
            (tilt_angle_deg - free_tilt) / (180.0 - free_tilt),
        )
    )
    cost = _finite_result(
        "upright diagnostic cost",
        normalized_depth * normalized_depth,
    )
    yaw_gate = (
        1.0
        if upright_cos >= 0.0
        else smoothstep01(1.0 + upright_cos)
    )

    return BaiyePowerUprightTerms(
        roll_rad=roll,
        pitch_rad=pitch,
        upright_cos=upright_cos,
        tilt_angle_deg=tilt_angle_deg,
        free_tilt_deg=free_tilt,
        free_cos=free_cos,
        normalized_depth=normalized_depth,
        cost=cost,
        yaw_gate=yaw_gate,
        active=(normalized_depth > 0.0),
        inverted=(upright_cos < 0.0),
        deep_inverted=(upright_cos < -0.5),
    )


def speed_magnitude_mps(u_mps, v_mps, w_mps):
    """Return the stable Euclidean magnitude of finite body-axis velocities."""

    u = _finite_float("u_mps", u_mps)
    v = _finite_float("v_mps", v_mps)
    w = _finite_float("w_mps", w_mps)
    return _finite_result("speed magnitude", math.hypot(u, v, w))


def specific_energy_height_m(
    altitude_m,
    speed_mps,
    gravity_mps2=STANDARD_GRAVITY_MPS2,
):
    """Return ``h + V**2 / (2 g)`` in metres."""

    altitude = _finite_float("altitude_m", altitude_m)
    speed = _finite_float("speed_mps", speed_mps)
    gravity = _positive_float("gravity_mps2", gravity_mps2)
    if speed < 0.0:
        raise ValueError("speed_mps cannot be negative")
    kinetic_height = _finite_result(
        "kinetic energy height",
        speed * speed / (2.0 * gravity),
    )
    return _finite_result(
        "specific energy height",
        altitude + kinetic_height,
    )


def kinetic_height_scale_m(
    target_speed_mps,
    speed_cost_scale_mps=DEFAULT_SPEED_COST_SCALE_MPS,
    gravity_mps2=STANDARD_GRAVITY_MPS2,
):
    """Map one downward speed tolerance to an equivalent kinetic-height scale.

    The denominator depends only on target speed:

    ``S_k = (V*² - max(V* - S_V, 0)²) / (2g)``.
    """

    target_speed = _positive_float("target_speed_mps", target_speed_mps)
    speed_scale = _positive_float(
        "speed_cost_scale_mps",
        speed_cost_scale_mps,
    )
    gravity = _positive_float("gravity_mps2", gravity_mps2)
    lower_speed = max(target_speed - speed_scale, 0.0)
    scale = (
        target_speed * target_speed - lower_speed * lower_speed
    ) / (2.0 * gravity)
    scale = _finite_result("kinetic height scale", scale)
    if scale <= 0.0:
        raise ValueError("kinetic height scale must be strictly positive")
    return scale


def rational_cost(value):
    """Return the frozen robust cost ``rho(x) = 2 x² / (1 + |x|)``."""

    x = _finite_float("rational cost input", value)
    magnitude = abs(x)
    # This ordering delays overflow compared with forming x*x directly.
    result = 2.0 * (magnitude / (1.0 + magnitude)) * magnitude
    return _finite_result("rational cost", result)


def compute_energy_cost_terms(
    altitude_m,
    target_altitude_m,
    u_mps,
    v_mps,
    w_mps,
    target_speed_mps,
    height_cost_scale_m=DEFAULT_HEIGHT_COST_SCALE_M,
    speed_cost_scale_mps=DEFAULT_SPEED_COST_SCALE_MPS,
    gravity_mps2=STANDARD_GRAVITY_MPS2,
):
    """Compute the complete BaiyePower v2 energy state and costs in SI units."""

    altitude = _finite_float("altitude_m", altitude_m)
    target_altitude = _finite_float(
        "target_altitude_m",
        target_altitude_m,
    )
    target_speed = _positive_float("target_speed_mps", target_speed_mps)
    height_scale = _positive_float(
        "height_cost_scale_m",
        height_cost_scale_m,
    )
    speed_scale = _positive_float(
        "speed_cost_scale_mps",
        speed_cost_scale_mps,
    )
    gravity = _positive_float("gravity_mps2", gravity_mps2)

    speed = speed_magnitude_mps(u_mps, v_mps, w_mps)
    kinetic_height = _finite_result(
        "current kinetic energy height",
        speed * speed / (2.0 * gravity),
    )
    target_kinetic_height = _finite_result(
        "target kinetic energy height",
        target_speed * target_speed / (2.0 * gravity),
    )
    energy_height = _finite_result(
        "current energy height",
        altitude + kinetic_height,
    )
    target_energy_height = _finite_result(
        "target energy height",
        target_altitude + target_kinetic_height,
    )

    e_h = _finite_result(
        "altitude error",
        target_altitude - altitude,
    )
    e_k = _finite_result(
        "kinetic height error",
        target_kinetic_height - kinetic_height,
    )
    e_T = _finite_result("total energy error", e_h + e_k)
    e_B = _finite_result("energy balance error", e_h - e_k)

    kinetic_scale = kinetic_height_scale_m(
        target_speed,
        speed_cost_scale_mps=speed_scale,
        gravity_mps2=gravity,
    )
    total_scale = _finite_result(
        "total energy scale",
        height_scale + kinetic_scale,
    )
    z_T = _finite_result("normalized total energy error", e_T / total_scale)
    z_h_shortage = _finite_result(
        "normalized altitude shortage",
        max(e_h, 0.0) / height_scale,
    )
    z_k_shortage = _finite_result(
        "normalized kinetic shortage",
        max(e_k, 0.0) / kinetic_scale,
    )

    return BaiyePowerV2EnergyTerms(
        altitude_m=altitude,
        target_altitude_m=target_altitude,
        speed_mps=speed,
        target_speed_mps=target_speed,
        delta_speed_mps=_finite_result(
            "speed error",
            target_speed - speed,
        ),
        kinetic_height_m=kinetic_height,
        target_kinetic_height_m=target_kinetic_height,
        energy_height_m=energy_height,
        target_energy_height_m=target_energy_height,
        e_h_m=e_h,
        e_k_m=e_k,
        e_T_m=e_T,
        e_B_m=e_B,
        S_h_m=height_scale,
        S_k_m=kinetic_scale,
        S_T_m=total_scale,
        z_T=z_T,
        z_h_shortage=z_h_shortage,
        z_k_shortage=z_k_shortage,
        C_T=rational_cost(z_T),
        C_h=rational_cost(z_h_shortage),
        C_k=rational_cost(z_k_shortage),
    )


def wrap_angle_pi(angle_rad):
    """Wrap a finite angle to the periodic interval ``[-pi, pi]``."""

    angle = _finite_float("angle_rad", angle_rad)
    wrapped = math.remainder(angle, 2.0 * math.pi)
    return _finite_result("wrapped angle", wrapped)


def compute_yaw_kernel_terms(
    current_yaw_rad,
    target_yaw_rad,
    gaussian_scale_deg=DEFAULT_YAW_GAUSSIAN_SCALE_DEG,
    narrow_weight=DEFAULT_YAW_NARROW_WEIGHT,
):
    """Compute the frozen narrow-Gaussian plus wide-cosine yaw kernel."""

    current_yaw = _finite_float("current_yaw_rad", current_yaw_rad)
    target_yaw = _finite_float("target_yaw_rad", target_yaw_rad)
    gaussian_scale = math.radians(
        _positive_float("gaussian_scale_deg", gaussian_scale_deg)
    )
    weight = _finite_float("narrow_weight", narrow_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("narrow_weight must lie in [0, 1]")

    error = wrap_angle_pi(target_yaw - current_yaw)
    narrow = _finite_result(
        "yaw narrow component",
        math.exp(-((error / gaussian_scale) ** 2)),
    )
    wide = _finite_result(
        "yaw wide component",
        (1.0 + math.cos(error)) / 2.0,
    )
    kernel = _finite_result(
        "yaw kernel",
        weight * narrow + (1.0 - weight) * wide,
    )
    # Guard harmless floating-point excursions without hiding invalid inputs.
    kernel = _clip01(kernel)

    return BaiyePowerV2YawTerms(
        current_yaw_rad=current_yaw,
        target_yaw_rad=target_yaw,
        error_rad=error,
        gaussian_scale_rad=gaussian_scale,
        narrow_weight=weight,
        narrow_component=narrow,
        wide_component=wide,
        kernel=kernel,
    )


def course_reliability(
    horizontal_speed_mps,
    mode="hard",
    minimum_speed_mps=DEFAULT_COURSE_MIN_SPEED_MPS,
    fade_start_mps=None,
    fade_end_mps=None,
):
    """Return course-direction reliability without a global speed reward gate.

    ``hard`` reproduces the first-pilot rule: reliability is zero below
    ``minimum_speed_mps`` and one at or above it.

    ``smoothstep`` is a local transition with explicit zero/one plateaus.
    ``fade_start_mps`` and ``fade_end_mps`` must be supplied and the latter
    must be greater than the former.  Above the end speed, reliability is
    exactly one, so normal-speed course costs are not continuously attenuated.
    """

    speed = _finite_float("horizontal_speed_mps", horizontal_speed_mps)
    if speed < 0.0:
        raise ValueError("horizontal_speed_mps cannot be negative")
    if not isinstance(mode, str):
        raise TypeError("course reliability mode must be a string")
    normalized_mode = mode.strip().lower()

    if normalized_mode == "hard":
        threshold = _positive_float(
            "minimum_speed_mps",
            minimum_speed_mps,
        )
        return 1.0 if speed >= threshold else 0.0

    if normalized_mode == "smoothstep":
        if fade_start_mps is None or fade_end_mps is None:
            raise ValueError(
                "smoothstep course reliability requires fade_start_mps "
                "and fade_end_mps"
            )
        start = _finite_float("fade_start_mps", fade_start_mps)
        end = _finite_float("fade_end_mps", fade_end_mps)
        if start < 0.0:
            raise ValueError("fade_start_mps cannot be negative")
        if end <= start:
            raise ValueError("fade_end_mps must be greater than fade_start_mps")
        return smoothstep01((speed - start) / (end - start))

    raise ValueError(
        "course reliability mode must be 'hard' or 'smoothstep'"
    )


def compute_course_cost_terms(
    north_velocity_mps,
    east_velocity_mps,
    target_course_rad,
    half_cost_angle_deg=DEFAULT_COURSE_HALF_COST_ANGLE_DEG,
    reliability_mode="hard",
    minimum_speed_mps=DEFAULT_COURSE_MIN_SPEED_MPS,
    fade_start_mps=None,
    fade_end_mps=None,
):
    """Compute periodic horizontal-course cost and low-speed reliability."""

    north_velocity = _finite_float(
        "north_velocity_mps",
        north_velocity_mps,
    )
    east_velocity = _finite_float(
        "east_velocity_mps",
        east_velocity_mps,
    )
    target_course = _finite_float("target_course_rad", target_course_rad)
    half_cost_angle_deg = _positive_float(
        "half_cost_angle_deg",
        half_cost_angle_deg,
    )
    if half_cost_angle_deg >= 180.0:
        raise ValueError("half_cost_angle_deg must be less than 180 degrees")
    half_cost_angle = math.radians(half_cost_angle_deg)

    horizontal_speed = _finite_result(
        "horizontal speed",
        math.hypot(north_velocity, east_velocity),
    )
    course = _finite_result(
        "course angle",
        math.atan2(east_velocity, north_velocity),
    )
    error = wrap_angle_pi(target_course - course)
    numerator = math.sin(error / 2.0) ** 2
    scale_squared = math.sin(half_cost_angle / 2.0) ** 2
    raw_cost = _finite_result(
        "raw course cost",
        numerator / (numerator + scale_squared),
    )
    reliability = course_reliability(
        horizontal_speed,
        mode=reliability_mode,
        minimum_speed_mps=minimum_speed_mps,
        fade_start_mps=fade_start_mps,
        fade_end_mps=fade_end_mps,
    )
    cost = _finite_result(
        "reliability-weighted course cost",
        raw_cost * reliability,
    )

    return BaiyePowerV2CourseTerms(
        north_velocity_mps=north_velocity,
        east_velocity_mps=east_velocity,
        horizontal_speed_mps=horizontal_speed,
        course_rad=course,
        target_course_rad=target_course,
        error_rad=error,
        half_cost_angle_rad=half_cost_angle,
        raw_cost=raw_cost,
        reliability_mode=reliability_mode.strip().lower(),
        reliability=reliability,
        cost=cost,
        valid=(reliability > 0.0),
    )


def compute_low_altitude_warning_terms(
    altitude_m,
    altitude_rate_mps,
    warning_altitude_m=DEFAULT_LOW_ALTITUDE_WARNING_M,
    terminal_altitude_m=DEFAULT_LOW_ALTITUDE_TERMINAL_M,
    sink_speed_scale_mps=DEFAULT_SINK_SPEED_SCALE_MPS,
    sink_eta=DEFAULT_SINK_ETA,
):
    """Compute the bounded smooth low-altitude warning cost."""

    altitude = _finite_float("altitude_m", altitude_m)
    altitude_rate = _finite_float("altitude_rate_mps", altitude_rate_mps)
    warning_altitude = _finite_float(
        "warning_altitude_m",
        warning_altitude_m,
    )
    terminal_altitude = _finite_float(
        "terminal_altitude_m",
        terminal_altitude_m,
    )
    if warning_altitude <= terminal_altitude:
        raise ValueError(
            "warning_altitude_m must be greater than terminal_altitude_m"
        )
    sink_scale = _positive_float(
        "sink_speed_scale_mps",
        sink_speed_scale_mps,
    )
    eta = _finite_float("sink_eta", sink_eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("sink_eta must lie in [0, 1]")

    warning_band = _finite_result(
        "low-altitude warning band",
        warning_altitude - terminal_altitude,
    )
    warning_depth = _finite_result(
        "low-altitude warning depth",
        warning_altitude - altitude,
    )
    normalized_depth = _clip01(
        _finite_result(
            "normalized low-altitude warning depth",
            warning_depth / warning_band,
        )
    )
    altitude_barrier = smoothstep01(normalized_depth)
    sink_speed = max(-altitude_rate, 0.0)
    sink_factor = _finite_result(
        "sink factor",
        sink_speed / (sink_speed + sink_scale),
    )
    cost = _finite_result(
        "low-altitude warning cost",
        altitude_barrier
        * (
            1.0
            + eta
            * (1.0 - altitude_barrier)
            * sink_factor
        ),
    )
    # The analytic range is [0, 1] for eta in [0, 1].
    cost = _clip01(cost)

    return BaiyePowerV2LowAltitudeTerms(
        altitude_m=altitude,
        altitude_rate_mps=altitude_rate,
        warning_altitude_m=warning_altitude,
        terminal_altitude_m=terminal_altitude,
        normalized_depth=normalized_depth,
        altitude_barrier=altitude_barrier,
        sink_speed_mps=sink_speed,
        sink_factor=sink_factor,
        sink_eta=eta,
        cost=cost,
    )

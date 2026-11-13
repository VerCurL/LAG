import numpy as np


TRACKING_WEIGHTS = {
    "heading": 0.45,
    "altitude": 0.30,
    "speed": 0.25,
}

DEFAULT_TRACKING_SCALES = {
    "heading": 20.0,
    "altitude": 150.0,
    "speed": 30.0,
}

DEFAULT_TRACKING_TOLERANCES = {
    "heading": 15.0,
    "altitude": 200.0,
    "speed": 20.0,
}


def finite_values(values):
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def tracking_quality(errors, scale):
    values = finite_values(errors)
    if values.size == 0:
        return float("nan")
    normalized = np.abs(values) / max(float(scale), 1e-9)
    pseudo_huber = np.sqrt(1.0 + normalized * normalized) - 1.0
    return float(np.mean(np.exp(-pseudo_huber)))


def compute_tracking_components(
    heading_errors,
    altitude_errors,
    speed_errors,
    heading_scale_deg=DEFAULT_TRACKING_SCALES["heading"],
    altitude_scale_m=DEFAULT_TRACKING_SCALES["altitude"],
    speed_scale_mps=DEFAULT_TRACKING_SCALES["speed"],
):
    return {
        "heading": tracking_quality(heading_errors, heading_scale_deg),
        "altitude": tracking_quality(altitude_errors, altitude_scale_m),
        "speed": tracking_quality(speed_errors, speed_scale_mps),
    }


def compute_tracking_score(components):
    values = {
        name: components.get(name, float("nan"))
        for name in TRACKING_WEIGHTS
    }
    if not all(np.isfinite(value) for value in values.values()):
        return float("nan")
    return float(sum(TRACKING_WEIGHTS[name] * values[name] for name in TRACKING_WEIGHTS))


def compute_tracking_tolerance_mask(
    heading_errors,
    altitude_errors,
    speed_errors,
    heading_tolerance_deg=DEFAULT_TRACKING_TOLERANCES["heading"],
    altitude_tolerance_m=DEFAULT_TRACKING_TOLERANCES["altitude"],
    speed_tolerance_mps=DEFAULT_TRACKING_TOLERANCES["speed"],
):
    heading = np.asarray(heading_errors, dtype=np.float64)
    altitude = np.asarray(altitude_errors, dtype=np.float64)
    speed = np.asarray(speed_errors, dtype=np.float64)
    if not (heading.shape == altitude.shape == speed.shape):
        raise ValueError("Tracking error arrays must have matching shapes")
    valid = np.isfinite(heading) & np.isfinite(altitude) & np.isfinite(speed)
    return (
        valid
        & (np.abs(heading) <= max(float(heading_tolerance_deg), 0.0))
        & (np.abs(altitude) <= max(float(altitude_tolerance_m), 0.0))
        & (np.abs(speed) <= max(float(speed_tolerance_mps), 0.0))
    )


def compute_error_overshoot_proxy(heading_errors, altitude_errors, speed_errors):
    heading = np.asarray(heading_errors, dtype=np.float64)
    altitude = np.asarray(altitude_errors, dtype=np.float64)
    speed = np.asarray(speed_errors, dtype=np.float64)
    if not (heading.shape == altitude.shape == speed.shape):
        raise ValueError("Tracking error arrays must have matching shapes")
    valid = np.isfinite(heading) & np.isfinite(altitude) & np.isfinite(speed)
    if not np.any(valid):
        return float("nan")
    normalized = (
        np.minimum(np.abs(heading[valid]) / 180.0, 1.0)
        + np.minimum(np.abs(altitude[valid]) / 1000.0, 1.0)
        + np.minimum(np.abs(speed[valid]) / 100.0, 1.0)
    ) / 3.0
    return float(np.mean(normalized))

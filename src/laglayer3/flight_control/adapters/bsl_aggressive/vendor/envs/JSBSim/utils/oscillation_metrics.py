"""Pure NumPy metrics for uniformly sampled control and response signals.

The reversal detector deliberately distinguishes a reversal *event rate* from
an oscillation frequency.  A frequency is reported only when there are enough
alternating events, their half-cycle intervals are sufficiently regular, and
same-sign event pairs provide valid full-cycle intervals.

NaNs split a signal into independent contiguous records.  Differences and
reversals are never formed across such gaps.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


def _as_1d_float(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _validate_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _finite_runs(mask: np.ndarray) -> List[np.ndarray]:
    """Return index arrays for non-empty contiguous True runs."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    return [run for run in np.split(indices, cuts) if run.size]


def _run_duration_sec(run: np.ndarray, dt_sec: float) -> float:
    """Elapsed time spanned by a uniformly sampled contiguous run."""
    return float(max(int(run.size) - 1, 0) * dt_sec)


def _resolve_baseline(samples: np.ndarray, baseline: Optional[ArrayLike]) -> tuple:
    finite = np.isfinite(samples)
    if baseline is None:
        center = float(np.median(samples[finite])) if np.any(finite) else math.nan
        baseline_array = np.full(samples.shape, center, dtype=np.float64)
        return baseline_array, center, "median"

    candidate = np.asarray(baseline, dtype=np.float64)
    if candidate.ndim == 0:
        center = float(candidate)
        if not np.isfinite(center):
            raise ValueError("scalar baseline must be finite")
        return np.full(samples.shape, center, dtype=np.float64), center, "scalar"
    if candidate.ndim != 1 or candidate.shape != samples.shape:
        raise ValueError("array baseline must have the same one-dimensional shape as samples")
    return candidate.copy(), math.nan, "array"


def _start_candidate(sign: int, index: int, residual: float) -> Dict[str, float]:
    return {
        "sign": int(sign),
        "start_index": int(index),
        "count": 1,
        "peak": float(sign * residual),
        "peak_index": int(index),
    }


def _update_candidate(candidate: Dict[str, float], index: int, residual: float) -> None:
    candidate["count"] = int(candidate["count"]) + 1
    magnitude = float(int(candidate["sign"]) * residual)
    if magnitude > float(candidate["peak"]):
        candidate["peak"] = magnitude
        candidate["peak_index"] = int(index)


def _confirmed_lobes(
    residual: np.ndarray,
    finite_runs: List[np.ndarray],
    hysteresis_on: float,
    hysteresis_off: float,
    dwell_samples: int,
) -> List[Dict[str, float]]:
    """Extract confirmed positive/negative lobes with Schmitt hysteresis."""
    lobes: List[Dict[str, float]] = []

    for run_id, indices in enumerate(finite_runs):
        current = None
        candidate = None

        for index in indices:
            index = int(index)
            value = float(residual[index])

            if current is None:
                if candidate is None:
                    if value >= hysteresis_on:
                        candidate = _start_candidate(1, index, value)
                    elif value <= -hysteresis_on:
                        candidate = _start_candidate(-1, index, value)
                else:
                    sign = int(candidate["sign"])
                    if sign * value >= hysteresis_off:
                        _update_candidate(candidate, index, value)
                    elif abs(value) >= hysteresis_on:
                        candidate = _start_candidate(1 if value > 0.0 else -1, index, value)
                    else:
                        candidate = None

                if candidate is not None and int(candidate["count"]) >= dwell_samples:
                    current = {
                        **candidate,
                        "run_id": int(run_id),
                        "confirmed_index": int(index),
                    }
                    candidate = None
                continue

            current_sign = int(current["sign"])
            current_magnitude = current_sign * value
            if current_magnitude > float(current["peak"]):
                current["peak"] = float(current_magnitude)
                current["peak_index"] = int(index)

            opposite_sign = -current_sign
            if candidate is None:
                if opposite_sign * value >= hysteresis_on:
                    candidate = _start_candidate(opposite_sign, index, value)
            else:
                candidate_sign = int(candidate["sign"])
                if candidate_sign * value >= hysteresis_off:
                    _update_candidate(candidate, index, value)
                else:
                    # Re-entering the old lobe or spending a sample in the
                    # neutral band breaks the required consecutive dwell.
                    candidate = None

            if candidate is not None and int(candidate["count"]) >= dwell_samples:
                current["end_index"] = int(candidate["start_index"]) - 1
                lobes.append(current)
                current = {
                    **candidate,
                    "run_id": int(run_id),
                    "confirmed_index": int(index),
                }
                candidate = None

        if current is not None:
            current["end_index"] = int(indices[-1])
            lobes.append(current)

    return lobes


def detect_valid_reversals(
    samples: ArrayLike,
    dt_sec: float,
    *,
    baseline: Optional[ArrayLike] = None,
    hysteresis_on: float = 0.02,
    hysteresis_off: float = 0.01,
    min_amplitude: float = 0.04,
    min_side_amplitude: Optional[float] = None,
    min_dwell_sec: float = 0.0,
    refractory_sec: float = 0.0,
) -> Dict[str, object]:
    """Detect amplitude- and dwell-qualified reversals.

    ``min_amplitude`` is the minimum peak-to-peak excursion of two adjacent
    confirmed lobes.  ``min_side_amplitude`` optionally requires each lobe to
    reach a minimum magnitude relative to the supplied baseline.

    ``min_dwell_sec`` is elapsed time, not a sample count: at spacing ``dt`` a
    positive dwell requires ``1 + ceil(min_dwell_sec / dt)`` samples so that
    the span from the first through the last sample reaches the requested
    duration.  Refractory history is local to each contiguous finite run.
    """
    signal = _as_1d_float(samples, "samples")
    dt_sec = float(dt_sec)
    if not np.isfinite(dt_sec) or dt_sec <= 0.0:
        raise ValueError("dt_sec must be finite and positive")
    hysteresis_on = _validate_nonnegative(hysteresis_on, "hysteresis_on")
    hysteresis_off = _validate_nonnegative(hysteresis_off, "hysteresis_off")
    if hysteresis_off > hysteresis_on:
        raise ValueError("hysteresis_off must not exceed hysteresis_on")
    min_amplitude = _validate_nonnegative(min_amplitude, "min_amplitude")
    min_dwell_sec = _validate_nonnegative(min_dwell_sec, "min_dwell_sec")
    refractory_sec = _validate_nonnegative(refractory_sec, "refractory_sec")
    if min_side_amplitude is None:
        min_side_amplitude = hysteresis_on
    min_side_amplitude = _validate_nonnegative(min_side_amplitude, "min_side_amplitude")

    baseline_array, baseline_center, baseline_kind = _resolve_baseline(signal, baseline)
    residual = signal - baseline_array
    finite = np.isfinite(signal) & np.isfinite(baseline_array)
    runs = _finite_runs(finite)
    invalid_runs = _finite_runs(~finite)
    internal_gap_count = int(
        sum(
            int(run[0]) > 0 and int(run[-1]) < finite.size - 1
            for run in invalid_runs
        )
    )
    dwell_samples = (
        1
        if min_dwell_sec <= 0.0
        else int(math.ceil(min_dwell_sec / dt_sec - 1e-12)) + 1
    )
    refractory_samples = max(0, int(math.ceil(refractory_sec / dt_sec - 1e-12)))
    lobes = _confirmed_lobes(
        residual,
        runs,
        hysteresis_on,
        hysteresis_off,
        dwell_samples,
    )

    events: List[Dict[str, float]] = []
    rejected_amplitude = 0
    rejected_refractory = 0
    last_event_index = None
    last_event_run_id = None
    for old, new in zip(lobes[:-1], lobes[1:]):
        if int(old["run_id"]) != int(new["run_id"]):
            continue
        if int(old["sign"]) * int(new["sign"]) >= 0:
            continue
        old_peak = float(old["peak"])
        new_peak = float(new["peak"])
        peak_to_peak = old_peak + new_peak
        if (
            old_peak < min_side_amplitude
            or new_peak < min_side_amplitude
            or peak_to_peak < min_amplitude
        ):
            rejected_amplitude += 1
            continue

        event_index = int(new["start_index"])
        event_run_id = int(new["run_id"])
        if (
            last_event_index is not None
            and last_event_run_id == event_run_id
            and event_index - last_event_index < refractory_samples
        ):
            rejected_refractory += 1
            continue
        events.append(
            {
                "index": event_index,
                "time_sec": float(event_index * dt_sec),
                "run_id": event_run_id,
                "from_sign": int(old["sign"]),
                "to_sign": int(new["sign"]),
                "old_peak": old_peak,
                "new_peak": new_peak,
                "peak_to_peak_amplitude": float(peak_to_peak),
            }
        )
        last_event_index = event_index
        last_event_run_id = event_run_id

    run_durations = [_run_duration_sec(run, dt_sec) for run in runs]
    valid_duration = float(sum(run_durations))
    longest_valid_duration = float(max(run_durations, default=0.0))
    return {
        "available": bool(valid_duration > 0.0),
        "baseline_kind": baseline_kind,
        "baseline_center": baseline_center,
        "valid_sample_count": int(np.sum(finite)),
        "invalid_sample_count": int(np.sum(~finite)),
        "finite_run_count": int(len(runs)),
        "gap_count": internal_gap_count,
        "valid_duration_sec": valid_duration,
        "longest_contiguous_valid_duration_sec": longest_valid_duration,
        "dwell_samples": int(dwell_samples),
        "refractory_samples": int(refractory_samples),
        "confirmed_lobe_count": int(len(lobes)),
        "event_count": int(len(events)),
        "rejected_amplitude_count": int(rejected_amplitude),
        "rejected_refractory_count": int(rejected_refractory),
        "events": events,
    }


def compute_oscillation_metrics(
    samples: ArrayLike,
    dt_sec: float,
    *,
    baseline: Optional[ArrayLike] = None,
    hysteresis_on: float = 0.02,
    hysteresis_off: float = 0.01,
    min_amplitude: float = 0.04,
    min_side_amplitude: Optional[float] = None,
    min_dwell_sec: float = 0.0,
    refractory_sec: float = 0.0,
    min_periodic_reversals: int = 4,
    max_interval_cv: float = 0.25,
) -> Dict[str, object]:
    """Compute reversal, variation, slew, curvature, and periodicity metrics."""
    signal = _as_1d_float(samples, "samples")
    dt_sec = float(dt_sec)
    if not np.isfinite(dt_sec) or dt_sec <= 0.0:
        raise ValueError("dt_sec must be finite and positive")
    min_periodic_reversals = int(min_periodic_reversals)
    if min_periodic_reversals < 3:
        raise ValueError("min_periodic_reversals must be at least 3")
    max_interval_cv = _validate_nonnegative(max_interval_cv, "max_interval_cv")

    reversal = detect_valid_reversals(
        signal,
        dt_sec,
        baseline=baseline,
        hysteresis_on=hysteresis_on,
        hysteresis_off=hysteresis_off,
        min_amplitude=min_amplitude,
        min_side_amplitude=min_side_amplitude,
        min_dwell_sec=min_dwell_sec,
        refractory_sec=refractory_sec,
    )
    # Use the same validity mask as reversal detection.  In particular, an
    # array-valued baseline may itself contain gaps; derivatives must not
    # bridge those gaps even when the raw signal happens to be finite there.
    baseline_array, _, _ = _resolve_baseline(signal, baseline)
    finite = np.isfinite(signal) & np.isfinite(baseline_array)
    runs = _finite_runs(finite)
    first_differences = []
    second_differences = []
    for run in runs:
        values = signal[run]
        if values.size >= 2:
            first_differences.append(np.diff(values))
        if values.size >= 3:
            second_differences.append(np.diff(values, n=2))
    delta = np.concatenate(first_differences) if first_differences else np.empty(0)
    delta2 = np.concatenate(second_differences) if second_differences else np.empty(0)

    duration = float(reversal["valid_duration_sec"])
    amplitudes = np.asarray(
        [event["peak_to_peak_amplitude"] for event in reversal["events"]],
        dtype=np.float64,
    )
    event_times = np.asarray(
        [event["time_sec"] for event in reversal["events"]],
        dtype=np.float64,
    )
    event_signs = np.asarray(
        [event["to_sign"] for event in reversal["events"]],
        dtype=np.int8,
    )

    event_runs = np.asarray(
        [event["run_id"] for event in reversal["events"]],
        dtype=np.int64,
    )
    eligible_runs = {
        int(run_id)
        for run_id in np.unique(event_runs)
        if int(np.sum(event_runs == run_id)) >= min_periodic_reversals
    }
    periodicity_available = bool(eligible_runs)
    adjacent_same_run = (
        (event_runs[1:] == event_runs[:-1])
        & np.asarray(
            [int(run_id) in eligible_runs for run_id in event_runs[1:]],
            dtype=bool,
        )
    )
    intervals = (
        np.diff(event_times)[adjacent_same_run]
        if periodicity_available
        else np.empty(0)
    )
    interval_median = float(np.median(intervals)) if intervals.size else math.nan
    interval_mad = (
        float(np.median(np.abs(intervals - interval_median)))
        if intervals.size
        else math.nan
    )
    interval_robust_cv = (
        float(1.4826 * interval_mad / interval_median)
        if intervals.size and interval_median > 0.0
        else math.nan
    )
    interval_mean = float(np.mean(intervals)) if intervals.size else math.nan
    # Classical CV is intentionally the primary periodicity guard.  MAD is a
    # useful robust descriptor, but a single long missed/irregular half-cycle
    # can leave MAD exactly zero and must not be hidden.
    interval_cv = (
        float(np.std(intervals) / interval_mean)
        if intervals.size and interval_mean > 0.0
        else math.nan
    )
    interval_max_relative_deviation = (
        float(np.max(np.abs(intervals - interval_median)) / interval_median)
        if intervals.size and interval_median > 0.0
        else math.nan
    )
    interval_p10 = float(np.percentile(intervals, 10.0)) if intervals.size else math.nan
    interval_p90_p10_ratio = (
        float(np.percentile(intervals, 90.0) / interval_p10)
        if intervals.size and interval_p10 > 0.0
        else math.nan
    )
    alternating = bool(
        periodicity_available
        and intervals.size >= min_periodic_reversals - 1
        and np.all(
            event_signs[1:][adjacent_same_run]
            * event_signs[:-1][adjacent_same_run]
            == -1
        )
    )
    periodic_detected = bool(
        periodicity_available
        and alternating
        and np.isfinite(interval_cv)
        and interval_cv <= max_interval_cv
        and np.isfinite(interval_max_relative_deviation)
        and interval_max_relative_deviation <= 2.0 * max_interval_cv
        and np.isfinite(interval_p90_p10_ratio)
        and interval_p90_p10_ratio <= 1.0 + 2.0 * max_interval_cv
    )
    same_run_full_cycle = (
        (event_runs[2:] == event_runs[:-2])
        & (event_signs[2:] == event_signs[:-2])
        & np.asarray(
            [int(run_id) in eligible_runs for run_id in event_runs[2:]],
            dtype=bool,
        )
    )
    full_cycle_intervals = (
        (event_times[2:] - event_times[:-2])[same_run_full_cycle]
        if periodicity_available and event_times.size >= 3
        else np.empty(0)
    )
    full_cycle_interval_median = (
        float(np.median(full_cycle_intervals))
        if full_cycle_intervals.size
        else math.nan
    )
    periodic_detected = bool(
        periodic_detected
        and full_cycle_intervals.size >= max(min_periodic_reversals - 2, 1)
        and np.isfinite(full_cycle_interval_median)
        and full_cycle_interval_median > 0.0
    )
    # Use same-sign event spacing (a full cycle), not twice the median of
    # pooled positive/negative half-cycles.  The latter is biased when the two
    # half-cycles are asymmetric and their finite-window counts differ.
    periodic_frequency = (
        float(1.0 / full_cycle_interval_median)
        if periodic_detected
        else math.nan
    )

    result = dict(reversal)
    result.update(
        {
            "reversal_rate_per_sec": float(len(amplitudes) / duration) if duration > 0.0 else math.nan,
            "reversal_amplitude_median": float(np.median(amplitudes)) if amplitudes.size else math.nan,
            "reversal_amplitude_p90": float(np.percentile(amplitudes, 90.0)) if amplitudes.size else math.nan,
            "reversal_amplitude_burden_per_sec": float(np.sum(amplitudes) / duration) if duration > 0.0 else math.nan,
            "total_variation_per_sec": float(np.sum(np.abs(delta)) / duration) if duration > 0.0 else math.nan,
            "slew_rms_per_sec": float(np.sqrt(np.mean(np.square(delta / dt_sec)))) if delta.size else math.nan,
            "curvature_rms_per_sec2": float(np.sqrt(np.mean(np.square(delta2 / (dt_sec * dt_sec))))) if delta2.size else math.nan,
            "first_difference_count": int(delta.size),
            "second_difference_count": int(delta2.size),
            "periodicity_available": periodicity_available,
            "periodic_detected": periodic_detected,
            "periodic_alternating": alternating,
            "reversal_interval_median_sec": interval_median,
            "reversal_interval_cv": interval_cv,
            "reversal_interval_robust_cv": interval_robust_cv,
            "reversal_interval_max_relative_deviation": interval_max_relative_deviation,
            "reversal_interval_p90_p10_ratio": interval_p90_p10_ratio,
            "full_cycle_interval_count": int(full_cycle_intervals.size),
            "full_cycle_interval_median_sec": full_cycle_interval_median,
            "periodic_frequency_hz": periodic_frequency,
        }
    )
    return result


def welch_psd(
    samples: ArrayLike,
    dt_sec: float,
    *,
    nperseg: int = 128,
    overlap: float = 0.5,
    min_windows: int = 2,
    detrend: Optional[str] = "constant",
) -> Dict[str, object]:
    """Estimate a one-sided Welch PSD without joining across invalid gaps.

    A short record returns ``available=False`` and empty frequency/PSD arrays.
    Supported detrending modes are ``None``, ``"constant"``, and ``"linear"``.
    """
    signal = _as_1d_float(samples, "samples")
    dt_sec = float(dt_sec)
    if not np.isfinite(dt_sec) or dt_sec <= 0.0:
        raise ValueError("dt_sec must be finite and positive")
    nperseg = int(nperseg)
    min_windows = int(min_windows)
    overlap = float(overlap)
    if nperseg < 4:
        raise ValueError("nperseg must be at least 4")
    if min_windows < 1:
        raise ValueError("min_windows must be positive")
    if not np.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be within [0, 1)")
    if detrend not in (None, "constant", "linear"):
        raise ValueError("detrend must be None, 'constant', or 'linear'")

    noverlap = int(math.floor(nperseg * overlap))
    step = nperseg - noverlap
    finite = np.isfinite(signal)
    runs = _finite_runs(finite)
    run_durations = [_run_duration_sec(run, dt_sec) for run in runs]
    longest_valid_duration = float(max(run_durations, default=0.0))
    windows = []
    windows_per_run = []
    for run in runs:
        values = signal[run]
        run_window_count = 0
        for start in range(0, values.size - nperseg + 1, step):
            windows.append(values[start : start + nperseg].copy())
            run_window_count += 1
        windows_per_run.append(run_window_count)

    empty = {
        "available": False,
        "frequencies_hz": np.empty(0, dtype=np.float64),
        "psd": np.empty(0, dtype=np.float64),
        "window_count": int(len(windows)),
        "finite_run_count": int(len(runs)),
        "fragmented": bool(len(runs) > 1),
        "max_windows_in_single_run": int(max(windows_per_run, default=0)),
        "single_run_min_windows_available": bool(
            max(windows_per_run, default=0) >= min_windows
        ),
        "valid_sample_count": int(np.sum(finite)),
        "longest_contiguous_valid_duration_sec": longest_valid_duration,
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "frequency_resolution_hz": math.nan,
    }
    if len(windows) < min_windows:
        return empty

    fs = 1.0 / dt_sec
    taper = np.hanning(nperseg)
    taper_energy = float(np.sum(np.square(taper)))
    if taper_energy <= 0.0:
        return empty
    time_index = np.arange(nperseg, dtype=np.float64)
    centered_time = time_index - np.mean(time_index)
    time_energy = float(np.dot(centered_time, centered_time))
    estimates = []
    for window_values in windows:
        if detrend == "constant":
            window_values -= np.mean(window_values)
        elif detrend == "linear":
            centered_values = window_values - np.mean(window_values)
            slope = float(np.dot(centered_time, centered_values) / time_energy)
            window_values = centered_values - slope * centered_time

        transform = np.fft.rfft(window_values * taper)
        estimate = np.square(np.abs(transform)) / (fs * taper_energy)
        if nperseg % 2 == 0:
            estimate[1:-1] *= 2.0
        else:
            estimate[1:] *= 2.0
        estimates.append(estimate)

    frequencies = np.fft.rfftfreq(nperseg, d=dt_sec)
    result = dict(empty)
    result.update(
        {
            "available": True,
            "frequencies_hz": frequencies,
            "psd": np.mean(np.asarray(estimates, dtype=np.float64), axis=0),
            "window_count": int(len(estimates)),
            "frequency_resolution_hz": float(fs / nperseg),
        }
    )
    return result

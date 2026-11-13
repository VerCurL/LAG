"""V1 observable init/reset signatures, compared to archived E6 init frames.

These observations contain no hidden trim fields. Passing this comparison
does not prove unobserved simulator/FCS state was reset, nor that thrust at a
post-trim offline equilibrium should equal this untrimmed initialization.
"""
import math

INITIAL_FIELDS = ("simulation/sim-time-sec", "position/h-sl-ft", "position/lat-geod-deg", "position/long-gc-deg",
    "velocities/ve-fps", "velocities/u-fps", "velocities/v-fps", "velocities/w-fps",
    "attitude/roll-rad", "attitude/pitch-rad", "attitude/psi-deg", "aero/alpha-deg", "aero/beta-deg",
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",
    "inertia/mass-slugs", "inertia/ixx-slugs_ft2", "inertia/iyy-slugs_ft2", "inertia/izz-slugs_ft2",
    "inertia/ixy-slugs_ft2", "inertia/ixz-slugs_ft2", "inertia/iyz-slugs_ft2",
    "propulsion/tank/contents-lbs", "propulsion/tank/pct-full", "propulsion/engine/thrust-lbs",
    "fcs/left-aileron-pos-norm", "fcs/right-aileron-pos-norm", "fcs/elevator-pos-norm", "fcs/rudder-pos-norm",
    "fcs/throttle-pos-norm", "fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm", "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm",
    "gear/gear-pos-norm", "gear/gear-cmd-norm", "LifeCurrent", "DeathEvent")

DEFAULT_TOLERANCES = dict(default_absolute=1e-5, default_relative=1e-6, overrides={
    "simulation/sim-time-sec": [0., 0.], "position/h-sl-ft": [.01, 0.],
    "position/lat-geod-deg": [1e-7, 0.], "position/long-gc-deg": [1e-7, 0.],
    "velocities/ve-fps": [.001, 0.], "velocities/u-fps": [.01, 0.],
    "velocities/v-fps": [.01, 0.], "velocities/w-fps": [.01, 0.],
    "attitude/psi-deg": [.0001, 0.], "attitude/roll-rad": [1e-6, 0.], "attitude/pitch-rad": [1e-6, 0.],
    "velocities/p-rad_sec": [1e-6, 0.], "velocities/q-rad_sec": [1e-6, 0.], "velocities/r-rad_sec": [1e-6, 0.],
    "propulsion/tank/contents-lbs": [.01, 0.], "propulsion/tank/pct-full": [.001, 0.],
    "propulsion/engine/thrust-lbs": [.1, 0.], "LifeCurrent": [0., 0.], "DeathEvent": [0., 0.],
    **{path: [1e-6, 0.] for path in INITIAL_FIELDS if path.startswith("fcs/") or path.startswith("gear/")}})


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_initial(observation, reference_observation, tolerances=None):
    tolerances = DEFAULT_TOLERANCES if tolerances is None else tolerances
    result = dict(passed=True, aircraft={}, source="C49-E6 archived untrimmed init observations",
        hidden_trim_observed=False, unobserved_state_reset_proven=False)
    for side in ("red", "blue"):
        expected = reference_observation[side][side+"_0"]
        actual = observation.get(side, {}).get(side+"_0", {})
        fields, failed = {}, []
        for name in INITIAL_FIELDS:
            value, reference = actual.get(name), expected[name]
            if not _numeric(reference):
                raise ValueError("invalid frozen E6 reference field: " + name)
            absolute, relative = tolerances["overrides"].get(name,
                (tolerances["default_absolute"], tolerances["default_relative"]))
            bound = absolute + relative*abs(reference)
            if not _numeric(value):
                fields[name] = dict(actual=value if value is None or isinstance(value, (str, bool)) else None,
                    expected=reference, tolerance=bound, passed=False, error="missing_or_nonfinite_numeric_field")
                failed.append(name)
                continue
            difference = abs(value-reference)
            # Heading is periodic even at an initialization boundary.
            if name == "attitude/psi-deg":
                difference = abs((value-reference+180.)%360.-180.)
            fields[name] = dict(actual=value, expected=reference, tolerance=bound,
                abs_difference=difference, passed=difference <= bound)
            if difference > bound:
                failed.append(name)
        result["aircraft"][side] = dict(passed=not failed, fields=fields, failed_fields=failed,
            fields_checked=len(INITIAL_FIELDS))
        result["passed"] = result["passed"] and not failed
    return result

"""Run the first dependency-light LAGLayer3 local loop."""

import argparse

from laglayer3.domain.models import AircraftState, ScenarioInitialization, TeamObservation
from laglayer3.policies.fixed_execution import FixedRouteExecutionProvider
from laglayer3.policies.fixed_head import FixedRouteHeadProvider
from laglayer3.runtime.loop import LayeredRuntime
from laglayer3.simulation.local_jsbsim import LocalJSBSimAdapter


def build_initialization(count: int = 4) -> ScenarioInitialization:
    aircraft = tuple(
        AircraftState(
            aircraft_id="red_{}".format(index),
            side="red",
            position_ned_m=(float(index * 1000), 0.0, -8000.0),
            velocity_ned_mps=(220.0, 0.0, 0.0),
            heading_deg=0.0,
            altitude_m=8000.0,
            speed_mps=220.0,
        )
        for index in range(count)
    )
    return ScenarioInitialization(scenario_id="smoke", aircraft=aircraft)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--backend", choices=("kinematic", "jsbsim"), default="kinematic")
    args = parser.parse_args()

    simulation = LocalJSBSimAdapter(
        step_seconds=1.0 / 60.0,
        time_limit_seconds=20.0,
        backend=args.backend,
        data_root="third_party/jsbsim_data" if args.backend == "jsbsim" else None,
    )
    runtime = LayeredRuntime(
        simulation=simulation,
        head=FixedRouteHeadProvider(horizon_seconds=20.0, interval_seconds=1.0),
        execution=FixedRouteExecutionProvider(),
    )
    snapshot = runtime.reset(build_initialization())
    for _ in range(args.steps):
        observation = TeamObservation(own_aircraft=snapshot.aircraft, timestamp=snapshot.timestamp)
        snapshot = runtime.step(observation)
        if simulation.termination_status().terminated:
            break
    print(
        "timestamp={:.3f} aircraft={} termination={}".format(
            snapshot.timestamp,
            len(snapshot.aircraft),
            simulation.termination_status().reason,
        )
    )


if __name__ == "__main__":
    main()

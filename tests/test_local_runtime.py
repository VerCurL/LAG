from laglayer3.domain.models import AircraftState, ScenarioInitialization, TeamObservation
from laglayer3.policies.fixed_execution import FixedRouteExecutionProvider
from laglayer3.policies.fixed_head import FixedRouteHeadProvider
from laglayer3.runtime.loop import LayeredRuntime
from laglayer3.simulation.local_jsbsim import LocalJSBSimAdapter


def test_local_runtime_produces_snapshot():
    """中文说明：执行 test_local_runtime_produces_snapshot 的公开流程。"""
    aircraft = AircraftState(
        aircraft_id="red_0",
        side="red",
        position_ned_m=(0.0, 0.0, -8000.0),
        velocity_ned_mps=(200.0, 0.0, 0.0),
        heading_deg=0.0,
        altitude_m=8000.0,
        speed_mps=200.0,
    )
    init = ScenarioInitialization(scenario_id="test", aircraft=(aircraft,))
    sim = LocalJSBSimAdapter(step_seconds=0.1, time_limit_seconds=1.0)
    runtime = LayeredRuntime(sim, FixedRouteHeadProvider(2.0, 1.0), FixedRouteExecutionProvider())
    snapshot = runtime.reset(init)
    snapshot = runtime.step(TeamObservation(own_aircraft=snapshot.aircraft, timestamp=snapshot.timestamp))
    assert snapshot.timestamp == 0.1
    assert snapshot.aircraft[0].position_ned_m != aircraft.position_ned_m

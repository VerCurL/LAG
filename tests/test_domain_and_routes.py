from laglayer3.domain.models import AircraftState, RoutePrediction
from laglayer3.planning.frames import ned_to_route_frame, route_frame_to_ned
from laglayer3.planning.routes import RouteValidator


def test_body_aligned_frame_round_trip():
    """中文说明：执行 test_body_aligned_frame_round_trip 的公开流程。"""
    origin = (10.0, 20.0, -100.0)
    point = (15.0, 25.0, -95.0)
    local = ned_to_route_frame(point, origin, 90.0)
    restored = route_frame_to_ned(local, origin, 90.0)
    assert restored == point


def test_route_validator_rejects_invalid_speed():
    """中文说明：执行 test_route_validator_rejects_invalid_speed 的公开流程。"""
    route = RoutePrediction(
        aircraft_id="red_0",
        points=((10.0, 0.0, 8000.0, 10.0),),
        interval_seconds=1.0,
        origin_position_ned_m=(0.0, 0.0, -8000.0),
        origin_heading_deg=0.0,
    )
    result = RouteValidator(min_speed_mps=40.0).validate(route)
    assert not result.valid
    assert "speed_out_of_range_at_0" in result.reasons

"""Route construction and validation kept behind a small interface."""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from laglayer3.domain.models import RoutePrediction


def make_route(
    aircraft_id: str,
    points: Iterable[Tuple[float, float, float, float]],
    interval_seconds: float,
    origin_position_ned_m: Tuple[float, float, float],
    origin_heading_deg: float,
    confidence: float = 1.0,
    route_mode: str = "unknown",
) -> RoutePrediction:
    return RoutePrediction(
        aircraft_id=aircraft_id,
        points=tuple(tuple(float(value) for value in point) for point in points),
        interval_seconds=float(interval_seconds),
        origin_position_ned_m=tuple(float(x) for x in origin_position_ned_m),
        origin_heading_deg=float(origin_heading_deg),
        confidence=float(confidence),
        route_mode=route_mode,
    )


@dataclass(frozen=True)
class RouteValidationResult:
    valid: bool
    reasons: Tuple[str, ...] = ()


class RouteValidator:
    def __init__(
        self,
        min_altitude_m: float = 100.0,
        max_altitude_m: float = 20000.0,
        min_speed_mps: float = 40.0,
        max_speed_mps: float = 800.0,
    ) -> None:
        self.min_altitude_m = float(min_altitude_m)
        self.max_altitude_m = float(max_altitude_m)
        self.min_speed_mps = float(min_speed_mps)
        self.max_speed_mps = float(max_speed_mps)

    def validate(self, route: RoutePrediction) -> RouteValidationResult:
        reasons: List[str] = []
        previous_forward = None
        for index, (forward, right, altitude, speed) in enumerate(route.points):
            if altitude < self.min_altitude_m or altitude > self.max_altitude_m:
                reasons.append("altitude_out_of_range_at_{}".format(index))
            if speed < self.min_speed_mps or speed > self.max_speed_mps:
                reasons.append("speed_out_of_range_at_{}".format(index))
            if previous_forward is not None and forward < previous_forward - 1e-5:
                reasons.append("route_backtracks_at_{}".format(index))
            previous_forward = forward
        return RouteValidationResult(valid=not reasons, reasons=tuple(reasons))

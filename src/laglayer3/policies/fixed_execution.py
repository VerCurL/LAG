"""Simple route-following Provider for smoke tests and contract validation."""

from laglayer3.domain.models import ExecutionAction, LocalObservation, RoutePrediction


class FixedRouteExecutionProvider:
    def reset(self, aircraft_id: str) -> None:
        return None

    def act(self, observation: LocalObservation, route: RoutePrediction) -> ExecutionAction:
        aircraft = observation.aircraft
        if not route.points:
            return ExecutionAction(1, 2, 1)
        forward, right, altitude, speed = route.points[0]
        altitude_action = 0 if aircraft.altitude_m < altitude - 50 else 2 if aircraft.altitude_m > altitude + 50 else 1
        speed_action = 0 if aircraft.speed_mps < speed - 10 else 2 if aircraft.speed_mps > speed + 10 else 1
        heading_action = 1 if abs(right) < 100 else (3 if right > 0 else 1)
        return ExecutionAction(altitude_action, heading_action, speed_action)

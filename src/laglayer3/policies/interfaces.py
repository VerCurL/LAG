from typing import Any, Mapping, Protocol, Sequence

from laglayer3.domain.models import (
    ExecutionAction,
    ExecutionStatus,
    LocalObservation,
    RoutePrediction,
    TeamObservation,
    WeaponIntent,
)


class HeadPolicyProvider(Protocol):
    def reset(self) -> None: ...

    def decide(
        self, observation: TeamObservation, timestamp: float
    ) -> Mapping[str, object]: ...


class ExecutionPolicyProvider(Protocol):
    def reset(self, aircraft_id: str) -> None: ...

    def act(
        self, observation: LocalObservation, route: RoutePrediction
    ) -> ExecutionAction: ...


class FlightControlAdapter(Protocol):
    def reset(self) -> None: ...

    def control(self, aircraft_state: Mapping[str, Any], action: ExecutionAction): ...

from typing import Protocol

from laglayer3.domain.models import (
    ScenarioInitialization,
    SimulationSnapshot,
    TerminationStatus,
)


class SimulationAdapter(Protocol):
    def reset(self, initialization: ScenarioInitialization) -> SimulationSnapshot: ...

    def step(self, commands) -> SimulationSnapshot: ...

    def snapshot(self) -> SimulationSnapshot: ...

    def restore(self, snapshot: SimulationSnapshot) -> None: ...

    def termination_status(self) -> TerminationStatus: ...

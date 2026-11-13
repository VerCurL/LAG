"""Stable domain values shared by runtime, adapters, policies and learning."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class AircraftState:
    aircraft_id: str
    side: str
    position_ned_m: Vector3
    velocity_ned_mps: Vector3
    heading_deg: float
    altitude_m: float
    speed_mps: float
    alive: bool = True
    fuel: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutePrediction:
    """Future route in the decision-time, body-aligned local frame."""

    aircraft_id: str
    points: Tuple[Tuple[float, float, float, float], ...]
    interval_seconds: float
    origin_position_ned_m: Vector3
    origin_heading_deg: float
    confidence: float = 1.0
    route_mode: str = "unknown"
    schema_version: str = "route.v1"

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("route interval must be positive")
        if not self.points:
            raise ValueError("route must contain at least one point")
        if any(len(point) != 4 for point in self.points):
            raise ValueError("route points must be (forward, right, altitude, speed)")


@dataclass(frozen=True)
class ExecutionAction:
    altitude: int
    heading: int
    speed: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.altitude, self.heading, self.speed)


@dataclass(frozen=True)
class WeaponIntent:
    aircraft_id: str
    request: str = "none"
    weapon_type: Optional[str] = None
    target_id: Optional[str] = None
    fire_permission: bool = False
    valid_until: Optional[float] = None


@dataclass(frozen=True)
class LocalObservation:
    aircraft: AircraftState
    route: Optional[RoutePrediction] = None
    features: Tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeamObservation:
    own_aircraft: Tuple[AircraftState, ...]
    enemy_tracks: Tuple[Mapping[str, Any], ...] = ()
    missile_tracks: Tuple[Mapping[str, Any], ...] = ()
    timestamp: float = 0.0
    schema_version: str = "team-observation.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationEvent:
    name: str
    timestamp: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationSnapshot:
    timestamp: float
    aircraft: Tuple[AircraftState, ...]
    missiles: Tuple[Mapping[str, Any], ...] = ()
    events: Tuple[SimulationEvent, ...] = ()
    random_state: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioInitialization:
    scenario_id: str
    aircraft: Tuple[AircraftState, ...]
    timestamp: float = 0.0
    missiles: Tuple[Mapping[str, Any], ...] = ()
    sensor_state: Mapping[str, Any] = field(default_factory=dict)
    communication_state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TerminationStatus:
    terminated: bool
    reason: Optional[str] = None
    winner: Optional[str] = None
    events: Tuple[SimulationEvent, ...] = ()


@dataclass(frozen=True)
class ExecutionStatus:
    state: str
    task_id: Optional[str] = None
    event: Optional[str] = None
    replan_requested: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

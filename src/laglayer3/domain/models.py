"""运行时、适配器、策略和训练模块共享的领域数据对象。"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class AircraftState:
    """描述单架飞机在某一时刻的动力学和生存状态。"""
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
    """描述首脑层在决策时刻生成的机体对齐局部航迹。"""

    aircraft_id: str
    points: Tuple[Tuple[float, float, float, float], ...]
    interval_seconds: float
    origin_position_ned_m: Vector3
    origin_heading_deg: float
    confidence: float = 1.0
    route_mode: str = "unknown"
    schema_version: str = "route.v1"

    def __post_init__(self) -> None:
        """校验航迹间隔、点数量和每个点的四维结构。"""
        if self.interval_seconds <= 0:
            raise ValueError("route interval must be positive")
        if not self.points:
            raise ValueError("route must contain at least one point")
        if any(len(point) != 4 for point in self.points):
            raise ValueError("route points must be (forward, right, altitude, speed)")


@dataclass(frozen=True)
class ExecutionAction:
    """表示执行层输出的高度、航向和速度离散调整。"""
    altitude: int
    heading: int
    speed: int

    def as_tuple(self) -> Tuple[int, int, int]:
        """把离散动作转换为便于适配器使用的三元组。"""
        return (self.altitude, self.heading, self.speed)


@dataclass(frozen=True)
class WeaponIntent:
    """表示首脑层发出的武器请求及其目标和有效期。"""
    aircraft_id: str
    request: str = "none"
    weapon_type: Optional[str] = None
    target_id: Optional[str] = None
    fire_permission: bool = False
    valid_until: Optional[float] = None


@dataclass(frozen=True)
class LocalObservation:
    """表示单架飞机执行层可见的局部观测和当前航迹。"""
    aircraft: AircraftState
    route: Optional[RoutePrediction] = None
    features: Tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeamObservation:
    """表示首脑层可见的己方飞机、目标轨迹和来袭导弹信息。"""
    own_aircraft: Tuple[AircraftState, ...]
    enemy_tracks: Tuple[Mapping[str, Any], ...] = ()
    missile_tracks: Tuple[Mapping[str, Any], ...] = ()
    timestamp: float = 0.0
    schema_version: str = "team-observation.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationEvent:
    """记录仿真过程中发生的可追踪事件。"""
    name: str
    timestamp: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationSnapshot:
    """保存可用于观测、回放和恢复的仿真状态快照。"""
    timestamp: float
    aircraft: Tuple[AircraftState, ...]
    missiles: Tuple[Mapping[str, Any], ...] = ()
    events: Tuple[SimulationEvent, ...] = ()
    random_state: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioInitialization:
    """描述一局仿真的声明式初始飞机、导弹和感知状态。"""
    scenario_id: str
    aircraft: Tuple[AircraftState, ...]
    timestamp: float = 0.0
    missiles: Tuple[Mapping[str, Any], ...] = ()
    sensor_state: Mapping[str, Any] = field(default_factory=dict)
    communication_state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TerminationStatus:
    """描述仿真是否结束、结束原因、胜方和终止事件。"""
    terminated: bool
    reason: Optional[str] = None
    winner: Optional[str] = None
    events: Tuple[SimulationEvent, ...] = ()


@dataclass(frozen=True)
class ExecutionStatus:
    """描述单机航迹执行的阶段、事件和重规划请求。"""
    state: str
    task_id: Optional[str] = None
    event: Optional[str] = None
    replan_requested: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

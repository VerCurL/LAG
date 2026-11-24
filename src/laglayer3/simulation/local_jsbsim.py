"""本地仿真适配器。

适配器向上层提供统一的仿真快照、事件、恢复和终止接口。内部可以使用
轻量运动学后端进行契约测试，也可以切换到真实 JSBSim 飞机模型。
"""

import copy
from dataclasses import replace
from typing import Dict, Mapping, Optional

from laglayer3.domain.models import (
    AircraftState,
    ExecutionAction,
    ScenarioInitialization,
    SimulationEvent,
    SimulationSnapshot,
    TerminationStatus,
)


class LocalJSBSimAdapter:
    """管理本地仿真的生命周期，并统一不同仿真后端的输出格式。"""
    def __init__(
        self,
        step_seconds: float = 1.0 / 60.0,
        time_limit_seconds: float = 900.0,
        backend: str = "kinematic",
        data_root: Optional[str] = None,
        model: str = "f16",
    ) -> None:
        """设置步长、终止时间和内部仿真后端。"""
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        self.step_seconds = float(step_seconds)
        self.time_limit_seconds = float(time_limit_seconds)
        if backend not in {"kinematic", "jsbsim"}:
            raise ValueError("backend must be kinematic or jsbsim")
        self.backend = backend
        self.data_root = data_root
        self.model = model
        self._jsbsim_aircraft = {}
        self._snapshot: Optional[SimulationSnapshot] = None
        self._done = False
        self._reason: Optional[str] = None

    def reset(self, initialization: ScenarioInitialization) -> SimulationSnapshot:
        """根据声明式初始化创建一局新的仿真。"""
        self._snapshot = SimulationSnapshot(
            timestamp=float(initialization.timestamp),
            aircraft=tuple(copy.deepcopy(initialization.aircraft)),
            missiles=tuple(copy.deepcopy(initialization.missiles)),
            events=(),
            metadata={
                "backend": "local_jsbsim_contract",
                "scenario_id": initialization.scenario_id,
                "schema_version": "zk-compatible.v1",
            },
        )
        self._done = False
        self._reason = None
        self._jsbsim_aircraft = {}
        if self.backend == "jsbsim":
            if not self.data_root:
                raise ValueError("data_root is required for the jsbsim backend")
            from .jsbsim_backend import JSBSimAircraft

            self._jsbsim_aircraft = {
                aircraft.aircraft_id: JSBSimAircraft(
                    aircraft, self.data_root, model=self.model, dt=self.step_seconds
                )
                for aircraft in initialization.aircraft
            }
        return self._snapshot

    def step(self, commands: Mapping[str, ExecutionAction]) -> SimulationSnapshot:
        """执行一个仿真步，更新飞机状态并生成终止事件。"""
        if self._snapshot is None:
            raise RuntimeError("reset must be called before step")
        if self._done:
            raise RuntimeError("cannot step a terminated simulation")
        dt = self.step_seconds
        next_aircraft = []
        for aircraft in self._snapshot.aircraft:
            action = commands.get(aircraft.aircraft_id, ExecutionAction(1, 2, 1))
            if self.backend == "jsbsim":
                next_aircraft.append(self._jsbsim_aircraft[aircraft.aircraft_id].step(action))
            else:
                next_aircraft.append(self._integrate_aircraft(aircraft, action, dt))
        timestamp = self._snapshot.timestamp + dt
        events = []
        if timestamp >= self.time_limit_seconds:
            self._done = True
            self._reason = "time_limit"
            events.append(SimulationEvent("time_limit", timestamp))
        if not any(aircraft.alive for aircraft in next_aircraft):
            self._done = True
            self._reason = "all_aircraft_destroyed"
            events.append(SimulationEvent("all_aircraft_destroyed", timestamp))
        self._snapshot = SimulationSnapshot(
            timestamp=timestamp,
            aircraft=tuple(next_aircraft),
            missiles=self._snapshot.missiles,
            events=tuple(events),
            metadata=self._snapshot.metadata,
        )
        return self._snapshot

    def snapshot(self) -> SimulationSnapshot:
        """返回当前仿真状态的深拷贝。"""
        if self._snapshot is None:
            raise RuntimeError("simulation has not been reset")
        return copy.deepcopy(self._snapshot)

    def restore(self, snapshot: SimulationSnapshot) -> None:
        """恢复快照中的状态和终止标记。"""
        self._snapshot = copy.deepcopy(snapshot)
        self._done = any(event.name in {"time_limit", "all_aircraft_destroyed"} for event in snapshot.events)
        self._reason = snapshot.events[-1].name if self._done and snapshot.events else None

    def termination_status(self) -> TerminationStatus:
        """返回当前是否终止以及对应的原因和事件。"""
        return TerminationStatus(
            terminated=self._done,
            reason=self._reason,
            winner=None,
            events=() if self._snapshot is None else self._snapshot.events,
        )

    @staticmethod
    def _integrate_aircraft(aircraft: AircraftState, action: ExecutionAction, dt: float) -> AircraftState:
        """使用轻量运动学模型推进单架飞机状态。"""
        altitude_delta = float(action.altitude - 1) * 10.0
        heading_delta = float(action.heading - 2) * 5.0
        speed_delta = float(action.speed - 1) * 5.0
        heading = (aircraft.heading_deg + heading_delta * dt) % 360.0
        speed = max(1.0, aircraft.speed_mps + speed_delta * dt)
        altitude = max(50.0, aircraft.altitude_m + altitude_delta * dt)
        import math

        angle = math.radians(heading)
        velocity = (speed * math.cos(angle), speed * math.sin(angle), 0.0)
        position = tuple(aircraft.position_ned_m[i] + velocity[i] * dt for i in range(3))
        return replace(
            aircraft,
            position_ned_m=position,
            velocity_ned_mps=velocity,
            heading_deg=heading,
            altitude_m=altitude,
            speed_mps=speed,
        )

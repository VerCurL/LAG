"""Three-frequency runtime loop for the first end-to-end smoke path."""

from typing import Dict, Mapping

from laglayer3.domain.models import LocalObservation, TeamObservation
from laglayer3.policies.fixed_execution import FixedRouteExecutionProvider
from laglayer3.policies.fixed_head import FixedRouteHeadProvider
from laglayer3.simulation.interface import SimulationAdapter


class LayeredRuntime:
    """中文说明：定义本模块负责的 LayeredRuntime 数据或行为。"""
    def __init__(
        self,
        simulation: SimulationAdapter,
        head=None,
        execution=None,
        head_period_seconds: float = 1.0,
        execution_period_seconds: float = 0.2,
    ) -> None:
        """中文说明：初始化对象配置、依赖和运行状态。"""
        self.simulation = simulation
        self.head = head or FixedRouteHeadProvider()
        self.execution = execution or FixedRouteExecutionProvider()
        self.head_period_seconds = float(head_period_seconds)
        self.execution_period_seconds = float(execution_period_seconds)
        self.routes = {}
        self.next_head_time = 0.0
        self.next_execution_time = 0.0

    def reset(self, initialization):
        """中文说明：清理本模块的运行状态和循环记忆。"""
        snapshot = self.simulation.reset(initialization)
        self.head.reset()
        self.routes = {}
        self.next_head_time = snapshot.timestamp
        self.next_execution_time = snapshot.timestamp
        return snapshot

    def step(self, observation: TeamObservation):
        """中文说明：推进一次仿真或执行周期并返回最新状态。"""
        timestamp = observation.timestamp
        if timestamp >= self.next_head_time - 1e-9:
            decision = self.head.decide(observation, timestamp)
            self.routes = dict(decision.get("routes", {}))
            self.next_head_time = timestamp + self.head_period_seconds
        commands = {}
        if timestamp >= self.next_execution_time - 1e-9:
            for aircraft in observation.own_aircraft:
                route = self.routes.get(aircraft.aircraft_id)
                if route is None:
                    continue
                local = LocalObservation(aircraft=aircraft, route=route)
                commands[aircraft.aircraft_id] = self.execution.act(local, route)
            self.next_execution_time = timestamp + self.execution_period_seconds
        return self.simulation.step(commands)

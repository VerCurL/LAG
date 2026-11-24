"""分层运行时调度器。

该模块统一推进仿真、首脑层和执行层的不同时间尺度。首脑层默认每秒
重新规划一次，执行层默认每 0.2 秒生成一次动作，仿真器由适配器按自身
步长推进。
"""

from typing import Dict, Mapping

from laglayer3.domain.models import LocalObservation, TeamObservation
from laglayer3.policies.fixed_execution import FixedRouteExecutionProvider
from laglayer3.policies.fixed_head import FixedRouteHeadProvider
from laglayer3.simulation.interface import SimulationAdapter


class LayeredRuntime:
    """协调首脑 Provider、执行 Provider 和仿真适配器。"""
    def __init__(
        self,
        simulation: SimulationAdapter,
        head=None,
        execution=None,
        head_period_seconds: float = 1.0,
        execution_period_seconds: float = 0.2,
    ) -> None:
        """注入运行依赖并设置首脑层和执行层的决策周期。"""
        self.simulation = simulation
        self.head = head or FixedRouteHeadProvider()
        self.execution = execution or FixedRouteExecutionProvider()
        self.head_period_seconds = float(head_period_seconds)
        self.execution_period_seconds = float(execution_period_seconds)
        self.routes = {}
        self.next_head_time = 0.0
        self.next_execution_time = 0.0

    def reset(self, initialization):
        """重置仿真、首脑 Provider、路线缓存和两个调度时钟。"""
        snapshot = self.simulation.reset(initialization)
        self.head.reset()
        self.routes = {}
        self.next_head_time = snapshot.timestamp
        self.next_execution_time = snapshot.timestamp
        return snapshot

    def step(self, observation: TeamObservation):
        """根据当前时间决定是否重规划、是否更新执行动作，并推进仿真。

        首脑层更新时会覆盖当前团队路线。执行层只消费已生成的路线，
        没有路线的飞机本轮不生成动作。关键事件触发重规划的接口会在
        后续事件调度模块接入后复用这里的首脑更新时间。
        """
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

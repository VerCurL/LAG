from typing import Any, Mapping, Protocol

from laglayer3.domain.models import (
    ExecutionAction,
    LocalObservation,
    RoutePrediction,
    TeamObservation,
)


class HeadPolicyProvider(Protocol):
    """首脑层策略提供者的接口。"""

    # 清理首脑策略的记忆状态。
    def reset(self) -> None: ...

    # 根据团队观测生成当前首脑决策。
    def decide(
        self, observation: TeamObservation, timestamp: float
    ) -> Mapping[str, object]: ...


class ExecutionPolicyProvider(Protocol):
    """执行层策略提供者的接口。"""

    # 清理指定飞机的执行策略记忆状态。
    def reset(self, aircraft_id: str) -> None: ...

    # 根据航迹和局部观测生成执行动作。
    def act(
        self, observation: LocalObservation, route: RoutePrediction
    ) -> ExecutionAction: ...


class FlightControlAdapter(Protocol):
    """飞控适配器的接口声明。"""

    # 清理飞控适配器的每机隐藏状态。
    def reset(self) -> None: ...

    # 将执行动作转换为平台控制量。
    def control(self, aircraft_state: Mapping[str, Any], action: ExecutionAction): ...

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
    """中文说明：定义本模块负责的 HeadPolicyProvider 数据或行为。"""
    # 中文说明：清理本模块的运行状态和循环记忆。
    # 中文说明：清理首脑策略的记忆状态。
    def reset(self) -> None: ...

    # 中文说明：根据团队观测生成当前首脑决策。
    def decide(
        self, observation: TeamObservation, timestamp: float
    ) -> Mapping[str, object]: ...


class ExecutionPolicyProvider(Protocol):
    """中文说明：定义本模块负责的 ExecutionPolicyProvider 数据或行为。"""
    # 中文说明：清理本模块的运行状态和循环记忆。
    # 中文说明：清理指定飞机的执行策略记忆状态。
    def reset(self, aircraft_id: str) -> None: ...

    # 中文说明：根据局部观测和首脑航迹生成单机执行动作。
    # 中文说明：根据航迹和局部观测生成执行动作。
    def act(
        self, observation: LocalObservation, route: RoutePrediction
    ) -> ExecutionAction: ...


class FlightControlAdapter(Protocol):
    """中文说明：定义本模块负责的 FlightControlAdapter 数据或行为。"""
    # 中文说明：清理本模块的运行状态和循环记忆。
    # 中文说明：清理飞控适配器的每机隐藏状态。
    def reset(self) -> None: ...

    # 中文说明：把执行动作转换为飞控层可消费的控制输出。
    # 中文说明：将执行动作转换为平台控制量。
    def control(self, aircraft_state: Mapping[str, Any], action: ExecutionAction): ...

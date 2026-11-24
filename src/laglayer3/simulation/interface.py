from typing import Protocol

from laglayer3.domain.models import (
    ScenarioInitialization,
    SimulationSnapshot,
    TerminationStatus,
)


class SimulationAdapter(Protocol):
    """中文说明：定义本模块负责的 SimulationAdapter 数据或行为。"""
    # 中文说明：清理本模块的运行状态和循环记忆。
    def reset(self, initialization: ScenarioInitialization) -> SimulationSnapshot: ...

    # 中文说明：推进一次仿真或执行周期并返回最新状态。
    def step(self, commands) -> SimulationSnapshot: ...

    # 中文说明：返回当前仿真状态的独立快照。
    def snapshot(self) -> SimulationSnapshot: ...

    # 中文说明：从快照恢复仿真内部状态。
    def restore(self, snapshot: SimulationSnapshot) -> None: ...

    # 中文说明：返回当前仿真是否结束及结束原因。
    def termination_status(self) -> TerminationStatus: ...

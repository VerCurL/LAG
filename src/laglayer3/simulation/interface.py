from typing import Protocol

from laglayer3.domain.models import (
    ScenarioInitialization,
    SimulationSnapshot,
    TerminationStatus,
)


class SimulationAdapter(Protocol):
    """仿真适配器的统一接口。

    本接口屏蔽本地 JSBSim、远程 ZK 平台和回放适配器之间的实现差异。
    调用方只需要处理领域快照、事件和终止状态。
    """

    # 按声明式初始化创建一局新的仿真。
    def reset(self, initialization: ScenarioInitialization) -> SimulationSnapshot: ...

    # 推进一个仿真步并返回最新快照。
    def step(self, commands) -> SimulationSnapshot: ...

    # 返回当前仿真状态的独立副本。
    def snapshot(self) -> SimulationSnapshot: ...

    # 从快照恢复仿真内部状态。
    def restore(self, snapshot: SimulationSnapshot) -> None: ...

    # 返回当前终止状态、原因和相关事件。
    def termination_status(self) -> TerminationStatus: ...

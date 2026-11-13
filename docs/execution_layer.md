# 执行层设计

执行层是首脑航迹的单机黑盒跟踪器。它不重新规划任务、不改变首脑目标、不负责发弹。

## 输入

```text
LocalObservation
WaypointRouteTask
SafetyContext
PreviousIntent
ControllerFeedback
```

## 输出

```text
ExecutionAction
ExecutionStatus
```

执行层负责：

- 跟踪决策时刻固定局部坐标系中的航迹；
- 根据当前飞机状态学习高度、航向和速度调整；
- 在奖励中优化航迹拟合、能量、平滑性和稳定性；
- 报告完成、失败、过期和重规划事件。

执行层默认使用 MultiDiscrete([3, 5, 3])，不输出武器动作和最终舵面。

## 规则和模型

推荐内部结构：

```text
ExecutionCoordinator
├── RouteEncoder
├── ExecutionPolicy
├── RouteTrackingReward
└── ExecutionStatusReporter
```

初版使用航迹编码器、当前状态编码器、GRU 和共享 Actor。模型只负责局部航迹跟踪动作，首脑层和 WeaponManager 负责任务与武器。

建议训练路径为：

```text
规则或几何教师
    ↓
行为克隆
    ↓
DAgger
    ↓
Residual PPO
    ↓
条件 PPO 微调
```


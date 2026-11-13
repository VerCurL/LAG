# 架构总览

LAGLayer3 采用分层控制架构，并把仿真、感知、武器、安全和学习流程分开。默认仿真和飞控 60 Hz、执行层 5 Hz、首脑层 1 Hz，关键事件可以触发首脑重规划。完整设计决策见 [重构设计决策](./restructuring_decisions.md)，源码边界见 [目录结构设计](./directory_structure.md)。

```text
World State
    ↓
Radar / Fog of War / Communication
    ↓
Team Observation
    ↓
Head Policy Provider
    ↓
Team Tactical Plan
    ↓
Aircraft Task
    ↓
Execution Coordinator
    ↓
Team Route + Weapon Intent
    ↓                 ↓
Safety Supervisor    Weapon Manager
    ↓                 ↓
Flight Controller  Low-level Weapon Command
    ↓                 ↓
Combat Simulation
```

## 模块职责

### `domain`

定义稳定的领域对象和接口，例如 `TeamObservation`、`TeamTacticalPlan`、`AircraftTask`、`RoutePrediction`、`ExecutionAction`、`DecisionBundle` 和事件对象。这里不依赖 PyTorch、JSBSim 或具体平台。

### `simulation`

负责推进本地 JSBSim 或连接外部 ZK 平台，管理飞机、导弹、仿真时间、重置和快照。它不负责 PPO、CFC 或策略奖励。

### `perception`

把世界状态转换成受雷达、战争迷雾和通信限制影响的团队观测和局部观测。

### `policies`

提供可替换的决策实现。FKR v3 是 `rule_policy` 下的一个策略提供者，后续的模仿学习和 PPO 也实现同一决策接口。

### `execution`

管理单机航迹跟踪、执行模型和执行状态。它只消费首脑航迹，不重新分配团队目标、不改变角色、不规划武器，也不直接输出舵面。默认动作空间为 MultiDiscrete([3, 5, 3])。

### `layers/flight_control`

提供 BSL、`bsl_aggressive`、PID 和其他飞控 Adapter，把飞行意图转成平台控制量。

### `learning`

包含解析和仿真教师数据、行为克隆、DAgger、Residual PPO、条件 PPO、分层 MoE、独立 Trainer 和计划级 CFC。训练由 TrainingGraph 调度任意固定或可训练层。

## 依赖方向

推荐依赖方向为：

```text
domain ← simulation / perception / weapons / safety
domain ← policies / execution / flight_control
simulation + perception + weapons + safety → environments
policies + execution + environments → learning
所有运行模块 → runtime / evaluation
```

规则策略可以依赖领域对象、感知结果、武器状态和安全约束，但不应直接依赖 `bsl_aggressive` 的内部类。飞控连接通过 Adapter 完成。


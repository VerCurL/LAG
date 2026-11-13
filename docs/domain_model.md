# 领域模型

## 主要对象关系

```text
SimulationSnapshot
    ├── AircraftState[*]
    ├── MissileState[*]
    ├── SensorState
    └── SimulationEvent[*]

TeamObservation
    ├── TeamAircraftView[*]
    ├── EnemyTrack[*]
    ├── MissileTrack[*]
    └── CommunicationView

TeamTacticalPlan
    └── AircraftTask[*]

AircraftTask
    ├── role
    ├── target
    ├── skill
    ├── geometry_constraints
    ├── support_relation
    ├── route_reference
    └── weapon_permission

DecisionBundle
    ├── RoutePrediction
    ├── WeaponIntent
    ├── control_source
    ├── priority
    └── DecisionTrace
```

## 状态的三个视图

### `WorldState`

仿真平台中的真实状态。只有仿真和授权的训练 Critic 可以访问完整世界状态。

### `TacticalObservation`

经过雷达、战争迷雾和通信处理后的团队观测，供首脑层使用。

### `LocalObservation`

单架飞机能够使用的局部观测，供执行层使用。

这三个对象不能混用。特别是，策略不能因为训练方便而直接读取 `WorldState`。

## 计划和意图的区别

`TeamTacticalPlan` 描述团队准备做什么，例如“红 1 主攻、红 2 支援、红 3 牵制”。

`AircraftTask` 描述分配给单机的任务。

`RoutePrediction` 描述首脑层给单机的相对局部航迹。它以决策时刻飞机为原点，以飞机航向为 x 轴，包含未来多个 forward、right、altitude 和 speed 点。

`ExecutionAction` 是执行层的局部动作，默认是高度、航向和速度三维离散调整。它不包含武器动作。

`LowLevelCommand` 才是飞控层输出给平台的副翼、升降舵、方向舵和油门。

保持这四个层次分开，是首脑、执行和飞控可以独立替换的前提。


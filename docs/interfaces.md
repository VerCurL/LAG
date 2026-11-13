# 接口契约

## 策略提供者

所有规则、模仿学习、强化学习和混合策略都应满足同一概念接口：

```text
decide(
    tactical_observation,
    plan_context,
    policy_memory,
    constraints
) -> DecisionBundle
```

首脑 Provider 是团队级 Provider，一次为所有己方飞机输出相对航迹和武器意图。航迹使用决策时刻的机体对齐局部坐标，武器意图只是请求。

接口必须说明：

- 输入观测是否带有延迟和噪声；
- 规则状态或 RNN 状态如何 reset；
- 决策的有效时间；
- 输出是否可以被安全规则覆盖；
- 是否允许输出紧急直接控制；
- 输出中的武器意图是否只是请求。

## 执行层

```text
step(
    local_observation,
    aircraft_task,
    safety_context,
    previous_intent,
    controller_feedback
) -> ExecutionAction, ExecutionStatus
```

执行层输入固定长度航迹窗口和当前飞机在该坐标系中的状态，默认输出 MultiDiscrete([3, 5, 3]) 的高度、航向和速度调整。执行层不输出武器动作、最终舵面，也不绕过武器管理器。

## 飞控层

```text
control(
    aircraft_state,
    execution_action,
    controller_memory
) -> LowLevelCommand, ControllerFeedback
```

`bsl_aggressive` 是这个接口的一个 Adapter。它可以内部使用 40 维观测、GRU、名义控制和 QP，但上层只需要认识执行动作和飞控 Adapter。

## 仿真接口

```text
reset(initialization)
step(command)
snapshot()
events()
restore(snapshot)
termination_status()
```

本地 JSBSim 遵循 ZK 的状态、时间、事件、初始化和终止语义。初始化和终止条件由配置和 Provider 组合，不能写死在环境 reset 中。

## 决策记录

每个决策周期都应保留：

```text
observation
candidate_decisions
selected_decision
proposed_intent
accepted_intent
executed_intent
active_rules
override_reason
weapon_request
platform_feedback
```

这组记录既用于调试，也用于后续教师数据和反事实分析。


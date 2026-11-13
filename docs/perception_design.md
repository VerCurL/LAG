# 感知与战争迷雾

感知模块把仿真真值转换为策略能够使用的观测。它是 `WorldState` 和 `TeamObservation`、`LocalObservation` 之间的唯一入口。

## 数据流

```text
WorldState
    ↓
Radar Model
    ↓
Detection and Track Manager
    ↓
Fog of War
    ↓
Team Communication
    ↓
TeamObservation / LocalObservation
```

## 感知状态

每条目标轨迹至少需要记录：

- 目标身份或身份置信度；
- 相对位置和相对速度；
- 最近一次观测时间；
- 轨迹年龄；
- 当前可见性；
- 跟踪置信度；
- 是否由队友共享；
- 是否存在冲突或过期信息。

## 首脑层和执行层的区别

首脑层使用经过队内通信融合的 `TeamObservation`，可以看到队友状态和团队目标摘要，但仍受延迟、丢失和置信度限制。

执行层使用单架飞机的 `LocalObservation`，重点关注自身状态、当前任务、目标轨迹、附近队友和来袭导弹。

两层都不应在部署模式下读取没有经过感知流程的仿真真值。

## 训练要求

训练时应覆盖：

- 探测延迟；
- 目标短暂丢失；
- 雷达噪声；
- 队友通信延迟；
- 错误或过期轨迹；
- 不同探测距离和扫描周期。

否则规则策略或学习策略可能只适应理想观测，正式平台运行时无法稳定执行计划。


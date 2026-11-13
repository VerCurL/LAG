# 开发路线

## 阶段一：领域、网络模板和接口

- 固定术语；
- 定义 `SimulationSnapshot`；
- 定义 `TacticalObservation`；
- 定义 `DecisionBundle`；
- 定义 `RoutePrediction`；
- 定义 `ExecutionAction`；
- 定义 `WeaponIntent`；
- 定义 `LowLevelCommand`。
- 定义相对局部航迹 schema、武器意图 schema、ActionSpec 和 Provider；
- 建立首脑实体编码、分层 MoE、连续航迹头和武器头模板；
- 建立执行层航迹编码器、GRU 和 MultiDiscrete([3,5,3]) Actor 模板；
- 建立 TrainingGraph、冻结策略和分层 checkpoint manifest。

## 阶段二：仿真和平台 Adapter

- 接入遵循 ZK 协议的本地 JSBSIM；
- 固定 ZK 状态、指令、事件、初始化和终止契约；
- 建立官方初始化和 `combat_start`；
- 建立快照、事件和恢复状态。
- 建立可组合 TerminationProvider 和 custom 初始化。

## 阶段三：FKR v3 规则策略

- 迁移观测 Adapter；
- 迁移规则状态机；
- 迁移优先级仲裁；
- 接入武器管理器；
- 接入 `bsl_aggressive` Adapter；
- 验证规则输出和比赛行为一致。

## 阶段四：规则教师数据

- 记录候选规则和最终决策；
- 生成首脑层任务标签；
- 生成执行层意图标签；
- 建立规则分歧和覆盖分析。
- 分别生成解析教师航迹和仿真教师航迹，带数据集指纹和 schema 版本。

## 阶段五：执行层学习

- 行为克隆；
- DAgger；
- Residual PPO；
- 条件执行策略；
- 保留安全规则。

## 阶段六：首脑层学习和 CFC

- 规则计划模仿；
- 首脑层强化学习；
- 计划级反事实；
- 计划贡献估计；
- 多机联合贡献和置信度门控。


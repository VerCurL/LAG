# 新 AI 实现交接说明

本文面向没有当前对话上下文的开发者或 AI。开始实现前必须阅读：

1. GLOSSARY.md；
2. docs/README.md；
3. docs/reference_sources.md；
4. docs/architecture.md；
5. docs/interfaces.md；
6. docs/policy_framework.md；
7. docs/migration_map.md。
8. docs/restructuring_decisions.md。

## 当前任务

在空的 LAGLayer3 项目中，逐步重构 LAGCFC 和比赛 FKR v3 的可复用能力，形成：

~~~text
团队观测
    ↓
首脑策略提供者
    ↓
团队战术计划
    ↓
单机执行层
    ↓
相对航迹
    ↓
航迹执行动作
    ↓
bsl_aggressive / 其他飞控
    ↓
仿真平台
~~~

## 已确定的设计

- FKR v3 是规则策略提供者，不是整个系统的固定架构。
- bsl_aggressive 是飞控 Adapter，不属于首脑规则。
- 首脑层输出 RoutePrediction，执行层输出 ExecutionAction，不直接输出 A/E/R/T。
- 首脑层输出相对航迹和武器意图，执行层默认只输出 MultiDiscrete([3,5,3]) 航迹跟踪动作。
- 武器意图交给独立 WeaponManager。
- CFC 迁移为计划级反事实，第一版作为独立分析器，不直接修改训练奖励。
- 教师航迹分解析生成和仿真生成两类，训练只读取匹配缓存。
- 首脑使用团队实体编码和分层 MoE，执行层使用航迹编码器和 GRU。
- 规则决策必须记录教师轨迹，供后续行为克隆、DAgger 和残差 PPO 使用。
- 安全规则可以覆盖模型输出，但必须记录覆盖原因。

## 尚未确定的设计

- 首脑航迹和武器头的最终 loss 权重；
- MoE 专家规模、任务映射和负载均衡系数；
- 执行层路线跟踪奖励权重；
- ZK 字段到 JSBSIM 状态的最终映射和参数标定；
- 计划级 CFC 何时升级为训练辅助优势。

这些问题不能在没有实验依据时被静默定案。应记录为实验或 ADR。

## 第一阶段建议

第一阶段实现接口、网络模板和最小规则闭环：

1. 定义 TacticalObservation、DecisionBundle、RoutePrediction、ExecutionAction 和 WeaponIntent；
2. 建立 FKR v3 的输入 Adapter；
3. 迁移一个最小的进场、边界和高度规则；
4. 接入 bsl_aggressive Adapter；
5. 用固定观测回放验证规则输出；
6. 记录规则教师轨迹；
7. 生成解析教师航迹并保存数据集；
8. 使用固定 FKR、固定执行 Provider 和 bsl_aggressive 跑通 JSBSim 闭环；
9. 再开始执行层 BC/PPO 和首脑层 BC。

## 约束

- 不修改 LAGCFC；
- 不把比赛提交包的绝对路径写进运行时代码；
- 不直接复制 FKR v3 的单体 runtime.py 作为新核心；
- 不让规则直接依赖飞控内部实现；
- 不把安全覆盖后的动作冒充模型原始动作；
- 不用正式评估结果替代训练专用 combat_start 结果。

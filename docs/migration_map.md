# 迁移映射

本文把参考项目中的主要内容映射到 LAGLayer3 的目标模块。它是迁移指导，不是要求一次性全部复制。

| 参考内容 | LAGLayer3 目标位置 | 迁移方式 |
|---|---|---|
| FKR v3 runtime.py | policies/rule_policy/fkr_v3/ | 按观测、规则状态、候选、仲裁和编译拆分 |
| FKR v3 coordination.py | policies/rule_policy/fkr_v3/rules/coordination/ | 提取双翼、支援、诱导和角色状态 |
| envs/rules/formation* | policies/rule_policy/fkr_v3/rules/opening/ | 保留编队几何，输出任务或意图候选 |
| envs/rules/missile_defense.py | policies/rule_policy/fkr_v3/rules/defense/ | 保留威胁状态机，输出紧急命令或防御意图 |
| envs/rules/weapons.py | weapons/ | 迁移为独立武器管理器，避免规则直接发射 |
| envs/rules/boundary.py、altitude.py、speed.py | safety/ | 提取为安全接管和可行域投影 |
| envs/model/bsl_aggressive/ | layers/flight_control/、adapters/intent/ | 只暴露统一飞控接口 |
| MultipleCombatEnv | simulation/、environments/ | 拆分仿真推进和训练环境包装 |
| SituationExtractor | simulation/core/snapshot_builder.py | 形成统一快照，不保留旧 info 依赖 |
| FieldCalculator | learning/credit/ 或 evaluation/ | 作为态势标签和计划结果评估工具 |
| mappoCFC/ppo_trainer.py | learning/credit/、learning/head/ | 重写为计划级反事实，不直接复制奖励重分配 |

## 迁移原则

1. 先迁移行为语义，再迁移代码。
2. 平台原始状态先经过 Adapter，再进入规则或模型。
3. 规则输出先进入 DecisionBundle，再编译成 BSL 或紧急控制。
4. 武器请求与飞行意图分开。
5. 记录规则决策过程，保留教师数据。
6. 每迁移一组规则，就建立对应的单元测试和短局回归。

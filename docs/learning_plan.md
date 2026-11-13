# 学习与教师数据

## 规则教师

FKR v3 规则运行时同时产生决策和教师轨迹：

```text
observation
candidate decisions
selected decision
active rules
intent
weapon intent
override reason
outcome
```

这些数据可用于：

- 首脑层行为克隆；
- 执行层行为克隆；
- DAgger；
- 规则与模型分歧分析；
- 安全规则覆盖统计。

## 分阶段学习

推荐顺序：

1. 先运行规则策略验证仿真和接口；
2. 使用规则数据训练执行层行为克隆；
3. 使用 Residual PPO 学习规则之外的局部改进；
4. 训练首脑层模仿规则任务分配；
5. 再使用强化学习微调首脑层；
6. 最后加入计划级 CFC。

## 教师数据

教师航迹有两种独立来源：解析教师不启动 JSBSim，用 FKR 参考量和运动学积分生成 planned route；仿真教师使用固定执行层和飞控在 LocalJSBSim 中生成 executed route、误差和结果。两类数据由独立脚本提前生成，训练时只读取严格匹配的数据集。

首脑和执行层训练都通过 Provider 和 TrainingGraph 配置固定或训练节点。每层独立拥有 Trainer、optimizer、loss、buffer、checkpoint 和 metrics。

## CFC 迁移

CFC 的反事实对象从底层动作上移到：

- 角色；
- 目标分配；
- 技能；
- 支援关系；
- 团队战术计划。

事实计划和反事实计划在相同态势下比较未来威胁、攻击机会、存活率和团队收益，再估计计划贡献。第一版 CFC 作为独立分析器和指标插件，不直接重分配首脑层训练奖励。


# 参考项目与源码定位

这份文档说明 LAGLayer3 的设计参考来自哪里。路径是当前工作区的绝对路径；新环境中的 AI 必须先检查路径是否存在，再读取或迁移内容。

## 1. 比赛项目：IntelligentAirCombatChallenge

根目录：

~~~text
C:/Users/byai/Documents/Projects/IntelligentAirCombatChallenge
~~~

### FKR v3 部署入口

~~~text
submission/fkr/v3/fkr_v3/policy_rule/agent/main_versus.py
submission/fkr/v3/fkr_v3/policy_rule/agent/red_agent.py
submission/fkr/v3/fkr_v3/policy_rule/agent/blue_agent.py
submission/fkr/v3/fkr_v3/policy_rule/agent/base_agent.py
submission/fkr/v3/fkr_v3/policy_rule/agent/runtime.py
submission/fkr/v3/fkr_v3/policy_rule/agent/coordination.py
submission/fkr/v3/fkr_v3/policy_rule/agent/platform_profile.json
~~~

runtime.py 是比赛提交包中的整合运行时，包含观测编码、状态跟踪、规则、仲裁、战术机动和平台命令编排。它是行为参考和回归参考，不建议原样作为 LAGLayer3 的最终模块。

### FKR v3 飞控

~~~text
submission/fkr/v3/fkr_v3/policy_rule/agent/bsl_aggressive/
envs/model/bsl_aggressive/
~~~

关键文件：

~~~text
controller.py
fast_runtime.py
airspeed.py
source_manifest.json
vendor/assets/actor_best.pt
vendor/competition/flight_control_ppo_delivery/
vendor/scripts/delivery/ppo_windows/
~~~

FKR v3 的 model_manifest.json 明确记录 flight_controller 为 bsl_aggressive。bsl_aggressive 是下层飞控执行器，不是 FKR v3 的首脑策略。

### FKR v3 行为说明

~~~text
submission/fkr/v3/fkr_v3/自测结果/README.md
submission/fkr/v3/fkr_v3/自测结果/local_self_check.json
submission/fkr/v3/fkr_v3/validation/
submission/fkr/v3/fkr_v3/policy_rule/model_manifest.json
~~~

这些文件说明双翼牵制、支援、目标绑定、失联处理、导弹防御和当前已验证范围。测试记录不能自动视为新架构的验收标准，迁移后必须重新验证。

### 比赛工程中的可拆分规则来源

相比提交包内的单体 runtime.py，比赛工程中的规则文件更适合作为迁移参考：

~~~text
envs/rules/schema.py
envs/rules/router.py
envs/rules/tactical.py
envs/rules/approach.py
envs/rules/attack.py
envs/rules/attack_align.py
envs/rules/engagement.py
envs/rules/formation.py
envs/rules/spacing.py
envs/rules/maneuver.py
envs/rules/turn.py
envs/rules/missile_defense.py
envs/rules/sliceback.py
envs/rules/split_s.py
envs/rules/escape.py
envs/rules/post_launch.py
envs/rules/return_heading.py
envs/rules/boundary.py
envs/rules/altitude.py
envs/rules/speed.py
envs/rules/weapons.py
envs/rules/formations/
~~~

这些文件分别代表几何计算、任务规则、导弹防御、边界保护、编队和武器逻辑。迁移时应先理解行为，再映射到 TacticalObservation、DecisionBundle、RoutePrediction 和 ExecutionAction，不要直接复制 import 关系。

## 2. 原 LAGCFC 项目

根目录：

~~~text
C:/Users/byai/Documents/Projects/LAGCFC
~~~

### MAPPO-CFC 与 AeroTAF

~~~text
algorithms/mappoCFC/ppo_trainer.py
algorithms/mappoCFC/online_dataset.py
algorithms/mappoCFC/ppo_policy.py
algorithms/mappoCFC/ppo_AeroTAF.py
algorithms/utils/AeroTAF_ATNN_Fast.py
~~~

这些文件是计划级 CFC 的算法参考，不应直接成为新项目的首脑或执行层实现。当前设计要把反事实对象从底层动作迁移到团队计划、角色、目标、技能和支援关系。

### LAGCFC 仿真和场值

~~~text
envs/JSBSim/envs/multiplecombat_env.py
envs/JSBSim/envs/env_base.py
envs/JSBSim/core/simulatior.py
envs/JSBSim/situation/extractor.py
envs/JSBSim/situation/field.py
envs/JSBSim/tasks/Task4v4/fkr/multiplecombat_task.py
envs/env_wrappers.py
~~~

这些文件提供飞机、导弹、任务、快照和本地 JSBSIM 的参考实现。迁移目标是抽取仿真内核和状态语义，重新设计环境接口。

## 3. 参考和迁移的区别

| 来源 | 可以参考 | 不能直接假设 |
|---|---|---|
| FKR v3 | 规则行为、优先级、战术状态、武器条件 | 单体 runtime 的目录和部署耦合 |
| bsl_aggressive | 飞行意图到 A/E/R/T 的执行链 | 执行层可以直接访问其内部 GRU/QP |
| LAGCFC | MAPPO、CFC、AeroTAF 和 JSBSIM 语义 | 旧环境的奖励、buffer 和训练流程可以原样迁移 |
| 比赛验证记录 | 场景、失败原因和回归方法 | 旧测试通过等于新架构正确 |

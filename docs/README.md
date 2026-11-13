# LAGLayer3 设计文档

这些文档描述 LAGLayer3 的架构基线和当前设计决策。它们服务于后续实现，不代表所有模块已经完成。

## 文档索引

- [架构总览](./architecture.md)：模块职责、依赖方向和运行链路。
- [领域模型](./domain_model.md)：核心对象及其关系。
- [接口契约](./interfaces.md)：仿真、策略、执行层和飞控层的输入输出。
- [仿真设计](./simulation_design.md)：本地 JSBSim、ZK Socket、快照和初始化模式。
- [感知与战争迷雾](./perception_design.md)：雷达、跟踪、通信和观测生成。
- [规则策略框架](./policy_framework.md)：规则、学习策略、混合策略和 FKR v3 的位置。
- [执行层设计](./execution_layer.md)：执行层职责、模型、规则和训练方式。
- [武器与安全设计](./weapons_and_safety.md)：导弹、发射状态机和安全接管。
- [学习与教师数据](./learning_plan.md)：规则教师、模仿学习、PPO 和 CFC 的迁移路径。
- [评估与测试](./evaluation_plan.md)：分层评估指标和验证协议。
- [开发路线](./roadmap.md)：推荐实现顺序。
- [参考源码定位](./reference_sources.md)：比赛项目、FKR v3、bsl_aggressive 和 LAGCFC 的原始路径。
- [迁移映射](./migration_map.md)：参考代码如何映射到 LAGLayer3。
- [新 AI 交接说明](./implementation_handoff.md)：没有当前对话上下文时的阅读顺序和实现约束。
- [待决策问题](./open_questions.md)：不能自行假设的未决设计。
- [重构设计决策](./restructuring_decisions.md)：Q1–Q110 已确认的分层、网络、仿真、训练和数据设计。
- [目录结构设计](./directory_structure.md)：源码模块、Provider、训练图、工具和数据目录边界。

架构性的取舍记录在 [ADR](./adr/README.md) 中。领域术语统一见仓库根目录的 [GLOSSARY.md](../GLOSSARY.md)。

## 当前边界

LAGLayer3 是新项目。LAGCFC 只作为算法、环境和模型的参考来源，后续迁移必须经过接口适配和回归验证，不直接把旧项目的训练脚本作为新项目的运行核心。


---
status: accepted
---

# 按模块和 seam 组织目录

LAGLayer3 按稳定领域模块、运行时 Adapter、可替换 Provider、分层训练图和工具入口组织目录。domain、simulation、perception、planning、weapons、safety、flight_control、policies、execution、learning、runtime 和 evaluation 各自拥有明确 Interface。

旧 LAGCFC 的文件不直接按目录复制。迁移先建立新 Interface，再将 FKR、JSBSim、bsl_aggressive、PPO 和 CFC 作为 Adapter 或 Implementation 接入。训练数据、checkpoint 和运行结果与源码分离，使用 schema 和配置指纹校验兼容性。


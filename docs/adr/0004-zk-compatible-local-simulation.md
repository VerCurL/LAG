---
status: accepted
---

# 使用 ZK 语义契约实现本地 JSBSim

第一版直接运行本地 JSBSim，但统一遵循 ZK 的状态、指令、时间、事件、初始化、快照恢复和终止语义。上层模块只依赖统一仿真接口，不依赖 JSBSim 内部对象。

初始化采用声明式 ScenarioInitialization，终止采用可组合 TerminationProvider。LocalJSBSimAdapter 和未来的 ZKSocketAdapter 使用同一套协议契约。


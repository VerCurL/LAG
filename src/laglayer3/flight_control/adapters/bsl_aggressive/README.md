# 独立激进版 BSL：bsl_aggressive

本目录完整复制自相邻 `bsl/`，含231个源码、配置和资源文件（复制时排除Python缓存）。它不导入原 `envs.model.bsl` 运行链，权重字节保持一致，策略/GRU/配平/QP状态和缓存均独立。

默认配置与修改说明见 [bsl_aggressive.md](../../../docs/bsl_aggressive.md)。

- `controller.py`：独立类 `AggressiveBSLFlightController`，保持40维测量输入和原高层动作维度，传递加速请求。
- `vendor/competition/flight_control_ppo_delivery/default_config.json`：更快航向、坡度和能量时间常数。
- `vendor/scripts/delivery/ppo_windows/guarded_controller.py`：加速动作在QP之前设置0.9油门提议；舵面、变化率限制和原一次QP继续生效。
- `fast_runtime.py`：训练精简路径使用相同油门提议；与完整路径数值一致。
- `source_manifest.json`：保留原来源哈希，更新副本集成哈希和改动登记。副本修改vendor文件后必须更新集成SHA；原目录无需修改。

原始文件来自 `files/BSLPlatform/flight_control`。副本的行为与原版不同，原版飞行验证不能自动视为副本全包线验证。

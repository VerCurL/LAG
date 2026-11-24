# 独立激进版 BSL：bsl_aggressive

本目录从比赛项目复制后，按 LAGLayer3 的完整控制入口裁剪。当前保留 63 个 manifest 文件和必要资源。它不导入原 `envs.model.bsl` 运行链，权重字节保持一致，策略、GRU、配平、QP 状态和缓存均独立。

默认配置与修改说明见 [bsl_aggressive.md](../../../docs/bsl_aggressive.md)。

- `controller.py`：独立类 `AggressiveBSLFlightController`，保持40维测量输入和原高层动作维度，传递加速请求。
- `vendor/competition/flight_control_ppo_delivery/default_config.json`：更快航向、坡度和能量时间常数。
- `vendor/scripts/delivery/ppo_windows/guarded_controller.py`：加速动作在QP之前设置0.9油门提议；舵面、变化率限制和原一次QP继续生效。
- `fast_runtime.py`：精简路径使用相同油门提议；与完整路径数值一致。
- `source_manifest.json`：保留原来源哈希和当前裁剪后的集成哈希。副本修改 vendor 文件后必须更新集成 SHA。

裁剪移除了旧 PPO/MAPPO 训练器、历史环境任务注册、奖励和终止条件注册、人工控制、战斗环境、EAS 研究分支、平台构建与审计脚本。当前入口闭包保留 Actor、40D 控制器、GRU、名义参考、残差组合、动作投影器、必要的能量与几何计算。

原始文件来自 `files/BSLPlatform/flight_control`。副本的行为与原版不同，原版飞行验证不能自动视为副本全包线验证。

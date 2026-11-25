# 独立激进版 BSL：bsl_aggressive

本目录从比赛项目复制后，按 LAGLayer3 的部署控制入口裁剪。它只保留 Actor、GRU、40D 观测、名义控制、残差组合、QP 和每架飞机运行状态。它不导入原 `envs.model.bsl` 运行链，权重字节保持一致。

默认配置与修改说明见 [bsl_aggressive.md](../../../docs/bsl_aggressive.md)。

- `controller.py`：独立类 `AggressiveBSLFlightController`，保持 40 维测量输入和原高层动作维度，传递加速请求。
- `vendor/competition/flight_control_ppo_delivery/default_config.json`：更快航向、坡度和能量时间常数。
- `vendor/scripts/delivery/ppo_windows/guarded_controller.py`：加速动作在QP之前设置0.9油门提议；舵面、变化率限制和原一次QP继续生效。
- 当前只保留部署推理路径。训练、审计和第二套快速运行入口已经移除。
- `source_manifest.json`：保留原来源哈希和当前裁剪后的集成哈希。副本修改 vendor 文件后必须更新集成 SHA。

裁剪移除了训练器、诊断输出、审计脚本、历史环境任务、奖励和终止条件注册、人工控制、战斗环境、EAS 研究分支和第二套快速控制实现。当前入口闭包保留 Actor、40D 控制器、GRU、名义参考、残差组合、动作投影器、必要的能量与几何计算。

原始文件来自 `files/BSLPlatform/flight_control`。副本的行为与原版不同，原版飞行验证不能自动视为副本全包线验证。

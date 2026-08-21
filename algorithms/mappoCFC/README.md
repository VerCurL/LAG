# mappoCFC

新版 `mappoCFC` 使用 `AeroTAF_ATTN_Fast` 预测事实动作和反事实动作对应的未来威胁场、进攻场，并据此重新分配 MAPPO 团队奖励。

## 每轮训练流程

1. 32 个并行环境完成 rollout，保存 MAPPO buffer 和每步 `AeroTAF_snapshot`。
2. 按 `masks` 将 rollout 切分为独立对局，尾部未结束的截断局不进入 AeroTAF 训练集。
3. 32 个环境在本轮 rollout 中产生的所有完整对局都进入阈值拟合和 AeroTAF 训练候选集。
4. 使用 `FieldCalculator` 计算每个时间点的 K 步威胁场和进攻场标签。
5. 在所有完整对局上拟合 detail 阈值，并按以下优先级标注时间点：
   `event > high_change > high_field > stable`。
6. 保留全部前三类样本，并按 `AeroTAF_stable_sample_ratio` 随机采样 stable 样本。
7. 使用不跨越对局边界的历史窗口训练 `AeroTAF_ATTN_Fast`。
8. 一次编码整轮事实轨迹并缓存空间特征和时间 K/V；事实动作及逐架飞机反事实动作共享历史缓存，并在 variant batch 中推理。
9. 根据威胁降低量和进攻提升量计算贡献率。正奖励偏向高贡献飞机，负奖励偏向低贡献飞机；失活飞机不参与 softmax。
10. 保持每个环境、每个时间点的团队奖励总量不变，再执行标准 MAPPO return 计算和 PPO 更新。

## 反事实动作

`CFC_counterfactual_actions` 支持：

- `previous`
- `no_op`
- `invert_maneuver`
- `invert_heading`
- `invert_altitude`
- `invert_velocity`

多个动作使用英文逗号分隔。每次只修改一架己方飞机，其他飞机保持事实动作；所有反事实动作都保留当前时刻的事实发弹维度。配置多个动作时，对各反事实预测取均值后计算该飞机贡献率。

## 主要参数

- `--n-rollout-threads 32`：并行环境数量；所有环境在本轮产生的完整对局都会用于阈值拟合和 AeroTAF 训练。
- `--AeroTAF-spatial-head-num 4`：AeroTAF 每个时间点内部的飞机空间注意力头数。
- `--AeroTAF-history-windows 100`：最大历史窗口。
- `--AeroTAF-kstep 100`：场标签未来长度 K。
- `--AeroTAF-stable-sample-ratio 0.05`：stable 类保留比例。
- `--AeroTAF-mini-batch-size 256`：AeroTAF 训练 batch，独立于 PPO mini-batch。
- `--AeroTAF-inference-batch-size 512`：事实/反事实场推理 batch。
- `--AeroTAF-inference-amp`：在 CUDA 上使用 FP16 混合精度执行缓存式场值推理。
- `--AeroTAF-pretrained-model PATH`：可选的离线 Fast 模型检查点。
- `--CFC-counterfactual-actions previous`：贡献率基线动作集合。
- `--CFC-reward-blend 1.0`：原奖励与 CFC 奖励的混合比例。
- `--CFC-warmup-rollouts 0`：开始使用 CFC 奖励前，仅训练 AeroTAF 的 rollout 数。

旧的单时刻实现保存在 `algorithms/mappoCFC_legacy`，可通过 `--algorithm-name mappoCFC_legacy` 运行。

# utils/logger.py
import csv
import os

class AlgorithmsLogger:
    NEW_CFC_COLUMNS = [
        "AeroTAF_loss",
        "AeroTAF_grad_norm",
        "AeroTAF_threat_loss",
        "AeroTAF_attack_loss",
        "AeroTAF_updates",
        "AeroTAF_train_samples",
        "AeroTAF_train_episodes",
        "AeroTAF_dataset_time",
        "AeroTAF_train_time",
        "AeroTAF_event_points",
        "AeroTAF_high_field_points",
        "AeroTAF_high_change_points",
        "AeroTAF_stable_points",
        "AeroTAF_event_threat_loss",
        "AeroTAF_event_attack_loss",
        "AeroTAF_high_field_threat_loss",
        "AeroTAF_high_field_attack_loss",
        "AeroTAF_high_change_threat_loss",
        "AeroTAF_high_change_attack_loss",
        "AeroTAF_stable_threat_loss",
        "AeroTAF_stable_attack_loss",
        "CFC_applied",
        "CFC_fact_threat_mean",
        "CFC_fact_attack_mean",
        "CFC_threat_delta_mean",
        "CFC_attack_delta_mean",
        "CFC_contribution_mean",
        "CFC_contribution_std",
        "CFC_weight_min",
        "CFC_weight_max",
        "CFC_reward_sum_error",
        "CFC_inference_time",
        "MAPPO_update_time",
    ]

    LEGACY_CFC_COLUMNS = [
        "AeroTAF_loss",
        "AeroTAF_grad_norm",
        "threat_loss",
        "attack_loss",
        "fact_threat_mean",
        "fact_attack_mean",
        "threat_delta_mean",
        "attack_delta_mean",
        "contribution_mean",
        "contribution_std",
        "weight_min",
        "weight_max",
    ]

    def __init__(self, save_dir, filename="training_log.csv", algorithm_name="ppo"):
        # 自动创建目录（如果不存在）
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.filepath = os.path.join(save_dir, filename)

        # 判断是否是首次写入（用于写表头）
        self.first_write = not os.path.exists(self.filepath)

        # 以追加模式打开文件（如果不存在会自动创建文件）
        self.csv_file = open(self.filepath, 'a', newline='')
        self.writer = csv.writer(self.csv_file)

        # 记录当前所记录的算法名称
        self.algorithm_name = algorithm_name

    def _write_header(self, info):
        """
        在第一次 log 时写入表头
        """
        # 写入表头
        if self.first_write:
            header = [
                "episode",              # 训练的episode数
                "policy_loss",          # 策略损失
                "policy_entropy_loss",  # 策略熵损失
                "value_loss",           # 价值损失
                "actor_grad_norm",      # 策略网络梯度
                "critic_grad_norm",     # 价值网络梯度
                "average_episode_rewards",      # 每局奖励
                "env_time",             # 虚拟仿真时间
                "train_time",           # 训练时间
                "fps",                  # 每秒帧数
            ]
            # 按照算法添加额外列
            if self.algorithm_name in ["mappo"]:
                pass
            elif self.algorithm_name == "mappoCFC":
                header += self.NEW_CFC_COLUMNS
            elif self.algorithm_name == "mappoCFC_legacy":
                header += self.LEGACY_CFC_COLUMNS
            self.writer.writerow(header)

        self.csv_file.flush()

    def log(self, episode, data):
        """
        写入每一行数据
        """
        expert_usage = data.get("expert_usage", None)

        # 在第一次 log 时写入表头（因为这时才能拿到 expert_usage 的长度）
        if self.first_write:
            info = {}

            # expert_usage 必须可迭代
            self._write_header(info)
            self.first_write = False

        # 写入数据
        row = [
            episode,
            data.get("policy_loss", None),
            data.get("policy_entropy_loss", None),
            data.get("value_loss", None),
            data.get("actor_grad_norm", None),
            data.get("critic_grad_norm", None),
            data.get("average_episode_rewards", None),
            data.get("env_time", None),
            data.get("train_time", None),
            data.get("fps", None),

        ]

        if self.algorithm_name == "mappoCFC":
            row += [data.get(column, None) for column in self.NEW_CFC_COLUMNS]
        elif self.algorithm_name == "mappoCFC_legacy":
            row += [data.get(column, None) for column in self.LEGACY_CFC_COLUMNS]

        self.writer.writerow(row)
        self.csv_file.flush()

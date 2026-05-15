# utils/logger.py
import csv
import os

class AlgorithmsLogger:
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
            if self.algorithm_name in ["mappo-v1"]:
                pass
            elif self.algorithm_name in ["mappoMoE-v1"]:
                self.expert_usage_len = info["expert_usage_len"]
                header += [
                    "expert_out_loss",      # 专家交叉损失
                    "gate_entropy",
                    "gate_max_prob",
                ]
                # 添加 expert_usage 的展开列
                header += [f"expert_usage_{i + 1}" for i in range(self.expert_usage_len)]
            elif self.algorithm_name in ["mappoPCAN-v1"]:
                header += [
                    "pcan_loss",
                    "pcan_grad_norm",
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
            elif self.algorithm_name in ["mappoPCAN-v2"]:
                header += [
                    "rewards_pred_loss",    # 奖励预测损失
                    "credit_diag_mean",     # credit对角线均值
                    "credit_entropy",       # credit行熵的均值
                ]
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
            if self.algorithm_name in ["mappoMoE-v1"]:
                info["expert_usage_len"] = len(expert_usage)

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

        if self.algorithm_name in ["mappoMoE-v1"]:
            row += [
                data.get("expert_out_loss", None),
                data.get("gate_entropy", None),
                data.get("gate_max_prob", None),
            ]
            # 写入 expert_usage 展开
            if expert_usage is None:
                row += [None] * self.expert_usage_len
            else:
                row += list(expert_usage)
        elif self.algorithm_name in ["mappoPCAN-v1"]:
            row += [
                data.get("pcan_loss", None),
                data.get("pcan_grad_norm", None),
                data.get("threat_loss", None),
                data.get("attack_loss", None),
                data.get("fact_threat_mean", None),
                data.get("fact_attack_mean", None),
                data.get("threat_delta_mean", None),
                data.get("attack_delta_mean", None),
                data.get("contribution_mean", None),
                data.get("contribution_std", None),
                data.get("weight_min", None),
                data.get("weight_max", None),
            ]
        elif self.algorithm_name in ["mappoPCAN-v2"]:
            row += [
                data.get("rewards_pred_loss", None),
                data.get("credit_diag_mean", None),
                data.get("credit_entropy", None),
            ]

        self.writer.writerow(row)
        self.csv_file.flush()

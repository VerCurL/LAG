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
            if self.algorithm_name in ["ppo", "mappo"]:
                header = [
                    "episode",                      # 训练的episode数
                    "policy_loss",                  # 策略损失
                    "policy_entropy_loss",          # 策略熵损失
                    "value_loss",                   # 价值损失
                    "average_episode_rewards",      # 每局奖励
                    "fps",                          # 每秒帧数

                    # "value_mean",                   # 价值均值
                    # "value_std",                    # 价值标准差
                    # "approx_kl",                    # 近似KL散度
                    # "win_rate"                      # 胜率
                ]
                self.writer.writerow(header)
            elif self.algorithm_name in ["ppoMoE", "mappoMoE"]:
                self.expert_usage_len = info["expert_usage_len"]
                header = [
                    "episode",
                    "policy_loss",
                    "policy_entropy_loss",
                    "value_loss",
                    "average_episode_rewards",
                    "fps",
                    "gate_entropy",
                    "gate_max_prob",
                ]
                # 添加 expert_usage 的展开列
                header += [f"expert_usage_{i + 1}" for i in range(self.expert_usage_len)]
                # # 继续后面的指标
                # header += [
                #     "value_mean",
                #     "value_std",
                #     "approx_kl",
                #     "win_rate"
                # ]
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
            if self.algorithm_name in ["ppoMoE", "mappoMoE"]:
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
            data.get("average_episode_rewards", None),
            data.get("fps", None),
            data.get("gate_entropy", None),
            data.get("gate_max_prob", None),
        ]

        if self.algorithm_name in ["ppoMoE", "mappoMoE"]:
            # 写入 expert_usage 展开
            if expert_usage is None:
                row += [None] * self.expert_usage_len
            else:
                row += list(expert_usage)

        # # 写后续字段
        # row += [
        #     data.get("value_mean", None),
        #     data.get("value_std", None),
        #     data.get("approx_kl", None),
        #     data.get("win_rate", None)
        # ]

        self.writer.writerow(row)
        self.csv_file.flush()

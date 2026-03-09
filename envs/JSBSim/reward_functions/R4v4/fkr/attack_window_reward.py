import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R, get_near_offset_of_multi_air

class AttackWindowReward(BaseRewardFunction):
    """
    攻击窗口奖励函数，基于攻击窗口获取优势站位的奖励
    在原实现基础上做小幅修改：
    1. 将 score 做归一化，降低不同项之间的量级差异
    2. 将原先线性放大的增量奖励改成 tanh 有界形式，减少单步奖励震荡
    3. 适当减小静态窗口分数奖励
    4. 多目标聚合由 p=3 Power Mean 改为 softmax 加权平均，降低目标切换抖动
    5. 其余结构尽量保持不变
    """
    def __init__(self, config):
        super().__init__(config)
        # 原始评分权重保持不变
        self.weight_attack_angle = 10
        self.weight_distance = 4
        self.weight_threat_angle = 5

        # 奖励项权重：改为更平滑、更小的量级
        self.weight_reward_grad = 4.0
        self.weight_reward_score = 0.05

        # tanh 增益
        self.grad_tanh_gain = 1.2

        # softmax 聚合温度，越大越偏向高分目标
        self.score_softmax_tau = 0.3

        # 单目标理论最大绝对分数，用于归一化
        # max = 10 * 1 + 4 * 1 - 5 * (-1) = 19
        self.score_norm = 19.0

        # 记录上局得分
        self.pre_scores = {}

    def reset(self, task, env):
        self.pre_scores.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])

        reward = 0.0
        if agent.num_left_missiles > 0:
            scores = []
            states = {}  # 记录所有存活敌人的位置

            for enm in agent.enemies:
                if enm.is_alive:
                    enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
                    AO, TA, R = get_AO_TA_R(enm_feature, ego_feature)
                    score = self.get_score(task, AO, TA, R / 1000.0)

                    # 归一化到大致 [-1, 1]
                    score = np.clip(score / self.score_norm, -1.0, 1.0)

                    scores.append(score)
                    states[enm.uid] = enm.get_position()

            if len(states) > 0:
                # 最近敌人相对加权质心的偏移位置
                ego_position = agent.get_position()
                enm_positions = np.array([position for position in states.values()])
                near_offset_position, near_offset_vector = get_near_offset_of_multi_air(ego_position, enm_positions)

                # 计算我机和偏移位置的距离
                ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
                center_feature = np.hstack([near_offset_position, np.array([0, 0, 0])])
                _, _, R_offset = get_AO_TA_R(ego_feature, center_feature)
                R_offset /= 1000.0

                # softmax 加权平均聚合，替代 p=3 Power Mean，减少抖动
                scores = np.asarray(scores, dtype=np.float64)
                logits = self.score_softmax_tau * scores
                logits = logits - np.max(logits)  # 数值稳定
                weights = np.exp(logits)
                weights = weights / (np.sum(weights) + 1e-8)
                result_score = float(np.sum(weights * scores))

                distance_max = 1.3 * task.max_missile_attack_distance

                if agent_id not in self.pre_scores:
                    self.pre_scores[agent_id] = result_score

                delta_score = result_score - self.pre_scores[agent_id]

                # 改为有界增量奖励，避免单步过大震荡
                reward_grad = self.weight_reward_grad * np.tanh(self.grad_tanh_gain * delta_score)

                # 保留静态窗口奖励，但减小权重
                reward_score = self.weight_reward_score * max(0.0, result_score)

                reward = (R_offset <= distance_max) * (reward_grad + reward_score)

                self.pre_scores[agent_id] = result_score

            # 被攻击时优先规避
            if len(agent.check_all_missile_warning()) > 0:
                reward = 0.0

        reward = float(reward)
        return self._process(reward, agent_id)

    def get_score(self, task, AO, TA, R):
        def get_angle_score(angle):
            max_AO = task.max_missile_attack_AO
            return ((angle <= max_AO) * (1 - angle / max_AO) +
                    (angle > max_AO) * (-(angle - max_AO) / (np.pi - max_AO)))

        def get_distance_score(distance):
            min_R = 0.8 * task.min_missile_attack_distance
            max_R = 1.2 * task.max_missile_attack_distance
            mean_R = (min_R + max_R) / 2.0
            return (
                (distance < min_R) * (-1 + np.exp(-(min_R - distance))) +
                (distance > max_R) * (-1 + np.exp(-(distance - max_R))) +
                (min_R <= distance <= max_R) * (1 - (2 * (distance - mean_R) / (max_R - min_R)) ** 2)
            )

        score = (
            self.weight_attack_angle * get_angle_score(AO) +
            self.weight_distance * get_distance_score(R) -
            self.weight_threat_angle * get_angle_score(np.pi - TA)
        )
        return float(score)
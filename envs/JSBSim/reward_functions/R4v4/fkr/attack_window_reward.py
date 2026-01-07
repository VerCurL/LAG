import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R, get_near_offset_of_multi_air

class AttackWindowReward(BaseRewardFunction):
    """
    攻击窗口奖励函数，基于攻击窗口获取优势站位的奖励
    引入了评分系统，约优势的位置评分越高
    - 奖励计算采用评分变化获得，在评分变高时基于奖励
    - 在规避时，不强制寻找优势站位，奖励为 0
    """
    def __init__(self, config):
        super().__init__(config)
        # 定义奖励权重
        self.weight_attack_angle = 10
        self.weight_distance = 4
        self.weight_threat_angle = 5
        self.weight_reward_grad = 20
        self.weight_reward_score = 0.1

        # 记录上局得分
        self.pre_scores = {}

    def reset(self, task, env):
        self.pre_scores.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])

        # 如果飞机还有导弹，则基于攻击窗口寻找优势站位
        reward = 0.0
        if agent.num_left_missiles > 0:
            # 遍历敌机，获得针对每个敌机的优势站位得分
            scores = []
            states = {}         # 记录所有敌人的态势信息
            for enm in agent.enemies:
                enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
                AO, TA, R = get_AO_TA_R(enm_feature, ego_feature)
                if enm.is_alive:
                    scores.append(self.get_score(task, AO, TA, R / 1000))
                    states[enm.uid] = enm.get_position()
                else:
                    scores.append(0.0)

            if len(states) > 0:
                # 获得最近敌人相对加权质心的偏移位置
                ego_position = agent.get_position()
                enm_positions = np.array([position for position in states.values()])
                near_offset_position, near_offset_vector = get_near_offset_of_multi_air(ego_position, enm_positions)

                # 获得我机和偏移位置的参数关系
                ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
                center_feature = np.hstack([near_offset_position, np.array([0, 0, 0])])
                _, _, R_offset = get_AO_TA_R(ego_feature, center_feature)
                R_offset /= 1000.0

                # 使用 p=3 的 Power Mean 来获得综合优势站位得分
                risks = -np.asarray(scores, dtype=np.float64)
                result_score = -np.cbrt(np.mean(risks ** 3))

                # 计算奖励函数，要求飞机倾向于获得高得分
                distance_max = 1.3 * task.max_missile_attack_distance
                if agent_id not in self.pre_scores:
                    self.pre_scores[agent_id] = result_score

                reward = (R_offset <= distance_max) * (self.weight_reward_grad * np.maximum(0, result_score - self.pre_scores[agent_id]) +
                                                       self.weight_reward_score * np.maximum(0, result_score))
                self.pre_scores[agent_id] = result_score

            # 如果被攻击，则不管优势站位，先躲避导弹活下来
            if len(agent.check_all_missile_warning()) > 0:
               reward = 0.0

        reward = float(reward)
        return self._process(reward, agent_id)

    def get_score(self, task, AO, TA, R):
        def get_angle_score(angle):
            max_AO = task.max_missile_attack_AO
            return (angle <= max_AO) * (1 - angle / max_AO) + (angle > max_AO) * (-(angle - max_AO) / (np.pi - max_AO))
        def get_distance_score(distance):
            min_R = task.min_missile_attack_distance
            max_R = task.max_missile_attack_distance
            meal_R = (min_R + max_R) / 2
            return ((distance < min_R) * (-1 + np.exp(-(min_R - distance))) + (distance > max_R) * (-1 + np.exp(-(distance - max_R))) +
                    (min_R <= distance <= max_R) * (1 - (2 * (distance - meal_R) / (max_R - min_R)) ** 2))

        score = (self.weight_attack_angle * get_angle_score(AO) +
                self.weight_distance * get_distance_score(R) -
                self.weight_threat_angle * get_angle_score(np.pi - TA))
        return score
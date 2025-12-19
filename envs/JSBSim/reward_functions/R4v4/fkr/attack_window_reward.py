import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R

class AttackWindowReward(BaseRewardFunction):
    """
    攻击窗口奖励函数，基于攻击窗口获取优势站位的奖励
    引入了评分系统，约优势的位置评分越高
    - 奖励计算采用评分变化获得，在评分变高时基于奖励
    - 在规避时，不强制寻找优势站位，奖励为 0
    """
    def __init__(self, config):
        super().__init__(config)
        # 定义攻击窗口
        self.min_missile_attack_distance = getattr(self.config, "min_missile_attack_distance", 4000) / 1000     # unit: km
        self.max_missile_attack_distance = getattr(self.config, "max_missile_attack_distance", 14000) / 1000    # unit: km
        self.max_missile_attack_AO = np.radians(getattr(self.config, "missile_attack_AO", 60))                  # unit: rad

        # 定义奖励权重
        self.weight_attack_angle = 10
        self.weight_distance = 4
        self.weight_threat_angle = 5

        # 记录上局得分
        self.pre_scores = {}
        self.pre_R_min = {}

    def reset(self, task, env):
        self.pre_scores.clear()
        self.pre_R_min.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        reward = 0.0
        agent = env.agents[agent_id]
        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])

        # 如果飞机还有导弹，则基于攻击窗口寻找优势站位
        if agent.num_left_missiles > 0:
            # 遍历敌机，获得针对每个敌机的优势站位得分
            scores = []
            Rs = []
            for enm in agent.enemies:
                enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
                AO, TA, R = get_AO_TA_R(enm_feature, ego_feature)
                if enm.is_alive:
                    scores.append(self.get_score(AO, TA, R / 1000))
                    Rs.append(R / 1000)

            # 使用soft-min来获得综合优势站位得分
            if len(scores) > 0:
                kappa = 5
                scores = np.asarray(scores, dtype=np.float64)
                x = -kappa * scores
                x_max = np.max(x)
                result_score = -(x_max + np.log(np.sum(np.exp(x - x_max))))

            # 计算奖励函数，要求飞机倾向于获得高得分
            if len(Rs) > 0:
                R_min = min(Rs)
                distance_max = 1.2 * self.max_missile_attack_distance
                if agent_id not in self.pre_scores:
                    self.pre_scores[agent_id] = result_score
                else:
                    reward = (R_min <= distance_max) * 20 * (result_score - self.pre_scores[agent_id])
                    self.pre_scores[agent_id] = result_score

                if agent_id not in self.pre_R_min:
                    self.pre_R_min[agent_id] = R_min
                reward += (R_min > distance_max) * 20 * (-np.exp(-R_min / distance_max) + (R_min - self.pre_R_min[agent_id]))
                self.pre_R_min[agent_id] = R_min

            # 如果被攻击，则不管优势站位，先躲避导弹活下来
            if len(agent.check_all_missile_warning()) > 0:
               reward = 0.0

        # if agent_id == "A0100":
        # print("[attack_window] reward: ", reward)
            # print("                pre_scores: ", self.pre_scores[agent_id])
            # print("                scores: ", result_score)

        return self._process(reward, agent_id)

    def get_score(self, AO, TA, R):
        def get_angle_score(angle):
            max_AO = self.max_missile_attack_AO
            return (angle <= max_AO) * (1 - angle / max_AO) + (angle > max_AO) * (-(angle - max_AO) / (np.pi - max_AO))
        def get_distance_score(distance):
            min_R = self.min_missile_attack_distance
            max_R = self.max_missile_attack_distance
            meal_R = (min_R + max_R) / 2
            return ((distance < min_R) * (-1 + np.exp(-(min_R - distance))) + (distance > max_R) * (-1 + np.exp(-(distance - max_R))) +
                    (min_R <= distance <= max_R) * (1 - (2 * (distance - meal_R) / (max_R - min_R)) ** 2))

        reward = (self.weight_attack_angle * get_angle_score(AO) +
                self.weight_distance * get_distance_score(R) -
                self.weight_threat_angle * get_angle_score(np.pi - TA))
        return reward
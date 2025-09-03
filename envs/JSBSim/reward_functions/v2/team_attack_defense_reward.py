from typing import List, Dict

import numpy as np
import math

from envs.JSBSim.core.simulatior import AircraftSimulator
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R  # 导入角度计算工具

class TeamAttackDefenseReward(BaseRewardFunction):
    """
    攻击和防御的相关奖励
    """
    def __init__(self, config):
        super().__init__(config)
        # 最佳射距（单位：km）
        self.shoot_opt_dist = getattr(self.config, f'{self.__class__.__name__}_opt_dist', 12)
        # 理想攻击角度（单位：弧度 ）
        self.attack_angle = getattr(self.config, f'{self.__class__.__name__}_attack_angle', 45)
        self.attack_angle = math.radians(self.attack_angle)

        # 每个敌机对应友方所有飞机角色分配（射手shooter、压侧/拖引assist）
        self.enemies_allocation = {}    # type: Dict[str, Dict[int, List[str]]]
        # 每个敌机对应友方所有飞机得分记录
        self.score_values = {}          # type: Dict[str, Dict[str, float]]
        # 我机对每个敌机的角色定义（shooter/assist）
        self.ego_self_role = {}         # type: Dict[str, int]

        # 存储上一时刻和敌机的距离的变量
        self.R_pre_time = {}                # type: Dict[str, float]

    def reset(self, task, env):
        self.enemies_allocation.clear()
        self.score_values.clear()
        self.ego_self_role.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        计算团队的战术奖励
        """
        # 给团队角色进行分类（射手shooter和压侧/拖引assist）
        self.allocation(env.agents[agent_id])

        # 计算各项奖励值
        shooter_attack_reward = 0
        assist_approach_reward = 0

        # 获取自身状态：位置(北,东,下) + 速度(北,东,下)
        ego_feature = np.hstack([env.agents[agent_id].get_position(), env.agents[agent_id].get_velocity()])

        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

            if self.ego_self_role[enm.uid] == 1:    # shooter
                shooter_attack_reward += 10. * self.shooter_attack_function(env.agents[agent_id], enm)
            elif self.ego_self_role[enm.uid] == 0:  # assist
                assist_approach_reward += 2. * self.assist_approach_function(enm, R / 1000)

            self.R_pre_time[enm.uid] = R / 1000

        new_reward = shooter_attack_reward + assist_approach_reward
        self.reset(task, env)

        return self._process(new_reward, agent_id)

    def allocation(self, ego_self):
        """
        对我方所有的飞机分配，定位身份
        """
        # 计算对于每个敌机对应的所有我方飞机的可射性得分
        enemies_scores = {}
        for enm in ego_self.enemies:
            enemies_scores[enm.uid] = {}
            enemies_scores[enm.uid][ego_self.uid] = self.shoot_score(ego_self, enm)
            self.enemies_allocation[enm.uid] = {0:[], 1:[]}     # 0:assist, 1:shooter
        for partner in ego_self.partners:
            for enm in partner.enemies:
                enemies_scores[enm.uid][partner.uid] = self.shoot_score(partner, enm)

        # 根据得分给我方飞机分配角色
        for enm_id, scores in enemies_scores.items():
            self.score_values[enm_id] = {}

            for ego_id, score in scores.items():
                self.score_values[enm_id][ego_id] = score

            # 将评分超过0.1的判定为shooter，低于0.1的判定为assist
            self.enemies_allocation[enm_id][0] = \
                [ego_id for ego_id, softmax_score in self.score_values[enm_id].items() if softmax_score < 0.1]
            self.enemies_allocation[enm_id][1] = \
                [ego_id for ego_id, softmax_score in self.score_values[enm_id].items() if softmax_score >= 0.1]

        # 返回我机的所有身份信息
        enm_ids = []
        for enm in ego_self.enemies:
            enm_ids.append(enm.uid)

        for enm in ego_self.enemies:
            self.ego_self_role[enm.uid] = 1 if ego_self.uid in self.enemies_allocation[enm.uid][1] else 0

    def shoot_score(self, ego, enm):
        """
        计算“就位/可射性”评分
        """
        ego_feature = np.hstack([ego.get_position(), ego.get_velocity()])
        enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
        AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
        return np.exp(-(R / 1000 - self.shoot_opt_dist) ** 2 / 4) * ((1 + np.cos(TA)) / 2) * np.exp(- AO ** 2 / (np.pi / 3) ** 2)

    def shooter_attack_function(self, ego_self, enm):
        """
        射手进攻咬尾奖励设置
        """
        return self.score_values[enm.uid][ego_self.uid]

    def assist_approach_function(self, enm, R):
        """
        辅助机需要靠近敌机，避免只在远处分散。
        """
        # 如果已有射手存在，则辅助只做合围，不加此奖励
        if 1 in self.ego_self_role.values() or enm.uid not in self.R_pre_time:
            return 0

        # 光滑负奖励：在 R > R_opt 时变负
        return (R > self.shoot_opt_dist * 1.2) * np.tanh((R - self.shoot_opt_dist * 1.2) ** 2 - (self.R_pre_time[enm.uid] - self.shoot_opt_dist * 1.2) ** 2 - 50)
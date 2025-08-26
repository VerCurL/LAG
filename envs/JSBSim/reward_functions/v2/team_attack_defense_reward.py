from typing import List, Dict

import numpy as np
import math

from networkx.generators import ego
from wandb import agent

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
        self.enemies_allocation = {}    # type: Dict[AircraftSimulator, Dict[str, List[AircraftSimulator]]]
        # 每个敌机对应友方所有飞机得分记录
        self.score_values = {}          # type: Dict[AircraftSimulator, Dict[AircraftSimulator, float]]
        # 我机对每个敌机的角色定义（shooter/assist）
        self.ego_self_role = {}         # type: Dict[AircraftSimulator, str]

    def get_reward(self, task, env, agent_id):
        """
        计算团队的战术奖励
        """
        # 给团队角色进行分类（射手shooter和压侧/拖引assist）
        self.allocation(env.agents[agent_id])

        # 计算奖励值
        new_reward = 0

        # 获取自身状态：位置(北,东,下) + 速度(北,东,下)
        ego_feature = np.hstack([env.agents[agent_id].get_position(), env.agents[agent_id].get_velocity()])

        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

            if self.ego_self_role[enm] == 'shooter':
                new_reward += self.shooter_attack_function(env.agents[agent_id], enm)
            elif self.ego_self_role[enm] == 'assist':
                new_reward += self.assist_pincer_function(enm)

        return self._process(new_reward, agent_id)


    def allocation(self, ego_self):
        """
        对我方所有的飞机分配，定位身份
        """
        # 计算对于每个敌机对应的所有我方飞机的可射性得分
        enemies_scores = {}
        for enm in ego_self.enemies:
            enemies_scores[enm] = {}
            enemies_scores[enm][ego_self] = self.shoot_score(ego_self, enm)
            self.enemies_allocation[enm] = {'shooters':[], 'assists':[]}
        for partner in ego_self.partners:
            for enm in partner.enemies:
                enemies_scores[enm][partner] = self.shoot_score(partner, enm)

        # 根据得分给我方飞机分配角色
        for enm, scores in enemies_scores.items():
            sum_exp = np.sum(np.exp(scores.values()))
            for ego, score in scores.items():
                self.score_values[enm] = {}
                self.score_values[enm][ego] = np.exp(score) / sum_exp

            # 将评分超过0.6的判定为shooter，低于0.6的判定为assist
            self.enemies_allocation[enm]['shooters'] = \
                [ego for ego, softmax_score in self.score_values[enm].items() if softmax_score >= 0.6]
            self.enemies_allocation[enm]['assists'] = \
                [ego for ego, softmax_score in self.score_values[enm].items() if softmax_score < 0.6]

        # 返回我机的所有身份信息
        for enm in ego_self.enemies:
            self.ego_self_role[enm] = 'shooter' if ego_self in self.enemies_allocation[enm]['shooters'] else 'assist'

    def shoot_score(self, ego, enm):
        """
        计算“就位/可射性”评分
        """
        ego_feature = np.hstack([ego.get_position(), ego.get_velocity()])
        enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
        AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
        return np.exp(-(R - self.shoot_opt_dist) ** 2 / 4) * ((1 + np.cos(TA)) / 2) * np.exp(- AO ** 2 / (np.pi / 3) ** 2)

    def shooter_attack_function(self, ego_self, enm):
        """
        射手进攻咬尾奖励设置
        """
        return self.score_values[enm][ego_self]

    def assist_pincer_function(self, enm):
        """
        压侧形成合围攻势的奖励函数
        """
        # 计算每个敌机对战的所有压侧飞机的方位奖励
        enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
        assists = self.enemies_allocation[enm]['assists']

        # 计算该敌机对应的所有团队飞机的平均相对位置得分
        position = 0
        for assist in assists:
            assist_feature = np.hstack([assist.get_position(), assist.get_velocity()])
            _, _, _, side_flag = get_AO_TA_R(assist_feature, enm_feature, return_side=True)
            position += side_flag
        position /= len(assists)

        return 1. - np.abs(position)


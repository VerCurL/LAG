import os
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
        self.R_pre_time = {}                # type: Dict[str, Dict[str, float]]

        # 设置评分系统的阈值
        self.score_threshold = 0.35

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
        self.allocation(env, agent_id)
        if agent_id not in self.R_pre_time:
            self.R_pre_time[agent_id] = {}

        # 各项奖励值初始化
        shooter_attack_reward = 0
        approach_reward = 0
        runner_escape_reward = 0
        assist_to_shooter_reward = 0

        # 获取自身状态：位置(北,东,下) + 速度(北,东,下)
        ego_feature = np.hstack([env.agents[agent_id].get_position(), env.agents[agent_id].get_velocity()])

        # 计算各项奖励
        for enm in env.agents[agent_id].enemies:
            # 获得和敌机的状态
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            R /= 1000

            # 无导弹飞机逃跑的奖励
            if self.ego_self_role[enm.uid] == 2:
                runner_escape_reward += 20. * self.runner_escape_function(agent_id, enm.uid, AO, R)
            # 有导弹飞机战斗的奖励
            else:
                # 射手专属奖励
                if self.ego_self_role[enm.uid] == 1:    # shooter
                    shooter_attack_reward += 10. * self.shooter_attack_function(env.agents[agent_id], enm)
                # 辅助专属奖励
                elif self.ego_self_role[enm.uid] == 0:
                    assist_to_shooter_reward += 4. * self.assist_to_shooter_function(env.agents[agent_id], enm)
                # 有导弹飞机的通用奖励
                approach_reward += 2. * self.approach_function(agent_id, enm.uid, AO, R)

            # 记录上一时刻和敌机的距离
            self.R_pre_time[agent_id][enm.uid] = R

        new_reward = shooter_attack_reward + assist_to_shooter_reward + approach_reward + runner_escape_reward

        self.reset(task, env)
        return self._process(new_reward, agent_id)

    def allocation(self, env, agent_id):
        """
        对我方所有的飞机分配，定位身份
        """
        # 计算对于每个敌机对应的所有我方飞机的可射性得分
        ego_self = env.agents[agent_id]
        for enm in ego_self.enemies:
            self.score_values[enm.uid] = {}
            self.score_values[enm.uid][ego_self.uid] = self.shoot_score(ego_self, enm)
            self.enemies_allocation[enm.uid] = {0:[], 1:[], 2:[]}     # 0: assist, 1: shooter, 2: runner
        for partner in ego_self.partners:
            for enm in partner.enemies:
                self.score_values[enm.uid][partner.uid] = self.shoot_score(partner, enm)

        # 根据得分给我方飞机分配角色
        for enm_id, scores in self.score_values.items():
            # 将装有导弹的飞机评分超过阈值的判定为shooter，低于阈值的判定为assist，没有导弹的飞机判定为runner
            for ego_id, score in scores.items():
                if env.agents[ego_id].num_remaining_missiles == 0:
                    self.enemies_allocation[enm_id][2].append(ego_id)
                elif score > self.score_threshold:
                    self.enemies_allocation[enm_id][1].append(ego_id)
                elif score <= self.score_threshold:
                    self.enemies_allocation[enm_id][0].append(ego_id)

        # 返回我机的所有身份信息
        for enm in ego_self.enemies:
            if ego_self.uid in self.enemies_allocation[enm.uid][0]:
                self.ego_self_role[enm.uid] = 0
            elif ego_self.uid in self.enemies_allocation[enm.uid][1]:
                self.ego_self_role[enm.uid] = 1
            else:
                self.ego_self_role[enm.uid] = 2

    def shoot_score(self, ego, enm):
        """
        计算“就位/可射性”评分
        """
        ego_feature = np.hstack([ego.get_position(), ego.get_velocity()])
        enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
        AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
        return np.exp(-(R / 1000 - self.shoot_opt_dist) ** 2 / 40) * ((1 + np.cos(TA)) / 2) * ((1 + np.cos(AO)) / 2)

    def shooter_attack_function(self, ego_self, enm):
        """
        射手进攻咬尾奖励设置
        """
        return self.score_values[enm.uid][ego_self.uid]

    def assist_to_shooter_function(self, ego_self, enm):
        """
        辅助要尽可能寻找优势占位
        """
        return self.score_values[enm.uid][ego_self.uid] - self.score_threshold

    def approach_function(self, ego_id, enm_id, AO, R):
        """
        飞机需要靠近敌机，避免只在远处分散。
        """
        if enm_id not in self.R_pre_time[ego_id]:
            return 0

        # 当距离过远，则让飞机有个靠近的趋势
        return (R > self.shoot_opt_dist * 1.25) * (np.tanh(self.R_pre_time[ego_id][enm_id] - R) + np.cos(AO))

    def runner_escape_function(self, ego_id, enm_id, AO, R):
        """
        无导弹逃跑的奖励
        """
        return np.tanh(R - self.R_pre_time[ego_id][enm_id]) - np.cos(AO)
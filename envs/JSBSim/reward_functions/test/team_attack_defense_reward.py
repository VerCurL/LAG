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
        # print("==========================")
        # print("self.enemies_allocation: ", self.enemies_allocation)
        # print("self.score_values: ", self.score_values)
        # print("self.ego_self_role: ", self.ego_self_role)
        # print("==========================")

        # 计算各项奖励值
        shooter_attack_reward = 0
        shooter_increase_reward = 0
        assist_pincer_reward = 0
        assist_approach_reward = 0

        # 获取自身状态：位置(北,东,下) + 速度(北,东,下)
        ego_feature = np.hstack([env.agents[agent_id].get_position(), env.agents[agent_id].get_velocity()])


        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

            if self.ego_self_role[enm.uid] == 1:    # shooter
                shooter_attack_reward += 5. * self.shooter_attack_function(env.agents[agent_id], enm)
                shooter_increase_reward += 2. * self.shooter_increase_function()
            elif self.ego_self_role[enm.uid] == 0:  # assist
                assist_pincer_reward += 2. * self.assist_pincer_function(env, enm)
                assist_approach_reward += 2. * self.assist_approach_function(enm, R / 1000)

            self.R_pre_time[enm.uid] = R / 1000

        # file_path = "/mnt/d/FastProjects/ModelFlight/LAG/scripts/results/log/reward/team_attack_defense_reward.txt"
        # with open(file_path, "a", encoding="utf-8") as f:
        #     f.write(str(shooter_attack_reward) + ", " + str(shooter_increase_reward) + ", "
        #             + str(assist_pincer_reward) + ", " + str(assist_approach_reward) + "\n")

        new_reward = shooter_attack_reward + shooter_increase_reward + assist_pincer_reward + assist_approach_reward
        self.reset(task, env)

        # reward_child = {"shooter_attack_reward": 5. * shooter_attack_reward, "shooter_increase_reward": 2. * shooter_increase_reward,
        #                 "assist_pincer_reward": 2. * assist_pincer_reward, "assist_approach_reward": 2. * assist_approach_reward}

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
        # print(enemies_scores)

        # 根据得分给我方飞机分配角色
        for enm_id, scores in enemies_scores.items():
            self.score_values[enm_id] = {}

            sum_exp = 0
            for score in scores.values():
                sum_exp += np.exp(score)

            for ego_id, score in scores.items():
                self.score_values[enm_id][ego_id] = np.exp(score) / sum_exp

            # 将评分超过0.6的判定为shooter，低于0.6的判定为assist
            self.enemies_allocation[enm_id][0] = \
                [ego_id for ego_id, softmax_score in self.score_values[enm_id].items() if softmax_score < 0.6]
            self.enemies_allocation[enm_id][1] = \
                [ego_id for ego_id, softmax_score in self.score_values[enm_id].items() if softmax_score >= 0.6]

        # 返回我机的所有身份信息
        enm_ids = []
        for enm in ego_self.enemies:
            enm_ids.append(enm.uid)
        # print("self.enm_ids: ", enm_ids)

        for enm in ego_self.enemies:
            self.ego_self_role[enm.uid] = 1 if ego_self in self.enemies_allocation[enm.uid][1] else 0

    def shoot_score(self, ego, enm):
        """
        计算“就位/可射性”评分
        """
        ego_feature = np.hstack([ego.get_position(), ego.get_velocity()])
        enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
        AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
        # print("AO = ", AO, ", TA = ", TA, ", R = ", R)
        return np.exp(-(R / 1000 - self.shoot_opt_dist) ** 2 / 4) * ((1 + np.cos(TA)) / 2) * np.exp(- AO ** 2 / (np.pi / 3) ** 2)

    def shooter_attack_function(self, ego_self, enm):
        """
        射手进攻咬尾奖励设置
        """
        return self.score_values[enm.uid][ego_self.uid]

    def shooter_increase_function(self):
        """
        尽量增加射手的数量
        """
        return len([role for role in self.ego_self_role.values() if role == 1])

    def assist_pincer_function(self, env, enm):
        """
        压侧形成合围攻势的奖励函数
        """
        # 如果没有assist则返回0
        assist_ids = self.enemies_allocation[enm.uid][0]
        if len(assist_ids) == 0:
            return 0

        # 计算每个敌机对战的所有压侧飞机的方位奖励
        enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])

        # 计算该敌机对应的所有团队飞机的平均相对位置得分
        position = 0
        dist_reward = 0
        for assist_id in assist_ids:
            # 计算相对位置的奖励
            assist = env.agents[assist_id]
            assist_feature = np.hstack([assist.get_position(), assist.get_velocity()])
            _, _, R, side_flag = get_AO_TA_R(assist_feature, enm_feature, return_side=True)
            position += side_flag

            # 保持队友之间距离不要过远
            for other_assist_id in assist_ids:
                if other_assist_id == assist_id:
                    continue
                other_assist = env.agents[other_assist_id]
                other_assist_feature = np.hstack([other_assist.get_position(), other_assist.get_velocity()])
                _, _, R_assist = get_AO_TA_R(assist_feature, other_assist_feature)
                dist_reward += ((R_assist >= self.shoot_opt_dist) * np.exp(-(R_assist / 1000 - self.shoot_opt_dist) ** 2 / 4)
                                + (R_assist < self.shoot_opt_dist) * 1.)

        position /= len(assist_ids)
        return (1. - np.abs(position)) * (dist_reward / len(assist_ids))

    def assist_approach_function(self, enm, R):
        """
        辅助机需要靠近敌机，避免只在远处分散。
        """
        # 如果已有射手存在，则辅助只做合围，不加此奖励
        if 1 in self.ego_self_role.values() or enm.uid not in self.R_pre_time:
            return 0

        # 光滑负奖励：在 R > R_opt 时变负
        return (R > self.shoot_opt_dist * 1.2) * np.tanh((R - self.shoot_opt_dist * 1.2) ** 2 - (self.R_pre_time[enm.uid] - self.shoot_opt_dist * 1.2) ** 2 - 50)
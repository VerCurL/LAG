import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R

class EnergyReward(BaseRewardFunction):
    """
    能量奖励函数，基于当前风险等级的优势能力获取机制
    分三个风险等级：
    - 被导弹锁定：生死级，直接放弃能量维持，逃跑
    - 优势位置不利：交战级，适当减少能量奖励，注重优势站位
    - 数量风险：压力级，一定范围内敌机越多，能量维持系数越低
    """
    def __init__(self, config):
        super().__init__(config)
        self.safe_distance = getattr(self.config, "max_missile_attack_distance", 14000) / 1000  # unit: km
        self.min_energy = 5000
        self.pre_energies = {}

    def reset(self, task, env):
        self.pre_energies.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        # 获得我机对象
        agent = env.agents[agent_id]

        # 计算当前我机的能量值
        ego_velocity = np.linalg.norm(agent.get_velocity())
        ego_height = agent.get_position()[-1]
        SE = self.calculate_SE(ego_velocity, ego_height)

        # 计算相较于上一时刻我机能量值的变化量
        if agent_id not in self.pre_energies:
            self.pre_energies[agent_id] = SE
        Delta_SE = SE - self.pre_energies[agent_id]

        # 防止能量突变
        if abs(Delta_SE) > 3000:
            Delta_SE = 0

        # 计算风险等级获得最终奖励
        reward_base_energy = Delta_SE
        reward_low_energy = -max(0, self.min_energy - SE)
        reward = (1 - self.calculate_risk(agent)) * (0.4 * reward_base_energy + 0.1 * reward_low_energy)

        # 更新上一时刻的能量值
        self.pre_energies[agent_id] = SE
        # if agent_id == "A0100":
        # print("[energy_reward] reward: ", reward)
        # print("                SE: ", SE)
        return self._process(reward, agent_id)

    def calculate_SE(self, v, h):
        return (v ** 2) / 19.62 + h

    def calculate_risk(self, agent):
        # 1. 如果被导弹锁定或者没有导弹，则风险最高
        risk_missile = 0.0
        if len(agent.check_all_missile_warning()) > 0 or agent.num_left_missiles <= 0:
            risk_missile = 1.0

        # 2. 计算空战危险等级
        risk_geometries = []
        enm_num = 0
        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        for enm in agent.enemies:
            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            # 1) 敌机约靠近我机尾部威胁越大，也就是AO越大威胁越大
            # 2) 敌机机头约指向我机威胁越大
            risk = (1 + np.cos(AO)) / 2 * np.exp(-(R / 1000) / (0.8 * self.safe_distance))
            risk_geometries.append(risk)
            if R / 1000.0 < 1.2 * self.safe_distance:
                enm_num += 1

        risk_geometry = 1.0
        for r in risk_geometries:
            risk_geometry *= 1.0 - np.clip(r, 0.0, 1.0)
        risk_geometry = 1.0 - risk_geometry

        # 3. 计算数量风险等级，最高为0.3
        risk_num = 1 - np.exp(-0.1 * enm_num)

        # print("[energy_reward] risk_missile: ", risk_missile)
        # print("                risk_geometry: ", risk_geometry)
        # print("                risk_num: ", risk_num)

        return min(1, risk_missile + risk_geometry + risk_num)





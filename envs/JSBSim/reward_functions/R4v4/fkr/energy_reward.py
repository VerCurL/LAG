import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R, get_near_offset_of_multi_air

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
        self.min_energy = 5000
        self.pre_energies = {}                  # 上一时刻的能量
        self.pre_near_offset_states = {}             # 上一时刻我机的相对敌机加权质心参数：{ego: [AO, 0, R]}

    def reset(self, task, env):
        self.pre_energies.clear()
        self.pre_near_offset_states.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        # 获得我机对象
        agent = env.agents[agent_id]

        # 获取最近敌机的信息
        states = {}
        for enm in agent.enemies:
            if enm.is_alive:
                states[enm.uid] = enm.get_position()

        # 如果有敌人，则开始奖励计算
        reward = 0.0
        if len(states) > 0:
            # 获得最近敌人相对加权质心的偏移位置
            ego_position = agent.get_position()
            enm_positions = np.array([position for position in states.values()])
            near_offset_position, near_offset_vector = get_near_offset_of_multi_air(ego_position, enm_positions)

            # 获得我机和偏移位置的参数关系
            ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
            center_feature = np.hstack([near_offset_position, np.array([0, 0, 0])])
            AO, _, R = get_AO_TA_R(ego_feature, center_feature)

            near_offset_state = [AO, 0, R / 1000]
            if agent_id not in self.pre_near_offset_states:
                self.pre_near_offset_states[agent_id] = near_offset_state

            # 计算攻击角变化了
            pre_AO = self.pre_near_offset_states[agent_id][0]
            Delta_AO = pre_AO - AO                                  # >0：做转向质心的机动

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
            reward_base_energy = (Delta_SE < 0) * np.tanh(Delta_SE / 800) + (Delta_SE >= 0) * Delta_SE / 10
            reward_turn_enm = (R > 1.3 * task.max_missile_attack_distance) * max(0, Delta_AO)
            reward_low_energy = -max(0, self.min_energy - SE)
            reward = ((1 - self.calculate_risk(agent, task.safe_distance)) *
                      (0.4 * (reward_base_energy + 20 * reward_turn_enm) + 0.1 * reward_low_energy))

            # 更新上一时刻的值
            self.pre_energies[agent_id] = SE
            self.pre_near_offset_states[agent_id] = near_offset_state

            # if agent_id == "A0100":
            #     print("[energy_reward] reward: ", reward)
            #     print("                Delta_SE >= 0: ", Delta_SE >= 0)
            #     print("                reward_base_energy: ", 0.4 * reward_base_energy)
            #     print("                reward_turn_enm: ", 8 * reward_turn_enm)
                # print("                pre_AO: ", pre_AO)
                # print("                cur_AO: ", AO)
                # near_offset_positions_plot(ego_position, enm_positions, near_offset_position)
        return self._process(reward, agent_id)

    def calculate_SE(self, v, h):
        return (v ** 2) / 19.62 + h

    def calculate_risk(self, agent, safe_distance):
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
            risk = (1 + np.cos(AO)) / 2 * np.exp(-(R / 1000) / (0.8 * safe_distance))
            risk_geometries.append(risk)
            if R / 1000.0 < 1.2 * safe_distance:
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





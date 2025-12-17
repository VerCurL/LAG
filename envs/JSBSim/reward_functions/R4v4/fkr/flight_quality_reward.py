import numpy as np
import os
import math
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction  # 从基础奖励类继承

class FlightQualityReward(BaseRewardFunction):
    """
    飞行质量奖励：保持平稳姿态，避免极端动作
    包含惩罚项：
    - 角速度变化过大惩罚：sqrt((角速度(t) - 角速度(t-1))²)
    - 姿态角变化过大惩罚：sqrt((姿态角(t) - 姿态角(t-1))²)
    - 不强制确保飞行质量，优先活下来
    """

    def __init__(self, config):
        super().__init__(config)
        self.pre_rpy = {}            # 记录上一时刻各飞机的姿态角
        self.pre_rpy_v = {}          # 记录上一时刻各飞机的角速度
        self.step_count = 0

    def reset(self, task, env):
        self.step_count = 0
        self.pre_rpy.clear()
        self.pre_rpy_v.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        计算当前智能体的飞行质量相关奖励

        参数:
            task: 当前任务实例
            env: 环境实例
            agent_id: 智能体ID

        返回:
            float: 计算出的奖励值
        """
        # 初始化上一时刻的角速度记录
        if agent_id not in self.pre_rpy:
            self.pre_rpy[agent_id] = np.zeros(3)
        if agent_id not in self.pre_rpy_v:
            self.pre_rpy_v[agent_id] = np.zeros(3)

        # 获得当前时刻角速度信息
        cur_rpy = env.agents[agent_id].get_rpy()
        cur_rpy_v = cur_rpy - self.pre_rpy[agent_id]

        # 计算奖励，控制角速度变化防止抖动
        reward = 0.0
        if self.step_count > 2 and len(env.agents[agent_id].check_all_missile_warning()) == 0:
            reward = np.clip(-2.0 * (np.linalg.norm(cur_rpy_v) ** 2 + np.linalg.norm(cur_rpy_v - self.pre_rpy_v[agent_id]) ** 2),
                             -2.0, 0.0)
        elif agent_id == "A0100":
            self.step_count += 1

        # if agent_id == "A0100":
        # print("[flight_quality_reward] reward: ", reward)
        #     # print("                        pre_rpy: ", {agent_id: self.pre_rpy[agent_id]})
        #     # print("                        cur_rpy: ", {agent_id: cur_rpy})
        #     # print("                        pre_rpy_v: ", {agent_id: self.pre_rpy_v[agent_id]})
        #     print("                        cur_rpy_v: ", {agent_id: cur_rpy_v})

        # 更新当前时刻为上一时刻
        self.pre_rpy[agent_id] = cur_rpy.copy()
        self.pre_rpy_v[agent_id] = cur_rpy_v.copy()

        return self._process(reward, agent_id)
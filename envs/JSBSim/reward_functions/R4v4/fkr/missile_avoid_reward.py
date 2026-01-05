import numpy as np
import os
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R

class MissileAvoidReward(BaseRewardFunction):
    """
    导弹躲避奖励
    """
    def __init__(self, config):
        super().__init__(config)
        self.pre_missiles_info = {}

    def reset(self, task, env):
        self.pre_missiles_info.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):

        new_reward = 0
        if agent_id not in self.pre_missiles_info:
            self.pre_missiles_info[agent_id] = {}
        agent = env.agents[agent_id]
        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
        missile_sims = agent.check_all_missile_warning()                        # 获取锁定我方的所有导弹信息

        for sim in missile_sims:
            # 判断是否是一枚还在运行的导弹
            if not sim.is_alive:
                continue

            # 1. 鼓励我放飞机和导弹形成交叉角
            relative_angle = np.degrees(np.arccos(np.dot(sim.get_velocity(), agent.get_velocity()) /
                                (np.linalg.norm(sim.get_velocity()) * np.linalg.norm(agent.get_velocity()))))
            if 60 <= relative_angle <= 110:
                new_reward += 10
            else:
                new_reward -= 15

            # 存储导弹相关信息
            sim_feature = np.hstack([sim.get_position(), sim.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, sim_feature)                       # 获取我机和导弹的关系
            relative_velocity = sim.get_velocity() - agent.get_velocity()           # 获取导弹和我机的相对速度
            missile_to_aircraft_direction = (agent.get_position() - sim.get_position()) / R         # 获得导弹指向我机的单位向量
            velocity_component = np.dot(sim.get_velocity(), missile_to_aircraft_direction)          # 获取导弹速度在direction_to_aircraft连线上的分量

            if sim.uid not in self.pre_missiles_info[agent_id]:
                # 其中：0存储relative_velocity，来计算导弹是否有飞向我机的趋势
                #      1存储velocity_component，来计算导弹当前时刻是否飞向我机
                self.pre_missiles_info[agent_id][sim.uid] = [relative_velocity, velocity_component]
                continue

            # 2. 看导弹是否有飞向我机的趋势
            relative_acceleration = relative_velocity - self.pre_missiles_info[agent_id][sim.uid][0]  # 获取相对速度的变化了
            acceleration_component = np.dot(relative_acceleration, missile_to_aircraft_direction)
            pre_velocity_component = self.pre_missiles_info[agent_id][sim.uid][1]
            if acceleration_component < 0:
                new_reward += 10    # 导弹有远离的趋势的话，奖励
            elif velocity_component < pre_velocity_component:
                new_reward += 5     # 导弹没有远离的趋势，但是正在减速接近我机的话，也奖励
            else:
                new_reward -= 10    # 如果没有远离的趋势也没有减速接近，惩罚

        return self._process(new_reward, agent_id)




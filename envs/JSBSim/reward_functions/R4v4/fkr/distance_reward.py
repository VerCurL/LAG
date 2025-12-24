import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R, get_near_offset_of_multi_air

class DistanceReward(BaseRewardFunction):
    """
    距离奖励函数，控制我机能够在进攻状态下靠近敌机
    """
    def __init__(self, config):
        super().__init__(config)
        # 记录上局和最近敌机的距离
        self.pre_near_offset_states = {}        # {enm_id: [AO, TA, R]}

    def reset(self, task, env):
        self.pre_near_offset_states.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        # ======================================================
        # 一、 如果飞机还有导弹，进行奖励计算
        # ======================================================
        reward = 0.0
        agent = env.agents[agent_id]
        if agent.num_left_missiles > 0:
            # 遍历敌机，获得针对每个敌机的优势站位得分
            states = {}
            for enm in agent.enemies:
                if enm.is_alive:
                    states[enm.uid] = enm.get_position()

            # 如果有敌人，则开始奖励计算
            if len(states) > 0:
                # 获得最近敌人相对加权质心的偏移位置
                ego_position = agent.get_position()
                enm_positions = np.array([position for position in states.values()])
                near_offset_position, near_offset_vector = get_near_offset_of_multi_air(ego_position, enm_positions)

                # 获得我机和偏移位置的参数关系
                ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])
                center_feature = np.hstack([near_offset_position, np.array([0, 0, 0])])
                AO, _, R = get_AO_TA_R(ego_feature, center_feature)

                near_off_state = [AO, 0, R / 1000]
                if agent_id not in self.pre_near_offset_states:
                    self.pre_near_offset_states[agent_id] = near_off_state

                distance_max = 1.3 * task.max_missile_attack_distance
                distance_min = 1.0 * task.max_missile_attack_distance

                # 奖励1：我现在是否朝向敌机
                gate = ((near_off_state[2] > distance_max) * 1.0 + (distance_min < near_off_state[2] <= distance_max) *
                        ((near_off_state[2] - distance_min) / (distance_max - distance_min)))

                forward = (np.pi / 2 - near_off_state[0]) / (np.pi / 2)
                reward_forward = max(0, forward)

                # 奖励2：如果敌机在侧后方，则引导我机转向
                reward_turn = (near_off_state[0] >= np.pi / 2) * (self.pre_near_offset_states[agent_id][0] - near_off_state[0])

                # 综合奖励值
                reward += gate * (reward_forward + reward_turn)

                self.pre_near_offset_states[agent_id] = near_off_state

            # 如果被攻击，则不管优势站位，先躲避导弹活下来
            if len(agent.check_all_missile_warning()) > 0:
               reward = 0.0

        # print("[distance_window] reward: ", reward)

        return self._process(reward, agent_id)
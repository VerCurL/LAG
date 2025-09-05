import numpy as np
from gymnasium import spaces
from typing import Tuple
import torch

from ..tasks import SingleCombatTask
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward, \
    TeamPostureReward, MissileAvoidReward, TeamAttackDefenseReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn, PartnerSafe
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor


class MultipleCombatTask(SingleCombatTask):
    """
    多飞机空战任务基类，继承自单机空战任务类
    实现了4架飞机（2v2）的基本空战对抗环境，使用底层直接控制方式
    """
    def __init__(self, config):
        """
        初始化多飞机空战任务
        
        参数:
            config: 配置对象，包含任务的各种参数设置
        """
        super().__init__(config)

        # 设置奖励函数列表
        self.reward_functions = [
            AltitudeReward(self.config),          # 高度奖励：鼓励智能体保持合适的飞行高度
            PostureReward(self.config),           # 姿态奖励：鼓励智能体保持有利的战斗姿态
            EventDrivenReward(self.config),       # 事件驱动奖励：基于特定事件（如击落敌机）给予奖励
            TeamPostureReward(self.config),       # 团队姿态奖励：鼓励团队协作保持有利战术位置
        ]

        # 设置终止条件列表
        self.termination_conditions = [
            SafeReturn(self.config),              # 安全返航：飞机成功返回指定区域
            ExtremeState(self.config),            # 极端状态：飞机处于危险状态（如超高速、极限G值等）
            Overload(self.config),                # 过载：飞机超过最大过载限制
            LowAltitude(self.config),             # 低高度：飞机高度过低（有坠机风险）
            Timeout(self.config),                 # 超时：任务时间超过最大限制
            # PartnerSafe(self.config),           # 队友安全：确保队友安全（当前未启用）
        ]

    @property
    def num_agents(self) -> int:
        """
        返回智能体数量（固定为4架飞机，表示2v2空战）
        """
        return 4

    def load_variables(self):
        """
        加载飞机状态变量、控制变量和渲染变量
        这些变量是与JSBSim仿真器交互的关键
        """
        # 状态变量列表
        self.state_var = [
            c.position_long_gc_deg,               # 0. 经度 (单位: °)
            c.position_lat_geod_deg,              # 1. 纬度 (单位: °)
            c.position_h_sl_m,                    # 2. 高度 (单位: m)
            c.attitude_roll_rad,                  # 3. 滚转角 (单位: rad)
            c.attitude_pitch_rad,                 # 4. 俯仰角 (单位: rad)
            c.attitude_heading_true_rad,          # 5. 航向角 (单位: rad)
            c.velocities_v_north_mps,             # 6. 北向速度 (单位: m/s)
            c.velocities_v_east_mps,              # 7. 东向速度 (单位: m/s)
            c.velocities_v_down_mps,              # 8. 下向速度 (单位: m/s)
            c.velocities_u_mps,                   # 9. 机体坐标系x轴速度 (单位: m/s)
            c.velocities_v_mps,                   # 10. 机体坐标系y轴速度 (单位: m/s)
            c.velocities_w_mps,                   # 11. 机体坐标系z轴速度 (单位: m/s)
            c.velocities_vc_mps,                  # 12. 校正空速 (单位: m/s)
            c.accelerations_n_pilot_x_norm,       # 13. 飞行员x轴过载 (单位: G)
            c.accelerations_n_pilot_y_norm,       # 14. 飞行员y轴过载 (单位: G)
            c.accelerations_n_pilot_z_norm,       # 15. 飞行员z轴过载 (单位: G)
        ]
        # 控制变量列表
        self.action_var = [
            c.fcs_aileron_cmd_norm,               # 副翼控制 [-1., 1.]
            c.fcs_elevator_cmd_norm,              # 升降舵控制 [-1., 1.]
            c.fcs_rudder_cmd_norm,                # 方向舵控制 [-1., 1.]
            c.fcs_throttle_cmd_norm,              # 油门控制 [0.4, 0.9]
        ]
        # 渲染变量列表（用于可视化）
        self.render_var = [
            c.position_long_gc_deg,               # 经度
            c.position_lat_geod_deg,              # 纬度
            c.position_h_sl_m,                    # 高度
            c.attitude_roll_rad,                  # 滚转角
            c.attitude_pitch_rad,                 # 俯仰角
            c.attitude_heading_true_rad,          # 航向角
        ]

    def load_observation_space(self):
        """
        定义观察空间（状态空间）
        包括自身信息和其他飞机的相对信息
        """
        # 观察向量长度 = 9(自身信息) + (总数-1)×6(其他飞机相对信息)
        self.obs_length = 9 + (self.num_agents - 1) * 6
        # 定义观察空间，使用Box表示连续空间
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        # 定义共享观察空间（用于CTDE训练范式）
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        """
        定义动作空间
        使用MultiDiscrete表示多个离散动作维度
        """
        # 四个控制面：副翼(41维)、升降舵(41维)、方向舵(41维)、油门(30维)
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """
        获取指定智能体的观察向量
        
        参数:
            env: 环境对象
            agent_id: 智能体ID
            
        返回:
            归一化后的观察向量
        """
        # 初始化观察向量
        norm_obs = np.zeros(self.obs_length)
        
        # (1) 自身信息归一化
        # 获取自身状态
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        # 将经纬高坐标转换为NED（北东下）坐标
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        # 构建特征向量（位置+速度）
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        
        # 归一化各状态量并存入观察向量
        norm_obs[0] = ego_state[2] / 5000           # 0. 高度归一化（单位：5km）
        norm_obs[1] = np.sin(ego_state[3])          # 1. 滚转角正弦值
        norm_obs[2] = np.cos(ego_state[3])          # 2. 滚转角余弦值
        norm_obs[3] = np.sin(ego_state[4])          # 3. 俯仰角正弦值
        norm_obs[4] = np.cos(ego_state[4])          # 4. 俯仰角余弦值
        norm_obs[5] = ego_state[9] / 340            # 5. 机体x轴速度（单位：马赫）
        norm_obs[6] = ego_state[10] / 340           # 6. 机体y轴速度（单位：马赫）
        norm_obs[7] = ego_state[11] / 340           # 7. 机体z轴速度（单位：马赫）
        norm_obs[8] = ego_state[12] / 340           # 8. 校正空速（单位：马赫）
        
        # (2) 添加队友和敌人的相对信息
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            # 获取目标飞机状态
            state = np.array(sim.get_property_values(self.state_var))
            # 将目标位置转换为NED坐标
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            # 构建目标特征向量
            feature = np.array([*cur_ned, *(state[6:9])])
            # 计算目标的方位角、俯仰角、距离和相对侧向位置
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            
            # 归一化目标相对信息并存入观察向量
            norm_obs[offset + 1] = (state[9] - ego_state[9]) / 340    # 相对速度（单位：马赫）
            norm_obs[offset + 2] = (state[2] - ego_state[2]) / 1000   # 相对高度（单位：1km）
            norm_obs[offset + 3] = AO                                 # 方位角
            norm_obs[offset + 4] = TA                                 # 俯仰角
            norm_obs[offset + 5] = R / 10000                          # 距离（单位：10km）
            norm_obs[offset + 6] = side_flag                          # 侧向标志（左/右）
            offset += 6
            
        # 裁剪观察向量到合法范围
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """
        将离散动作索引转换为连续控制值
        
        参数:
            env: 环境对象
            agent_id: 智能体ID
            action: 离散动作向量
            
        返回:
            归一化后的连续动作向量
        """
        norm_act = np.zeros(4)
        # 将离散动作转换为连续值，范围映射如下：
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.   # 副翼：[-1, 1]
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.   # 升降舵：[-1, 1]
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.   # 方向舵：[-1, 1]
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4 # 油门：[0.4, 0.9]
        return norm_act

    # def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
    #     """
    #     计算智能体的奖励
    #
    #     参数:
    #         env: 环境对象
    #         agent_id: 智能体ID
    #         info: 附加信息字典
    #
    #     返回:
    #         (奖励值, 更新后的信息字典)
    #     """
    #     # 只有智能体存活时才计算奖励，否则返回0
    #     if env.agents[agent_id].is_alive:
    #         return super().get_reward(env, agent_id, info=info)
    #     else:
    #         return 0.0, info


class HierarchicalMultipleCombatTask(MultipleCombatTask):
    """
    分层多飞机空战任务类
    使用高级别控制（如高度、航向、速度变化）作为动作空间，
    通过预训练的低级别策略将高级别指令转换为底层控制命令
    """
    def __init__(self, config: str):
        """
        初始化分层多飞机空战任务
        
        参数:
            config: 配置对象
        """
        super().__init__(config)
        
        # 加载预训练的低级别控制策略
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(
            torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()  # 设置为评估模式
        
        # 定义高级别控制的归一化变化量
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])                           # 高度变化：上升/保持/下降
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])  # 航向变化：大左转/小左转/直飞/小右转/大右转
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])                         # 速度变化：加速/保持/减速

    def load_action_space(self):
        """
        定义高级别动作空间
        """
        # 3个控制维度：高度变化(3)、航向变化(5)、速度变化(3)
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def normalize_action(self, env, agent_id, action):
        """
        将高级别动作转换为低级别控制命令
        
        参数:
            env: 环境对象
            agent_id: 智能体ID
            action: 高级别动作向量
            
        返回:
            归一化后的底层控制量
        """
        # 生成低级别控制器的输入观察向量
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        
        # (1) 设置目标变化量（高度/航向/速度）
        input_obs[0] = self.norm_delta_altitude[action[0]]  # 高度变化指令
        input_obs[1] = self.norm_delta_heading[action[1]]   # 航向变化指令
        input_obs[2] = self.norm_delta_velocity[action[2]]  # 速度变化指令
        
        # (2) 加入自身状态信息
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)  # 增加批次维度
        
        # 使用低级别策略生成底层控制动作
        _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        
        # 归一化低级别动作到合适范围
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.    # 副翼：[-1, 1]
        norm_act[1] = action[1] / 20 - 1.    # 升降舵：[-1, 1]
        norm_act[2] = action[2] / 20 - 1.    # 方向舵：[-1, 1]
        norm_act[3] = action[3] / 58 + 0.4   # 油门：[0.4, 0.9]
        return norm_act

    def reset(self, env):
        """
        任务重置函数，初始化RNN隐藏状态
        
        参数:
            env: 环境对象
            
        返回:
            父类reset的返回值
        """
        # 为每个智能体初始化RNN隐藏状态
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)


class HierarchicalMultipleCombatShootTask(HierarchicalMultipleCombatTask):
    """
    分层多飞机导弹发射任务类
    在分层控制基础上增加了导弹发射能力
    """
    def __init__(self, config: str):
        """
        初始化分层多飞机导弹发射任务
        
        参数:
            config: 配置对象
        """
        super().__init__(config)
        
        # 设置导弹发射的约束条件
        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)           # 最大攻击角度
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)  # 最大攻击距离
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)     # 最小攻击间隔（冷却时间）
        
        # 设置奖励函数列表（增加了导弹相关奖励）
        self.reward_functions = [
            AltitudeReward(self.config),            # 高度奖励
            MissilePostureReward(self.config),      # 导弹姿态奖励
            EventDrivenReward(self.config),         # 事件驱动奖励
            TeamPostureReward(self.config),         # 团队姿态奖励
            TeamAttackDefenseReward(self.config),   # 团队攻防奖励
            MissileAvoidReward(self.config)         # 导弹躲避奖励
        ]

    def load_observation_space(self):
        """
        定义观察空间（增加了导弹信息）
        """
        # 观察向量长度 = 9(自身信息) + 总数×6(其他飞机+导弹信息)
        self.obs_length = 9 + self.num_agents * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        """
        定义动作空间（增加了发射导弹动作）
        """
        # 4个控制维度：高度变化(3)、航向变化(5)、速度变化(3)、导弹发射(2)
        self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])

    def get_obs(self, env, agent_id):
        """
        获取指定智能体的观察向量（增加了导弹信息）
        
        参数:
            env: 环境对象
            agent_id: 智能体ID
            
        返回:
            归一化后的观察向量
        """
        norm_obs = np.zeros(self.obs_length)
        
        # (1) 自身信息归一化（与父类相同）
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000           # 0. 高度归一化（单位：5km）
        norm_obs[1] = np.sin(ego_state[3])          # 1. 滚转角正弦值
        norm_obs[2] = np.cos(ego_state[3])          # 2. 滚转角余弦值
        norm_obs[3] = np.sin(ego_state[4])          # 3. 俯仰角正弦值
        norm_obs[4] = np.cos(ego_state[4])          # 4. 俯仰角余弦值
        norm_obs[5] = ego_state[9] / 340            # 5. 机体x轴速度（单位：马赫）
        norm_obs[6] = ego_state[10] / 340           # 6. 机体y轴速度（单位：马赫）
        norm_obs[7] = ego_state[11] / 340           # 7. 机体z轴速度（单位：马赫）
        norm_obs[8] = ego_state[12] / 340           # 8. 校正空速（单位：马赫）
        
        # (2) 添加队友和敌人的相对信息（与父类相同）
        offset = 8
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset + 1] = (state[9] - ego_state[9]) / 340
            norm_obs[offset + 2] = (state[2] - ego_state[2]) / 1000
            norm_obs[offset + 3] = AO
            norm_obs[offset + 4] = TA
            norm_obs[offset + 5] = R / 10000
            norm_obs[offset + 6] = side_flag
            offset += 6
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        
        # (3) 添加导弹信息（检测到的最近导弹）
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            # 获取导弹的位置和速度信息
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            # 计算导弹相对于飞机的方位角、俯仰角、距离和侧向位置
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            
            # 归一化导弹信息并存入观察向量
            norm_obs[offset + 1] = (np.linalg.norm(missile_sim.get_velocity()) - ego_state[9]) / 340  # 相对速度
            norm_obs[offset + 2] = (missile_feature[2] - ego_state[2]) / 1000                         # 相对高度
            norm_obs[offset + 3] = ego_AO                                                            # 方位角
            norm_obs[offset + 4] = ego_TA                                                            # 俯仰角
            norm_obs[offset + 5] = R / 10000                                                         # 距离
            norm_obs[offset + 6] = side_flag                                                         # 侧向标志
        return norm_obs

    def reset(self, env):
        """
        任务重置函数，初始化导弹状态和计时器
        
        参数:
            env: 环境对象
            
        返回:
            父类reset的返回值
        """
        # 初始化上次发射导弹的时间（设为负值表示初始可发射）
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        # 初始化剩余导弹数量
        self._remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        # 初始化发射动作标志
        self._shoot_action = {agent_id: False for agent_id in env.agents.keys()}
        return super().reset(env)

    def normalize_action(self, env, agent_id, action):
        """
        处理高级别动作，包括导弹发射控制
        
        参数:
            env: 环境对象
            agent_id: 智能体ID
            action: 高级别动作向量
            
        返回:
            归一化后的底层控制量
        """
        # 提取导弹发射动作
        self._shoot_action[agent_id] = action[3] > 0
        # 将其余动作传递给父类处理
        return super().normalize_action(env, agent_id, action[:3])

    def step(self, env):
        """
        环境步进函数，处理导弹发射逻辑
        
        参数:
            env: 环境对象
        """
        # 先执行父类的step方法
        SingleCombatTask.step(self, env)
        
        # 处理每个智能体的导弹发射
        for agent_id, agent in env.agents.items():
            # [基于RL的导弹发射，带限制条件]
            # 确定是否可以向最近的敌机发射导弹
            target_list = list(map(lambda x: x.get_position() - agent.get_position(), agent.enemies))   # 所有敌机的相对位置
            target_distance = list(map(np.linalg.norm, target_list))                                    # 所有敌机的和距离
            target_index = np.argmin(target_distance)
            target = target_list[target_index]                                                          # 最近的敌机
            heading = agent.get_velocity()                                                              # 本机航向
            distance = target_distance[target_index]                                                    # 本机与目标的距离

            attack_angle = np.rad2deg(                                                                  # 攻击角度（飞机航向与目标方向的夹角）
                np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]                         # 距离上次发射的时间间隔
            velocity = np.linalg.norm(heading)                                                          # 本机的速度值

            # 判断是否满足发射条件
            shoot_flag = (agent.is_alive and self._shoot_action[agent_id] and self._remaining_missiles[agent_id] > 0 \
                         and attack_angle <= self.max_attack_angle and distance <= self.max_attack_distance
                         and shoot_interval >= self.min_attack_interval and velocity >= 150)
            # shoot_flag = True
            # 如果满足发射条件，创建新导弹
            if shoot_flag:
                # 创建唯一导弹ID
                new_missile_uid = agent_id + str(self._remaining_missiles[agent_id])
                # 添加导弹仿真器
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[target_index], uid=new_missile_uid))
                # 减少剩余导弹数量
                self._remaining_missiles[agent_id] -= 1
                # 减少飞机类的剩余导弹数量
                env.agents[agent_id].num_remaining_missiles -= 1
                # 更新上次发射时间
                self._last_shoot_time[agent_id] = env.current_step

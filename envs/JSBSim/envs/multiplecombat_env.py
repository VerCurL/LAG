import numpy as np
from typing import Tuple, Dict, Any
import math
import random
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask


class MultipleCombatEnv(BaseEnv):
    """
    多飞机空战环境类
    实现了多智能体（2v2）的竞争性对抗环境
    """
    def __init__(self, config_name: str, fix_position = False):
        """
        初始化多飞机空战环境
        
        参数:
            config_name: 配置文件名称，指定要加载的场景配置
        """
        super().__init__(config_name)
        # 环境特定初始化
        self._create_records = False
        self.fix_position = fix_position

    @property
    def share_observation_space(self):
        """
        获取共享观察空间
        用于集中式训练分散式执行（CTDE）范式
        
        返回:
            共享观察空间，包含所有智能体的信息
        """
        return self.task.share_observation_space

    def load_task(self):
        """
        根据配置加载对应的任务类
        包括基础的多飞机空战任务、分层控制任务和带导弹发射功能的分层任务
        """
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            # 基础多飞机空战任务
            self.task = MultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat':
            # 分层控制多飞机空战任务
            self.task = HierarchicalMultipleCombatTask(self.config)
        elif taskname == 'hierarchical_multiplecombat_shoot':
            # 分层控制多飞机带导弹发射空战任务
            self.task = HierarchicalMultipleCombatShootTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        重置环境状态并返回初始观察
        
        返回:
            obs: 各智能体的初始观察，字典形式 {agent_id: 观察数组}
            share_obs: 各智能体的初始共享状态，字典形式 {agent_id: 共享状态数组}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        share_obs = self.get_state()
        return self._pack(obs), self._pack(share_obs)

    def reset_simulators(self):
        """
        重置所有模拟器的状态
        包括飞机模拟器和临时模拟器（如导弹）
        """
        if self.fix_position:
            # 重新加载所有飞机模拟器
            for sim in self._jsbsims.values():
                sim.reload()
            # 清空临时模拟器（如导弹）
            self._tempsims.clear()
        else:
            self.random_reset_simulators()

    def random_reset_simulators(self):
        # --- 常量定义 ---
        KM_PER_DEG_LAT = 111.132  # 每度纬度对应的公里数 (近似值)
        KM_PER_DEG_LON_AT_EQ = 111.320  # 赤道上每度经度对应的公里数 (近似值)
        FT_PER_METER = 3.28084

        # --- 基地和距离设置 ---
        red_base_lon_deg = 120.0
        red_base_lat_deg = 60.0
        inner_radius_km = 5.0  # 队伍内部散布半径
        min_base_separation_km = 10.0  # 队伍基地最小间距
        max_base_separation_km = 40.0  # (可选) 队伍基地最大间距，增加随机性

        # --- 计算红队纬度处的经度换算因子 ---
        # 注意：math.cos() 需要弧度
        km_per_deg_lon_red = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(red_base_lat_deg))

        # --- 计算蓝队基地的随机位置 ---
        # 1. 随机选择一个方向 (角度)
        angle_rad = random.uniform(0, 2 * math.pi)
        # 2. 随机选择一个距离 (大于等于最小间距)
        distance_km = random.uniform(min_base_separation_km, max_base_separation_km)

        # 3. 计算经纬度偏移量 (使用平面近似，对于几十公里通常足够)
        delta_lat_deg = (distance_km * math.cos(angle_rad)) / KM_PER_DEG_LAT
        # 使用红队基地的经度换算因子作为近似
        delta_lon_deg = (distance_km * math.sin(angle_rad)) / km_per_deg_lon_red

        # 4. 计算蓝队基准点
        blue_base_lon_deg = red_base_lon_deg + delta_lon_deg
        blue_base_lat_deg = red_base_lat_deg + delta_lat_deg

        # --- 计算蓝队纬度处的经度换算因子 ---
        km_per_deg_lon_blue = KM_PER_DEG_LON_AT_EQ * math.cos(math.radians(blue_base_lat_deg))

        # --- 计算内部散布的最大经纬度偏移量 ---
        #   (单位: 度)
        max_lat_offset_deg = inner_radius_km / KM_PER_DEG_LAT
        max_lon_offset_deg_red = inner_radius_km / km_per_deg_lon_red
        max_lon_offset_deg_blue = inner_radius_km / km_per_deg_lon_blue

        # --- 循环设置每个单位的初始条件 ---
        for sim_id, sim in self._jsbsims.items():
            # 为每个单位生成独立的随机属性
            altitude_m = random.randint(5000, 10000)  # 先用米，方便理解
            heading_deg = random.randint(0, 359)  # 0-359 更常用
            speed_mps = random.randint(100, 300)  # 先用米/秒

            if sim_id.startswith('A'):  # 红队
                # 在基准点周围随机偏移 (允许负值)
                offset_lat = random.uniform(-max_lat_offset_deg, max_lat_offset_deg)
                offset_lon = random.uniform(-max_lon_offset_deg_red, max_lon_offset_deg_red)

                sim.reload({
                    "ic_long_gc_deg": red_base_lon_deg + offset_lon,
                    "ic_lat_geod_deg": red_base_lat_deg + offset_lat,
                    "ic_h_sl_ft": altitude_m * FT_PER_METER,
                    "ic_psi_true_deg": heading_deg,
                    "ic_u_fps": speed_mps * FT_PER_METER,  # 假设 ic_u_fps 是总速度标量
                })
            elif sim_id.startswith('B'):  # 蓝队
                # 在基准点周围随机偏移 (允许负值)
                offset_lat = random.uniform(-max_lat_offset_deg, max_lat_offset_deg)
                offset_lon = random.uniform(-max_lon_offset_deg_blue, max_lon_offset_deg_blue)

                sim.reload({
                    "ic_long_gc_deg": blue_base_lon_deg + offset_lon,
                    "ic_lat_geod_deg": blue_base_lat_deg + offset_lat,
                    "ic_h_sl_ft": altitude_m * FT_PER_METER,
                    "ic_psi_true_deg": heading_deg,
                    "ic_u_fps": speed_mps * FT_PER_METER,
                })

        self._tempsims.clear()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        环境的单步执行，接收所有智能体的动作并返回结果
        
        参数:
            action: 包含所有智能体动作的数组
            
        返回:
            (元组):
                obs: 各智能体对当前环境的观察
                share_obs: 各智能体的共享观察
                rewards: 各智能体执行动作后获得的奖励
                dones: 各智能体是否完成任务的标志
                info: 额外信息字典
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # 应用所有智能体的动作
        action = self._unpack(action)
        for agent_id in self.agents.keys():
            # 将智能体的动作归一化后设置到飞机控制系统
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
            
        # 运行模拟器
        for _ in range(self.agent_interaction_steps):
            # 更新所有飞机模拟器
            for sim in self._jsbsims.values():
                sim.run()
            # 更新所有临时模拟器（如导弹）
            for sim in self._tempsims.values():
                sim.run()
                
        # 执行任务特定的步进逻辑
        self.task.step(self)
        
        # 获取观察和共享状态
        obs = self.get_obs()
        share_obs = self.get_state()

        # 计算各智能体的奖励
        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]
            
        # 计算团队平均奖励（红/蓝队分别取平均）
        ego_reward = np.mean([rewards[ego_id] for ego_id in self.ego_ids])
        enm_reward = np.mean([rewards[enm_id] for enm_id in self.enm_ids])
        
        # 将团队平均奖励分配给团队中的每个成员
        for ego_id in self.ego_ids:
            rewards[ego_id] = [ego_reward]
        for enm_id in self.enm_ids:
            rewards[enm_id] = [enm_reward]

        # 判断各智能体是否完成任务
        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        return self._pack(obs), self._pack(share_obs), self._pack(rewards), self._pack(dones), info

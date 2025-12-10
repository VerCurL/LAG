import numpy as np
import math
from .env_base import BaseEnv
from ..tasks import HierarchicalSingleCombatShootTask, LC_HierarchicalSingleCombatShootTask

class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str, policy_type: str):
        super().__init__(config_name, policy_type)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)

        if taskname == 'hierarchical_singlecombat_shoot':
            if self.config.policy_type == "default":
                self.task = HierarchicalSingleCombatShootTask(self.config)
            elif self.config.policy_type == "lc":
                self.task = LC_HierarchicalSingleCombatShootTask(self.config)
            else:
                raise NotImplementedError(f"Policy type {self.config.policy_type} not implemented!")
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        """
        1v1 飞机重置逻辑：
        - 第一次重置时保存两个飞机的基本初始状态（位置、朝向、速度等）
        - 每次重置时：
            1. 随机扰动（经纬度、高度、航向、速度）
            2. 随机交换红蓝机位置（swap）
        """
        # ========== 常量 ==========
        KM_PER_DEG_LAT = 111.132
        KM_PER_DEG_LON_EQ = 111.320
        FT_PER_METER = 3.28084

        # 扰动参数：你可以根据需要调整
        max_radius_km = 3.0  # 随机散布半径
        min_alt_ft = 18000
        max_alt_ft = 24000
        min_speed_fps = 300  # 飞机速度扰动（fps）
        max_speed_fps = 500

        # ========== 1. 首次重置时，保存初始状态 ==========
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]

        # 两个飞机的原始初始状态（蓝、红）
        base_states = [s.copy() for s in self.init_states]

        # ========== 2. 随机扰动每个飞机的初始点 ==========
        # 注意：此处先扰动 base_states，然后再执行 swap

        new_states = []
        for state in base_states:
            # 取初始经纬度
            base_lat = state["ic_lat_geod_deg"]
            base_lon = state["ic_long_gc_deg"]

            # 计算当前纬度处经度换算因子
            km_per_deg_lon = KM_PER_DEG_LON_EQ * math.cos(math.radians(base_lat))

            # -------- 生成随机散布 --------
            offset_angle = self.np_random.uniform(0, 2 * math.pi)
            offset_dist = self.np_random.uniform(0, max_radius_km)

            dlat = (offset_dist * math.cos(offset_angle)) / KM_PER_DEG_LAT
            dlon = (offset_dist * math.sin(offset_angle)) / km_per_deg_lon

            # -------- 随机高度/航向/速度 --------
            new_alt_ft = self.np_random.uniform(min_alt_ft, max_alt_ft)
            new_heading = self.np_random.uniform(0, 360)
            new_speed = self.np_random.uniform(min_speed_fps, max_speed_fps)

            # -------- 构造新状态 --------
            new_state = state.copy()
            new_state.update({
                "ic_lat_geod_deg": base_lat + dlat,
                "ic_long_gc_deg": base_lon + dlon,
                "ic_h_sl_ft": new_alt_ft,
                "ic_psi_true_deg": new_heading,
                "ic_u_fps": new_speed,
            })

            new_states.append(new_state)

        # ========== 3. swap（随机交换蓝红机位置） ==========
        self.np_random.shuffle(new_states)

        # ========== 4. 加载到环境中的两架飞机 ==========
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(new_states[idx])

        # 清理导弹/临时模拟器
        self._tempsims.clear()

    # def reset_simulators(self):
    #     # switch side
    #     if self.init_states is None:
    #         self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
    #     # self.init_states[0].update({
    #     #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
    #     #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
    #     # })
    #     init_states = self.init_states.copy()
    #     self.np_random.shuffle(init_states)
    #     for idx, sim in enumerate(self.agents.values()):
    #         sim.reload(init_states[idx])
    #     self._tempsims.clear()

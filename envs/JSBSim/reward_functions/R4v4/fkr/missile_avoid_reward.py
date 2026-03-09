import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class MissileAvoidReward(BaseRewardFunction):
    """
    平滑版导弹规避奖励

    设计思路：
    1. 鼓励我机速度方向与导弹来袭 LOS 方向尽量正交（beam/横向脱离）
    2. 鼓励导弹闭合速度下降
    3. 鼓励导弹距离停止减小甚至开始增大
    4. 对多导弹取平均并裁剪，避免奖励爆炸
    """
    def __init__(self, config):
        super().__init__(config)

        # 历史信息：{agent_id: {missile_uid: {"range":..., "closing_speed":...}}}
        self.pre_missiles_info = {}

        # 权重
        self.w_angle = getattr(self.config, f'{self.__class__.__name__}_w_angle', 0.8)
        self.w_closing = getattr(self.config, f'{self.__class__.__name__}_w_closing', 1.2)
        self.w_range = getattr(self.config, f'{self.__class__.__name__}_w_range', 0.8)

        # 归一化尺度
        self.closing_speed_scale = getattr(
            self.config, f'{self.__class__.__name__}_closing_speed_scale', 80.0
        )   # m/s
        self.range_scale = getattr(
            self.config, f'{self.__class__.__name__}_range_scale', 80.0
        )   # m

        # 总奖励裁剪
        self.max_reward_clip = getattr(
            self.config, f'{self.__class__.__name__}_max_reward_clip', 2.5
        )

        self.reward_item_names = [
            self.__class__.__name__ + item
            for item in ['', '_angle', '_closing', '_range']
        ]

    def reset(self, task, env):
        self.pre_missiles_info.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]

        if agent_id not in self.pre_missiles_info:
            self.pre_missiles_info[agent_id] = {}

        missiles = agent.check_all_missile_warning()

        # 只保留当前仍存活的来袭导弹
        alive_missiles = [m for m in missiles if m.is_alive]

        if len(alive_missiles) == 0:
            # 清掉已经失效的历史信息，避免字典无限增长
            self.pre_missiles_info[agent_id].clear()
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0))

        ego_pos = agent.get_position()
        ego_vel = agent.get_velocity()

        per_missile_rewards = []
        angle_terms = []
        closing_terms = []
        range_terms = []

        current_alive_ids = set()

        eps = 1e-6

        for sim in alive_missiles:
            current_alive_ids.add(sim.uid)

            sim_pos = sim.get_position()
            sim_vel = sim.get_velocity()

            rel_pos = ego_pos - sim_pos
            R = np.linalg.norm(rel_pos) + eps
            los_dir = rel_pos / R

            rel_vel = sim_vel - ego_vel

            # 闭合速度：正值表示导弹沿 LOS 正在逼近我机
            closing_speed = float(np.dot(rel_vel, los_dir))

            # -----------------------------
            # 1) 横向脱离几何奖励
            # 我机速度方向与 LOS 方向越接近 90° 越好
            # sin^2(theta) = 1 - cos^2(theta)
            # -----------------------------
            ego_speed = np.linalg.norm(ego_vel) + eps
            cos_theta = np.clip(np.dot(ego_vel, los_dir) / ego_speed, -1.0, 1.0)
            angle_reward = 1.0 - cos_theta ** 2   # in [0, 1]

            # 首次观测该导弹时，不计算趋势项，只记录
            if sim.uid not in self.pre_missiles_info[agent_id]:
                self.pre_missiles_info[agent_id][sim.uid] = {
                    "range": R,
                    "closing_speed": closing_speed
                }
                closing_reward = 0.0
                range_reward = 0.0
            else:
                prev_R = self.pre_missiles_info[agent_id][sim.uid]["range"]
                prev_closing = self.pre_missiles_info[agent_id][sim.uid]["closing_speed"]

                # -----------------------------
                # 2) 闭合速度改善奖励
                # 若当前闭合速度比上一时刻更小，则奖励为正
                # -----------------------------
                delta_closing = prev_closing - closing_speed
                closing_reward = np.tanh(delta_closing / self.closing_speed_scale)

                # -----------------------------
                # 3) 距离变化趋势奖励
                # 若距离开始增大，则奖励为正
                # -----------------------------
                delta_range = R - prev_R
                range_reward = np.tanh(delta_range / self.range_scale)

                # 更新历史
                self.pre_missiles_info[agent_id][sim.uid]["range"] = R
                self.pre_missiles_info[agent_id][sim.uid]["closing_speed"] = closing_speed

            missile_reward = (
                self.w_angle * angle_reward +
                self.w_closing * closing_reward +
                self.w_range * range_reward
            )

            per_missile_rewards.append(missile_reward)
            angle_terms.append(angle_reward)
            closing_terms.append(closing_reward)
            range_terms.append(range_reward)

        # 删除本步已经不再存活/告警的导弹历史信息
        stale_ids = [
            uid for uid in self.pre_missiles_info[agent_id].keys()
            if uid not in current_alive_ids
        ]
        for uid in stale_ids:
            del self.pre_missiles_info[agent_id][uid]

        # 多导弹取平均，避免奖励线性爆炸
        raw_reward = float(np.mean(per_missile_rewards))

        # 总裁剪，保证 reward scale 稳定
        new_reward = float(np.clip(raw_reward, -self.max_reward_clip, self.max_reward_clip))

        mean_angle = float(np.mean(angle_terms)) if angle_terms else 0.0
        mean_closing = float(np.mean(closing_terms)) if closing_terms else 0.0
        mean_range = float(np.mean(range_terms)) if range_terms else 0.0

        return self._process(new_reward, agent_id, (mean_angle, mean_closing, mean_range))
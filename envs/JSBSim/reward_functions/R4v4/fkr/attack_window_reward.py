import numpy as np

from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R, get_near_offset_of_multi_air


class AttackWindowReward(BaseRewardFunction):
    """
    平滑版攻击窗口奖励

    设计目标：
    1. 奖励低 AO / 高 TA / 合适距离的优势占位
    2. 用“绝对得分 + 小幅趋势奖励”替代“大权重差分奖励”
    3. 用 softmax 聚合多敌机，减少目标切换尖峰
    4. 用 EMA 平滑时间序列，降低单步方差
    5. 用 sigmoid 门控替代硬距离门控
    """

    def __init__(self, config):
        super().__init__(config)

        # 单敌机评分权重（总和建议为 1）
        self.w_ao = getattr(self.config, f'{self.__class__.__name__}_w_ao', 0.45)
        self.w_ta = getattr(self.config, f'{self.__class__.__name__}_w_ta', 0.30)
        self.w_r = getattr(self.config, f'{self.__class__.__name__}_w_r', 0.25)

        # 总奖励：绝对得分 + 趋势项
        self.w_score = getattr(self.config, f'{self.__class__.__name__}_w_score', 1.2)
        self.w_progress = getattr(self.config, f'{self.__class__.__name__}_w_progress', 0.8)
        self.w_reward = 2.5

        # 趋势项的温度参数（越小越敏感）
        self.progress_scale = getattr(
            self.config, f'{self.__class__.__name__}_progress_scale', 0.05
        )

        # 多敌机 softmax 聚合温度
        self.softmax_beta = getattr(
            self.config, f'{self.__class__.__name__}_softmax_beta', 4.0
        )

        # EMA 平滑系数
        self.ema_decay = getattr(
            self.config, f'{self.__class__.__name__}_ema_decay', 0.8
        )

        # 距离门控参数（单位：km）
        self.gate_center = getattr(
            self.config, f'{self.__class__.__name__}_gate_center', 24.0
        )
        self.gate_width = getattr(
            self.config, f'{self.__class__.__name__}_gate_width', 3.0
        )

        # 总奖励裁剪
        self.reward_clip = getattr(
            self.config, f'{self.__class__.__name__}_reward_clip', 3
        )

        # 历史平滑分数
        self.pre_scores = {}

        self.reward_item_names = [
            self.__class__.__name__ + item
            for item in ['', '_score', '_progress', '_gate']
        ]

    def reset(self, task, env):
        self.pre_scores.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        reward = 0.0
        score_term = 0.0
        progress_term = 0.0
        gate = 0.0

        # 无导弹时才继续占位；有导弹威胁时交给 MissileAvoidReward
        if agent.num_left_missiles <= 0:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0))

        if len(agent.check_all_missile_warning()) > 0:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0))

        ego_feature = np.hstack([agent.get_position(), agent.get_velocity()])

        scores = []
        states = {}

        for enm in agent.enemies:
            if not enm.is_alive:
                continue

            enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(enm_feature, ego_feature)
            R_km = R / 1000.0

            s = self.get_score(task, AO, TA, R_km)
            scores.append(s)
            states[enm.uid] = enm.get_position()

        if len(scores) == 0:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0))

        # 计算 offset 距离，用于平滑门控
        ego_position = agent.get_position()
        enm_positions = np.array([position for position in states.values()])
        near_offset_position, _ = get_near_offset_of_multi_air(ego_position, enm_positions)

        center_feature = np.hstack([near_offset_position, np.array([0.0, 0.0, 0.0])])
        _, _, R_offset = get_AO_TA_R(ego_feature, center_feature)
        R_offset_km = R_offset / 1000.0

        # -----------------------------
        # 1) 多敌机 softmax 聚合
        # -----------------------------
        scores = np.asarray(scores, dtype=np.float64)
        logits = self.softmax_beta * scores
        logits = logits - np.max(logits)  # 数值稳定
        weights = np.exp(logits)
        weights = weights / (np.sum(weights) + 1e-8)

        current_score = float(np.sum(weights * scores))

        # -----------------------------
        # 2) 时间 EMA 平滑
        # -----------------------------
        if agent_id not in self.pre_scores:
            self.pre_scores[agent_id] = current_score

        prev_score = self.pre_scores[agent_id]
        smooth_score = self.ema_decay * prev_score + (1.0 - self.ema_decay) * current_score

        # -----------------------------
        # 3) 平滑距离门控
        # gate in (0, 1)
        # -----------------------------
        gate = 1.0 / (1.0 + np.exp((R_offset_km - self.gate_center) / self.gate_width))

        # -----------------------------
        # 4) 当前站位奖励 + 趋势奖励（限幅）
        # -----------------------------
        score_term = self.w_score * smooth_score
        progress_term = self.w_progress * np.tanh(
            (smooth_score - prev_score) / self.progress_scale
        )

        reward = self.w_reward * gate * (score_term + progress_term)
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        # 更新历史
        self.pre_scores[agent_id] = smooth_score

        return self._process(reward, agent_id, (score_term, progress_term, gate))

    def get_score(self, task, AO, TA, R):
        """
        返回 [0, 1] 附近的平滑站位评分
        越接近理想攻击窗口，分数越高
        """

        # -----------------------------
        # 1) AO：越小越好
        # -----------------------------
        sigma_ao = 0.6 * task.max_missile_attack_AO
        s_ao = np.exp(- (AO / (sigma_ao + 1e-8)) ** 2)

        # -----------------------------
        # 2) TA：越接近 pi 越好
        # -----------------------------
        sigma_ta = 0.5
        s_ta = np.exp(- ((np.pi - TA) / sigma_ta) ** 2)

        # -----------------------------
        # 3) R：越接近窗口中部越好
        # 你也可以把 center 改成 10.0
        # -----------------------------
        r_min = task.min_missile_attack_distance
        r_max = task.max_missile_attack_distance
        r_center = 0.5 * (r_min + r_max)
        sigma_r = 0.35 * (r_max - r_min)

        s_r = np.exp(- ((R - r_center) / (sigma_r + 1e-8)) ** 2)

        score = self.w_ao * s_ao + self.w_ta * s_ta + self.w_r * s_r
        return float(np.clip(score, 0.0, 1.0))
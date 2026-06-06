import math
import numpy as np
from ..utils.utils import get_AO_TA_R


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

A_TEAM = 0
A_ALIVE = 1
A_SHOTDOWN = 2
A_CRASH = 3
A_LEFT_MISSILES = 4
A_POS = slice(5, 8)
A_VEL = slice(8, 11)
A_SPEED = 14
A_THREAT_MISSILE_EXIST = 15
A_THREAT_MISSILE_POS = slice(16, 19)
A_THREAT_MISSILE_VEL = slice(19, 22)
A_SHOOT_MISSILE_0 = 22
A_SHOOT_MISSILE_1 = 23

M_ALIVE = 0
M_SUCCESS = 1
M_DONE = 2
M_STATUS = 3
M_PARENT = 4
M_TARGET = 5
M_POS = slice(6, 9)
M_VEL = slice(9, 12)
M_SPEED = 12
M_TARGET_DISTANCE = 13

class FieldCalculator:
    def __init__(self, k_step=20, gamma=0.95, ego_team=0.0, r_min=4000.0, r_attack=14000.0, r_nez=10000.0,
                 theta_attack=np.deg2rad(60.0), theta_nez=np.deg2rad(30.0)):
        self.k_step = k_step
        self.gamma = gamma
        self.ego_team = ego_team
        self.r_min = r_min
        self.r_attack = r_attack
        self.r_nez = r_nez
        self.theta_attack = theta_attack
        self.theta_nez = theta_nez

    def build_targets(self, snapshots, shared_buffer):
        """
        snapshots: List[n_envs][T]，每个元素是 {"aircraft": ..., "missiles": ...}
        shared_buffer.obs[:-1]: [T, n_envs, n_agents, obs_dim]

        Returns:
            threat_targets: [n_envs * T, 1]
            attack_targets: [n_envs * T, 1]
        """
        T = shared_buffer.actions.shape[0]
        n_envs = shared_buffer.actions.shape[1]
        masks = shared_buffer.masks

        threat_targets = np.zeros((n_envs * T, 1), dtype=np.float32)
        attack_targets = np.zeros((n_envs * T, 1), dtype=np.float32)

        geom_cache = self._precompute_all_geometry_vectorized(
            snapshots=snapshots,
            T=T,
            n_envs=n_envs,
        )

        for env_i in range(n_envs):
            env_snapshots = snapshots[env_i]

            instant_threat = np.zeros(T, dtype=np.float32)
            instant_attack = np.zeros(T, dtype=np.float32)

            # 1. 每个 step 只计算一次瞬时团队威胁场 / 进攻场
            for t in range(T):
                geom_t = (
                    geom_cache["AO"][env_i, t],
                    geom_cache["TA"][env_i, t],
                    geom_cache["R"][env_i, t],
                )

                prev_snapshot = None
                if t > 0 and not np.all(masks[t, env_i] <= 0.0):
                    prev_snapshot = env_snapshots[t - 1]

                instant_threat[t], instant_attack[t] = self.instant_team_field(
                    env_snapshots[t],
                    geom_t,
                    prev_snapshot = prev_snapshot
                )

            # 2. 根据 masks 切分 episode，避免 K 步窗口跨越新一局
            base = env_i * T
            for start, end in self._episode_segments(T, masks, env_i):
                threat_seg = self._discounted_k_window(instant_threat[start:end])
                attack_seg = self._discounted_k_window(instant_attack[start:end])

                threat_targets[base + start:base + end, 0] = threat_seg
                attack_targets[base + start:base + end, 0] = attack_seg

        return threat_targets, attack_targets

    def instant_team_field(self, snapshot, geom, prev_snapshot=None):
        """
        计算当前 step 的团队瞬时场值。
        这里只对 ego_team 的飞机取平均。
        """
        aircraft = snapshot["aircraft"]
        team_mask = aircraft[:, A_TEAM] == self.ego_team
        team_indices = np.where(team_mask)[0]

        threat_vals = []
        attack_vals = []

        if len(team_indices) == 0:
            return 0.0, 0.0

        for ego_idx in team_indices:
            threat_vals.append(self.instant_threat_agent(snapshot, ego_idx, geom, prev_snapshot=prev_snapshot))
            attack_vals.append(self.instant_attack_agent(snapshot, ego_idx, geom, prev_snapshot=prev_snapshot))

        return float(np.mean(threat_vals)), float(np.mean(attack_vals))

    def instant_threat_agent(self, snapshot, ego_idx, geom, prev_snapshot=None):
        aircraft = snapshot["aircraft"]
        ego = aircraft[ego_idx]
        AO_mat, TA_mat, R_mat = geom

        # ⭐如果我机被击落，返回 1 点威胁值
        if ego[A_SHOTDOWN] > 0.5:
            return 1.0

        # 得到飞机ego的敌机列表
        enemies = [
            j for j, row in enumerate(aircraft)
            if row[A_TEAM] != ego[A_TEAM] and row[A_ALIVE] > 0.5
        ]

        # 获取具有威胁的飞机
        threat_enemies = [
            j for j in enemies
            if aircraft[j, A_LEFT_MISSILES] > 0
        ]

        # 计算每个敌机产生的威胁场
        n_enemy = len(threat_enemies)
        if n_enemy == 0:
            nez_score = 0.0
            attack_score = 0.0
        else:
            c_nez = 0
            c_attack = 0

            for j in threat_enemies:
                enm = aircraft[j]

                # 1. 计算和敌机的位置关系
                TA = float(TA_mat[ego_idx, j])
                R = float(R_mat[ego_idx, j])
                closing = float(
                    np.dot(
                        enm[A_VEL] - ego[A_VEL],
                        (ego[A_POS] - enm[A_POS]) / (R + 1e-8),
                    )
                )

                # 2. 是否进入敌机的攻击区
                c_attack += int(
                    self.r_min <= R <= self.r_attack
                    and np.pi - TA <= self.theta_attack
                )

                # 3. 是否进入敌机的不可逃逸区
                c_nez += int(
                    R <= self.r_nez
                    and np.pi - TA <= self.theta_nez
                    and closing > 0.0
                )

            # 平均每个敌机产生的威胁场
            nez_score = min(c_nez / n_enemy, 1.0)
            attack_score = min(c_attack / n_enemy, 1.0)

        missile_score = self.incoming_missile_score(snapshot, ego_idx)

        return float(np.clip(0.45 * missile_score + 0.35 * nez_score + 0.2 * attack_score, 0.0, 1.0))

    def instant_attack_agent(self, snapshot, ego_idx, geom, prev_snapshot=None):
        aircraft = snapshot["aircraft"]
        ego = aircraft[ego_idx]
        AO_mat, TA_mat, R_mat = geom

        # ⭐如果我机阵亡，不产生进攻场
        if ego[A_ALIVE] <= 0.5:
            return 0.0

        # 初始敌机集合：不管当前是否存活，只按队伍归属统计
        all_enemies = [
            j for j, row in enumerate(aircraft)
            if row[A_TEAM] != ego[A_TEAM]
        ]

        n_enemy_total = len(all_enemies)
        if n_enemy_total == 0:
            return 0.0

        # 当前仍存活的敌机集合
        alive_enemies = [
            j for j in all_enemies
            if aircraft[j, A_ALIVE] > 0.5
        ]

        n_enemy_alive = len(alive_enemies)

        # 战果进度分：敌方少一架，基础进攻分至少增加 1 / 初始敌机数
        progress_score = (n_enemy_total - n_enemy_alive) / n_enemy_total
        progress_score = float(np.clip(progress_score, 0.0, 1.0))

        # 敌方全灭，进攻场达到最大
        if n_enemy_alive == 0:
            return 1.0

        c_nez = 0
        c_attack = 0

        # 如果我机有剩余导弹，才计算当前对存活敌机的攻击包线 / NEZ 覆盖
        if ego[A_LEFT_MISSILES] > 0:
            for j in alive_enemies:
                enm = aircraft[j]

                # 直接从预计算矩阵中读取 AO / R
                AO = float(AO_mat[ego_idx, j])
                R = float(R_mat[ego_idx, j])
                closing = float(
                    np.dot(
                        ego[A_VEL] - enm[A_VEL],
                        (enm[A_POS] - ego[A_POS]) / (R + 1e-8),
                    )
                )

                # 敌机是否进入我机攻击区
                c_attack += int(
                    self.r_min <= R <= self.r_attack
                    and AO <= self.theta_attack
                )

                # 敌机是否进入我机不可逃逸区
                c_nez += int(
                    R <= self.r_nez
                    and AO <= self.theta_nez
                    and closing > 0.0
                )

        # hit_score 建议按初始敌机数归一化，而不是当前存活敌机数
        hit_score = min(self.hit_count(snapshot, prev_snapshot, ego_idx) / n_enemy_total, 1.0)

        # missile / NEZ / attack 仍然表达当前瞬时攻击能力
        missile_score = self.outgoing_missile_score(snapshot, ego_idx)
        nez_score = min(c_nez / n_enemy_alive, 1.0)
        attack_score = min(c_attack / n_enemy_alive, 1.0)

        dynamic_score = float(
            np.clip(
                0.35 * hit_score
                + 0.3 * missile_score
                + 0.2 * nez_score
                + 0.15 * attack_score,
                0.0,
                1.0,
            )
        )

        # 最终进攻场 = 已取得战果基础分 + 剩余空间内的瞬时进攻能力
        final_score = progress_score + (1.0 - progress_score) * dynamic_score

        return float(np.clip(final_score, 0.0, 1.0))

    def hit_count(self, snapshot, prev_snapshot, ego_idx):
        missiles = snapshot["missiles"]
        parent_match = missiles[:, M_PARENT].astype(int) == ego_idx
        success_now = missiles[:, M_SUCCESS] > 0.5

        if prev_snapshot is None:
            return int(np.sum(parent_match & success_now))

        prev_missiles = prev_snapshot["missiles"]
        n = min(len(missiles), len(prev_missiles))

        success_prev = np.zeros(len(missiles), dtype=bool)
        success_prev[:n] = prev_missiles[:n, M_SUCCESS] > 0.5

        return int(np.sum(parent_match & success_now & (~success_prev)))

    def incoming_missile_score(self, snapshot, ego_idx):
        aircraft = snapshot["aircraft"]
        ego = aircraft[ego_idx]

        if ego[A_THREAT_MISSILE_EXIST] <= 0.5:
            return 0.0

        m_pos = ego[A_THREAT_MISSILE_POS]
        m_vel = ego[A_THREAT_MISSILE_VEL]

        rel = ego[A_POS] - m_pos
        dist = np.linalg.norm(rel)

        closing = float(np.dot(m_vel - ego[A_VEL], rel / (dist + 1e-8)))

        dist_score = np.clip(1.0 - dist / self.r_attack, 0.0, 1.0)
        closing_score = _sigmoid(closing / 300.0)

        return float(np.clip(dist_score * closing_score, 0.0, 1.0))

    def outgoing_missile_score(self, snapshot, ego_idx):
        aircraft = snapshot["aircraft"]
        missiles = snapshot["missiles"]

        parent_match = missiles[:, M_PARENT].astype(int) == ego_idx
        alive = missiles[:, M_ALIVE] > 0.5
        success = missiles[:, M_SUCCESS] > 0.5

        # 只看当前 ego 发射、仍在飞行、尚未命中的导弹
        valid = parent_match & alive & (~success)

        if not np.any(valid):
            return 0.0

        scores = []

        for m in missiles[valid]:
            target_idx = int(m[M_TARGET])

            if target_idx < 0 or target_idx >= len(aircraft):
                continue

            target = aircraft[target_idx]

            if target[A_ALIVE] <= 0.5:
                continue

            m_pos = m[M_POS]
            m_vel = m[M_VEL]
            tgt_pos = target[A_POS]
            tgt_vel = target[A_VEL]

            rel = tgt_pos - m_pos
            dist = np.linalg.norm(rel)

            # 导弹相对目标的接近速度
            # > 0 表示导弹正在接近目标
            closing = float(
                np.dot(
                    m_vel - tgt_vel,
                    rel / (dist + 1e-8),
                )
            )

            # 距离越近，得分越高
            dist_score = np.clip(1.0 - dist / self.r_attack, 0.0, 1.0)

            # 接近速度越大，得分越高
            # 300 可理解为速度尺度，可后续调参
            closing_score = _sigmoid(closing / 300.0)

            # 如果导弹已经远离目标，直接压低得分
            if closing <= 0.0:
                closing_score *= 0.2

            scores.append(float(dist_score * closing_score))

        if not scores:
            return 0.0

        # 取 max 表示最有威胁的一枚我方在途导弹
        return float(np.clip(max(scores), 0.0, 1.0))

    def _discounted_k_window(self, values):
        """
        对一个 episode 段内的瞬时场值做 K 步折扣窗口递推。

        values: [L]
        return: [L]
        """
        L = len(values)
        out = np.zeros(L, dtype=np.float32)
        gamma_k = self.gamma ** self.k_step

        nums = np.zeros(L + 1, dtype=np.float32)
        dens = np.zeros(L + 1, dtype=np.float32)

        for t in range(L - 1, -1, -1):
            num = float(values[t]) + self.gamma * float(nums[t + 1])
            den = 1.0 + self.gamma * float(dens[t + 1])

            drop_idx = t + self.k_step
            if drop_idx < L:
                num -= gamma_k * float(values[drop_idx])
                den -= gamma_k

            nums[t] = num
            dens[t] = den
            out[t] = num / (den + 1e-6)

        return out

    def _episode_segments(self, T, masks, env_i):
        """
        根据 masks 把 rollout 切成若干个 episode 段，每个 episode 段是一局战斗

        约定：
        masks[t, env_i] 全 0 表示 t 是 reset 后新 episode 的起点，
        因此上一段应该在 t 前结束。
        """
        segments = []
        start = 0

        for t in range(1, T):
            if np.all(masks[t, env_i] <= 0.0):
                if start < t:
                    segments.append((start, t))
                start = t

        if start < T:
            segments.append((start, T))

        return segments

    def _precompute_all_geometry_vectorized(self, snapshots, T, n_envs):
        """
        一次性预计算所有 env、所有 step 中飞机之间的 AO / TA / R。

        返回:
            {
                "AO": [n_envs, T, n_aircraft, n_aircraft],
                "TA": [n_envs, T, n_aircraft, n_aircraft],
                "R":  [n_envs, T, n_aircraft, n_aircraft],
            }

        说明：
        - 只显式计算 ego_team 与 enemy_team 之间的一半 pair。
        - 反方向通过几何关系补齐：
            AO(j, i) = pi - TA(i, j)
            TA(j, i) = pi - AO(i, j)
            R(j, i)  = R(i, j)
        """
        # aircraft_all: [n_envs, T, n_aircraft, obs_dim]
        aircraft_all = np.stack(
            [
                np.stack(
                    [snapshots[env_i][t]["aircraft"] for t in range(T)],
                    axis=0,
                )
                for env_i in range(n_envs)
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        n_aircraft = aircraft_all.shape[2]

        # 默认假设 aircraft index 和 team 归属在 rollout 内固定
        aircraft0 = aircraft_all[0, 0]
        ego_indices = np.where(aircraft0[:, A_TEAM] == self.ego_team)[0]
        enemy_indices = np.where(aircraft0[:, A_TEAM] != self.ego_team)[0]

        # ego: [E, T, n_ego, obs_dim]
        # enm: [E, T, n_enemy, obs_dim]
        ego = aircraft_all[:, :, ego_indices, :]
        enm = aircraft_all[:, :, enemy_indices, :]

        # 广播成所有 ego-enemy pair
        # ego_pos: [E, T, n_ego, 1, 3]
        # enm_pos: [E, T, 1, n_enemy, 3]
        ego_pos = ego[:, :, :, None, A_POS]
        ego_vel = ego[:, :, :, None, A_VEL]

        enm_pos = enm[:, :, None, :, A_POS]
        enm_vel = enm[:, :, None, :, A_VEL]

        # delta: ego -> enemy
        # shape: [E, T, n_ego, n_enemy, 3]
        delta = enm_pos - ego_pos

        # R: [E, T, n_ego, n_enemy]
        R = np.linalg.norm(delta, axis=-1)

        # ego_v: [E, T, n_ego, 1]
        # enm_v: [E, T, 1, n_enemy]
        ego_v = np.linalg.norm(ego_vel, axis=-1)
        enm_v = np.linalg.norm(enm_vel, axis=-1)

        ego_proj = np.sum(delta * ego_vel, axis=-1)
        enm_proj = np.sum(delta * enm_vel, axis=-1)

        AO = np.arccos(
            np.clip(
                ego_proj / (R * ego_v + 1e-8),
                -1.0,
                1.0,
            )
        )

        TA = np.arccos(
            np.clip(
                enm_proj / (R * enm_v + 1e-8),
                -1.0,
                1.0,
            )
        )

        AO = AO.astype(np.float32, copy=False)
        TA = TA.astype(np.float32, copy=False)
        R = R.astype(np.float32, copy=False)

        # 填入完整的 [E, T, n_aircraft, n_aircraft] 矩阵
        AO_all = np.zeros((n_envs, T, n_aircraft, n_aircraft), dtype=np.float32)
        TA_all = np.zeros((n_envs, T, n_aircraft, n_aircraft), dtype=np.float32)
        R_all = np.zeros((n_envs, T, n_aircraft, n_aircraft), dtype=np.float32)

        # 正方向：ego_team aircraft -> enemy aircraft
        # AO_all[:, :, ego_indices[a], enemy_indices[b]] = AO[:, :, a, b]
        AO_all[:, :, ego_indices[:, None], enemy_indices[None, :]] = AO
        TA_all[:, :, ego_indices[:, None], enemy_indices[None, :]] = TA
        R_all[:, :, ego_indices[:, None], enemy_indices[None, :]] = R

        # 反方向：enemy aircraft -> ego_team aircraft
        # AO(enemy, ego) = pi - TA(ego, enemy)
        # TA(enemy, ego) = pi - AO(ego, enemy)
        # R(enemy, ego)  = R(ego, enemy)
        AO_all[:, :, enemy_indices[:, None], ego_indices[None, :]] = (
            np.pi - np.swapaxes(TA, -1, -2)
        )
        TA_all[:, :, enemy_indices[:, None], ego_indices[None, :]] = (
            np.pi - np.swapaxes(AO, -1, -2)
        )
        R_all[:, :, enemy_indices[:, None], ego_indices[None, :]] = np.swapaxes(
            R,
            -1,
            -2,
        )

        return {
            "AO": AO_all,
            "TA": TA_all,
            "R": R_all,
            "ego_indices": ego_indices,
            "enemy_indices": enemy_indices,
        }
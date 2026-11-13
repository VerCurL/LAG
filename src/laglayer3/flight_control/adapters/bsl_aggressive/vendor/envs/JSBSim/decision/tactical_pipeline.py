import numpy as np

from .adaptive_kf import AdaptiveCVKalmanFilter3D, compute_relative_features, sigmoid
from .threat_assessment import ThreatAssessment
from .mode_manager import ModeManager


class TacticalPipeline:
    """
    Orchestrates:
      EnemyKF + MissileKF -> ThreatAssessment -> ModeManager -> permit
    """

    def __init__(self, config, dt: float):
        self.config = config
        self.dt = float(dt)

        self.attack_ao_deg = float(getattr(config, "Decision_attack_ao_deg", 38.0))
        self.attack_range_m = float(getattr(config, "Decision_attack_range_m", 6800.0))
        self.tau_ao_deg = float(getattr(config, "Decision_tau_ao_deg", 4.0))
        self.tau_range_m = float(getattr(config, "Decision_tau_range_m", 700.0))

        self.enemy_kf = AdaptiveCVKalmanFilter3D(
            dt=self.dt,
            use_vel_meas=bool(getattr(config, "KF_use_vel_meas", True)),
            sigma_a_base=float(getattr(config, "KF_enemy_sigma_a_base", 6.0)),
            sigma_a_min=float(getattr(config, "KF_enemy_sigma_a_min", 3.0)),
            sigma_a_max=float(getattr(config, "KF_enemy_sigma_a_max", 12.0)),
            sigma_pos_meas=float(getattr(config, "KF_enemy_sigma_pos_meas", 20.0)),
            sigma_vel_meas=float(getattr(config, "KF_enemy_sigma_vel_meas", 3.0)),
            p0_pos_std=float(getattr(config, "KF_enemy_p0_pos_std", 60.0)),
            p0_vel_std=float(getattr(config, "KF_enemy_p0_vel_std", 15.0)),
            k_lambda=float(getattr(config, "KF_enemy_k_lambda", 3.0)),
            k_acc=float(getattr(config, "KF_enemy_k_acc", 0.8)),
            k_nis=float(getattr(config, "KF_enemy_k_nis", 0.25)),
            q_scale_max=float(getattr(config, "KF_enemy_q_scale_max", 4.0)),
        )

        self.missile_kf = AdaptiveCVKalmanFilter3D(
            dt=self.dt,
            use_vel_meas=bool(getattr(config, "KF_use_vel_meas", True)),
            sigma_a_base=float(getattr(config, "KF_missile_sigma_a_base", 16.0)),
            sigma_a_min=float(getattr(config, "KF_missile_sigma_a_min", 8.0)),
            sigma_a_max=float(getattr(config, "KF_missile_sigma_a_max", 30.0)),
            sigma_pos_meas=float(getattr(config, "KF_missile_sigma_pos_meas", 15.0)),
            sigma_vel_meas=float(getattr(config, "KF_missile_sigma_vel_meas", 6.0)),
            p0_pos_std=float(getattr(config, "KF_missile_p0_pos_std", 80.0)),
            p0_vel_std=float(getattr(config, "KF_missile_p0_vel_std", 30.0)),
            k_lambda=float(getattr(config, "KF_missile_k_lambda", 6.0)),
            k_acc=float(getattr(config, "KF_missile_k_acc", 1.2)),
            k_nis=float(getattr(config, "KF_missile_k_nis", 0.25)),
            q_scale_max=float(getattr(config, "KF_missile_q_scale_max", 6.0)),
        )

        self.threat = ThreatAssessment(
            w_tgo=float(getattr(config, "Threat_w_tgo", 0.35)),
            w_dcpa=float(getattr(config, "Threat_w_dcpa", 0.35)),
            w_los_rate=float(getattr(config, "Threat_w_los", 0.20)),
            w_alert=float(getattr(config, "Threat_w_alert", 0.10)),
            tgo_ref_s=float(getattr(config, "Threat_tgo_ref_s", 8.0)),
            dcpa_ref_m=float(getattr(config, "Threat_dcpa_ref_m", 2500.0)),
            los_rate_ref=float(getattr(config, "Threat_los_ref", 0.08)),
            score_slope=float(getattr(config, "Threat_score_slope", 5.0)),
            score_bias=float(getattr(config, "Threat_score_bias", 0.5)),
            use_t_dist=bool(getattr(config, "Threat_use_t_dist", False)),
            t_nu=float(getattr(config, "Threat_t_nu", 5.0)),
        )

        self.mode = ModeManager(
            dt=self.dt,
            defend_enter=float(getattr(config, "Mode_defend_enter", 0.72)),
            defend_exit=float(getattr(config, "Mode_defend_exit", 0.55)),
            attack_ao_deg=self.attack_ao_deg,
            attack_range_m=self.attack_range_m,
            attack_threat_max=float(getattr(config, "Mode_attack_threat_max", 0.55)),
            recover_delta_es=float(getattr(config, "Mode_recover_delta_es", -1500.0)),
            recover_safety_margin=float(getattr(config, "Mode_recover_safety_margin", 0.20)),
            dwell_time_s=float(getattr(config, "Mode_dwell_time_s", 1.2)),
        )

        self.prev_enemy_rel_v = None
        self.prev_missile_rel_v = None
        self.last_enemy_los = 0.0
        self.last_missile_los = 0.0

    def reset(self):
        self.enemy_kf.reset()
        self.missile_kf.reset()
        self.mode.reset()

        self.prev_enemy_rel_v = None
        self.prev_missile_rel_v = None
        self.last_enemy_los = 0.0
        self.last_missile_los = 0.0

    def _rel_acc_norm(self, prev_v, cur_v):
        if prev_v is None:
            return 0.0
        return float(np.linalg.norm((cur_v - prev_v) / max(self.dt, 1e-6)))

    def update(
        self,
        step_idx: int,
        ego_pos: np.ndarray,
        ego_vel: np.ndarray,
        enemy_pos: np.ndarray,
        enemy_vel: np.ndarray,
        ao_deg: float,
        delta_es: float,
        safety_margin: float,
        missile_alert: bool,
        missile_pos: np.ndarray = None,
        missile_vel: np.ndarray = None,
    ):
        enemy_r = enemy_pos - ego_pos
        enemy_v = enemy_vel - ego_vel
        enemy_z = np.concatenate([enemy_r, enemy_v]).astype(np.float64)

        enemy_rel_acc = self._rel_acc_norm(self.prev_enemy_rel_v, enemy_v)
        enemy_kf_out = self.enemy_kf.step(
            enemy_z,
            los_rate=self.last_enemy_los,
            rel_acc_norm=enemy_rel_acc,
        )
        enemy_feat = compute_relative_features(enemy_kf_out["x_hat"])
        self.last_enemy_los = enemy_feat["los_rate_rad_s"]
        self.prev_enemy_rel_v = enemy_v.copy()

        missile_feat = None
        missile_kf_out = None
        if missile_alert and missile_pos is not None and missile_vel is not None:
            missile_r = missile_pos - ego_pos
            missile_v = missile_vel - ego_vel
            missile_z = np.concatenate([missile_r, missile_v]).astype(np.float64)

            missile_rel_acc = self._rel_acc_norm(self.prev_missile_rel_v, missile_v)
            missile_kf_out = self.missile_kf.step(
                missile_z,
                los_rate=self.last_missile_los,
                rel_acc_norm=missile_rel_acc,
            )
            missile_feat = compute_relative_features(missile_kf_out["x_hat"])
            self.last_missile_los = missile_feat["los_rate_rad_s"]
            self.prev_missile_rel_v = missile_v.copy()

        threat_base = missile_feat if (missile_alert and missile_feat is not None) else enemy_feat
        s_threat, threat_items = self.threat.score(threat_base, missile_alert=missile_alert)

        s_ao = float(sigmoid((self.attack_ao_deg - ao_deg) / max(self.tau_ao_deg, 1e-6)))
        s_r = float(sigmoid((self.attack_range_m - enemy_feat["range_m"]) / max(self.tau_range_m, 1e-6)))
        a_t = s_ao * s_r

        permit_gate = 1.0 - (1.0 - float(bool(missile_alert))) * (1.0 - a_t) * (1.0 - s_threat)

        mode_name, permit_mode = self.mode.step(
            step_idx=step_idx,
            s_threat=s_threat,
            ao_deg=ao_deg,
            range_m=enemy_feat["range_m"],
            missile_alert=missile_alert,
            delta_es=delta_es,
            safety_margin=safety_margin,
        )
        permit = float(max(permit_gate, permit_mode))

        return {
            "mode": mode_name,
            "permit": permit,
            "s_threat": float(s_threat),
            "gate_geo": float(a_t),
            "enemy_feat": enemy_feat,
            "missile_feat": missile_feat,
            "enemy_kf": enemy_kf_out,
            "missile_kf": missile_kf_out,
            "threat_items": threat_items,
        }

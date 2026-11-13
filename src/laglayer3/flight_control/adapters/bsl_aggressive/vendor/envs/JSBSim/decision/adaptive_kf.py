import numpy as np

EPS = 1e-6


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def chi2_gate_threshold(obs_dim: int, conf: float = 0.99) -> float:
    # 99% chi-square thresholds for common dimensions
    table = {
        3: 11.345,  # position-only
        6: 16.812,  # position+velocity
    }
    return table.get(obs_dim, 16.812)


def compute_relative_features(x_hat: np.ndarray, vc_eps: float = 1.0, t_cpa_max: float = 20.0):
    r = x_hat[:3]
    v = x_hat[3:]
    range_m = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))

    vc = float(-(np.dot(r, v)) / (range_m + EPS))
    if vc > vc_eps:
        tgo_s = float(range_m / vc)
    else:
        tgo_s = float(np.inf)

    los_rate = float(np.linalg.norm(np.cross(r, v)) / (range_m * range_m + EPS))

    t_cpa = float(np.clip(-(np.dot(r, v)) / (v_norm * v_norm + EPS), 0.0, t_cpa_max))
    d_cpa = float(np.linalg.norm(r + t_cpa * v))

    return {
        "range_m": range_m,
        "closing_speed_mps": vc,
        "tgo_s": tgo_s,
        "los_rate_rad_s": los_rate,
        "dcpa_m": d_cpa,
        "tcpa_s": t_cpa,
        "r": r.copy(),
        "v": v.copy(),
    }


class AdaptiveCVKalmanFilter3D:
    """
    6D relative-state CV Kalman Filter:
        x = [rx, ry, rz, vx, vy, vz]
    """

    def __init__(
        self,
        dt: float,
        use_vel_meas: bool = True,
        sigma_a_base: float = 6.0,
        sigma_a_min: float = 3.0,
        sigma_a_max: float = 12.0,
        sigma_pos_meas: float = 20.0,
        sigma_vel_meas: float = 3.0,
        p0_pos_std: float = 60.0,
        p0_vel_std: float = 15.0,
        k_lambda: float = 3.0,
        k_acc: float = 0.8,
        k_nis: float = 0.25,
        q_scale_max: float = 4.0,
        gate_confidence: float = 0.99,
        enable_gating: bool = True,
    ):
        self.dt = float(dt)
        self.use_vel_meas = bool(use_vel_meas)

        self.sigma_a_base = float(sigma_a_base)
        self.sigma_a_min = float(sigma_a_min)
        self.sigma_a_max = float(sigma_a_max)

        self.sigma_pos_meas = float(sigma_pos_meas)
        self.sigma_vel_meas = float(sigma_vel_meas)

        self.p0_pos_std = float(p0_pos_std)
        self.p0_vel_std = float(p0_vel_std)

        self.k_lambda = float(k_lambda)
        self.k_acc = float(k_acc)
        self.k_nis = float(k_nis)
        self.q_scale_max = float(q_scale_max)

        self.enable_gating = bool(enable_gating)

        self.F = np.eye(6, dtype=np.float64)
        self.F[:3, 3:] = np.eye(3, dtype=np.float64) * self.dt

        if self.use_vel_meas:
            self.H = np.eye(6, dtype=np.float64)
            self.R = np.diag(
                [self.sigma_pos_meas**2] * 3 + [self.sigma_vel_meas**2] * 3
            ).astype(np.float64)
        else:
            self.H = np.zeros((3, 6), dtype=np.float64)
            self.H[:, :3] = np.eye(3, dtype=np.float64)
            self.R = np.diag([self.sigma_pos_meas**2] * 3).astype(np.float64)

        self.obs_dim = self.H.shape[0]
        self.gate_threshold = chi2_gate_threshold(self.obs_dim, gate_confidence)

        self.nis_ema = float(self.obs_dim)
        self.last_sigma_a = self.sigma_a_base

        self.reset()

    def reset(self, x0: np.ndarray = None):
        self.x = np.zeros(6, dtype=np.float64) if x0 is None else x0.astype(np.float64).copy()

        p_diag = [self.p0_pos_std**2] * 3 + [self.p0_vel_std**2] * 3
        self.P = np.diag(p_diag).astype(np.float64)
        self.initialized = x0 is not None

        self.nis_ema = float(self.obs_dim)
        self.last_sigma_a = self.sigma_a_base

    def _build_Q(self, sigma_a: float) -> np.ndarray:
        dt = self.dt
        q = sigma_a**2
        q_block = np.array(
            [
                [dt**4 / 4.0, dt**3 / 2.0],
                [dt**3 / 2.0, dt**2],
            ],
            dtype=np.float64,
        ) * q

        Q = np.zeros((6, 6), dtype=np.float64)
        for axis in range(3):
            i = axis
            j = axis + 3
            Q[np.ix_([i, j], [i, j])] = q_block
        return Q

    def _adaptive_sigma_a(self, los_rate: float, rel_acc_norm: float) -> float:
        nis_ratio = max(0.0, self.nis_ema / max(float(self.obs_dim), 1.0) - 1.0)
        scale = 1.0 + self.k_lambda * abs(los_rate) + self.k_acc * abs(rel_acc_norm) + self.k_nis * nis_ratio
        scale = float(np.clip(scale, 1.0, self.q_scale_max))
        sigma_a = self.sigma_a_base * scale
        sigma_a = float(np.clip(sigma_a, self.sigma_a_min, self.sigma_a_max))
        return sigma_a

    def predict(self, sigma_a: float):
        Q = self._build_Q(sigma_a)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, z: np.ndarray):
        z = z.astype(np.float64).reshape(-1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.pinv(S)

        nis = float(y.T @ S_inv @ y)
        accepted = True

        if self.enable_gating and nis > self.gate_threshold:
            accepted = False
        else:
            K = self.P @ self.H.T @ S_inv
            I = np.eye(6, dtype=np.float64)
            self.x = self.x + K @ y
            self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

        self.nis_ema = 0.9 * self.nis_ema + 0.1 * nis
        return accepted, nis

    def step(self, z: np.ndarray, los_rate: float = 0.0, rel_acc_norm: float = 0.0):
        z = z.astype(np.float64).reshape(-1)
        if not self.initialized:
            if self.use_vel_meas and z.shape[0] >= 6:
                self.x = z[:6].copy()
            else:
                self.x[:3] = z[:3].copy()
                self.x[3:] = 0.0
            self.initialized = True
            return {
                "x_hat": self.x.copy(),
                "P": self.P.copy(),
                "accepted": True,
                "nis": 0.0,
                "nis_ema": float(self.nis_ema),
                "sigma_a_used": self.sigma_a_base,
            }

        sigma_a = self._adaptive_sigma_a(los_rate=los_rate, rel_acc_norm=rel_acc_norm)
        self.last_sigma_a = sigma_a

        self.predict(sigma_a=sigma_a)
        accepted, nis = self.update(z=z)

        return {
            "x_hat": self.x.copy(),
            "P": self.P.copy(),
            "accepted": accepted,
            "nis": nis,
            "nis_ema": float(self.nis_ema),
            "sigma_a_used": sigma_a,
        }

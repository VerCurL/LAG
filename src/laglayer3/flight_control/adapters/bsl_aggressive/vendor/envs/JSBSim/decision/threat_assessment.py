import numpy as np

from .adaptive_kf import sigmoid

try:
    from scipy.stats import t as student_t
    HAS_SCIPY = True
except Exception:
    student_t = None
    HAS_SCIPY = False


class ThreatAssessment:
    """
    Convert kinematic features into a continuous threat score in [0, 1].
    """

    def __init__(
        self,
        w_tgo: float = 0.35,
        w_dcpa: float = 0.35,
        w_los_rate: float = 0.20,
        w_alert: float = 0.10,
        tgo_ref_s: float = 8.0,
        dcpa_ref_m: float = 2500.0,
        los_rate_ref: float = 0.08,
        score_slope: float = 5.0,
        score_bias: float = 0.5,
        use_t_dist: bool = False,
        t_nu: float = 5.0,
    ):
        self.w_tgo = float(w_tgo)
        self.w_dcpa = float(w_dcpa)
        self.w_los_rate = float(w_los_rate)
        self.w_alert = float(w_alert)

        self.tgo_ref_s = float(tgo_ref_s)
        self.dcpa_ref_m = float(dcpa_ref_m)
        self.los_rate_ref = float(los_rate_ref)

        self.score_slope = float(score_slope)
        self.score_bias = float(score_bias)
        self.use_t_dist = bool(use_t_dist)
        self.t_nu = float(max(t_nu, 2.1))

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _t_cdf(self, z: float) -> float:
        # If scipy is unavailable in runtime env, fallback to a smooth approximation.
        if HAS_SCIPY:
            return float(student_t.cdf(z, df=self.t_nu))
        return float(sigmoid(z))

    def score(self, features: dict, missile_alert: bool):
        tgo_s = float(features.get("tgo_s", np.inf))
        dcpa_m = float(features.get("dcpa_m", np.inf))
        los_rate = float(features.get("los_rate_rad_s", 0.0))

        if self.use_t_dist:
            # MVP: use reference values as scale anchors for t-based probability mapping.
            if np.isfinite(tgo_s):
                z_tgo = (self.tgo_ref_s - tgo_s) / max(self.tgo_ref_s, 1e-6)
                tgo_score = self._t_cdf(z_tgo)  # lower tgo => higher risk
            else:
                tgo_score = 0.0

            if np.isfinite(dcpa_m):
                z_dcpa = (self.dcpa_ref_m - dcpa_m) / max(self.dcpa_ref_m, 1e-6)
                dcpa_score = self._t_cdf(z_dcpa)  # lower dcpa => higher risk
            else:
                dcpa_score = 0.0

            z_los = (los_rate - self.los_rate_ref) / max(self.los_rate_ref, 1e-6)
            los_score = self._t_cdf(z_los)  # higher los_rate => higher risk

            raw = (
                self.w_tgo * tgo_score
                + self.w_dcpa * dcpa_score
                + self.w_los_rate * los_score
                + self.w_alert * (1.0 if missile_alert else 0.0)
            )
            s_threat = self._clip01(raw)
        else:
            if np.isfinite(tgo_s):
                tgo_score = self._clip01((self.tgo_ref_s - tgo_s) / max(self.tgo_ref_s, 1e-6))
            else:
                tgo_score = 0.0

            if np.isfinite(dcpa_m):
                dcpa_score = self._clip01((self.dcpa_ref_m - dcpa_m) / max(self.dcpa_ref_m, 1e-6))
            else:
                dcpa_score = 0.0

            los_score = self._clip01(los_rate / (los_rate + max(self.los_rate_ref, 1e-6)))

            raw = (
                self.w_tgo * tgo_score
                + self.w_dcpa * dcpa_score
                + self.w_los_rate * los_score
                + self.w_alert * (1.0 if missile_alert else 0.0)
            )
            s_threat = float(sigmoid(self.score_slope * (raw - self.score_bias)))

        return s_threat, {
            "raw": raw,
            "tgo_score": tgo_score,
            "dcpa_score": dcpa_score,
            "los_score": los_score,
            "alert_score": 1.0 if missile_alert else 0.0,
        }

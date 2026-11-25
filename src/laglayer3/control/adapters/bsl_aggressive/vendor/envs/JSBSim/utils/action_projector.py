import numpy as np


class ConvexActionProjector:
    """Small diagonal QP projector for decision-rate control targets.

    The first implementation solves the separable box/slew-constrained QP:

        min wr ||u - raw||^2 + rate ||u - prev||^2
            + jerk ||u - 2prev + prev2||^2 + eps ||u||^2

    Because the Hessian is diagonal and constraints are per-channel box bounds,
    the exact solution is a weighted average followed by interval projection.
    """

    def __init__(
        self,
        action_dim,
        action_min=None,
        action_max=None,
        slew_limit=None,
        raw_weight=1.0,
        rate_weight=0.25,
        jerk_weight=0.08,
        hessian_eps=1e-6,
        enabled_channels=None,
        channel_names=None,
    ):
        self.action_dim = int(action_dim)
        self.channel_names = self._make_channel_names(channel_names)
        self.action_min = self._make_vector(action_min, -1.0)
        self.action_max = self._make_vector(action_max, 1.0)
        self.slew_limit = self._make_vector(slew_limit, np.inf)
        self.raw_weight = self._make_vector(raw_weight, 1.0)
        self.rate_weight = self._make_vector(rate_weight, 0.0)
        self.jerk_weight = self._make_vector(jerk_weight, 0.0)
        self.hessian_eps = max(float(hessian_eps), 0.0)
        self.enabled_channels = self._make_channel_mask(enabled_channels)
        self._state = {}
        self._last_log = {}

    def reset_agent(self, agent_id, initial_action):
        initial = np.asarray(initial_action, dtype=np.float32).reshape(self.action_dim)
        initial = np.clip(initial, self.action_min, self.action_max).astype(np.float32)
        self._state[agent_id] = {
            "prev": initial.copy(),
            "prev2": initial.copy(),
            "fail_streak": 0,
        }
        zeros = np.zeros(self.action_dim, dtype=np.float32)
        self._last_log[agent_id] = self._make_log(initial, initial, initial, zeros, zeros, zeros, 0)

    def project(self, agent_id, raw_action):
        raw = np.asarray(raw_action, dtype=np.float32).reshape(self.action_dim)
        if agent_id not in self._state:
            self.reset_agent(agent_id, raw)
        state = self._state[agent_id]
        prev = np.asarray(state["prev"], dtype=np.float32)
        prev2 = np.asarray(state["prev2"], dtype=np.float32)

        try:
            if not np.all(np.isfinite(raw)):
                raise ValueError("raw action contains non-finite values")
            unconstrained = self._solve_unconstrained(raw, prev, prev2)
            lower = np.maximum(self.action_min, prev - self.slew_limit)
            upper = np.minimum(self.action_max, prev + self.slew_limit)
            projected = np.clip(unconstrained, lower, upper)
            projected = np.where(self.enabled_channels, projected, raw)
            projected = projected.astype(np.float32, copy=False)
            box_mask, slew_mask, active_mask = self._constraint_masks(projected, unconstrained, prev, lower, upper)
            state["fail_streak"] = 0
        except Exception:
            projected = prev.copy()
            raw = np.where(np.isfinite(raw), raw, prev).astype(np.float32, copy=False)
            box_mask = np.zeros(self.action_dim, dtype=np.float32)
            slew_mask = np.zeros(self.action_dim, dtype=np.float32)
            active_mask = np.zeros(self.action_dim, dtype=np.float32)
            state["fail_streak"] = int(state.get("fail_streak", 0)) + 1

        state["prev2"] = prev.copy()
        state["prev"] = projected.copy()
        self._last_log[agent_id] = self._make_log(raw, projected, prev, active_mask, box_mask, slew_mask, state["fail_streak"])
        return projected.copy()

    def get_last_log(self, agent_id):
        log = self._last_log.get(agent_id)
        return {} if log is None else dict(log)

    def get_observation_state(self, agent_id):
        """Return the exact decision-boundary QP history as read-only copies.

        ``prev`` and ``prev2`` are the complete dynamic state used by the
        current diagonal projector.  Both are returned directly so the
        observation does not rely on reconstructing history by subtraction.
        """
        state = self._state.get(agent_id)
        if state is None:
            return None
        prev = np.asarray(state["prev"], dtype=np.float32).reshape(self.action_dim)
        prev2 = np.asarray(state["prev2"], dtype=np.float32).reshape(self.action_dim)
        return {
            "prev": prev.copy(),
            "prev2": prev2.copy(),
        }

    def _solve_unconstrained(self, raw, prev, prev2):
        jerk_ref = 2.0 * prev - prev2
        denom = self.raw_weight + self.rate_weight + self.jerk_weight + self.hessian_eps
        denom = np.maximum(denom, 1e-9)
        numerator = (
            self.raw_weight * raw
            + self.rate_weight * prev
            + self.jerk_weight * jerk_ref
        )
        return numerator / denom

    def _constraint_masks(self, projected, unconstrained, prev, lower, upper):
        tol = 1e-6
        box_active = (
            (np.abs(projected - self.action_min) <= tol)
            | (np.abs(projected - self.action_max) <= tol)
        )
        slew_active = (
            np.isfinite(self.slew_limit)
            & (
                (np.abs(projected - (prev - self.slew_limit)) <= tol)
                | (np.abs(projected - (prev + self.slew_limit)) <= tol)
            )
        )
        interval_active = (
            (unconstrained < lower - tol)
            | (unconstrained > upper + tol)
        )
        box_active &= self.enabled_channels
        slew_active &= self.enabled_channels
        active = (interval_active | box_active | slew_active) & self.enabled_channels
        return box_active.astype(np.float32), slew_active.astype(np.float32), active.astype(np.float32)

    def _make_log(self, raw, projected, prev, active_mask, box_mask, slew_mask, fail_streak):
        residual = projected - raw
        delta = projected - prev
        log = {
            # An empty channel mask is a dimension-matched history shell, not
            # an active projector.  Report functional status so QP-off
            # ablations do not misleadingly appear enabled in training logs.
            "qp_projector_enabled": bool(np.any(self.enabled_channels)),
            "qp_projector_fail_streak": float(fail_streak),
            "qp_projector_active_count": float(np.sum(active_mask)),
            "qp_projector_box_active_count": float(np.sum(box_mask)),
            "qp_projector_slew_active_count": float(np.sum(slew_mask)),
            "qp_projector_residual_abs_mean": float(np.mean(np.abs(residual))),
            "qp_projector_delta_abs_mean": float(np.mean(np.abs(delta))),
        }
        for idx, name in enumerate(self.channel_names):
            log[f"qp_raw_{name}"] = float(raw[idx])
            log[f"qp_action_{name}"] = float(projected[idx])
            log[f"qp_residual_{name}"] = float(residual[idx])
            log[f"qp_active_{name}"] = float(active_mask[idx])
            log[f"qp_box_active_{name}"] = float(box_mask[idx])
            log[f"qp_slew_active_{name}"] = float(slew_mask[idx])
        return log

    def _make_vector(self, value, default):
        if value is None:
            return np.full(self.action_dim, float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            return np.full(self.action_dim, float(arr[0]), dtype=np.float32)
        out = np.full(self.action_dim, float(default), dtype=np.float32)
        out[: min(self.action_dim, arr.size)] = arr[: min(self.action_dim, arr.size)]
        return out

    def _make_channel_mask(self, enabled_channels):
        mask = np.ones(self.action_dim, dtype=bool)
        if enabled_channels is None:
            return mask
        mask[:] = False
        channel_to_index = {
            "aileron": 0,
            "elevator": 1,
            "rudder": 2,
            "throttle": 3,
        }
        for channel in enabled_channels:
            idx = channel_to_index.get(str(channel).strip().lower())
            if idx is not None and idx < self.action_dim:
                mask[idx] = True
        return mask

    def _make_channel_names(self, channel_names):
        default_names = ("aileron", "elevator", "rudder", "throttle")
        names = list(default_names[: self.action_dim])
        if channel_names is not None:
            supplied = [str(name).strip().lower() for name in channel_names]
            for idx, name in enumerate(supplied[: self.action_dim]):
                names[idx] = name
        while len(names) < self.action_dim:
            names.append(f"action_{len(names)}")
        return tuple(names[: self.action_dim])

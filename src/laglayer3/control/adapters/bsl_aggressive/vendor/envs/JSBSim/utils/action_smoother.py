import numpy as np


class MinimumJerkActionSmoother:
    """Smooth decision-rate actions into simulator-rate commands."""

    def __init__(
        self,
        action_dim,
        base_steps,
        enabled_channels=None,
        large_delta_threshold=0.65,
        large_delta_duration_scale=0.5,
        target_slew_enable=False,
        target_deadband=None,
        target_slew_base=None,
        urgency_enable=False,
        urgency_gain=1.0,
        aileron_turn_boost_enable=False,
        aileron_turn_boost_gain=0.7,
        aileron_max_gate=2.0,
        elevator_vertical_gate_enable=False,
        elevator_vertical_damping_gain=0.45,
        elevator_min_gate=0.45,
        elevator_max_gate=2.0,
        elevator_momentum_unload_enable=False,
        elevator_momentum_unload_gain=0.25,
        elevator_vertical_activity_blend=0.5,
    ):
        self.action_dim = int(action_dim)
        self.base_steps = max(1, int(base_steps))
        self.enabled_channels = self._make_channel_mask(enabled_channels)
        self.large_delta_threshold = float(large_delta_threshold)
        self.large_delta_duration_scale = float(np.clip(large_delta_duration_scale, 0.0, 1.0))
        self.target_slew_enable = bool(target_slew_enable)
        self.target_deadband = self._make_vector(target_deadband, 0.0)
        self.target_slew_base = self._make_vector(target_slew_base, 1.0)
        self.urgency_enable = bool(urgency_enable)
        self.urgency_gain = float(urgency_gain)
        self.aileron_turn_boost_enable = bool(aileron_turn_boost_enable)
        self.aileron_turn_boost_gain = float(aileron_turn_boost_gain)
        self.aileron_max_gate = float(aileron_max_gate)
        self.elevator_vertical_gate_enable = bool(elevator_vertical_gate_enable)
        self.elevator_vertical_damping_gain = float(elevator_vertical_damping_gain)
        self.elevator_min_gate = float(elevator_min_gate)
        self.elevator_max_gate = float(elevator_max_gate)
        self.elevator_momentum_unload_enable = bool(elevator_momentum_unload_enable)
        self.elevator_momentum_unload_gain = float(elevator_momentum_unload_gain)
        self.elevator_vertical_activity_blend = float(np.clip(elevator_vertical_activity_blend, 0.0, 1.0))
        self._state = {}

    def reset_agent(self, agent_id, initial_action):
        initial = np.asarray(initial_action, dtype=np.float32).reshape(self.action_dim)
        self._state[agent_id] = {
            "start": initial.copy(),
            "target": initial.copy(),
            "filtered_target": initial.copy(),
            "current": initial.copy(),
            "step": 0,
            "duration_steps": 1,
        }

    def set_target(self, agent_id, target_action, context=None):
        policy_target = np.asarray(target_action, dtype=np.float32).reshape(self.action_dim)
        if agent_id not in self._state:
            self.reset_agent(agent_id, policy_target)
        state = self._state[agent_id]
        target = self._filter_target(state, policy_target, context or {})
        start = np.asarray(state["current"], dtype=np.float32).copy()
        max_delta = float(np.max(np.abs(target - start))) if target.size else 0.0
        duration_steps = self.base_steps
        if max_delta >= self.large_delta_threshold and self.large_delta_duration_scale > 0.0:
            duration_steps = max(2, int(round(self.base_steps * self.large_delta_duration_scale)))
        state.update(
            {
                "start": start,
                "target": target.copy(),
                "filtered_target": target.copy(),
                "step": 0,
                "duration_steps": max(1, duration_steps),
            }
        )

    def sample(self, agent_id):
        state = self._state[agent_id]
        duration_steps = max(1, int(state["duration_steps"]))
        if duration_steps <= 1:
            smooth = np.asarray(state["target"], dtype=np.float32).copy()
        else:
            tau = min(float(state["step"] + 1) / float(duration_steps), 1.0)
            blend = self._minimum_jerk_blend(tau)
            start = np.asarray(state["start"], dtype=np.float32)
            target = np.asarray(state["target"], dtype=np.float32)
            smooth = start + blend * (target - start)
            smooth = np.where(self.enabled_channels, smooth, target)
        state["current"] = smooth.astype(np.float32, copy=True)
        state["step"] = int(state["step"]) + 1
        return state["current"].copy()

    def get_current(self, agent_id):
        if agent_id not in self._state:
            return None
        return np.asarray(self._state[agent_id]["current"], dtype=np.float32).copy()

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

    def _make_vector(self, value, default):
        if value is None:
            return np.full(self.action_dim, float(default), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            return np.full(self.action_dim, float(arr[0]), dtype=np.float32)
        out = np.full(self.action_dim, float(default), dtype=np.float32)
        out[: min(self.action_dim, arr.size)] = arr[: min(self.action_dim, arr.size)]
        return out

    def _filter_target(self, state, policy_target, context):
        if not self.target_slew_enable:
            return policy_target.copy()
        previous = np.asarray(state["filtered_target"], dtype=np.float32)
        delta = policy_target - previous
        desired = np.where(np.abs(delta) < self.target_deadband, previous, policy_target)
        max_delta = self._effective_slew_limits(context)
        limited = previous + np.clip(desired - previous, -max_delta, max_delta)
        return np.where(self.enabled_channels, limited, policy_target).astype(np.float32, copy=False)

    def _effective_slew_limits(self, context):
        gate = np.ones(self.action_dim, dtype=np.float32)
        if self.urgency_enable:
            urgency = float(np.clip(context.get("urgency", 0.0), 0.0, 1.0))
            gate *= 1.0 + self.urgency_gain * urgency
        if self.aileron_turn_boost_enable and self.action_dim > 0:
            turn_need = float(np.clip(context.get("aileron_turn_need", 0.0), 0.0, 1.0))
            aileron_gate = gate[0] * (1.0 + self.aileron_turn_boost_gain * turn_need)
            gate[0] = float(np.clip(aileron_gate, 1.0, self.aileron_max_gate))
        if self.elevator_vertical_gate_enable and self.action_dim > 1:
            vertical_activity = float(np.clip(context.get("vertical_activity", 0.0), 0.0, 1.0))
            gate_signal = vertical_activity
            if self.elevator_momentum_unload_enable:
                gate_signal = self.elevator_vertical_activity_blend * vertical_activity
            elevator_gate = gate[1] - self.elevator_vertical_damping_gain * gate_signal
            if self.elevator_momentum_unload_enable:
                momentum_excess = float(np.clip(context.get("momentum_envelope_excess", 0.0), 0.0, 1.0))
                elevator_gate -= self.elevator_momentum_unload_gain * momentum_excess
            gate[1] = float(np.clip(elevator_gate, self.elevator_min_gate, self.elevator_max_gate))
        return np.maximum(self.target_slew_base * gate, 1e-6)

    @staticmethod
    def _minimum_jerk_blend(tau):
        tau = float(np.clip(tau, 0.0, 1.0))
        return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

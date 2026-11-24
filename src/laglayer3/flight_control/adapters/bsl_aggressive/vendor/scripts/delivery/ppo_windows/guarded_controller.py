from ....competition.flight_control_ppo_delivery.controller import DeliveryController, UID
import numpy as np

EFFECTIVE_RESIDUAL_MULTIPLIERS = (.2, 1.0, 1.0, .01 / .22)
ELEVATOR_TRIM_FEEDFORWARD = -.18599601816358236


class GuardedController(DeliveryController):
    """Deployment-only residual composition and one QP projection."""

    def apply(self, residual):
        if self._context is None or self._faulted or not self._pending:
            raise RuntimeError("one fresh observation required per control action")
        try:
            raw = np.asarray(residual, dtype=np.float32).reshape(-1)
            if raw.shape != (4,) or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1.0):
                raise ValueError("four bounded finite actual PPO residuals required")
            effective = (raw * np.asarray(EFFECTIVE_RESIDUAL_MULTIPLIERS, dtype=np.float32)).astype(np.float32)
            env = self._context
            composed = env._apply_coordinated_residual(UID, effective)
            physical = np.asarray(env.task.normalize_action(env, UID, composed), dtype=np.float32).copy()
            if physical.shape != (4,) or not np.isfinite(physical).all():
                raise ValueError("normalized proposal must have four finite physical channels")
            physical[1] = np.float32(float(physical[1]) + ELEVATOR_TRIM_FEEDFORWARD)
            if bool(getattr(self, "_assault_power", False)):
                physical[3] = np.float32(1.0)
            elif bool(getattr(self, "_aggressive_accelerate", False)):
                physical[3] = np.float32(.9)
            physical[3] = np.float32(min(float(physical[3]), getattr(self, "_speed_throttle_ceiling", 1.0)))
            command = np.asarray(env._apply_action_projector(UID, physical), dtype=np.float32)
            qp = env._action_projector.get_last_log(UID)
            if command.shape != (4,) or not np.isfinite(command).all() or qp["qp_projector_fail_streak"] != 0:
                raise ValueError("invalid QP output")
            self._pending = False
            self._actions += 1
            self._last_result = {"command": command, "reference": self._reference}
            return self._last_result
        except Exception:
            self._faulted = True
            raise


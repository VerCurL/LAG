"""Isolated C1 task: U1 objectives plus coordinated action composition."""

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerCoordinatedResidualC1Task(BaiyePowerUprightV1Task):
    """Freeze U1 observations/rewards while changing action parameterization.

    C1's first gate is deliberately ``zero_residual``: the environment uses
    only the analytic coordinated nominal action.  PPO residual training is
    forbidden until that nominal path passes its pre-registered feasibility
    checks.
    """

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_coordinated_residual_c1"
        ):
            raise ValueError("C1 requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C1 must retain the U1 29D v4_qp_state")
        if bool(
            getattr(
                self.config,
                "BaiyePowerTargetRates_enable",
                False,
            )
        ):
            raise ValueError(
                "C1 must not inherit O1 target-rate observations"
            )
        if not bool(
            getattr(self.config, "CoordinatedResidual_enable", False)
        ):
            raise ValueError(
                "baiyepower_coordinated_residual_c1 requires "
                "CoordinatedResidual_enable=true"
            )
        mode = str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower()
        if mode != "zero_residual":
            raise ValueError(
                "C1 feasibility gate requires zero_residual; "
                "residual training needs a later isolated config"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_rate_source", "")
        ).strip().lower() != "route_analytic_v1":
            raise ValueError(
                "C1 target rates must use route_analytic_v1"
            )

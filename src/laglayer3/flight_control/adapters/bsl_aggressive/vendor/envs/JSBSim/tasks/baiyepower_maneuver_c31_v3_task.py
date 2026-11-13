"""C31-V3 steady-turn bank-feedback task with the frozen C28 flight contract."""

from .baiyepower_maneuver_c28_task import (
    BaiyePowerManeuverC28Task,
    EPISODE_STEPS,
    RESIDUAL_SCALE,
)
from .baiyepower_v2_task import BaiyePowerV2Task


SCENARIO = "1/baiyepower_maneuver_c31_v3"
TASK_NAME = "baiyepower_maneuver_c31_v3"


class BaiyePowerManeuverC31V3Task(BaiyePowerManeuverC28Task):
    """Isolate V3 without changing the frozen C28/V1/V2 task identities."""

    def _validate_isolation_config(self):
        # C28 checks its own task name, so invoke the common V2 contract and
        # assert every invariant of the maneuver task independently.
        BaiyePowerV2Task._validate_isolation_config(self)
        required = {
            "task": TASK_NAME,
            "CoordinatedResidual_enable": True,
            "CoordinatedResidual_mode": "actuator_residual_fixed_center",
            "CoordinatedResidual_flight_path_geometry": "atan2_horizontal",
            "CoordinatedResidual_max_course_rate_deg_s": 9.0,
            "CoordinatedResidual_max_bank_deg": 80.0,
            "CoordinatedResidual_turn_pitch_coupling_gain": .75,
            "CoordinatedResidual_pitch_rate_feedback_gain": .25,
            "CoordinatedResidual_direct_flight_path_feedback_gain_per_rad": 0.0,
            "CoordinatedResidual_direct_flight_path_feedback_max_action": 0.0,
            "CoordinatedResidual_steady_bank_feedback_enable": True,
            "CoordinatedResidual_turn_bank_feedback_enable": False,
            "CoordinatedResidual_turn_vertical_trim_enable": False,
            "DirectionInvariantObservation_mode": "signed_c28",
            "sim_freq": 60,
            "agent_interaction_steps": 12,
            "max_steps": EPISODE_STEPS,
            "BaiyePowerV2_unsafe_terminal_penalty": -20.0,
            "BaiyePowerV2_numerical_terminal_penalty": -20.0,
        }
        for key, expected in required.items():
            if getattr(self.config, key, None) != expected:
                raise ValueError("C31-V3 identity mismatch: {}".format(key))
        if tuple(self.config.CoordinatedResidual_residual_scale) != RESIDUAL_SCALE:
            raise ValueError("C31-V3 residual authority differs from C28")
        if getattr(self.config, "DirectStudentTeacher_enable", False):
            raise ValueError("C31-V3 may not use teacher forcing")

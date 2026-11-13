"""C50-M1-PID: change only the EAS speed proportional coefficient.

The existing rate loops, derivative filter, integrators, antiwindup, guidance,
information boundary and command limits remain in the frozen implementations.
"""
from ...competition.flight_control_eas.pid import PIDController
from ...competition.flight_control_eas.runtime import EASFlightController

EXPERIMENT = "C50-M1-PID"
BASELINE_KP = .023627101866982084
CANDIDATE_KP = .034839412687137215
UNCHANGED_KI = .003150280248930945


class SpeedProportionalPID(PIDController):
    def __init__(self, trim):
        original = PIDController(trim).config
        if original["speed"]["kp"] != BASELINE_KP or original["speed"]["ki"] != UNCHANGED_KI:
            raise ValueError("frozen PID speed gains differ from the reviewed baseline")
        super().__init__(trim, {"speed": {"kp": CANDIDATE_KP}})
        expected = PIDController(trim).config
        expected["speed"]["kp"] = CANDIDATE_KP
        if self.config != expected:
            raise ValueError("candidate may change only speed.kp")

    def step(self, state, reference, dt):
        output = super().step(state, reference, dt)
        output.diagnostics["development_candidate"] = dict(experiment=EXPERIMENT,
            changed_parameter="speed.kp", previous=BASELINE_KP, current=CANDIDATE_KP,
            speed_ki_unchanged=UNCHANGED_KI, original_controller_source_modified=False)
        return output


class CandidateFlightController(EASFlightController):
    def __init__(self, model):
        super().__init__("pid", model)
        self.inner = SpeedProportionalPID(self.trim)
        self.reset()


def make_controller(kind, model):
    if kind != "pid":
        raise ValueError("C50-M1-PID contains only a pure PID candidate")
    return CandidateFlightController(model)


def comparison_config(trim):
    previous = PIDController(trim).config
    candidate = SpeedProportionalPID(trim).config
    check = dict(candidate)
    check["speed"] = dict(candidate["speed"], kp=previous["speed"]["kp"])
    if check != previous:
        raise ValueError("parameters outside speed.kp changed")
    return dict(experiment=EXPERIMENT, original=previous, candidate=candidate,
                changed_paths=["speed.kp"], other_fields_identical=True,
                schema_metadata_unchanged=True, new_control_modules=0, PPO_enabled=False)

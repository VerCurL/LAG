"""C50-E2 select the fresh0s mapped trim center; no further integration.

The pitch-summer change of command basis is the frozen C50-E1 mechanism.
This wrapper changes only model-center selection and corrects its provenance;
the E1 two-second measured center remains a separate immutable artifact.
"""
from copy import deepcopy

from .actuator_basis import SendableCommandBackend, TRIM_PATHS
from .model import OfficialEASBackend, native

CENTER_SOURCE = "fresh_mapped_trim_t0_no_explicit_integration"


class FreshSendableTrimBackend(SendableCommandBackend):
    def __init__(self, model_root=None, fdm_factory=None):
        super().__init__(model_root, fdm_factory, trim_before_handover=True, map_hidden_trim=True)

    def reset(self):
        trim = super().reset()
        if abs(trim.state.time_sec) > 1e-10 or self.physics_steps != 0 or self.trim_calls != 1:
            self._ready = False
            raise ValueError("fresh mapped trim0s without explicit integration required")
        _, raw = self._sample()
        if any(raw[path] != 0. for path in TRIM_PATHS):
            self._ready = False
            raise ValueError("fresh model center may contain no hidden trim")
        self.initialization["model_center"] = dict(source=CENTER_SOURCE,
            trim=trim.as_dict(), native_time_sec=trim.state.time_sec,
            explicit_integration_steps_before_identification=0)
        return trim

    def linearize(self, linearization_factory=None):
        self._active()
        state, raw = self._sample()
        if abs(state.time_sec) > 1e-10 or self.physics_steps != 0 or self.trim_calls != 1:
            raise ValueError("fresh0s center must not be replaced by a propagated endpoint")
        if any(raw[path] != 0. for path in TRIM_PATHS):
            raise ValueError("zero hidden trim at fresh model identification required")
        basis = deepcopy(self.command_basis)
        basis.update(model_center_source=CENTER_SOURCE,
            identification_center_hidden_trim={path: raw[path] for path in TRIM_PATHS},
            identification_center_commands=list(native.read_commands(self.fdm)),
            model_center_native_time_sec=state.time_sec,
            explicit_integration_steps_before_identification=0)
        # Deliberately bypass E1's2s-center labeling, while retaining exactly the
        # same EAS transformation and native full12 identification implementation.
        model = OfficialEASBackend.linearize(self, linearization_factory)
        model["command_basis"] = basis
        model["provenance"]["command_basis"] = deepcopy(basis)
        model["provenance"]["model_center_selection"] = CENTER_SOURCE
        return model

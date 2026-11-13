"""C50-E1 explicit native A/E/R/T basis without hidden trim commands.

Only the XML-proven pitch summer is mapped. Roll/yaw hidden trim must already
be zero. This engineering basis change never changes gains or aircraft XML.
"""
from copy import deepcopy
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .contracts import EASTrim, finite
from .evidence import sha
from .model import OfficialEASBackend, native

SCHEMA = "competition-native-aert-zero-hidden-trim-v1"
TRIM_PATHS = ("fcs/roll-trim-cmd-norm", "fcs/pitch-trim-cmd-norm", "fcs/yaw-trim-cmd-norm")
OPTIONAL_PHYSICAL_PATHS = ("fcs/elevator-cmd-limiter", "forces/fbx-total-lbs", "forces/fby-total-lbs",
    "forces/fbz-total-lbs", "moments/l-total-lbsft", "moments/m-total-lbsft", "moments/n-total-lbsft")


def verify_pitch_summer(xml_path):
    path = Path(xml_path)
    tree = ET.parse(path)
    summers = [node for node in tree.findall(".//summer") if node.attrib.get("name") == "fcs/elevator-cmd-limiter"]
    if len(summers) != 1:
        raise ValueError("one declared elevator command summer required")
    summer = summers[0]
    inputs = [(node.text or "").strip() for node in summer.findall("input")]
    if inputs != ["fcs/elevator-cmd-norm", "fcs/pitch-trim-cmd-norm"]:
        raise ValueError("elevator/pitch trim must enter the declared direct same-sign sum")
    if float(summer.findtext("clipto/min", "nan")) != -1. or float(summer.findtext("clipto/max", "nan")) != .44:
        raise ValueError("declared native elevator summer clip must be [-1,.44]")
    consumers = [node for node in tree.findall(".//input") if (node.text or "").strip().lstrip("-") == "fcs/pitch-trim-cmd-norm"]
    if len(consumers) != 1:
        raise ValueError("pitch trim has other consumers; same-summer mapping is insufficient")
    return dict(xml_path=str(path.resolve()), xml_sha256=sha(path), summer="fcs/elevator-cmd-limiter",
        sum_inputs=inputs, clip_min=-1., clip_max=.44, pitch_trim_input_consumers=1,
        mapping="elevator_new=elevator_old+pitch_trim_old; pitch_trim_new=0",
        unrelated_roll_yaw_mapping_authorized=False)


def mapped_commands(commands, hidden_trim):
    values = tuple(finite(value, "A/E/R/T command") for value in commands)
    if len(values) != 4:
        raise ValueError("four native commands required")
    trims = tuple(finite(hidden_trim[path], path) for path in TRIM_PATHS)
    if trims[0] != 0. or trims[2] != 0.:
        raise ValueError("nonzero roll/yaw trim cannot be mapped without its own proven mechanism")
    mapped = (values[0], values[1] + trims[1], values[2], values[3])
    if any(not lo <= value <= hi for value, lo, hi in zip(mapped, (-1., -1., -1., 0.), (1., 1., 1., 1.))):
        raise ValueError("mapped physical command outside sendable A/E/R/T box")
    # Exactly preserve the pre-clip summer; do not absorb a clipped sum.
    if not -1. <= mapped[1] <= .44:
        raise ValueError("equivalent elevator baseline would touch native summer clipping")
    return mapped


class SendableCommandBackend(OfficialEASBackend):
    def __init__(self, model_root=None, fdm_factory=None, trim_before_handover=True, *, map_hidden_trim=True):
        if type(map_hidden_trim) is not bool:
            raise ValueError("explicit mapping policy required")
        super().__init__(model_root, fdm_factory, trim_before_handover)
        self.map_hidden_trim = map_hidden_trim
        self._extra_paths = ()
        self.command_basis = None

    def _sample(self):
        state, raw = super()._sample()
        raw.update({path: finite(self.fdm.get_property_value(path), path) for path in self._extra_paths})
        return state, raw

    def reset(self):
        self._extra_paths = ()
        self.command_basis = None
        original = super().reset()
        try:
            return self._set_basis(original)
        except Exception as exc:
            self._ready = False
            self.initialization.update(command_basis_error=dict(type=type(exc).__name__, message=str(exc)))
            raise

    def _set_basis(self, original):
        catalog = native._catalog(self.fdm)
        engine_paths = tuple(sorted(path for path, permission in catalog.items() if "R" in permission and
            re.fullmatch(r"propulsion/engine(?:\[\d+\])?/(?:n1|n2|thrust-lbs)", path, re.IGNORECASE)))
        self._extra_paths = tuple(dict.fromkeys(tuple(path for path in OPTIONAL_PHYSICAL_PATHS if "R" in catalog.get(path, "")) + engine_paths))
        state, raw = self._sample()
        hidden = {path: raw[path] for path in TRIM_PATHS}
        basis = dict(schema=SCHEMA, mapped=self.map_hidden_trim,
            before_commands=list(original.commands), before_hidden_trim=hidden,
            optional_physical_paths=list(self._extra_paths), sendable_commands=False,
            hidden_trim_zero=False, online_trim=False, XML_modified=False)
        basis["optional_engine_diagnostics"] = {name: [p for p in engine_paths if p.lower().endswith("/"+name)] or None
            for name in ("n1", "n2", "thrust-lbs")}
        basis["engine_running_status_directly_observed"] = False
        basis["initialization_scope"] = "mass/inertia/fuel match only; engine spool/thrust/startup equivalence with platform not established"
        self.command_basis = basis
        if self.map_hidden_trim:
            root = Path(self.initialization["model_root"])
            basis["xml_mechanism"] = verify_pitch_summer(root / "aircraft/f16/f16.xml")
            native._require_properties(catalog, TRIM_PATHS, "W")
            mapped = mapped_commands(original.commands, hidden)
            for path in TRIM_PATHS:
                self.fdm.set_property_value(path, 0.)
            actual = native.write_commands(self.fdm, mapped)
            _, after = self._sample()
            if any(after[path] != 0. for path in TRIM_PATHS):
                raise ValueError("hidden trim zero write/readback failed")
            self.trim = EASTrim(state, actual)
            basis.update(after_commands=list(actual), after_hidden_trim={path: after[path] for path in TRIM_PATHS},
                sendable_commands=True, hidden_trim_zero=True)
        else:
            basis.update(after_commands=list(original.commands), after_hidden_trim=hidden,
                hidden_trim_zero=all(v == 0. for v in hidden.values()))
        self.initialization["command_basis"] = deepcopy(basis)
        return self.trim

    def use_measured_endpoint_as_model_center(self):
        self._active()
        if not self.map_hidden_trim:
            raise ValueError("only mapped zero-hidden-trim aircraft may supply a sendable model")
        state, raw = self._sample()
        if any(raw[path] != 0. for path in TRIM_PATHS):
            raise ValueError("hidden trim changed during paired proof")
        self.trim = EASTrim(state, native.read_commands(self.fdm))
        self.initialization["model_center"] = dict(source="paired_mapped_2s_endpoint_no_additional_trim",
            state=state.as_dict(), commands=list(self.trim.commands), native_time_sec=state.time_sec)
        return self.trim

    def linearize(self, linearization_factory=None):
        if not self.map_hidden_trim or not self.command_basis or not self.command_basis.get("sendable_commands"):
            raise ValueError("sendable zero-hidden-trim basis required before identification")
        self._active()
        _, raw = self._sample()
        if any(raw[path] != 0. for path in TRIM_PATHS):
            raise ValueError("native hidden trim is not zero at identification")
        basis = deepcopy(self.command_basis)
        basis.update(identification_center_hidden_trim={path: raw[path] for path in TRIM_PATHS},
            identification_center_commands=list(native.read_commands(self.fdm)),
            model_center_source="paired_mapped_2s_endpoint_no_additional_trim")
        result = super().linearize(linearization_factory)
        result["command_basis"] = basis
        result["provenance"]["command_basis"] = deepcopy(basis)
        result["limitations"].append("Identification center is the measured2s matched endpoint, not an extra steady trim solve.")
        return result

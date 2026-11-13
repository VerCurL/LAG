"""Fail-closed identity of the plant actually selected by an environment.

No simulator is instantiated here. Aircraft-root resolution mirrors
AircraftSimulator.reload(): overrides are relative to envs/JSBSim, NOT cwd or
the YAML directory. Hashes are raw bytes, unlike normalized source hashes.

This is deliberately a bounded XML dependency resolver, not a universal
JSBSim loader. It supports the file consumers exercised by this project's
aircraft; unknown consumers, missing references, and asset-tree escapes stop
identity construction rather than claiming complete provenance.
"""

import hashlib
import importlib
import os
import platform
import re
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
import xml.etree.ElementTree as ET


SCHEMA = "jsbsim-flight-runtime-v1"
DEFAULT_JSBSIM_ROOT = Path(__file__).resolve().parents[1]
_MODEL_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
_MODEL_FILE_CONSUMERS = {
    "metrics", "mass_balance", "ground_reactions", "external_reactions",
    "buoyant_forces", "propulsion", "aerodynamics", "flight_control",
    "autopilot",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path, roots):
    return any(path == root or root in path.parents for root in roots)


def _directory(path):
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("identity directory is not a directory: {}".format(path))
    return path


def _file_record(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("identity resource is not a file: {}".format(path))
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def resolve_aircraft_directory(aircraft_config, jsbsim_root=None):
    """Resolve exactly the directory passed to set_aircraft_path by Simulator."""
    root = _directory(jsbsim_root or DEFAULT_JSBSIM_ROOT)
    override = aircraft_config.get("aircraft_path")
    if override is None:
        path = root / "data" / "aircraft"
    else:
        if not isinstance(override, (str, os.PathLike)) or not os.fspath(override):
            raise ValueError("aircraft_path must be a nonempty path or None")
        path = Path(override)
        if not path.is_absolute():
            path = root / path
    return _directory(path)


def _dependency_path(element, model_dir, data_root):
    """Mirror supported JSBSim 1.3.x model/propulsion/FCS lookup order.

    FGPropulsion::FindFullPathName searches Engines/engine then the global
    engine path for engine files. FGFCS searches the model root, Systems,
    then the global systems path. Thrusters share the engine search.
    """
    tag = element.tag
    filename = element.get("file")
    if not filename or not filename.strip():
        raise ValueError("empty JSBSim file reference")
    relative = Path(filename)
    if relative.suffix != ".xml":
        relative = Path(str(relative) + ".xml")
    if tag in ("engine", "thruster"):
        names = ("Engines", "engine") if os.name == "nt" else ("Engines", "engines", "Engine", "engine")
        search = [model_dir / name for name in names] + [data_root / "engine"]
    elif tag == "system":
        names = ("Systems",) if os.name == "nt" else ("Systems", "systems")
        search = [model_dir] + [model_dir / name for name in names] + [data_root / "systems"]
    elif tag in _MODEL_FILE_CONSUMERS:
        search = [model_dir]
    else:
        raise ValueError("unsupported JSBSim file consumer {!r}; provenance incomplete".format(tag))
    candidates = [relative] if relative.is_absolute() else [base / relative for base in search]
    roots = [model_dir, (data_root / "engine").resolve(), (data_root / "systems").resolve()]
    for candidate in candidates:
        resolved = candidate.resolve()
        if not _within(resolved, roots):
            raise ValueError("JSBSim dependency escapes supported asset roots: {}".format(candidate))
        if resolved.is_file():
            return resolved
    raise FileNotFoundError("unresolved JSBSim {} file={!r}; searched {}".format(tag, filename, candidates))


def actual_plant_identity(config, jsbsim_root=None):
    """Fingerprint actual aircraft trees plus referenced shared XML resources.

    The aircraft tree includes Systems and other companion resources, even
    when unused in the current XML. Shared engine/system resources are limited
    to recursive references, rather than every unrelated model in JSBSim data.
    """
    root = _directory(jsbsim_root or DEFAULT_JSBSIM_ROOT)
    aircraft = config.get("aircraft_configs") if isinstance(config, Mapping) else getattr(config, "aircraft_configs", None)
    if not isinstance(aircraft, Mapping) or not aircraft:
        raise ValueError("resolved config has no nonempty aircraft_configs mapping")
    data_root = _directory(root / "data")
    result = {}
    for uid, entry in sorted(aircraft.items()):
        if not isinstance(entry, Mapping):
            raise ValueError("aircraft config must be a mapping: {}".format(uid))
        model = entry.get("model", "f16")
        if not isinstance(model, str) or not _MODEL_NAME.fullmatch(model) or model in (".", ".."):
            raise ValueError("invalid JSBSim model name: {!r}".format(model))
        aircraft_dir = resolve_aircraft_directory(entry, root)
        model_dir = _directory(aircraft_dir / model)
        if not _within(model_dir, [aircraft_dir]):
            raise ValueError("model directory escapes aircraft root")
        model_xml = (model_dir / (model + ".xml")).resolve(strict=True)
        tree = {}
        for path in sorted(model_dir.rglob("*")):
            if path.is_symlink() and not _within(path.resolve(strict=True), [model_dir]):
                raise ValueError("aircraft resource symlink escapes model tree")
            if path.is_file():
                tree[path.relative_to(model_dir).as_posix()] = _sha256(path)
        if not model_xml.is_file() or not _within(model_xml, [model_dir]):
            raise ValueError("actual model XML is absent or outside model tree")
        dependencies, queue, visited = [], [model_xml], set()
        while queue:
            source = queue.pop()
            if source in visited:
                continue
            visited.add(source)
            for node in ET.parse(source).getroot().iter():
                if "file" not in node.attrib:
                    continue
                dependency = _dependency_path(node, model_dir, data_root)
                dependencies.append(dict(source=source.as_posix(), consumer=node.tag,
                                         reference=node.get("file"), **_file_record(dependency)))
                queue.append(dependency)
        result[str(uid)] = {
            "model": model, "aircraft_path": aircraft_dir.as_posix(),
            "model_directory": model_dir.as_posix(), "model_xml": _file_record(model_xml),
            "aircraft_tree_sha256": tree,
            "dependencies": sorted(dependencies, key=lambda item: (item["source"], item["consumer"], item["reference"])),
        }
    return result


def runtime_identity():
    """Record the imported runtime, not obsolete README version pins.

    The JSBSim extension hash distinguishes custom builds sharing a version.
    This is not a claim to hash every OS library or PyTorch dependency.
    """
    packages = {}
    for name in ("jsbsim", "torch", "numpy"):
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        if not isinstance(version, str) or not version:
            raise ValueError("missing imported runtime version: {}".format(name))
        packages[name] = {"version": version, "module_path": str(Path(module.__file__).resolve(strict=True))}
    extension = importlib.import_module("jsbsim._jsbsim")
    packages["jsbsim"]["extension"] = _file_record(extension.__file__)
    return {
        "python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
        "python_executable": _file_record(sys.executable), "pointer_bits": 8 * struct.calcsize("P"),
        "platform": platform.system(), "machine": platform.machine(), "packages": packages,
    }


def flight_runtime_identity(config, jsbsim_root=None):
    return {"schema": SCHEMA, "actual_plant": actual_plant_identity(config, jsbsim_root), "runtime": runtime_identity()}


def require_flight_runtime_identity(record, current):
    """Reject legacy/mismatched provenance; never auto-upgrade old results."""
    if not isinstance(current, dict) or current.get("schema") != SCHEMA or not current.get("actual_plant") or not current.get("runtime"):
        raise ValueError("current flight/runtime identity is incomplete")
    recorded = record.get("flight_runtime_identity")
    if recorded != current:
        raise ValueError("flight/runtime identity absent or changed; historical results cannot authorize a new run")

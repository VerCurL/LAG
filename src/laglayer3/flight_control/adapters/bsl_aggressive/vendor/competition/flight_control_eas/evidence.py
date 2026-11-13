"""Small append-free artifact utilities; no simulator or control imports."""
import hashlib
import json
from pathlib import Path
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON " + value)))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def snapshot(output, sources):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output / "registered_sources.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sources.items():
            if sha(ROOT / name) != expected:
                raise ValueError("source changed before snapshot: " + name)
            archive.writestr(name, (ROOT / name).read_bytes())


def verify_snapshot(output, registration):
    output = Path(output)
    if sha(output / "registered_sources.zip") != registration["source_snapshot_sha256"]:
        raise ValueError("source snapshot changed")
    with zipfile.ZipFile(output / "registered_sources.zip") as archive:
        if set(archive.namelist()) != set(registration["source_identity"]):
            raise ValueError("source inventory changed")
        for name, expected in registration["source_identity"].items():
            if sha(ROOT / name) != expected or hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("registered source changed: " + name)
    for name, expected in registration["input_identity"].items():
        if sha(name) != expected:
            raise ValueError("registered input changed: " + name)
    return registration

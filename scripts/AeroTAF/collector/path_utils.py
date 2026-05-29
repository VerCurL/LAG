import os
import posixpath
import re
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_project_root():
    return PROJECT_ROOT


def _normalize_path_text(path_text):
    return str(path_text).strip().replace("\\", "/")


def _is_windows_absolute(path_text):
    return re.match(r"^[A-Za-z]:/", path_text) is not None


def _normalize_relative_posix_text(path_text):
    normalized = posixpath.normpath(path_text)
    if normalized == ".":
        return "."
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _extract_project_relative_text(path_text):
    normalized_text = _normalize_path_text(path_text)
    if not normalized_text:
        return "."

    project_name = PROJECT_ROOT.name
    anchor = f"/{project_name}/"
    lower_text = normalized_text.lower()
    lower_anchor = anchor.lower()

    anchor_index = lower_text.find(lower_anchor)
    if anchor_index != -1:
        suffix = normalized_text[anchor_index + len(anchor):]
        return _normalize_relative_posix_text(suffix)

    project_tail = f"/{project_name}"
    if lower_text.endswith(project_tail.lower()):
        return "."

    if normalized_text == project_name:
        return "."

    if not normalized_text.startswith("/") and not _is_windows_absolute(normalized_text):
        return _normalize_relative_posix_text(normalized_text)

    return None


def resolve_project_path(path_text):
    project_relative = _extract_project_relative_text(path_text)
    if project_relative is not None:
        if project_relative == ".":
            return PROJECT_ROOT.resolve()
        posix_path = PurePosixPath(project_relative)
        return PROJECT_ROOT.joinpath(*posix_path.parts).resolve()

    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()

    normalized_text = _normalize_path_text(path_text)
    if _is_windows_absolute(normalized_text):
        return Path(normalized_text).expanduser().resolve()

    posix_path = PurePosixPath(normalized_text)
    return PROJECT_ROOT.joinpath(*posix_path.parts).resolve()


def to_project_relative_path(path_text):
    project_relative = _extract_project_relative_text(path_text)
    if project_relative is not None:
        return "." if project_relative == "." else project_relative

    resolved = resolve_project_path(path_text)
    resolved_str = os.path.normpath(str(resolved))
    root_str = os.path.normpath(str(PROJECT_ROOT))

    try:
        common = os.path.commonpath([os.path.normcase(resolved_str), os.path.normcase(root_str)])
    except ValueError:
        common = ""

    if common == os.path.normcase(root_str):
        return os.path.relpath(resolved_str, root_str).replace("\\", "/")

    return resolved.as_posix()


def normalize_path(path_text):
    return to_project_relative_path(path_text)


def canonicalize_task_key(task_key):
    text = str(task_key)
    parts = text.split("|")
    if len(parts) != 5:
        return text

    return "|".join([
        normalize_path(parts[0]),
        normalize_path(parts[1]),
        parts[2],
        parts[3],
        parts[4],
    ])

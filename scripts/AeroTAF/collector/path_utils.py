import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_project_root():
    return PROJECT_ROOT


def resolve_project_path(path_text):
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def to_project_relative_path(path_text):
    resolved = resolve_project_path(path_text)
    resolved_str = os.path.normpath(str(resolved))
    root_str = os.path.normpath(str(PROJECT_ROOT))

    try:
        common = os.path.commonpath([os.path.normcase(resolved_str), os.path.normcase(root_str)])
    except ValueError:
        common = ""

    if common == os.path.normcase(root_str):
        return os.path.relpath(resolved_str, root_str)

    return resolved_str


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

import json
import re
from pathlib import Path

import numpy as np

from .path_utils import canonicalize_task_key, normalize_path


def load_json_list(path):
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_json_list(path, rows):
    save_json(path, rows)


def parse_episode_index(path):
    match = re.match(r"episode_(\d+)\.npz$", path.name)
    if not match:
        return None
    return int(match.group(1))


def as_string(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def build_task_key(task):
    return canonicalize_task_key("|".join([
        normalize_path(task["ego_model_path"]),
        normalize_path(task["enm_model_path"]),
        str(task.get("scenario_id", "")),
        str(task.get("seed", "")),
        str(task.get("task_kind", "collect")),
    ]))


def load_existing_raw_state(raw_dir):
    completed_keys = set()
    existing_episode_ids = []

    if not raw_dir.exists():
        return completed_keys, existing_episode_ids

    for npz_path in sorted(raw_dir.glob("episode_*.npz")):
        episode_index = parse_episode_index(npz_path)
        if episode_index is not None:
            existing_episode_ids.append(episode_index)

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "task_key" in data.files:
                    completed_keys.add(canonicalize_task_key(as_string(data["task_key"])))
                    continue

                if "ego_model_path" in data.files and "enm_model_path" in data.files:
                    ego_model_path = normalize_path(as_string(data["ego_model_path"]))
                    enm_model_path = normalize_path(as_string(data["enm_model_path"]))
                    scenario_id = as_string(data["scenario_id"]) if "scenario_id" in data.files else ""
                    seed = as_string(data["random_seed"]) if "random_seed" in data.files else ""
                    task_kind = as_string(data["task_kind"]) if "task_kind" in data.files else "collect"
                    completed_keys.add(canonicalize_task_key("|".join([
                        ego_model_path,
                        enm_model_path,
                        scenario_id,
                        seed,
                        task_kind,
                    ])))
        except Exception:
            continue

    return completed_keys, existing_episode_ids


def extend_history(history_path, rows):
    current_rows = load_json_list(history_path)
    current_rows.extend(rows)
    save_json_list(history_path, current_rows)
    return current_rows

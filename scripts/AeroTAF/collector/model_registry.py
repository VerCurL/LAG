import re
from pathlib import Path


def normalize_path(path_text):
    return str(Path(path_text).expanduser().resolve())


def parse_checkpoint_step(path_text):
    path = Path(path_text)
    match = re.search(r"actor_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return -1


def _build_model_entry(path_text, source_index):
    normalized_path = normalize_path(path_text)
    checkpoint_path = Path(normalized_path)
    checkpoint_step = parse_checkpoint_step(normalized_path)
    return {
        "model_id": f"model_{source_index:03d}",
        "checkpoint_path": normalized_path,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_step": checkpoint_step,
        "source_run": str(checkpoint_path.parent),
    }


def discover_actor_models(model_roots):
    models = []
    seen_paths = set()

    for root in model_roots:
        path = Path(root).expanduser()
        if path.is_file() and path.suffix == ".pt":
            normalized_path = normalize_path(path)
            if normalized_path not in seen_paths:
                seen_paths.add(normalized_path)
                models.append(normalized_path)
            continue

        if path.is_dir():
            for pt in sorted(path.glob("actor*.pt")):
                normalized_path = normalize_path(pt)
                if normalized_path not in seen_paths:
                    seen_paths.add(normalized_path)
                    models.append(normalized_path)

    models = sorted(models, key=lambda item: (parse_checkpoint_step(item), item))
    if not models:
        raise RuntimeError("No actor*.pt models found.")

    return [_build_model_entry(path_text, index) for index, path_text in enumerate(models, start=1)]


def assign_provisional_stages(model_entries):
    sorted_entries = sorted(model_entries, key=lambda item: (item["checkpoint_step"], item["checkpoint_path"]))
    total = len(sorted_entries)
    if total == 0:
        return model_entries

    early_end = total // 3
    middle_end = (2 * total) // 3

    for index, entry in enumerate(sorted_entries):
        if index < early_end:
            stage_hint = "early"
        elif index < middle_end:
            stage_hint = "mid"
        else:
            stage_hint = "late"
        entry["stage_hint"] = stage_hint

    return sorted_entries


def build_model_lookup(model_entries):
    return {entry["checkpoint_path"]: entry for entry in model_entries}

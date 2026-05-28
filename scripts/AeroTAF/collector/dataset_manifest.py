import csv
import json


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_stage1_artifacts(
    metadata_dir,
    model_registry,
    model_profiles,
    pair_plan,
    profiling_plan,
    collection_plan,
    manifest,
):
    write_json(metadata_dir / "stage1_model_registry.json", model_registry)
    write_json(metadata_dir / "stage1_model_profiles.json", model_profiles)
    write_csv(metadata_dir / "stage1_model_profiles.csv", model_profiles)
    write_json(metadata_dir / "stage1_pair_plan.json", pair_plan)
    write_json(metadata_dir / "stage1_profiling_plan.json", profiling_plan)
    write_json(metadata_dir / "stage1_collection_plan.json", collection_plan)
    write_json(metadata_dir / "stage1_manifest.json", manifest)

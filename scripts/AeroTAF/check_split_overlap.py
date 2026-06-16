#!/usr/bin/env python
import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    import numpy as np
except ModuleNotFoundError as exc:
    logging.info(f"Error: missing dependency: {exc}")
    logging.info("Please activate the same Python environment used by this project, then run this script again.")
    sys.exit(1)

from scripts.AeroTAF.collector.path_utils import normalize_path, resolve_project_path


SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitData:
    name: str
    path: Path
    mode: str
    parent_indices: np.ndarray
    raw_file_indices: np.ndarray
    time_indices: np.ndarray
    episode_ids: np.ndarray
    sample_category: np.ndarray
    source_files: np.ndarray
    category_names: list

    @property
    def size(self):
        return int(self.time_indices.shape[0])


def object_array_to_strings(values):
    return np.asarray([str(v) for v in np.asarray(values, dtype=object).reshape(-1)], dtype=object)


def np_scalar_to_string(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0].item())
    return str(value)


def decode_category_names(data):
    if "sample_category_names" in data.files:
        return [str(x) for x in data["sample_category_names"].tolist()]
    return ["event", "high_field", "high_change", "stable"]


def require_keys(path, files, keys):
    missing = [key for key in keys if key not in files]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")


def load_all_target(all_target_path):
    with np.load(all_target_path, allow_pickle=True) as data:
        require_keys(
            all_target_path,
            data.files,
            ["raw_file_indices", "time_indices", "sample_category", "source_files"],
        )
        episode_ids = (
            data["episode_ids_per_step"].astype(np.int64, copy=False)
            if "episode_ids_per_step" in data.files
            else np.full(data["time_indices"].shape[0], -1, dtype=np.int64)
        )
        return {
            "raw_file_indices": data["raw_file_indices"].astype(np.int64, copy=False),
            "time_indices": data["time_indices"].astype(np.int64, copy=False),
            "episode_ids": episode_ids,
            "sample_category": data["sample_category"].astype(np.int64, copy=False),
            "source_files": object_array_to_strings(data["source_files"]),
            "category_names": decode_category_names(data),
        }


def split_uses_all_target_index(split_path):
    with np.load(split_path, allow_pickle=True) as data:
        return "all_target_indices" in data.files


def load_index_split(split_path, split_name, all_target):
    with np.load(split_path, allow_pickle=True) as data:
        require_keys(split_path, data.files, ["all_target_indices"])
        indices = data["all_target_indices"].astype(np.int64, copy=False).reshape(-1)

    parent_size = int(all_target["time_indices"].shape[0])
    if np.any(indices < 0) or np.any(indices >= parent_size):
        bad = indices[(indices < 0) | (indices >= parent_size)][:10].tolist()
        raise ValueError(f"{split_path}: all_target_indices out of range, examples={bad}, parent_size={parent_size}")

    return SplitData(
        name=split_name,
        path=split_path,
        mode="all_target_index",
        parent_indices=indices,
        raw_file_indices=all_target["raw_file_indices"][indices],
        time_indices=all_target["time_indices"][indices],
        episode_ids=all_target["episode_ids"][indices],
        sample_category=all_target["sample_category"][indices],
        source_files=all_target["source_files"],
        category_names=all_target["category_names"],
    )


def load_full_split(split_path, split_name):
    with np.load(split_path, allow_pickle=True) as data:
        require_keys(split_path, data.files, ["raw_file_indices", "time_indices", "sample_category", "source_files"])
        size = int(data["time_indices"].shape[0])
        parent_indices = (
            data["global_step_indices"].astype(np.int64, copy=False).reshape(-1)
            if "global_step_indices" in data.files
            else np.full(size, -1, dtype=np.int64)
        )
        episode_ids = (
            data["episode_ids_per_step"].astype(np.int64, copy=False).reshape(-1)
            if "episode_ids_per_step" in data.files
            else np.full(size, -1, dtype=np.int64)
        )
        return SplitData(
            name=split_name,
            path=split_path,
            mode="full_split",
            parent_indices=parent_indices,
            raw_file_indices=data["raw_file_indices"].astype(np.int64, copy=False).reshape(-1),
            time_indices=data["time_indices"].astype(np.int64, copy=False).reshape(-1),
            episode_ids=episode_ids,
            sample_category=data["sample_category"].astype(np.int64, copy=False).reshape(-1),
            source_files=object_array_to_strings(data["source_files"]),
            category_names=decode_category_names(data),
        )


def resolve_all_target_path(data_dir, train_split_path):
    with np.load(train_split_path, allow_pickle=True) as data:
        all_target_file = np_scalar_to_string(data["all_target_file"]) if "all_target_file" in data.files else "all_target.npz"
    candidate = data_dir / all_target_file
    if candidate.exists():
        return candidate
    return resolve_project_path(all_target_file)


def load_splits(data_dir):
    split_paths = {name: data_dir / f"{name}.npz" for name in SPLIT_NAMES}
    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files: {missing}")

    if split_uses_all_target_index(split_paths["train"]):
        all_target_path = resolve_all_target_path(data_dir, split_paths["train"])
        if not all_target_path.exists():
            raise FileNotFoundError(f"all_target file not found: {all_target_path}")
        all_target = load_all_target(all_target_path)
        splits = {name: load_index_split(path, name, all_target) for name, path in split_paths.items()}
        return splits, all_target_path

    splits = {name: load_full_split(path, name) for name, path in split_paths.items()}
    return splits, None


def category_name(split, category_id):
    category_id = int(category_id)
    if 0 <= category_id < len(split.category_names):
        return split.category_names[category_id]
    return str(category_id)


def raw_file_name(split, row):
    raw_idx = int(split.raw_file_indices[row])
    if 0 <= raw_idx < split.source_files.shape[0]:
        return str(split.source_files[raw_idx])
    return f"<bad-raw-index:{raw_idx}>"


def make_raw_time_key(split, row, include_episode_id):
    key = (raw_file_name(split, row), int(split.time_indices[row]))
    if include_episode_id:
        key = key + (int(split.episode_ids[row]),)
    return key


def build_raw_time_first_pos(split, include_episode_id):
    first_pos = {}
    duplicate_count = 0
    for row in range(split.size):
        key = make_raw_time_key(split, row, include_episode_id)
        if key in first_pos:
            duplicate_count += 1
        else:
            first_pos[key] = row
    return first_pos, duplicate_count


def build_parent_first_pos(split):
    valid = split.parent_indices >= 0
    first_pos = {}
    duplicate_count = 0
    for row, parent_idx in enumerate(split.parent_indices.tolist()):
        if parent_idx < 0:
            continue
        if parent_idx in first_pos:
            duplicate_count += 1
        else:
            first_pos[int(parent_idx)] = row
    return first_pos, duplicate_count, int(valid.sum())


def overlap_parent_rows(query, train_first_parent):
    rows = []
    if np.all(query.parent_indices < 0):
        return rows
    for row, parent_idx in enumerate(query.parent_indices.tolist()):
        parent_idx = int(parent_idx)
        if parent_idx in train_first_parent:
            rows.append((row, train_first_parent[parent_idx], parent_idx))
    return rows


def overlap_raw_time_rows(query, train_first_raw_time, include_episode_id):
    rows = []
    for row in range(query.size):
        key = make_raw_time_key(query, row, include_episode_id)
        train_row = train_first_raw_time.get(key)
        if train_row is not None:
            rows.append((row, train_row, key))
    return rows


def split_counts(split):
    counts = {}
    for category_id in sorted(np.unique(split.sample_category).astype(int).tolist()):
        counts[category_name(split, category_id)] = int((split.sample_category == category_id).sum())
    return counts


def write_overlap_csv(path, train, query_overlaps, max_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_split",
                "overlap_type",
                "query_row",
                "train_row",
                "parent_index",
                "query_raw_file",
                "query_time_index",
                "query_episode_id",
                "query_category",
                "train_raw_file",
                "train_time_index",
                "train_episode_id",
                "train_category",
            ],
        )
        writer.writeheader()
        for query, rows, overlap_type in query_overlaps:
            for item in rows[:max(0, int(max_rows) - written)]:
                query_row, train_row, value = item
                parent_index = int(value) if overlap_type == "parent_index" else ""
                writer.writerow(
                    {
                        "query_split": query.name,
                        "overlap_type": overlap_type,
                        "query_row": int(query_row),
                        "train_row": int(train_row),
                        "parent_index": parent_index,
                        "query_raw_file": raw_file_name(query, query_row),
                        "query_time_index": int(query.time_indices[query_row]),
                        "query_episode_id": int(query.episode_ids[query_row]),
                        "query_category": category_name(query, query.sample_category[query_row]),
                        "train_raw_file": raw_file_name(train, train_row),
                        "train_time_index": int(train.time_indices[train_row]),
                        "train_episode_id": int(train.episode_ids[train_row]),
                        "train_category": category_name(train, train.sample_category[train_row]),
                    }
                )
                written += 1
                if written >= max_rows:
                    return written
    return written


def get_parser():
    parser = argparse.ArgumentParser(description="Check whether AeroTAF val/test splits overlap with train time points.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing train.npz, val.npz, test.npz and optionally all_target.npz.")
    parser.add_argument("--output-csv", type=str, default="", help="CSV path for overlap examples. Defaults to data_dir/split_overlap_report.csv.")
    parser.add_argument("--max-report-rows", type=int, default=500, help="Max overlap rows written into CSV.")
    parser.add_argument("--include-episode-id", action="store_true", help="Use raw_file + time_index + episode_id as raw-time key. Default uses raw_file + time_index.")
    return parser


def main(args):
    all_args = get_parser().parse_args(args)
    data_dir = resolve_project_path(all_args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    output_csv = resolve_project_path(all_args.output_csv) if all_args.output_csv else data_dir / "split_overlap_report.csv"
    splits, all_target_path = load_splits(data_dir)
    train = splits["train"]

    logging.info("=" * 72)
    logging.info("AeroTAF Split Overlap Check")
    logging.info("=" * 72)
    logging.info(f"data dir      : {normalize_path(data_dir)}")
    logging.info(f"split mode    : {train.mode}")
    logging.info(f"all target    : {normalize_path(all_target_path) if all_target_path else '<not used>'}")
    logging.info(f"raw-time key  : raw_file + time_index" + (" + episode_id" if all_args.include_episode_id else ""))
    logging.info("-" * 72)

    train_first_parent, train_parent_dups, train_parent_valid = build_parent_first_pos(train)
    train_first_raw_time, train_raw_time_dups = build_raw_time_first_pos(train, all_args.include_episode_id)

    for split in splits.values():
        first_parent, parent_dups, parent_valid = build_parent_first_pos(split)
        first_raw_time, raw_time_dups = build_raw_time_first_pos(split, all_args.include_episode_id)
        logging.info(
            f"[{split.name:5s}] rows={split.size} | categories={split_counts(split)} | "
            f"parent_unique={len(first_parent)}/{parent_valid}, parent_dups={parent_dups} | "
            f"raw_time_unique={len(first_raw_time)}, raw_time_dups={raw_time_dups}"
        )

    logging.info("-" * 72)
    query_overlaps = []
    has_overlap = False
    for split_name in ("val", "test"):
        query = splits[split_name]
        parent_rows = overlap_parent_rows(query, train_first_parent)
        raw_time_rows = overlap_raw_time_rows(query, train_first_raw_time, all_args.include_episode_id)
        query_overlaps.append((query, parent_rows, "parent_index"))
        query_overlaps.append((query, raw_time_rows, "raw_time"))
        has_overlap = has_overlap or bool(parent_rows) or bool(raw_time_rows)
        logging.info(
            f"[{split_name} -> train] parent_overlap={len(parent_rows)} "
            f"({len(parent_rows) / max(query.size, 1):.6f}) | "
            f"raw_time_overlap={len(raw_time_rows)} ({len(raw_time_rows) / max(query.size, 1):.6f})"
        )

    written = write_overlap_csv(output_csv, train, query_overlaps, all_args.max_report_rows)
    logging.info("-" * 72)
    logging.info(f"overlap csv   : {normalize_path(output_csv)} | rows_written={written}")
    if has_overlap:
        logging.info("Result        : FOUND overlap between val/test and train.")
    else:
        logging.info("Result        : no val/test overlap with train was found.")


if __name__ == "__main__":
    default_args = [
        "--data-dir", "datasets/aerotaf/4v4_shoot_mappo_pool/stage1/processed_detail_index_k_target_K100",
        "--output-csv", "",
        "--max-report-rows", "500",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)

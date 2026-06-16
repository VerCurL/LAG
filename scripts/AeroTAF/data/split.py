import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def normalize_split_ratios(train_ratio, val_ratio, test_ratio):
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError(f"Invalid split ratios: {ratios.tolist()}")
    total = float(ratios.sum())
    if total <= 0.0:
        raise ValueError("At least one split ratio must be positive.")
    return ratios / total


def _largest_remainder_counts(size, ratios):
    size = int(size)
    if size <= 0:
        return np.zeros(len(ratios), dtype=np.int64)

    raw = ratios * size
    counts = np.floor(raw).astype(np.int64)
    remainder = size - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def split_indices_by_category(categories, category_names, train_ratio, val_ratio, test_ratio, seed):
    categories = np.asarray(categories).reshape(-1)
    ratios = normalize_split_ratios(train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(int(seed))

    split_indices = {name: [] for name in SPLIT_NAMES}
    category_split_counts = {}

    for category_id, category_name in enumerate(category_names):
        indices = np.flatnonzero(categories == int(category_id)).astype(np.int64, copy=False)
        rng.shuffle(indices)
        counts = _largest_remainder_counts(indices.shape[0], ratios)

        train_end = int(counts[0])
        val_end = train_end + int(counts[1])
        parts = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }
        for split_name, split_part in parts.items():
            split_indices[split_name].append(split_part)

        category_split_counts[str(category_name)] = {
            split_name: int(parts[split_name].shape[0])
            for split_name in SPLIT_NAMES
        }
        category_split_counts[str(category_name)]["total"] = int(indices.shape[0])

    for split_name in SPLIT_NAMES:
        if split_indices[split_name]:
            merged = np.concatenate(split_indices[split_name], axis=0).astype(np.int64, copy=False)
            rng.shuffle(merged)
        else:
            merged = np.asarray([], dtype=np.int64)
        split_indices[split_name] = merged

    return split_indices, category_split_counts, {name: float(value) for name, value in zip(SPLIT_NAMES, ratios)}

import numpy as np
from torch.utils.data import Dataset

from scripts.AeroTAF.data.schema import LABEL_EVENT, LABEL_HIGH_ATTACK, LABEL_HIGH_CHANGE, LABEL_HIGH_THREAT


def _fixed_window_bounds(center, length, episode_length):
    if episode_length <= length:
        return 0, episode_length
    start = int(center) - length // 2
    start = max(0, min(start, episode_length - length))
    return start, start + length


def build_priority_windows(store, chunk_length, key_stride, background_stride, mode="priority"):
    windows = []
    for episode_index in range(len(store)):
        episode = store.episode(episode_index)
        length = int(episode["length"])
        sample_multi_hot = episode.get("sample_multi_hot")
        if sample_multi_hot is None:
            raise RuntimeError("sample_multi_hot is required to build AeroTAF windows.")
        priorities = episode["sample_priority"].reshape(-1)

        seen = set()

        def add_window(start, end, window_type):
            key = (episode_index, int(start), int(end), window_type)
            if key in seen:
                return
            seen.add(key)
            local_priority = priorities[start:end]
            window_priority = float(np.max(local_priority) + 0.5 * np.mean(local_priority))
            if np.any(sample_multi_hot[start:end, LABEL_EVENT] > 0.5):
                window_priority += 2.0
            windows.append(
                {
                    "episode_index": episode_index,
                    "start": int(start),
                    "end": int(end),
                    "window_type": window_type,
                    "window_priority": float(np.clip(window_priority, 1.0, 12.0)),
                }
            )

        if mode == "natural":
            stride = max(1, int(background_stride))
            last_end = -1
            for start in range(0, max(length, 1), stride):
                end = min(start + int(chunk_length), length)
                if end > start:
                    add_window(start, end, "natural")
                    last_end = end
            if length > chunk_length and last_end != length:
                add_window(length - int(chunk_length), length, "natural")
            continue

        event_indices = np.where(sample_multi_hot[:, LABEL_EVENT] > 0.5)[0]
        high_change_indices = np.where(sample_multi_hot[:, LABEL_HIGH_CHANGE] > 0.5)[0]
        high_field_indices = np.where(
            (sample_multi_hot[:, LABEL_HIGH_THREAT] > 0.5)
            | (sample_multi_hot[:, LABEL_HIGH_ATTACK] > 0.5)
        )[0]

        for center in event_indices[:: max(1, int(key_stride))]:
            start, end = _fixed_window_bounds(center, int(chunk_length), length)
            add_window(start, end, "event")

        for center in high_change_indices[:: max(1, int(key_stride))]:
            start, end = _fixed_window_bounds(center, int(chunk_length), length)
            add_window(start, end, "high_change")

        for center in high_field_indices[:: max(1, int(key_stride))]:
            start, end = _fixed_window_bounds(center, int(chunk_length), length)
            add_window(start, end, "high_field")

        stride = max(1, int(background_stride))
        for start in range(0, length, stride):
            end = min(start + int(chunk_length), length)
            if end <= start:
                continue
            if np.any(sample_multi_hot[start:end] > 0.5):
                continue
            add_window(start, end, "background")

    return windows


class AeroTAFWindowDataset(Dataset):
    def __init__(self, store, chunk_length, key_stride, background_stride, mode="priority"):
        self.store = store
        self.windows = build_priority_windows(
            store=store,
            chunk_length=chunk_length,
            key_stride=key_stride,
            background_stride=background_stride,
            mode=mode,
        )
        if not self.windows:
            raise RuntimeError("No AeroTAF windows were generated.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        episode = self.store.episode(window["episode_index"])
        start = window["start"]
        end = window["end"]
        temporal_targets = episode.get("temporal_targets")
        sample_weight = episode.get("sample_weight")
        sample_multi_hot = episode.get("sample_multi_hot")
        return (
            episode["obs"][start:end],
            episode["actions"][start:end],
            episode["threat_targets"][start:end],
            episode["attack_targets"][start:end],
            temporal_targets[start:end] if temporal_targets is not None else None,
            sample_weight[start:end] if sample_weight is not None else None,
            sample_multi_hot[start:end] if sample_multi_hot is not None else None,
            end - start,
            start,
            np.asarray([window["window_priority"]], dtype=np.float32),
            window["window_type"],
        )

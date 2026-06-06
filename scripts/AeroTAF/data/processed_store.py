import numpy as np
from torch.utils.data import Dataset


class ProcessedEpisodeStore:
    def __init__(self, npz_path, require_temporal_targets=False):
        with np.load(npz_path, allow_pickle=True) as data:
            required = ["obs", "actions", "threat_targets", "attack_targets", "episode_lengths"]
            if require_temporal_targets:
                required.append("temporal_targets")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{npz_path} missing keys: {missing}")

            self.obs = data["obs"].astype(np.float32, copy=False)
            self.actions = data["actions"].astype(np.float32, copy=False)
            self.threat_targets = data["threat_targets"].astype(np.float32, copy=False)
            self.attack_targets = data["attack_targets"].astype(np.float32, copy=False)
            self.temporal_targets = data["temporal_targets"].astype(np.float32, copy=False) if "temporal_targets" in data.files else None
            self.sample_bucket = data["sample_bucket"].astype(np.int16, copy=False) if "sample_bucket" in data.files else np.zeros(self.obs.shape[0], dtype=np.int16)
            self.sample_priority = data["sample_priority"].astype(np.float32, copy=False) if "sample_priority" in data.files else np.ones((self.obs.shape[0], 1), dtype=np.float32)
            self.sample_weight = data["sample_weight"].astype(np.float32, copy=False) if "sample_weight" in data.files else np.ones((self.obs.shape[0], 1), dtype=np.float32)
            self.event_flags = data["event_flags"].astype(np.float32, copy=False) if "event_flags" in data.files else None
            self.field_delta_features = data["field_delta_features"].astype(np.float32, copy=False) if "field_delta_features" in data.files else None
            self.episode_lengths = data["episode_lengths"].astype(np.int32, copy=False)
            self.episode_ids = data["episode_ids"].astype(np.int32, copy=False) if "episode_ids" in data.files else None

        total_steps = int(self.episode_lengths.sum())
        if total_steps != self.obs.shape[0]:
            raise ValueError(f"{npz_path}: episode_lengths sum {total_steps} != obs steps {self.obs.shape[0]}")

        self.offsets = []
        start = 0
        for length in self.episode_lengths.tolist():
            end = start + int(length)
            self.offsets.append((start, end))
            start = end

    def __len__(self):
        return len(self.offsets)

    def episode(self, index):
        start, end = self.offsets[index]
        item = {
            "obs": self.obs[start:end],
            "actions": self.actions[start:end],
            "threat_targets": self.threat_targets[start:end],
            "attack_targets": self.attack_targets[start:end],
            "sample_bucket": self.sample_bucket[start:end],
            "sample_priority": self.sample_priority[start:end],
            "sample_weight": self.sample_weight[start:end],
            "length": end - start,
            "episode_id": int(self.episode_ids[index]) if self.episode_ids is not None else index,
        }
        if self.temporal_targets is not None:
            item["temporal_targets"] = self.temporal_targets[start:end]
        if self.event_flags is not None:
            item["event_flags"] = self.event_flags[start:end]
        if self.field_delta_features is not None:
            item["field_delta_features"] = self.field_delta_features[start:end]
        return item


class AeroTAFStepDataset(Dataset):
    def __init__(self, npz_path):
        self.store = ProcessedEpisodeStore(npz_path)

    def __len__(self):
        return self.store.obs.shape[0]

    def __getitem__(self, index):
        return (
            self.store.obs[index],
            self.store.actions[index],
            self.store.threat_targets[index],
            self.store.attack_targets[index],
            self.store.sample_weight[index],
            self.store.sample_bucket[index],
        )

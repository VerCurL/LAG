import numpy as np
from torch.utils.data import WeightedRandomSampler


def build_priority_sampler(dataset, num_samples=None, replacement=True):
    weights = np.asarray(dataset.store.sample_priority, dtype=np.float64).reshape(-1)
    weights = np.maximum(weights, 1e-6)
    if num_samples is None:
        num_samples = len(weights)
    return WeightedRandomSampler(weights=weights, num_samples=int(num_samples), replacement=replacement)


def bucket_counts(sample_bucket):
    sample_bucket = np.asarray(sample_bucket).reshape(-1)
    return {int(bucket): int((sample_bucket == bucket).sum()) for bucket in np.unique(sample_bucket)}


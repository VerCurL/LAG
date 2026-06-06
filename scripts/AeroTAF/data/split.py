import random
from collections import defaultdict


def choose_test_pair_keys(pair_groups, test_pair_ratio, split_seed):
    pair_type_to_keys = defaultdict(list)
    for pair_key, episodes in pair_groups.items():
        pair_type = episodes[0]["meta"].get("pair_type", "unknown")
        pair_type_to_keys[pair_type].append(pair_key)

    rng = random.Random(split_seed)
    test_pair_keys = set()
    for _, keys in pair_type_to_keys.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        if len(keys) <= 1:
            continue
        count = int(round(len(keys) * float(test_pair_ratio)))
        count = max(1, count)
        count = min(count, len(keys) - 1)
        test_pair_keys.update(keys[:count])
    return test_pair_keys


def split_stage1_episodes(episode_items, split_seed, test_pair_ratio, val_seed_ratio):
    pair_groups = defaultdict(list)
    for item in episode_items:
        pair_groups[item["meta"]["pair_key"]].append(item)

    for pair_key in pair_groups:
        pair_groups[pair_key] = sorted(
            pair_groups[pair_key],
            key=lambda item: (
                item["meta"].get("random_seed", -1),
                item["meta"].get("episode_id", -1),
                item["meta"].get("source_file", ""),
            ),
        )

    test_pair_keys = choose_test_pair_keys(pair_groups, test_pair_ratio, split_seed)
    train_items = []
    val_id_items = []
    test_pair_ood_items = []
    rng = random.Random(split_seed + 999)

    for pair_key, items in pair_groups.items():
        if pair_key in test_pair_keys:
            test_pair_ood_items.extend(items)
            continue

        pair_items = list(items)
        rng.shuffle(pair_items)
        val_count = int(round(len(pair_items) * float(val_seed_ratio)))
        if len(pair_items) >= 2:
            val_count = max(1, val_count)
            val_count = min(val_count, len(pair_items) - 1)
        else:
            val_count = 0

        val_id_items.extend(pair_items[:val_count])
        train_items.extend(pair_items[val_count:])

    split_info = {
        "num_pairs_total": len(pair_groups),
        "num_pairs_train_seen": len(pair_groups) - len(test_pair_keys),
        "num_pairs_test_pair_ood": len(test_pair_keys),
        "num_episodes_train": len(train_items),
        "num_episodes_val_id": len(val_id_items),
        "num_episodes_test_pair_ood": len(test_pair_ood_items),
    }
    return {
        "train": train_items,
        "val_id": val_id_items,
        "test_pair_ood": test_pair_ood_items,
        "split_info": split_info,
        "test_pair_keys": sorted(test_pair_keys),
    }


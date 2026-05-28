from .task_history import build_task_key


def build_stage1_profiling_tasks(
    profiling_pairs,
    scenarios,
    seeds_per_pair,
    seed_start,
    common_task_fields,
):
    tasks = []
    task_index = 0
    for pair in profiling_pairs:
        for scenario in scenarios:
            for seed_offset in range(seeds_per_pair):
                seed = seed_start + task_index
                task = {
                    "task_kind": "profile",
                    "episode_id": -1,
                    "seed": seed,
                    "save_raw": False,
                    **common_task_fields,
                    **pair,
                    **scenario,
                }
                task["task_key"] = build_task_key(task)
                tasks.append(task)
                task_index += 1
    return tasks


def build_stage1_collection_tasks(
    directed_pairs,
    scenarios,
    seeds_per_pair,
    seed_start,
    common_task_fields,
    start_episode_id,
    completed_keys,
):
    tasks = []
    task_index = 0
    pending_index = 0

    for pair in directed_pairs:
        for scenario in scenarios:
            for seed_offset in range(seeds_per_pair):
                seed = seed_start + task_index
                task = {
                    "task_kind": "collect",
                    "episode_id": start_episode_id + pending_index,
                    "seed": seed,
                    "save_raw": True,
                    **common_task_fields,
                    **pair,
                    **scenario,
                }
                task["task_key"] = build_task_key(task)
                if task["task_key"] not in completed_keys:
                    tasks.append(task)
                    pending_index += 1
                task_index += 1
    return tasks

from collections import defaultdict


def build_stage1_profiling_pairs(model_entries):
    stage_groups = defaultdict(list)
    for entry in sorted(model_entries, key=lambda item: (item["checkpoint_step"], item["checkpoint_path"])):
        stage_groups[entry["stage_hint"]].append(entry)

    stage_order = ("early", "mid", "late")
    stage_rotation = {stage_name: 0 for stage_name in stage_order}
    profiling_pairs = []

    for ego_entry in sorted(model_entries, key=lambda item: (item["checkpoint_step"], item["checkpoint_path"])):
        for target_stage in stage_order:
            candidates = [
                item for item in stage_groups[target_stage]
                if item["checkpoint_path"] != ego_entry["checkpoint_path"]
            ]
            if not candidates:
                continue

            chosen_index = stage_rotation[target_stage] % len(candidates)
            chosen_entry = candidates[chosen_index]
            stage_rotation[target_stage] += 1

            profiling_pairs.append({
                "ego_model_path": ego_entry["checkpoint_path"],
                "enm_model_path": chosen_entry["checkpoint_path"],
                "ego_model_id": ego_entry["model_id"],
                "enm_model_id": chosen_entry["model_id"],
                "ego_level": "unknown",
                "enm_level": "unknown",
                "ego_style": "unknown",
                "enm_style": "unknown",
                "ego_stage_hint": ego_entry["stage_hint"],
                "enm_stage_hint": chosen_entry["stage_hint"],
                "pair_type": f"profile_{ego_entry['stage_hint']}_vs_{chosen_entry['stage_hint']}",
            })

    return profiling_pairs


def _safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_profiling_results(model_entries, profiling_results):
    result_map = defaultdict(list)
    for result in profiling_results:
        if result.get("status") != "ok":
            continue
        result_map[result["ego_model_path"]].append(result)

    profiles = []
    for entry in sorted(model_entries, key=lambda item: (item["checkpoint_step"], item["checkpoint_path"])):
        rows = result_map.get(entry["checkpoint_path"], [])
        wins = sum(1 for row in rows if row.get("winner") == "ego")
        draws = sum(1 for row in rows if row.get("winner") == "draw")
        losses = sum(1 for row in rows if row.get("winner") == "enm")

        profile = {
            "model_id": entry["model_id"],
            "checkpoint_path": entry["checkpoint_path"],
            "checkpoint_name": entry["checkpoint_name"],
            "checkpoint_step": entry["checkpoint_step"],
            "stage_hint": entry.get("stage_hint", "unknown"),
            "profiling_episodes": len(rows),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / len(rows) if rows else 0.0,
            "draw_rate": draws / len(rows) if rows else 0.0,
            "loss_rate": losses / len(rows) if rows else 0.0,
            "avg_reward_margin": _safe_mean([row.get("reward_margin", 0.0) for row in rows]),
            "avg_alive_margin": _safe_mean([row.get("alive_margin", 0.0) for row in rows]),
            "avg_episode_steps": _safe_mean([row.get("steps", 0.0) for row in rows]),
            "avg_ego_speed_mps": _safe_mean([row.get("ego_speed_mps_mean", 0.0) for row in rows]),
            "avg_ego_nearest_enemy_distance_m": _safe_mean([row.get("ego_nearest_enemy_distance_m_mean", 0.0) for row in rows]),
            "avg_ego_attack_window_reward": _safe_mean([row.get("ego_attack_window_reward_mean", 0.0) for row in rows]),
            "avg_ego_missile_avoid_reward": _safe_mean([row.get("ego_missile_avoid_reward_mean", 0.0) for row in rows]),
        }
        profiles.append(profile)

    return profiles

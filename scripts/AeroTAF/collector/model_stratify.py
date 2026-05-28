def _rank_dict(items, key):
    sorted_items = sorted(items, key=lambda item: item[key])
    total = max(len(sorted_items) - 1, 1)
    return {
        item["checkpoint_path"]: index / total
        for index, item in enumerate(sorted_items)
    }


def _style_label(profile):
    distance = profile.get("avg_ego_nearest_enemy_distance_m", 0.0)
    attack_reward = profile.get("avg_ego_attack_window_reward", 0.0)
    avoid_reward = profile.get("avg_ego_missile_avoid_reward", 0.0)

    if attack_reward >= max(avoid_reward, 0.0) and distance <= 12000.0:
        return "aggressive_close"
    if avoid_reward > attack_reward:
        return "evasive"
    if distance >= 18000.0:
        return "long_range"
    return "balanced"


def assign_strength_tiers(model_profiles, tier_sizes=(10, 10, 10)):
    if sum(tier_sizes) != len(model_profiles):
        raise ValueError(
            f"tier_sizes {tier_sizes} do not match model count {len(model_profiles)}"
        )

    reward_ranks = _rank_dict(model_profiles, "avg_reward_margin")
    alive_ranks = _rank_dict(model_profiles, "avg_alive_margin")
    win_ranks = _rank_dict(model_profiles, "win_rate")
    checkpoint_ranks = _rank_dict(model_profiles, "checkpoint_step")

    for profile in model_profiles:
        path = profile["checkpoint_path"]
        profile["strength_score"] = (
            0.50 * profile["win_rate"]
            + 0.20 * reward_ranks[path]
            + 0.20 * alive_ranks[path]
            + 0.10 * checkpoint_ranks[path]
        )
        profile["style_label"] = _style_label(profile)

    sorted_profiles = sorted(
        model_profiles,
        key=lambda item: (
            item["strength_score"],
            item["win_rate"],
            item["avg_reward_margin"],
            item["checkpoint_step"],
        ),
        reverse=True,
    )

    high_count, mid_count, low_count = tier_sizes
    for index, profile in enumerate(sorted_profiles):
        if index < high_count:
            tier = "high"
        elif index < high_count + mid_count:
            tier = "mid"
        else:
            tier = "low"
        profile["strength_tier"] = tier

    return sorted_profiles

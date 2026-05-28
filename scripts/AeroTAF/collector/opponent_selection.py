from collections import defaultdict


TIER_ORDER = ("high", "mid", "low")


def _pair_type(ego_tier, enm_tier):
    return f"{ego_tier}_vs_{enm_tier}"


def _checkpoint_distance(ego_model, enm_model):
    return abs(ego_model.get("checkpoint_step", -1) - enm_model.get("checkpoint_step", -1))


def _candidate_score(ego_model, candidate_model, opponent_usage, prefer_style_diversity=True):
    same_style_penalty = 0
    if prefer_style_diversity and ego_model.get("style_label") == candidate_model.get("style_label"):
        same_style_penalty = 1

    return (
        opponent_usage[candidate_model["checkpoint_path"]],
        same_style_penalty,
        -_checkpoint_distance(ego_model, candidate_model),
        candidate_model["checkpoint_step"],
        candidate_model["checkpoint_path"],
    )


def select_stage1_directed_pairs(
    model_entries,
    opponents_per_tier=2,
    prefer_style_diversity=True,
):
    tier_groups = defaultdict(list)
    for entry in model_entries:
        tier_groups[entry["strength_tier"]].append(entry)

    for tier_name in TIER_ORDER:
        tier_groups[tier_name] = sorted(
            tier_groups[tier_name],
            key=lambda item: (item.get("checkpoint_step", -1), item["checkpoint_path"]),
        )

    directed_pairs = []
    opponent_usage = defaultdict(int)

    for ego_model in sorted(model_entries, key=lambda item: (item["strength_tier"], item["checkpoint_step"], item["checkpoint_path"])):
        selected_opponents = set()
        for target_tier in TIER_ORDER:
            candidates = [
                item for item in tier_groups[target_tier]
                if item["checkpoint_path"] != ego_model["checkpoint_path"]
                and item["checkpoint_path"] not in selected_opponents
            ]

            candidates = sorted(
                candidates,
                key=lambda item: _candidate_score(
                    ego_model,
                    item,
                    opponent_usage,
                    prefer_style_diversity=prefer_style_diversity,
                ),
            )

            quota = min(opponents_per_tier, len(candidates))
            for candidate in candidates[:quota]:
                directed_pairs.append({
                    "ego_model_path": ego_model["checkpoint_path"],
                    "enm_model_path": candidate["checkpoint_path"],
                    "ego_model_id": ego_model["model_id"],
                    "enm_model_id": candidate["model_id"],
                    "ego_level": ego_model["strength_tier"],
                    "enm_level": candidate["strength_tier"],
                    "ego_style": ego_model.get("style_label", "unknown"),
                    "enm_style": candidate.get("style_label", "unknown"),
                    "ego_stage_hint": ego_model.get("stage_hint", "unknown"),
                    "enm_stage_hint": candidate.get("stage_hint", "unknown"),
                    "pair_type": _pair_type(ego_model["strength_tier"], candidate["strength_tier"]),
                })
                selected_opponents.add(candidate["checkpoint_path"])
                opponent_usage[candidate["checkpoint_path"]] += 1

    return directed_pairs

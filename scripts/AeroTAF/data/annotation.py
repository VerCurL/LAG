import math

import numpy as np

from envs.JSBSim.situation.field import (
    A_ALIVE,
    A_LEFT_MISSILES,
    A_POS,
    A_SHOTDOWN,
    A_TEAM,
    A_THREAT_MISSILE_EXIST,
    A_VEL,
    M_ALIVE,
    M_PARENT,
    M_SUCCESS,
)
from scripts.AeroTAF.data.schema import (
    BUCKET_ACTION_CHANGE,
    BUCKET_BACKGROUND,
    BUCKET_EVENT,
    BUCKET_HIGH_ATTACK,
    BUCKET_HIGH_CHANGE,
    BUCKET_HIGH_THREAT,
    EVENT_NAMES,
    AnnotationConfig,
    bucket_summary,
)


def _safe_percentile(values, percentile, default=0.0):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, percentile))


def _episode_deltas(values, future_steps):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    length = values.shape[0]
    delta = np.zeros(length, dtype=np.float32)
    future_delta = np.zeros(length, dtype=np.float32)
    if length <= 1:
        return delta, future_delta

    delta[1:] = np.abs(values[1:] - values[:-1])
    for t in range(length):
        future_t = min(t + int(future_steps), length - 1)
        future_delta[t] = abs(float(values[future_t]) - float(values[t]))
    return delta, future_delta


def _episode_action_delta(actions):
    actions = np.asarray(actions, dtype=np.float32)
    length = actions.shape[0]
    out = np.zeros(length, dtype=np.float32)
    if length <= 1:
        return out
    diff = actions[1:] - actions[:-1]
    out[1:] = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    return out


def collect_threshold_stats(episode_items, config: AnnotationConfig):
    future_steps = max(1, min(10, int(config.field_k_step) // 2))
    stats = {
        "threat": [],
        "attack": [],
        "delta_threat": [],
        "delta_attack": [],
        "future_delta_threat": [],
        "future_delta_attack": [],
        "action_delta": [],
    }

    for item in episode_items:
        threat = item["threat_targets"].reshape(-1)
        attack = item["attack_targets"].reshape(-1)
        delta_threat, future_delta_threat = _episode_deltas(threat, future_steps)
        delta_attack, future_delta_attack = _episode_deltas(attack, future_steps)
        action_delta = _episode_action_delta(item["actions"])

        stats["threat"].append(threat)
        stats["attack"].append(attack)
        stats["delta_threat"].append(delta_threat)
        stats["delta_attack"].append(delta_attack)
        stats["future_delta_threat"].append(future_delta_threat)
        stats["future_delta_attack"].append(future_delta_attack)
        stats["action_delta"].append(action_delta)

    return {key: np.concatenate(value, axis=0) if value else np.asarray([], dtype=np.float32) for key, value in stats.items()}


def fit_annotation_thresholds(train_episode_items, config: AnnotationConfig):
    stats = collect_threshold_stats(train_episode_items, config)
    thresholds = {
        "high_threat_threshold": max(
            float(config.high_threat_floor),
            _safe_percentile(stats["threat"], config.high_field_percentile, config.high_threat_floor),
        ),
        "high_attack_threshold": max(
            float(config.high_attack_floor),
            _safe_percentile(stats["attack"], config.high_field_percentile, config.high_attack_floor),
        ),
        "very_high_threat_threshold": max(
            float(config.very_high_threat_floor),
            _safe_percentile(stats["threat"], config.very_high_field_percentile, config.very_high_threat_floor),
        ),
        "very_high_attack_threshold": max(
            float(config.very_high_attack_floor),
            _safe_percentile(stats["attack"], config.very_high_field_percentile, config.very_high_attack_floor),
        ),
        "delta_threat_threshold": max(
            float(config.delta_floor),
            _safe_percentile(stats["delta_threat"], config.delta_percentile, config.delta_floor),
        ),
        "delta_attack_threshold": max(
            float(config.delta_floor),
            _safe_percentile(stats["delta_attack"], config.delta_percentile, config.delta_floor),
        ),
        "future_delta_threat_threshold": max(
            float(config.future_delta_floor),
            _safe_percentile(stats["future_delta_threat"], config.future_delta_percentile, config.future_delta_floor),
        ),
        "future_delta_attack_threshold": max(
            float(config.future_delta_floor),
            _safe_percentile(stats["future_delta_attack"], config.future_delta_percentile, config.future_delta_floor),
        ),
        "action_change_threshold": _safe_percentile(stats["action_delta"], config.action_change_percentile, 0.0),
    }
    thresholds["future_delta_steps"] = max(1, min(10, int(config.field_k_step) // 2))
    return thresholds


def _team_indices(aircraft, ego_team):
    ego_indices = np.where(aircraft[:, A_TEAM] == ego_team)[0]
    enemy_indices = np.where(aircraft[:, A_TEAM] != ego_team)[0]
    return ego_indices, enemy_indices


def _angle_between(vec_a, vec_b):
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm <= 1e-8:
        return math.pi
    return float(np.arccos(np.clip(np.dot(vec_a, vec_b) / norm, -1.0, 1.0)))


def _closing_speed(own, target):
    rel = target[A_POS] - own[A_POS]
    dist = np.linalg.norm(rel)
    if dist <= 1e-8:
        return 0.0
    return float(np.dot(own[A_VEL] - target[A_VEL], rel / dist))


def _compute_geometry_events(snapshot, ego_team, config: AnnotationConfig):
    aircraft = snapshot["aircraft"]
    ego_indices, enemy_indices = _team_indices(aircraft, ego_team)
    theta_attack = np.deg2rad(float(config.theta_attack_deg))
    theta_nez = np.deg2rad(float(config.theta_nez_deg))

    attack_zone = False
    nez_zone = False
    close_range = False
    very_close_range = False

    for ego_idx in ego_indices:
        ego = aircraft[ego_idx]
        if ego[A_ALIVE] <= 0.5:
            continue
        for enemy_idx in enemy_indices:
            enemy = aircraft[enemy_idx]
            if enemy[A_ALIVE] <= 0.5:
                continue

            rel_ego_to_enemy = enemy[A_POS] - ego[A_POS]
            distance = float(np.linalg.norm(rel_ego_to_enemy))
            if distance <= config.r_attack:
                close_range = True
            if distance <= config.r_nez:
                very_close_range = True

            ego_ao = _angle_between(ego[A_VEL], rel_ego_to_enemy)
            enemy_ao = _angle_between(enemy[A_VEL], -rel_ego_to_enemy)
            ego_closing = _closing_speed(ego, enemy)
            enemy_closing = _closing_speed(enemy, ego)

            if (
                config.r_attack >= distance >= 0.0
                and ((ego[A_LEFT_MISSILES] > 0 and ego_ao <= theta_attack)
                     or (enemy[A_LEFT_MISSILES] > 0 and enemy_ao <= theta_attack))
            ):
                attack_zone = True

            if (
                distance <= config.r_nez
                and ((ego[A_LEFT_MISSILES] > 0 and ego_ao <= theta_nez and ego_closing > 0.0)
                     or (enemy[A_LEFT_MISSILES] > 0 and enemy_ao <= theta_nez and enemy_closing > 0.0))
            ):
                nez_zone = True

    return attack_zone, nez_zone, close_range, very_close_range


def build_event_flags(snapshots, ego_team, config: AnnotationConfig):
    length = len(snapshots)
    event_index = {name: idx for idx, name in enumerate(EVENT_NAMES)}
    flags = np.zeros((length, len(EVENT_NAMES)), dtype=np.float32)

    prev_aircraft = None
    prev_missiles = None
    prev_ego_left_missiles = None

    for t, snapshot in enumerate(snapshots):
        aircraft = snapshot["aircraft"]
        missiles = snapshot["missiles"]
        ego_indices, enemy_indices = _team_indices(aircraft, ego_team)

        if np.any(aircraft[ego_indices, A_THREAT_MISSILE_EXIST] > 0.5):
            flags[t, event_index["incoming_missile"]] = 1.0

        if missiles.size > 0:
            parent = missiles[:, M_PARENT].astype(int)
            alive = missiles[:, M_ALIVE] > 0.5
            success = missiles[:, M_SUCCESS] > 0.5
            ego_parent = np.isin(parent, ego_indices)
            if np.any(ego_parent & alive & (~success)):
                flags[t, event_index["outgoing_missile"]] = 1.0
            if np.any(ego_parent & success):
                if prev_missiles is None:
                    flags[t, event_index["hit_success"]] = 1.0
                else:
                    n = min(len(missiles), len(prev_missiles))
                    prev_success = np.zeros(len(missiles), dtype=bool)
                    prev_success[:n] = prev_missiles[:n, M_SUCCESS] > 0.5
                    if np.any(ego_parent & success & (~prev_success)):
                        flags[t, event_index["hit_success"]] = 1.0

        ego_left_missiles = aircraft[ego_indices, A_LEFT_MISSILES].astype(np.float32, copy=False)
        if prev_ego_left_missiles is not None and ego_left_missiles.shape == prev_ego_left_missiles.shape:
            if np.any(ego_left_missiles < prev_ego_left_missiles - 0.5):
                flags[t, event_index["shot_fired"]] = 1.0

        if prev_aircraft is not None:
            ego_shotdown_now = aircraft[ego_indices, A_SHOTDOWN] > 0.5
            ego_shotdown_prev = prev_aircraft[ego_indices, A_SHOTDOWN] > 0.5
            enemy_shotdown_now = aircraft[enemy_indices, A_SHOTDOWN] > 0.5
            enemy_shotdown_prev = prev_aircraft[enemy_indices, A_SHOTDOWN] > 0.5
            if np.any(ego_shotdown_now & (~ego_shotdown_prev)):
                flags[t, event_index["ego_shotdown"]] = 1.0
            if np.any(enemy_shotdown_now & (~enemy_shotdown_prev)):
                flags[t, event_index["enemy_shotdown"]] = 1.0

        attack_zone, nez_zone, close_range, very_close_range = _compute_geometry_events(snapshot, ego_team, config)
        flags[t, event_index["attack_zone_enter"]] = float(attack_zone)
        flags[t, event_index["nez_enter"]] = float(nez_zone)
        flags[t, event_index["close_range"]] = float(close_range)
        flags[t, event_index["very_close_range"]] = float(very_close_range)

        prev_aircraft = aircraft
        prev_missiles = missiles
        prev_ego_left_missiles = ego_left_missiles.copy()

    return flags


def expand_event_neighborhood(event_flags, pre_steps, post_steps):
    length = event_flags.shape[0]
    raw_event = np.any(event_flags > 0.5, axis=1)
    expanded = np.zeros(length, dtype=bool)
    event_indices = np.where(raw_event)[0]
    for idx in event_indices:
        start = max(0, int(idx) - int(pre_steps))
        end = min(length, int(idx) + int(post_steps) + 1)
        expanded[start:end] = True
    return expanded


def annotate_episode(item, thresholds, config: AnnotationConfig):
    threat = item["threat_targets"].reshape(-1)
    attack = item["attack_targets"].reshape(-1)
    future_steps = int(thresholds.get("future_delta_steps", max(1, min(10, int(config.field_k_step) // 2))))

    delta_threat, future_delta_threat = _episode_deltas(threat, future_steps)
    delta_attack, future_delta_attack = _episode_deltas(attack, future_steps)
    action_delta = _episode_action_delta(item["actions"])
    event_flags = build_event_flags(item.get("snapshots", []), float(config.ego_team), config)
    event_mask = expand_event_neighborhood(event_flags, config.event_pre_steps, config.event_post_steps)

    high_threat = threat >= float(thresholds["high_threat_threshold"])
    high_attack = attack >= float(thresholds["high_attack_threshold"])
    high_change = (
        (delta_threat >= float(thresholds["delta_threat_threshold"]))
        | (delta_attack >= float(thresholds["delta_attack_threshold"]))
        | (future_delta_threat >= float(thresholds["future_delta_threat_threshold"]))
        | (future_delta_attack >= float(thresholds["future_delta_attack_threshold"]))
    )
    action_change = action_delta >= float(thresholds["action_change_threshold"])

    sample_bucket = np.full(threat.shape[0], BUCKET_BACKGROUND, dtype=np.int16)
    sample_bucket[action_change] = BUCKET_ACTION_CHANGE
    sample_bucket[high_attack] = BUCKET_HIGH_ATTACK
    sample_bucket[high_threat] = BUCKET_HIGH_THREAT
    sample_bucket[high_change] = BUCKET_HIGH_CHANGE
    sample_bucket[event_mask] = BUCKET_EVENT

    priority = np.full(threat.shape[0], float(config.priority_base), dtype=np.float32)
    priority += high_threat.astype(np.float32) * float(config.priority_high_threat_bonus)
    priority += high_attack.astype(np.float32) * float(config.priority_high_attack_bonus)
    priority += high_change.astype(np.float32) * float(config.priority_high_change_bonus)
    priority += event_mask.astype(np.float32) * float(config.priority_event_bonus)
    priority += action_change.astype(np.float32) * float(config.priority_action_change_bonus)
    priority = np.clip(priority, float(config.priority_min), float(config.priority_max)).astype(np.float32)

    sample_weight = 1.0 + float(config.weight_from_priority_scale) * (priority - 1.0)
    sample_weight = np.clip(sample_weight, float(config.weight_min), float(config.weight_max)).astype(np.float32)

    item["event_flags"] = event_flags.astype(np.float32, copy=False)
    item["event_mask"] = event_mask.astype(np.float32).reshape(-1, 1)
    item["sample_bucket"] = sample_bucket
    item["sample_priority"] = priority.reshape(-1, 1)
    item["sample_weight"] = sample_weight.reshape(-1, 1)
    item["field_delta_features"] = np.stack(
        (delta_threat, delta_attack, future_delta_threat, future_delta_attack, action_delta),
        axis=-1,
    ).astype(np.float32, copy=False)
    item["annotation_summary"] = bucket_summary(sample_bucket)
    return item

def annotate_splits(split_result, thresholds, config: AnnotationConfig):
    annotated = {}
    for split_name in ("train", "val_id", "test_pair_ood"):
        annotated[split_name] = [annotate_episode(item, thresholds, config) for item in split_result[split_name]]
    return annotated

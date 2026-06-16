import numpy as np

from envs.JSBSim.situation.field import (
    A_LEFT_MISSILES,
    A_SHOTDOWN,
    A_TEAM,
    A_THREAT_MISSILE_EXIST,
    M_PARENT,
    M_SUCCESS,
)
from scripts.AeroTAF.data.schema import (
    CATEGORY_EVENT,
    CATEGORY_HIGH_CHANGE,
    CATEGORY_HIGH_FIELD,
    CATEGORY_NAMES,
    CATEGORY_STABLE,
    CONDITION_NAMES,
    DetailAnnotationConfig,
    EVENT_NAMES,
)


def safe_percentile(values, percentile, default=0.0):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, percentile))


def episode_delta(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    delta = np.zeros(values.shape[0], dtype=np.float32)
    if values.shape[0] > 1:
        delta[1:] = np.abs(values[1:] - values[:-1])
    return delta


def get_label_fields(item):
    return item["threat_targets"].reshape(-1), item["attack_targets"].reshape(-1)


def collect_threshold_stats(items, config: DetailAnnotationConfig):
    stats = {
        "threat": [],
        "attack": [],
        "delta_threat": [],
        "delta_attack": [],
    }

    for item in items:
        threat, attack = get_label_fields(item)
        stats["threat"].append(threat)
        stats["attack"].append(attack)
        stats["delta_threat"].append(episode_delta(threat))
        stats["delta_attack"].append(episode_delta(attack))

    return {
        key: np.concatenate(value, axis=0).astype(np.float32, copy=False) if value else np.asarray([], dtype=np.float32)
        for key, value in stats.items()
    }


def fit_detail_thresholds(train_items, config: DetailAnnotationConfig):
    stats = collect_threshold_stats(train_items, config)
    return {
        "label_field_source": "k_step_target",
        "high_threat_threshold": max(
            float(config.high_threat_floor),
            safe_percentile(stats["threat"], config.high_field_percentile, config.high_threat_floor),
        ),
        "high_attack_threshold": max(
            float(config.high_attack_floor),
            safe_percentile(stats["attack"], config.high_field_percentile, config.high_attack_floor),
        ),
        "delta_threat_threshold": max(
            float(config.delta_floor),
            safe_percentile(stats["delta_threat"], config.delta_percentile, config.delta_floor),
        ),
        "delta_attack_threshold": max(
            float(config.delta_floor),
            safe_percentile(stats["delta_attack"], config.delta_percentile, config.delta_floor),
        ),
    }


def _team_indices(aircraft, ego_team):
    ego_indices = np.where(aircraft[:, A_TEAM] == ego_team)[0]
    enemy_indices = np.where(aircraft[:, A_TEAM] != ego_team)[0]
    return ego_indices, enemy_indices


def build_event_flags(snapshots, ego_team):
    length = len(snapshots)
    event_index = {name: idx for idx, name in enumerate(EVENT_NAMES)}
    flags = np.zeros((length, len(EVENT_NAMES)), dtype=np.float32)

    prev_aircraft = None
    prev_missiles = None
    prev_ego_left_missiles = None
    prev_incoming_missile = False

    for t, snapshot in enumerate(snapshots):
        aircraft = snapshot["aircraft"]
        missiles = snapshot["missiles"]
        ego_indices, enemy_indices = _team_indices(aircraft, ego_team)

        incoming_missile = bool(
            ego_indices.size > 0 and np.any(aircraft[ego_indices, A_THREAT_MISSILE_EXIST] > 0.5)
        )
        if incoming_missile and not prev_incoming_missile:
            flags[t, event_index["incoming_missile"]] = 1.0

        if missiles.size > 0 and ego_indices.size > 0:
            parent = missiles[:, M_PARENT].astype(int)
            success = missiles[:, M_SUCCESS] > 0.5
            ego_parent = np.isin(parent, ego_indices)
            if np.any(ego_parent & success):
                if prev_missiles is None:
                    flags[t, event_index["hit_success"]] = 1.0
                else:
                    prev_success = np.zeros(len(missiles), dtype=bool)
                    n = min(len(missiles), len(prev_missiles))
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

        prev_aircraft = aircraft
        prev_missiles = missiles
        prev_ego_left_missiles = ego_left_missiles.copy()
        prev_incoming_missile = incoming_missile

    return flags


def annotate_episode_detail(item, thresholds, config: DetailAnnotationConfig):
    threat, attack = get_label_fields(item)
    delta_threat = episode_delta(threat)
    delta_attack = episode_delta(attack)

    event_flags = build_event_flags(item.get("snapshots", []), float(config.ego_team))
    if event_flags.shape[0] != threat.shape[0]:
        raise ValueError(f"event length {event_flags.shape[0]} != field length {threat.shape[0]}")
    event_point = np.any(event_flags > 0.5, axis=1)

    high_threat = threat >= float(thresholds["high_threat_threshold"])
    high_attack = attack >= float(thresholds["high_attack_threshold"])
    high_field = high_threat | high_attack
    high_change = (
        (delta_threat >= float(thresholds["delta_threat_threshold"]))
        | (delta_attack >= float(thresholds["delta_attack_threshold"]))
    )

    condition_multi_hot = np.stack(
        (
            event_point.astype(np.float32),
            high_field.astype(np.float32),
            high_change.astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)

    sample_category = np.full(threat.shape[0], CATEGORY_STABLE, dtype=np.int16)
    sample_category[high_field] = CATEGORY_HIGH_FIELD
    sample_category[high_change] = CATEGORY_HIGH_CHANGE
    sample_category[event_point] = CATEGORY_EVENT

    item["label_threat_fields"] = threat.reshape(-1, 1).astype(np.float32, copy=False)
    item["label_attack_fields"] = attack.reshape(-1, 1).astype(np.float32, copy=False)
    item["event_flags"] = event_flags.astype(np.float32, copy=False)
    item["condition_multi_hot"] = condition_multi_hot
    item["condition_names"] = np.asarray(CONDITION_NAMES, dtype=object)
    item["sample_category"] = sample_category
    item["sample_category_names"] = np.asarray(CATEGORY_NAMES, dtype=object)
    item["field_delta_features"] = np.stack((delta_threat, delta_attack), axis=-1).astype(np.float32, copy=False)
    return item


def annotate_split_items(split_items, thresholds, config: DetailAnnotationConfig):
    return [annotate_episode_detail(item, thresholds, config) for item in split_items]

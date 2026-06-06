from dataclasses import asdict, dataclass


LABEL_ACTION_CHANGE = 0
LABEL_HIGH_ATTACK = 1
LABEL_HIGH_THREAT = 2
LABEL_HIGH_CHANGE = 3
LABEL_EVENT = 4

SAMPLE_LABEL_NAMES = [
    "action_change",
    "high_attack",
    "high_threat",
    "high_change",
    "event",
]

EVENT_NAMES = [
    "incoming_missile",
    "outgoing_missile",
    "shot_fired",
    "hit_success",
    "ego_shotdown",
    "enemy_shotdown",
    "attack_zone_enter",
    "nez_enter",
    "close_range",
    "very_close_range",
]


@dataclass
class AnnotationConfig:
    field_k_step: int = 20
    ego_team: float = 0.0
    high_threat_floor: float = 0.20
    high_attack_floor: float = 0.15
    very_high_threat_floor: float = 0.35
    very_high_attack_floor: float = 0.30
    high_field_percentile: float = 75.0
    very_high_field_percentile: float = 90.0
    delta_floor: float = 0.03
    delta_percentile: float = 80.0
    future_delta_floor: float = 0.05
    future_delta_percentile: float = 80.0
    action_change_percentile: float = 80.0
    event_pre_steps: int = 20
    event_post_steps: int = 20
    r_attack: float = 14000.0
    r_nez: float = 10000.0
    theta_attack_deg: float = 60.0
    theta_nez_deg: float = 30.0
    priority_base: float = 1.0
    priority_high_threat_bonus: float = 1.5
    priority_high_attack_bonus: float = 1.5
    priority_high_change_bonus: float = 2.0
    priority_event_bonus: float = 3.0
    priority_action_change_bonus: float = 0.5
    priority_min: float = 1.0
    priority_max: float = 8.0
    weight_from_priority_scale: float = 0.5
    weight_min: float = 1.0
    weight_max: float = 4.0

    def to_dict(self):
        return asdict(self)


def multi_hot_summary(sample_multi_hot):
    summary = {}
    if sample_multi_hot.size == 0:
        for name in SAMPLE_LABEL_NAMES:
            summary[name] = 0
        summary["background"] = 0
        return summary

    for idx, name in enumerate(SAMPLE_LABEL_NAMES):
        summary[name] = int(sample_multi_hot[:, idx].sum())
    summary["background"] = int((sample_multi_hot.sum(axis=1) <= 0.5).sum())
    return summary

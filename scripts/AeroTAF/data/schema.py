from dataclasses import asdict, dataclass


BUCKET_BACKGROUND = 0
BUCKET_ACTION_CHANGE = 1
BUCKET_HIGH_ATTACK = 2
BUCKET_HIGH_THREAT = 3
BUCKET_HIGH_CHANGE = 4
BUCKET_EVENT = 5

BUCKET_NAMES = {
    BUCKET_BACKGROUND: "background",
    BUCKET_ACTION_CHANGE: "action_change",
    BUCKET_HIGH_ATTACK: "high_attack",
    BUCKET_HIGH_THREAT: "high_threat",
    BUCKET_HIGH_CHANGE: "high_change",
    BUCKET_EVENT: "event",
}

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


def bucket_summary(sample_bucket):
    summary = {}
    for bucket_id, name in BUCKET_NAMES.items():
        summary[name] = int((sample_bucket == bucket_id).sum())
    return summary

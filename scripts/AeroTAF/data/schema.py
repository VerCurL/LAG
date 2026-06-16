from dataclasses import asdict, dataclass


CATEGORY_EVENT = 0
CATEGORY_HIGH_FIELD = 1
CATEGORY_HIGH_CHANGE = 2
CATEGORY_STABLE = 3

CATEGORY_NAMES = [
    "event",
    "high_field",
    "high_change",
    "stable",
]

CONDITION_NAMES = [
    "event",
    "high_field",
    "high_change",
]

EVENT_NAMES = [
    "incoming_missile",
    "shot_fired",
    "hit_success",
    "ego_shotdown",
    "enemy_shotdown",
]

FIELD_DELTA_FEATURE_NAMES = [
    "delta_threat_target",
    "delta_attack_target",
]


@dataclass
class DetailAnnotationConfig:
    ego_team: float = 0.0
    high_threat_floor: float = 0.20
    high_attack_floor: float = 0.15
    high_field_percentile: float = 75.0
    delta_floor: float = 0.03
    delta_percentile: float = 80.0

    def to_dict(self):
        return asdict(self)

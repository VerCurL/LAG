# 第一阶段：先学会靠近敌机并平稳飞行
from .heading_reward import HeadingReward
from .event_driven_reward import EventDrivenReward
from .altitude_reward import AltitudeReward
from .distance_reward import DistanceReward


from .attack_window_reward import AttackWindowReward
from .energy_reward import EnergyReward
from .missile_avoid_reward import MissileAvoidReward
from .flight_quality_reward import FlightQualityReward
from .overall_situation_reward import OverallSituationReward
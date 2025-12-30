# 第一阶段：先学会靠近敌机并平稳飞行
from .heading_reward import HeadingReward
from .event_driven_reward import EventDrivenReward
from .altitude_reward import AltitudeReward
from .distance_reward import DistanceReward

# 第二阶段：寻找攻击窗口，维持能量和导弹躲避
from .event_driven_reward import EventDrivenReward
from .altitude_reward import AltitudeReward
from .distance_reward import DistanceReward
from .attack_window_reward import AttackWindowReward
from .energy_reward import EnergyReward
from .missile_avoid_reward import MissileAvoidReward

# 备用
from .flight_quality_reward import FlightQualityReward
from .overall_situation_reward import OverallSituationReward
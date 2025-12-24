"""
LAG平台最初奖励系统
"""
from .RewardOrigin.altitude_reward import AltitudeReward
from .RewardOrigin.event_driven_reward import EventDrivenReward
from .RewardOrigin.posture_reward import PostureReward
from .RewardOrigin.relative_altitude_reward import RelativeAltitudeReward
from .RewardOrigin.heading_reward import HeadingReward
from .RewardOrigin.missile_posture_reward import MissilePostureReward
from .RewardOrigin.shoot_penalty_reward import ShootPenaltyReward

"""
bxy-1v1奖励系统
"""
from .R1v1.bxy import AltitudeReward as BXY_AltitudeReward
from .R1v1.bxy import EventDrivenReward as BXY_EventDrivenReward
from .R1v1.bxy import PostureReward as BXY_PostureReward
from .R1v1.bxy import MissilePostureReward as BXY_MissilePostureReward
from .R1v1.bxy import ShootPenaltyReward as BXY_ShootPenaltyReward
from .R1v1.bxy import VelReward as BXY_VelReward

"""
lc-1v1奖励系统
"""
from .R1v1.lc import EventDrivenReward as LC_EventDrivenReward
from .R1v1.lc import EnergyReward as LC_EnergyReward
from .R1v1.lc import AttackWindowReward as LC_AttackWindowReward
from .R1v1.lc import DogdeAttackReward as LC_DogdeAttackReward

"""
fkr-4v4奖励系统
"""
from .R4v4.fkr import EventDrivenReward as FKR_4v4_EventDrivenReward
from .R4v4.fkr import AltitudeReward as FKR_4v4_AltitudeReward
from .R4v4.fkr import MissileAvoidReward as FKR_4v4_MissileAvoidReward
from .R4v4.fkr import DistanceReward as FKR_4v4_DistanceReward
from .R4v4.fkr import AttackWindowReward as FKR_4v4_AttackWindowReward
from .R4v4.fkr import EnergyReward as FKR_4v4_EnergyReward
from .R4v4.fkr import FlightQualityReward as FKR_4v4_FlightQualityReward
from .R4v4.fkr import OverallSituationReward as FKR_4v4_OverallSituationReward
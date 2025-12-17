"""
LAG平台最初tasks
"""
from .TaskOrigin.heading_task import HeadingTask
from .TaskOrigin.singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from .TaskOrigin.singlecombat_with_missle_task import SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask
from .TaskOrigin.multiplecombat_task import HierarchicalMultipleCombatShootTask, HierarchicalMultipleCombatTask, MultipleCombatTask

"""
lc-任务task
"""
from .Task1v1.lc import HierarchicalSingleCombatShootTask as LC_HierarchicalSingleCombatShootTask

"""
fkr-任务task
"""
from .Task4v4.fkr import HierarchicalMultipleCombatShootTask as FKR_4v4_HierarchicalMultipleCombatShootTask
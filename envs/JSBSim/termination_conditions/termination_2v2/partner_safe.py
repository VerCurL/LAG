import numpy as np

from ..termination_condition_base import BaseTerminationCondition
from ...utils.utils import get_AO_TA_R

class PartnerSafe(BaseTerminationCondition):
    '''
    队友安全的终止条件
    '''
    def __init__(self, config):
        BaseTerminationCondition.__init__(self, config)
        self.min_dist = getattr(self.config, f'TeamPostureReward_min_dist', 5)

    def get_termination(self, task, env, agent_id, info={}):
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        for partner in env.agents[agent_id].partners:
            partner_feature = np.hstack([partner.get_position(),
                                         partner.get_velocity()])
            _, _, R = get_AO_TA_R(ego_feature, partner_feature)

        done = R < self.min_dist
        success = False
        return done, success, info

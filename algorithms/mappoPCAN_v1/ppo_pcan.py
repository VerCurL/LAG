import torch
import torch.nn as nn

from ..utils.pcan_v1 import PCANBase
from ..utils.utils import check


class PPOPCAN(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOPCAN, self).__init__()
        # network config
        self.num_agents = args.num_agents
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.num_heads = args.num_heads
        self.KQ_hidden_size = args.KQ_hidden_size
        self.V_hidden_size = args.V_hidden_size
        self.PCANOut_hidden_size = args.PCANOut_hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        # pcan module
        self.pcan = PCANBase(obs_space, act_space, self.num_agents, self.num_heads, self.KQ_hidden_size, self.V_hidden_size,
                             self.PCANOut_hidden_size, self.activation_id, self.use_feature_normalization)

        self.to(device)

    def forward(self, obs, actions):
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)

        threat_output, attack_output = self.pcan(obs, actions)

        return threat_output, attack_output, self.pcan.record_info

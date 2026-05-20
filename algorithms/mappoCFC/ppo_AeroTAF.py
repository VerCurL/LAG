import torch
import torch.nn as nn

from ..utils.AeroTAF import AeroTAFBase
from ..utils.utils import check


class PPOAeroTAF(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOAeroTAF, self).__init__()
        # network config
        self.num_agents = args.num_agents
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.num_heads = args.num_heads
        self.KQ_hidden_size = args.KQ_hidden_size
        self.V_hidden_size = args.V_hidden_size
        self.AeroTAF_out_hidden_size = args.AeroTAF_out_hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        # pcan module
        self.AeroTAF = AeroTAFBase(obs_space, act_space, self.num_agents, self.num_heads, self.KQ_hidden_size, self.V_hidden_size,
                                   self.AeroTAF_out_hidden_size, self.activation_id, self.use_feature_normalization)

        self.to(device)

    def forward(self, obs, actions):
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)

        threat_output, attack_output = self.AeroTAF(obs, actions)

        return threat_output, attack_output, self.AeroTAF.record_info

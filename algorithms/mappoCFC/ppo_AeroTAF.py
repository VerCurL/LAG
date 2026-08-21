import torch
import torch.nn as nn

from ..utils.AeroTAF_ATNN_Fast import AeroTAFATTNFastBase
from ..utils.utils import check


class PPOAeroTAF(nn.Module):
    """Thin policy-side wrapper around the many-to-one ATTN Fast model."""

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__()
        self.num_agents = args.num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.AeroTAF = AeroTAFATTNFastBase(
            obs_space=obs_space,
            act_space=act_space,
            agent_num=args.num_agents,
            head_num=args.AeroTAF_spatial_head_num,
            time_head_num=args.AeroTAF_time_head_num,
            KQ_hidden_size=args.KQ_hidden_size,
            V_hidden_size=args.V_hidden_size,
            attn_output_hidden_size=args.AeroTAF_attn_output_hidden_size,
            field_output_hidden_size=args.AeroTAF_field_output_hidden_size,
            activation_id=args.activation_id,
            use_feature_normalization=args.use_feature_normalization,
        )
        self.to(device)

    def forward(self, obs, actions, seq_len, time_offset=0):
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        _, threat_output, attack_output = self.AeroTAF(
            obs,
            actions,
            seq_len=seq_len,
            time_offset=time_offset,
        )
        return threat_output, attack_output, self.AeroTAF.record_info

    def build_trajectory_cache(self, obs, actions):
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return self.AeroTAF.build_trajectory_cache(obs, actions)

    def predict_cached(
        self,
        cache,
        env_indices,
        time_indices,
        segment_starts,
        history_windows,
        action_variants,
    ):
        env_indices = check(env_indices).to(device=self.tpdv["device"], dtype=torch.long)
        time_indices = check(time_indices).to(device=self.tpdv["device"], dtype=torch.long)
        segment_starts = check(segment_starts).to(device=self.tpdv["device"], dtype=torch.long)
        action_variants = check(action_variants).to(**self.tpdv)
        return self.AeroTAF.predict_cached(
            cache,
            env_indices,
            time_indices,
            segment_starts,
            history_windows,
            action_variants,
        )

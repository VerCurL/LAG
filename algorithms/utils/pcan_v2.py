import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLPLayer
from .flatten import build_flattener

class PCANLayer(nn.Module):
    """
    Predictive Credit Assignment Network -- 预测式信用分配网络
    """
    def __init__(self, agent_num: int, head_num: int, KQ_input_dim: int, V_input_dim: int, activation_id,
                 KQ_hidden_size: str, V_hidden_size: str, output_hidden_size: str):
        super(PCANLayer, self).__init__()
        self.agent_num = agent_num
        self.head_num = head_num

        # -------- 各个模型的层次结构 --------
        self._KQ_hidden_size = [KQ_input_dim] + list(map(int, KQ_hidden_size.split(' ')))
        self._V_hidden_size = [V_input_dim] + list(map(int, V_hidden_size.split(' ')))
        self._output_hidden_size = [self._V_hidden_size[-1]] + list(map(int, output_hidden_size.split(' '))) + [1]

        # -------- K、Q、V模型和输出模型 --------
        self.K_module = MLPLayer(KQ_input_dim, KQ_hidden_size, activation_id)
        self.Q_module = MLPLayer(KQ_input_dim, KQ_hidden_size, activation_id)
        self.V_module = MLPLayer(V_input_dim, V_hidden_size, activation_id)
        self.weight_net = nn.Linear(self._V_hidden_size[-1], 1)
        self.output_module = MLPLayer(self._V_hidden_size[-1], output_hidden_size + " " + str(self._output_hidden_size[-1]), activation_id)

        # -------- 记录信息 --------
        self.record_info = {}

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        # n_rollout_threads/batch_size: B, agent_num: N, head_num: H
        batch_size = h.size(0)      # B * N
        size_B = batch_size // self.agent_num

        # size:(B * N, K/Q/V_dim) -> (B, N, K/Q/V_dim)
        KQ_dim, V_dim = self._KQ_hidden_size[-1], self._V_hidden_size[-1]
        K = self.K_module(h).reshape(size_B, self.agent_num, KQ_dim)
        Q = self.Q_module(h).reshape(size_B, self.agent_num, KQ_dim)
        V = self.V_module(s).reshape(size_B, self.agent_num, V_dim)

        # reshape成多头，size:(B, H, N, K/Q/V_head_dim)
        KQ_head_dim = KQ_dim // self.head_num
        V_head_dim = V_dim // self.head_num
        K = K.view(size_B, self.agent_num, self.head_num, KQ_head_dim).transpose(1, 2)
        Q = Q.view(size_B, self.agent_num, self.head_num, KQ_head_dim).transpose(1, 2)
        V = V.view(size_B, self.agent_num, self.head_num, V_head_dim).transpose(1, 2)

        # 缩放点积注意力，size:(B, H, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(KQ_head_dim)

        # softmax获得注意力权重，(B, H, N, N)
        attn_weights = F.softmax(scores, dim=-1)

        # ⭐获得奖励分配权重，(B, H, N, N) -> (B, N, N)
        credit = attn_weights.mean(dim=1)

        # 加权求和，(B, H, N, N) @ (B, H, N, VHD) -> (B, H, N, VHD)
        heads = torch.matmul(attn_weights, V)

        # 拼接heads，(B, H, N, VHD) -> (B, N, VD) -> (B * N, VD)
        ## (B, H, N, VHD) -> (B, N, VD)
        heads = heads.transpose(1, 2).contiguous().view(size_B, self.agent_num, V_dim)
        ## (B, N, 1)
        weights = torch.softmax(self.weight_net(heads), dim=1)
        ## 加权求和 (B, VD)
        global_feat = (weights * heads).mean(dim=1)

        # ⭐输出投影：(B, 1)
        output = self.output_module(global_feat)

        self.record_info = {
            "attn_weights": attn_weights.detach(),                      # (B, H, N, N)
            "credit": credit.detach(),                                  # (B, N, N)
            "pcan_output_norm": output.norm(dim=-1).mean().detach(),    # (B * N)
        }

        return output, credit

    @property
    def output_size(self):
        return self._output_hidden_size[-1]

    def get_info(self):
        return self.record_info

class PCANBase(nn.Module):
    def __init__(self, obs_space, agent_num: int, head_num: int, KQ_input_dim: int,
                 KQ_hidden_size: str, V_hidden_size: str, output_hidden_size: str,
                 activation_id, use_feature_normalization):
        super().__init__()
        self._use_feature_normalization = use_feature_normalization
        self.obs_flattener = build_flattener(obs_space)
        V_input_dim = self.obs_flattener.size
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(V_input_dim)

        self.PCAN = PCANLayer(
            agent_num=agent_num, head_num=head_num, KQ_input_dim=KQ_input_dim, V_input_dim=V_input_dim,
            activation_id=activation_id, KQ_hidden_size=KQ_hidden_size, V_hidden_size=V_hidden_size,
            output_hidden_size=output_hidden_size
        )

    def forward(self, h, s):
        if self._use_feature_normalization:
            s = self.feature_norm(s)
        output, credit = self.PCAN(h, s)
        return output, credit

    @property
    def output_size(self):
        return self.PCAN.output_size

    @property
    def record_info(self):
        return self.PCAN.get_info()
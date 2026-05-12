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
        # -------- 各个模型的层次结构 --------
        self._KQ_hidden_size = [KQ_input_dim] + list(map(int, KQ_hidden_size.split(' ')))
        self._V_hidden_size = [V_input_dim] + list(map(int, V_hidden_size.split(' ')))
        self._output_hidden_size = [self._V_hidden_size[-1]] + list(map(int, output_hidden_size.split(' '))) + [1]

        # -------- K、Q、V模型和输出模型 --------
        # 多头注意力机制模块
        self.head_num = head_num
        self.K_module = MLPLayer(KQ_input_dim, KQ_hidden_size, activation_id)
        self.Q_module = MLPLayer(KQ_input_dim, KQ_hidden_size, activation_id)
        self.V_module = MLPLayer(V_input_dim, V_hidden_size, activation_id)
        self.attn_output_module = nn.Linear(self._V_hidden_size[-1], V_input_dim)

        # 威胁场、攻击场预测模型
        self.threat_output_module = MLPLayer(V_input_dim, output_hidden_size + " " + str(self._output_hidden_size[-1]), activation_id)
        self.attack_output_module = MLPLayer(V_input_dim, output_hidden_size + " " + str(self._output_hidden_size[-1]), activation_id)

        # Transformer 用的 LayerNorm
        self.norm = nn.LayerNorm(V_input_dim)

        # -------- 记录信息 --------
        self.record_info = {}

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        # n_rollout_threads/batch_size: B, agent_num: N, head_num: H
        # (B * N, dim_s) + (B * N, dim_a) -cat-> (B * N, dim = dim_s + dim_a)
        x = torch.cat((s, a), dim=-1)
        batch_size, x_dim = x.shape
        size_B = batch_size // self.agent_num
        x = x.reshape(size_B, self.agent_num, x_dim)
        s = s.reshape(size_B, self.agent_num, s.shape[-1])

        # size:(B, N, x_dim) --> (B, N, KQ/V_dim)
        KQ_dim, V_dim = self._KQ_hidden_size[-1], self._V_hidden_size[-1]
        K = self.K_module(s)
        Q = self.Q_module(s)
        V = self.V_module(x)

        # reshape成多头，size:(B, N, H, KQ/V_head_dim) -T-> (B, H, N, KQ/V_head_dim)
        KQ_head_dim, V_head_dim = KQ_dim // self.head_num, V_dim // self.head_num
        K = K.view(size_B, self.agent_num, self.head_num, KQ_head_dim).transpose(1, 2)
        Q = Q.view(size_B, self.agent_num, self.head_num, KQ_head_dim).transpose(1, 2)
        V = V.view(size_B, self.agent_num, self.head_num, V_head_dim).transpose(1, 2)

        # 缩放点积注意力，size:(B, H, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(KQ_head_dim)

        # softmax获得注意力权重，(B, H, N, N)
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和，(B, H, N, N) @ (B, H, N, V_head_dim) -> (B, H, N, V_head_dim)
        heads = torch.matmul(attn_weights, V)

        # 拼接heads，(B, H, N, V_head_dim) -T-> (B, N, H, V_head_dim) -r-> (B, N, V_dim)
        heads = heads.transpose(1, 2).contiguous().view(size_B, self.agent_num, V_dim)

        # 计算注意力机制的输出，size: (B, N, V_dim) -> (B, N, dim)
        attn_output = self.attn_output_module(heads)

        # 残差连接 + LayerNorm，size: (B, N, dim)
        x = self.norm(x + attn_output)

        # ⭐输出投影：(B, N, 1) -mean-> (B, 1)
        threat_output = self.threat_output_module(x).mean(dim=1)
        attack_output = self.attack_output_module(x).mean(dim=1)

        return threat_output, attack_output

    @property
    def output_size(self):
        return self._output_hidden_size[-1]

    def get_info(self):
        return self.record_info

class PCANBase(nn.Module):
    def __init__(self, obs_space, act_space, agent_num: int, head_num: int, KQ_hidden_size: str, V_hidden_size: str,
                 output_hidden_size: str, activation_id, use_feature_normalization):
        super().__init__()
        self._use_feature_normalization = use_feature_normalization
        self.obs_flattener = build_flattener(obs_space)
        self.act_flattener = build_flattener(act_space)
        obs_input_dim = self.obs_flattener.size
        act_input_dim = self.act_flattener.size
        if self._use_feature_normalization:
            self.obs_feature_norm = nn.LayerNorm(obs_input_dim)
            self.act_feature_norm = nn.LayerNorm(act_input_dim)

        self.PCAN = PCANLayer(
            agent_num=agent_num, head_num=head_num, KQ_input_dim=obs_input_dim, V_input_dim=obs_input_dim + act_input_dim,
            activation_id=activation_id, KQ_hidden_size=KQ_hidden_size, V_hidden_size=V_hidden_size,
            output_hidden_size=output_hidden_size
        )

    def forward(self, s, a):
        if self._use_feature_normalization:
            s = self.obs_feature_norm(s)
            a = self.act_feature_norm(a)
        threat_output, attack_output = self.PCAN(s, a)
        return threat_output, attack_output

    @property
    def output_size(self):
        return self.PCAN.output_size

    @property
    def record_info(self):
        return self.PCAN.get_info()
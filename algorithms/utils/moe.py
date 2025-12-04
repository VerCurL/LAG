import torch
import torch.nn as nn
from .mlp import MLPLayer
from .flatten import build_flattener

class MoELayer(nn.Module):
    """
    通用专家 + top-k 专家混合 MoE

    参数：
    - num_general_experts：通用专家数量（每次都执行）
    - num_special_experts：专业专家数量（用于 gating top-k）
    - MoE_k：每次选择 top-k 个专业专家
    """
    def __init__(self, input_dim, hidden_size, activation_id, num_general_experts, num_special_experts, top_k):
        super(MoELayer, self).__init__()
        # 专家数量，选择专业专家数
        self.num_general_experts = num_general_experts
        self.num_special_experts = num_special_experts
        self.top_k = top_k
        self.total_experts = num_general_experts + num_special_experts

        # 专家层的层数，每层神经元个数，激活函数信息
        self._size = [input_dim] + list(map(int, hidden_size.split(' ')))
        expert_size = " ".join(map(str, [x // self.total_experts for x in self._size[1:]]))

        # 通用专家
        self.general_experts = nn.ModuleList([
            MLPLayer(input_dim, expert_size, activation_id)
            for _ in range(num_general_experts)
        ])

        # 专业专家
        self.special_experts = nn.ModuleList([
            MLPLayer(input_dim, expert_size, activation_id)
            for _ in range(num_special_experts)
        ])

        # gating
        self.gate = nn.Linear(input_dim, self.num_special_experts)

        # 记录信息
        self.record_info = {}

    def forward(self, x: torch.Tensor):
        # 记录输入数据的组数
        batch_size = x.size(0)

        # 通用专家（全部使用）
        general_outputs = []
        for expert in self.general_experts:
            general_outputs.append(expert(x).unsqueeze(1))

        general_outputs = torch.cat(general_outputs, dim=1).reshape(batch_size, -1)         # [batch_size, general_num * expert_output_dim]

        # 专业专家
        special_outputs = []
        for expert in self.special_experts:
            special_outputs.append(expert(x).unsqueeze(1))

        special_outputs = torch.cat(special_outputs, dim=1)         # [batch_size, special_num, expert_output_dim]

        # gating选择专业专家
        gate_logits = self.gate(x)                                  # [batch_size, special_num]
        gate_probs = torch.softmax(gate_logits, dim=1)              # [batch_size, special_num]

        top_k_vals, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)          # [batch_size, top_k]
        self.record_info = {
            "gate_probs": gate_probs,
            "top_k_idx": top_k_idx,
        }

        mask = torch.zeros_like(gate_probs)                         # 创建一个与gate_probs形状相同的全零张量
        mask.scatter_(1, top_k_idx, 1.0)                            # 将top_k_idx对应位置的值设为1.0

        masked_gate = gate_probs * mask                             # 将原始概论分布与掩码相乘，只保留top-k个专家的概论分布
        masked_gate = masked_gate / (masked_gate.sum(-1, keepdim=True) + 1e-8)          # 归一化，确保概论总和为1，添加1e-8防止除零错误

        # 加权求和专业专家输出
        weighted_special = torch.bmm(
            masked_gate.unsqueeze(1),               # [batch_size, 1, special_num]
            special_outputs                         # [batch_size, special_num, expert_dim]
        ).reshape(batch_size, -1)                   # [batch_size, special_num * expert_dim]

        # 将通用专家和专业专家输出拼接
        x = torch.cat([general_outputs, weighted_special], dim=-1)

        return x

    @property
    def output_size(self) -> int:
        return self._size[-1]

    def get_info(self):
        return self.record_info

class MoEBase(nn.Module):
    def __init__(self, obs_space, hidden_size, activation_id, use_feature_normalization,
                 num_general_experts, num_special_experts, top_k):
        super().__init__()
        self._hidden_size = hidden_size
        self._activation_id = activation_id
        self._use_feature_normalization = use_feature_normalization
        self._num_general_experts = num_general_experts
        self._num_special_experts = num_special_experts
        self._top_k = top_k

        self.obs_flattener = build_flattener(obs_space)
        input_dim = self.obs_flattener.size
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)
        self.MoE = MoELayer(
            input_dim=input_dim,  hidden_size=self._hidden_size, activation_id=self._activation_id,
            num_general_experts=self._num_general_experts, num_special_experts=self._num_special_experts, top_k=self._top_k
        )

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.MoE(x)
        return x

    @property
    def output_size(self):
        return self.MoE.output_size

    @property
    def record_info(self):
        return self.MoE.get_info()
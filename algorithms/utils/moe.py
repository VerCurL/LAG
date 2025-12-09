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
    def __init__(self, input_dim, expert_hidden_size, activation_id, num_general_experts, num_special_experts, top_k):
        super(MoELayer, self).__init__()
        # 专家数量，选择专业专家数
        self.num_general_experts = num_general_experts
        self.num_special_experts = num_special_experts
        self.top_k = top_k
        self.total_experts = num_general_experts + num_special_experts

        # # 专家层的层数，每层神经元个数，激活函数信息
        self._size = [input_dim] + list(map(int, expert_hidden_size.split(' ')))
        # self.expert_size = [x // self.total_experts for x in self._size[1:]]
        # expert_size = " ".join(map(str, self.expert_size))

        # 通用专家
        self.general_experts = nn.ModuleList([
            MLPLayer(input_dim, expert_hidden_size, activation_id)
            for _ in range(num_general_experts)
        ])

        # 专业专家
        self.special_experts = nn.ModuleList([
            MLPLayer(input_dim, expert_hidden_size, activation_id)
            for _ in range(num_special_experts)
        ])

        # gating
        self.gate = nn.Linear(input_dim, self.num_special_experts)

        # 记录信息
        self.record_info = {}

    def forward(self, x: torch.Tensor):
        # -------------------------------------------------
        # 0) 记录输入数据的组数
        # -------------------------------------------------
        batch_size = x.size(0)

        # -------------------------------------------------
        # 1) 通用专家并行执行（G个专家）
        # -------------------------------------------------
        # vectorized general experts forward
        general_outputs = []
        for expert in self.general_experts:
            general_outputs.append(expert(x).unsqueeze(1))

        # [batch_size, general_num * expert_output_dim]
        general_outputs = torch.cat(general_outputs, dim=1).reshape(batch_size, -1)

        # -------------------------------------------------
        # 2) gating → top-k 专家选择
        # -------------------------------------------------
        gate_logits = self.gate(x)                                  # [batch_size, special_num]
        gate_probs = torch.softmax(gate_logits, dim=1)              # [batch_size, special_num]
        top_k_vals, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)          # [batch_size, top_k]
        self.record_info = {
            "gate_probs": gate_probs,
            "top_k_idx": top_k_idx,
        }

        # -------------------------------------------------
        # 3) Sparse MoE：仅计算 top-k 专家（避免计算全部专家）
        # -------------------------------------------------
        # flatten B×k → one expert list
        unique_ids = torch.unique(top_k_idx)  # 去重后真正需要的专家数量
        # 专家一般重复率高：比如 top-k=2, S=16 → unique 数通常≈3–6个

        special_outputs = {}
        for idx in unique_ids.tolist():
            # 对每个真正选择到的专家执行一次 MLP
            out = self.special_experts[idx](x)  # [batch_size, H]
            special_outputs[idx] = out  # 缓存结果避免重复计算

        # -------------------------------------------------
        # 4) 根据 top-k 索引将专家结果组合成 [B, k*H]
        # -------------------------------------------------
        selected = []
        for i in range(self.top_k):
            expert_idx = top_k_idx[:, i]  # [batch_size]
            # vector gather：按 expert_idx 取出对应专家输出
            # 为避免 gather 的随机访问 → 用列表索引构造 batch 输出
            batch_out = torch.stack(
                [special_outputs[e.item()][b] for b, e in enumerate(expert_idx)],
                dim=0
            )  # [B, H]
            selected.append(batch_out)

        # 拼接 k 个专家输出 → [batch_size, top_k * expert_dim]
        special_out = torch.cat(selected, dim=-1)

        # -------------------------------------------------
        # 5) 拼接通用专家 + 专业专家输出 → [B, (G+k)*H]
        # -------------------------------------------------
        x = torch.cat([general_outputs, special_out], dim=-1)

        return x

    @property
    def output_size(self) -> int:
        return (self.num_general_experts + self.top_k) * self._size[-1]

    def get_info(self):
        return self.record_info

class MoEBase(nn.Module):
    def __init__(self, obs_space, expert_hidden_size, activation_id, use_feature_normalization,
                 num_general_experts, num_special_experts, top_k):
        super().__init__()
        self._expert_hidden_size = expert_hidden_size
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
            input_dim=input_dim,  expert_hidden_size=self._expert_hidden_size, activation_id=self._activation_id,
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
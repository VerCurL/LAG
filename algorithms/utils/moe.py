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
        # 展开 topk_idx → [B*k]
        flat_idx = top_k_idx.reshape(-1)
        # 对应的 batch id → [0,0,1,1,2,2,...]
        batch_ids = torch.arange(batch_size, device=x.device).repeat_interleave(self.top_k)

        # 对 flat_idx 排序，使得相同 expert_id 聚到一起
        sort_val, sort_idx = torch.sort(flat_idx)
        sorted_expert_ids = sort_val                    # [batch_size * top_k]
        sorted_batch_ids = batch_ids[sort_idx]          # 与 expert 对应的样本 id

        # 找出 unique expert 及其分组位置（完全 GPU）
        unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)

        # 拆分 sorted_batch_ids 为列表，每个列表对应同一个 expert 的 batch
        batch_splits = torch.split(sorted_batch_ids, counts.tolist())

        # -------------------------------------------------
        # 4) 对每个专业专家执行专业版 MLP（仅针对需要的 batch 子集）
        # -------------------------------------------------
        special_out_chunks = []
        for e_id, b_ids in zip(unique_experts.tolist(), batch_splits):
            selected_x = x[b_ids]  # [num_b, D]
            expert_out = self.special_experts[e_id](selected_x)     # [num_b, H]
            special_out_chunks.append((e_id, b_ids, expert_out))

        # ============================================================
        # 5) 将所有专家输出组装回 full batch 的 [B, k*H]
        # ============================================================
        # 用 zeros 初始化，然后把每个专家输出写入
        # [batch_size, top_k, output_size]
        special_out = torch.zeros(batch_size, self.top_k, self._size[-1], device=x.device)

        # 一个快速的专家 → topk位置映射表（避免重复计算）
        # positions[i, x] = 专家 i 在 batch x 中对应的 top_k 位置
        positions = torch.full((self.num_special_experts, batch_size), -1, device=x.device, dtype=torch.long)
        b_ids_all = torch.arange(batch_size, device=x.device)
        for j in range(self.top_k):
            e_ids = top_k_idx[:, j]             # [batch_size]，top_k为j的expert_id
            positions[e_ids, b_ids_all] = j     # 写入专家对应位置，[num_special_expert, batch_size, 1]
        # 根据专家输出填回 special_out
        for e_id, b_ids, expert_out in special_out_chunks:
            # 该专家 e_id 对这些 batch 的 top-k 位置
            pos = positions[e_id, b_ids]        # [num_b]，每个 batch 的 topk 位置
            # 直接写入
            special_out[b_ids, pos] = expert_out
        # reshape → [batch_size, k * output_dim]
        special_out = special_out.reshape(batch_size, -1)

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
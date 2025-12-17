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
    def __init__(self, input_dim, expert_hidden_size, activation_id,
                 num_general_experts, num_special_experts, top_k, bias_update_gamma=1e-3):
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

        # bias
        ## 初始化为0
        self.register_buffer("bias", torch.zeros(self.num_special_experts, dtype=torch.float32))
        self.bias_update_gamma = bias_update_gamma

        ## 本batch的选中次数（sample）
        self.register_buffer("load_counter", torch.zeros(self.num_special_experts, dtype=torch.float32))

        # 记录信息
        self.record_info = {}

    def update_bias(self):
        """
        根据 load_counter 调整 bias：
        - 使用次数高 → bias -= gamma * usage
        - 使用次数低 → bias += gamma * (avg - usage)

        实际等价于朝着"均匀使用"方向移动 bias
        """
        if self.load_counter.sum() == 0:
            return  # 第一次可能还没数据

        # 归一化 usage，使其总和=1
        usage = self.load_counter / (self.load_counter.sum() + 1e-6)

        # 理想均匀使用（每个专家相同）
        ideal = 1.0 / self.num_special_experts

        # Δbias = gamma * (ideal - usage)
        delta_bias = self.bias_update_gamma * (ideal - usage)

        self.bias += delta_bias

        # 重置 load_counter
        self.load_counter.zero_()

    def forward(self, x: torch.Tensor):
        # -------- 记录输入数据的组数 --------
        batch_size = x.size(0)

        # -------- 通用专家（全部执行） --------
        general_outputs = torch.cat(
            [expert(x).unsqueeze(1) for expert in self.general_experts], dim=1
        ).reshape(batch_size, -1)           # [batch_size, general_num * expert_output_dim]

        # -------- gating logits + bias --------
        gate_logits = self.gate(x) + self.bias                                  # [batch_size, special_num]
        gate_probs = torch.softmax(gate_logits, dim=1)                          # [batch_size, special_num]
        self.record_info["gate_probs"] = gate_probs.detach()

        # -------- top-k 选择 --------
        top_k_vals, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)      # [batch_size, top_k]
        self.record_info["top_k_idx"] = top_k_idx.detach()

        # 统计本 batch 中专家选中次数
        with torch.no_grad():
            # 对每个 token 选的 top-k 做计数
            flat_idx = top_k_idx.reshape(-1)
            with torch.no_grad():
                # flat_idx shape: [batch_size * top_k]
                counts = torch.bincount(flat_idx, minlength=self.num_special_experts)
                self.load_counter += counts.to(self.load_counter.dtype)

        # -------- Sparse MoE：仅计算 top-k 专家（避免计算全部专家） --------
        # 对应的 batch id → [0,0,1,1,2,2,...]
        batch_ids = torch.arange(batch_size, device=x.device).repeat_interleave(self.top_k)

        # 对 flat_idx 排序，使得相同 expert_id 聚到一起
        sort_val, sort_idx = torch.sort(flat_idx)
        sorted_expert_ids = sort_val  # [batch_size * top_k]
        sorted_batch_ids = batch_ids[sort_idx]  # 与 expert 对应的样本 id

        # 找出 unique expert 及其分组位置（完全 GPU）
        unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)

        # 拆分 sorted_batch_ids 为列表，每个列表对应同一个 expert 的 batch
        batch_splits = torch.split(sorted_batch_ids, counts.tolist())

        # -------- 对每个专业专家执行专业版 MLP（仅针对需要的 batch 子集） --------
        special_out_chunks = []
        for e_id, b_ids in zip(unique_experts.tolist(), batch_splits):
            selected_x = x[b_ids]  # [num_b, D]
            expert_out = self.special_experts[e_id](selected_x)  # [num_b, H]
            special_out_chunks.append((e_id, b_ids, expert_out))

        # -------- 将所有专家输出组装回 full batch 的 [B, k*H] --------
        # 用 zeros 初始化，然后把每个专家输出写入
        # [batch_size, top_k, output_size]
        special_out = torch.zeros(batch_size, self.top_k, self._size[-1], device=x.device)

        # 一个快速的专家 → topk位置映射表（避免重复计算）
        # positions[i, x] = 专家 i 在 batch x 中对应的 top_k 位置
        positions = torch.full((self.num_special_experts, batch_size), -1, device=x.device, dtype=torch.long)
        b_ids_all = torch.arange(batch_size, device=x.device)
        for j in range(self.top_k):
            e_ids = top_k_idx[:, j]  # [batch_size]，top_k为j的expert_id
            positions[e_ids, b_ids_all] = j  # 写入专家对应位置，[num_special_expert, batch_size, 1]
        # 根据专家输出填回 special_out
        for e_id, b_ids, expert_out in special_out_chunks:
            # 该专家 e_id 对这些 batch 的 top-k 位置
            pos = positions[e_id, b_ids]  # [num_b]，每个 batch 的 topk 位置
            # 直接写入
            special_out[b_ids, pos] = expert_out
        # reshape → [batch_size, k * output_dim]
        special_out = special_out.reshape(batch_size, -1)

        # -------- 拼接最终输出 --------
        x = torch.cat([general_outputs, special_out], dim=-1)

        # -------- 更新 bias（每个 forward 之后执行一次）--------
        self.update_bias()

        return x

    @property
    def output_size(self) -> int:
        return (self.num_general_experts + self.top_k) * self._size[-1]

    def get_info(self):
        return self.record_info

class MoEBase(nn.Module):
    def __init__(self, obs_space, expert_hidden_size, activation_id, use_feature_normalization,
                 num_general_experts, num_special_experts, top_k, bias_update_gamma=0.05):
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
            num_general_experts=self._num_general_experts, num_special_experts=self._num_special_experts, top_k=self._top_k,
            bias_update_gamma=bias_update_gamma
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
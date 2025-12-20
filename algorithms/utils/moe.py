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

        # -------- 通用专家 --------
        self.general_experts = nn.ModuleList([
            MLPLayer(input_dim, expert_hidden_size, activation_id)
            for _ in range(num_general_experts)
        ])

        # -------- 专业专家 --------
        self.special_experts = nn.ModuleList([
            MLPLayer(input_dim, expert_hidden_size, activation_id)
            for _ in range(num_special_experts)
        ])

        # -------- gating --------
        self.gate = nn.Linear(input_dim, self.num_special_experts)

        # -------- bias（用于负载均衡）--------
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
        batch_size = x.size(0)
        output_dim = self._size[-1]

        # -------- 用于记录所有专家输出（调试 / 可视化）--------
        experts_out = torch.zeros(batch_size, self.total_experts, output_dim, device=x.device)

        # ======================================================
        # 1. 通用专家：全部执行 + 等权平均
        # ======================================================
        general_outputs = []
        for i, expert in enumerate(self.general_experts):
            out = expert(x)                                     # [batch_size, output_dim]
            general_outputs.append(out)
            experts_out[:, i, :] = out                          # 填充到 experts_out
        general_outputs = torch.stack(general_outputs, dim=1)   # [batch_size, general_num, output_dim]
        general_out = general_outputs.mean(dim=1)               # [batch_size, output_dim]

        # ======================================================
        # 2. gating + top-k
        # ======================================================
        gate_logits = self.gate(x) + self.bias                                  # [batch_size, special_num]
        gate_probs = torch.softmax(gate_logits, dim=1)                          # [batch_size, special_num]
        self.record_info["gate_probs"] = gate_probs.detach()

        # -------- top-k 选择 --------
        top_k_vals, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)      # [batch_size, top_k]
        self.record_info["top_k_idx"] = top_k_idx.detach()

        # -------- 统计本 batch 中专家选中次数 --------
        with torch.no_grad():
            # 对每个 token 选的 top-k 做计数
            flat_idx = top_k_idx.reshape(-1)
            with torch.no_grad():
                # flat_idx shape: [batch_size * top_k]
                counts = torch.bincount(flat_idx, minlength=self.num_special_experts)
                self.load_counter += counts.to(self.load_counter.dtype)

        # ======================================================
        # 3. 专业专家：全部执行，但只用 top-k
        # ======================================================
        all_special_out = []
        for e_id, expert in enumerate(self.special_experts):
            out = expert(x)                                             # [batch_size, output_dim]
            experts_out[:, self.num_general_experts + e_id, :] = out    # 填充 experts_out
            all_special_out.append(out)
        all_special_out = torch.stack(all_special_out, dim=1)           # [batch_size, num_special_experts, output_dim]

        # -------- 取出 top-k 专家输出 --------
        special_out = torch.zeros(batch_size, self.top_k, output_dim, device=x.device)
        for j in range(self.top_k):
            e_ids = top_k_idx[:, j]                                     # [batch_size]
            special_out[torch.arange(batch_size), j, :] = all_special_out[torch.arange(batch_size), e_ids, :]

        # ======================================================
        # 4. 专业专家：softmax 加权求和
        # ======================================================
        special_weights = torch.softmax(top_k_vals, dim=-1)             # [batch_size, top_k]
        special_out = torch.sum(special_out * special_weights.unsqueeze(-1), dim=1)         # [batch_size, output_dim]

        # ======================================================
        # 5. 最终输出
        # ======================================================
        output = general_out + special_out

        # -------- 更新 bias（每个 forward 之后执行一次）--------
        self.update_bias()

        return output, experts_out

    @property
    def output_size(self) -> int:
        return self._size[-1]

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
        output, experts_out = self.MoE(x)
        return output, experts_out

    @property
    def output_size(self):
        return self.MoE.output_size

    @property
    def record_info(self):
        return self.MoE.get_info()
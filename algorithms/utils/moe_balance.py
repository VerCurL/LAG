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
        # 记录输入数据的组数
        batch_size = x.size(0)

        # -------- 通用专家（全部执行） --------
        general_outputs = torch.cat(
            [expert(x).unsqueeze(1) for expert in self.general_experts], dim=1
        ).reshape(batch_size, -1)           # [batch_size, general_num * expert_output_dim]

        # -------- 专业专家（预先运行所有专家）--------
        special_outputs = torch.cat(
            [expert(x).unsqueeze(1) for expert in self.special_experts], dim=1
        )                                   # [batch_size, special_num, expert_output_dim]

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

        # -------- gather 专业专家输出 --------
        selected_special = special_outputs.gather(                              # [batch_size, top_k, expert_dim]
            1, top_k_idx.unsqueeze(-1).expand(-1, -1, special_outputs.shape[-1])
        )
        # 加权求和专业专家输出
        weighted_special = selected_special.reshape(batch_size, -1)                 # [batch_size, top_k * expert_dim]

        # -------- 拼接最终输出 --------
        x = torch.cat([general_outputs, weighted_special], dim=-1)

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
"""用于航迹跟踪的执行层 GRU 网络模板。"""

import torch
from torch import nn


class ExecutionRouteNetwork(nn.Module):
    """分别编码航迹和飞机状态，再输出三组离散动作分布。"""
    def __init__(self, route_points: int = 20, state_dim: int = 16, hidden_dim: int = 128) -> None:
        """创建航迹编码器、状态编码器、GRU 和动作头。"""
        super().__init__()
        self.route_encoder = nn.Sequential(nn.Linear(route_points * 4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in (3, 5, 3)])

    def forward(self, route: torch.Tensor, state: torch.Tensor, hidden=None):
        """融合航迹和状态特征，并返回动作 logits 与新的隐藏状态。"""
        route_features = self.route_encoder(route.flatten(start_dim=-2))
        state_features = self.state_encoder(state)
        features = torch.cat((route_features, state_features), dim=-1).unsqueeze(1)
        output, hidden = self.gru(features, hidden)
        last = output[:, -1]
        return tuple(head(last) for head in self.action_heads), hidden

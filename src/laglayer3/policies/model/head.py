"""带分层 MoE 分支的团队首脑网络模板。"""

from typing import Dict, Tuple

import torch
from torch import nn


class SparseMoE(nn.Module):
    """使用可配置 Top-K 路由组合多个专家输出。"""
    def __init__(self, input_dim: int, hidden_dim: int, experts: int, top_k: int) -> None:
        """创建专家网络、路由器并校验 Top-K 配置。"""
        super().__init__()
        if experts <= 0 or top_k <= 0 or top_k > experts:
            raise ValueError("experts and top_k must satisfy 1 <= top_k <= experts")
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(experts)]
        )
        self.router = nn.Linear(input_dim, experts)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算路由权重、选择专家并返回混合特征和专家索引。"""
        logits = self.router(x)
        values, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        selected = outputs.gather(-2, indices.unsqueeze(-1).expand(*indices.shape, outputs.shape[-1]))
        result = (selected * weights.unsqueeze(-1)).sum(-2)
        return result, indices


class TeamHeadNetwork(nn.Module):
    """把团队实体编码为航迹和武器预测的首脑网络。"""

    def __init__(
        self,
        entity_dim: int,
        model_dim: int = 128,
        route_points: int = 20,
        general_experts: int = 2,
        route_experts: int = 8,
        weapon_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        """创建实体编码器、团队 Transformer、分支 MoE 和输出头。"""
        super().__init__()
        self.route_points = int(route_points)
        self.entity_encoder = nn.Sequential(nn.Linear(entity_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim))
        self.team_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, nhead=4, batch_first=True), num_layers=2
        )
        self.general = SparseMoE(model_dim, model_dim, general_experts, min(top_k, general_experts))
        self.route = SparseMoE(model_dim, model_dim, route_experts, min(top_k, route_experts))
        self.weapon = SparseMoE(model_dim, model_dim, weapon_experts, min(top_k, weapon_experts))
        self.route_head = nn.Linear(model_dim, route_points * 4)
        self.route_confidence = nn.Linear(model_dim, 1)
        self.weapon_request = nn.Linear(model_dim, 2)
        self.weapon_type = nn.Linear(model_dim, 3)
        self.target = nn.Linear(model_dim, 8)

    def forward(self, entities: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """编码团队实体，分别计算航迹、武器和专家路由结果。"""
        x = self.entity_encoder(entities)
        if mask is not None:
            x = self.team_encoder(x, src_key_padding_mask=~mask.bool())
        else:
            x = self.team_encoder(x)
        shared, general_ids = self.general(x)
        route_features, route_ids = self.route(shared)
        weapon_features, weapon_ids = self.weapon(shared)
        return {
            "route": self.route_head(route_features).view(*route_features.shape[:-1], self.route_points, 4),
            "route_confidence": torch.sigmoid(self.route_confidence(route_features)),
            "weapon_request_logits": self.weapon_request(weapon_features),
            "weapon_type_logits": self.weapon_type(weapon_features),
            "target_logits": self.target(weapon_features),
            "general_expert_ids": general_ids,
            "route_expert_ids": route_ids,
            "weapon_expert_ids": weapon_ids,
        }

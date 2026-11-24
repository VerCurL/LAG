"""用于管理冻结节点和可训练节点的分层训练图。"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


@dataclass
class TrainingNode:
    """记录一个 Provider 的训练状态、梯度边界和附加元数据。"""
    name: str
    provider: Any
    trainable: bool = False
    gradient_boundary: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_trainable(self, trainable: bool) -> None:
        """设置节点是否参与本轮梯度更新，并同步底层模型模式。"""
        self.trainable = bool(trainable)
        if self.trainable and hasattr(self.provider, "train"):
            self.provider.train()
        elif hasattr(self.provider, "eval"):
            self.provider.eval()
        for parameter in getattr(self.provider, "parameters", lambda: [])():
            parameter.requires_grad_(self.trainable)


class TrainingGraph:
    """协调多个分层 Provider 的冻结、训练和版本记录。"""
    def __init__(self, nodes: Iterable[TrainingNode] = ()) -> None:
        """注册初始训练节点并建立按名称索引。"""
        self.nodes = {node.name: node for node in nodes}

    def add(self, node: TrainingNode) -> None:
        """添加一个节点，并拒绝重复的节点名称。"""
        if node.name in self.nodes:
            raise ValueError("duplicate training node: {}".format(node.name))
        self.nodes[node.name] = node

    def configure(self, trainable: Iterable[str]) -> None:
        """根据节点名称设置本轮参与训练的 Provider。"""
        trainable_set = set(trainable)
        unknown = trainable_set.difference(self.nodes)
        if unknown:
            raise KeyError("unknown training nodes: {}".format(sorted(unknown)))
        for name, node in self.nodes.items():
            node.set_trainable(name in trainable_set)

    def manifest(self) -> Dict[str, Dict[str, Any]]:
        """生成记录训练状态、梯度边界和节点元数据的清单。"""
        return {
            name: {
                "trainable": node.trainable,
                "gradient_boundary": node.gradient_boundary,
                **node.metadata,
            }
            for name, node in self.nodes.items()
        }

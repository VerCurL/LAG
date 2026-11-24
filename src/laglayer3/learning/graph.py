"""Small configurable TrainingGraph for frozen/trainable Provider nodes."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class TrainingNode:
    """中文说明：定义本模块负责的 TrainingNode 数据或行为。"""
    name: str
    provider: Any
    trainable: bool = False
    gradient_boundary: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_trainable(self, trainable: bool) -> None:
        """中文说明：设置训练节点是否参与梯度更新。"""
        self.trainable = bool(trainable)
        if self.trainable and hasattr(self.provider, "train"):
            self.provider.train()
        elif hasattr(self.provider, "eval"):
            self.provider.eval()
        for parameter in getattr(self.provider, "parameters", lambda: [])():
            parameter.requires_grad_(self.trainable)


class TrainingGraph:
    """中文说明：定义本模块负责的 TrainingGraph 数据或行为。"""
    def __init__(self, nodes: Iterable[TrainingNode] = ()) -> None:
        """中文说明：初始化对象配置、依赖和运行状态。"""
        self.nodes = {node.name: node for node in nodes}

    def add(self, node: TrainingNode) -> None:
        """中文说明：向训练图注册一个新的训练节点。"""
        if node.name in self.nodes:
            raise ValueError("duplicate training node: {}".format(node.name))
        self.nodes[node.name] = node

    def configure(self, trainable: Iterable[str]) -> None:
        """中文说明：根据节点名称配置本轮训练对象。"""
        trainable_set = set(trainable)
        unknown = trainable_set.difference(self.nodes)
        if unknown:
            raise KeyError("unknown training nodes: {}".format(sorted(unknown)))
        for name, node in self.nodes.items():
            node.set_trainable(name in trainable_set)

    def manifest(self) -> Dict[str, Dict[str, Any]]:
        """中文说明：生成当前训练图的冻结和版本清单。"""
        return {
            name: {
                "trainable": node.trainable,
                "gradient_boundary": node.gradient_boundary,
                **node.metadata,
            }
            for name, node in self.nodes.items()
        }

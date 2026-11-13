"""Small configurable TrainingGraph for frozen/trainable Provider nodes."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class TrainingNode:
    name: str
    provider: Any
    trainable: bool = False
    gradient_boundary: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_trainable(self, trainable: bool) -> None:
        self.trainable = bool(trainable)
        if self.trainable and hasattr(self.provider, "train"):
            self.provider.train()
        elif hasattr(self.provider, "eval"):
            self.provider.eval()
        for parameter in getattr(self.provider, "parameters", lambda: [])():
            parameter.requires_grad_(self.trainable)


class TrainingGraph:
    def __init__(self, nodes: Iterable[TrainingNode] = ()) -> None:
        self.nodes = {node.name: node for node in nodes}

    def add(self, node: TrainingNode) -> None:
        if node.name in self.nodes:
            raise ValueError("duplicate training node: {}".format(node.name))
        self.nodes[node.name] = node

    def configure(self, trainable: Iterable[str]) -> None:
        trainable_set = set(trainable)
        unknown = trainable_set.difference(self.nodes)
        if unknown:
            raise KeyError("unknown training nodes: {}".format(sorted(unknown)))
        for name, node in self.nodes.items():
            node.set_trainable(name in trainable_set)

    def manifest(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "trainable": node.trainable,
                "gradient_boundary": node.gradient_boundary,
                **node.metadata,
            }
            for name, node in self.nodes.items()
        }

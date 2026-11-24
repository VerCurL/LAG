"""用于冻结规则或模型检查点的 Provider 包装器。"""

from typing import Any


class FixedProvider:
    """把任意策略对象包装成固定推理 Provider。"""
    def __init__(self, provider: Any) -> None:
        """保存被包装对象，并标记当前 Provider 不参与训练。"""
        self.provider = provider
        self.trainable = False

    def train(self, mode: bool = True) -> Any:
        """保持底层对象处于推理模式，兼容训练图的统一调用。"""
        if hasattr(self.provider, "eval"):
            self.provider.eval()
        return self.provider

    def eval(self) -> Any:
        """将底层对象切换到推理模式。"""
        if hasattr(self.provider, "eval"):
            self.provider.eval()
        return self.provider

    def reset(self) -> None:
        """转发底层策略的状态重置。"""
        reset = getattr(self.provider, "reset", None)
        if reset is not None:
            reset()

    def __getattr__(self, name: str) -> Any:
        """把未显式声明的属性访问转发给底层策略。"""
        return getattr(self.provider, name)

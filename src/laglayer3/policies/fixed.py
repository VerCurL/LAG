"""Fixed Provider wrapper used for frozen rules and model checkpoints."""

from typing import Any, Callable, Mapping, Optional


class FixedProvider:
    """中文说明：定义本模块负责的 FixedProvider 数据或行为。"""
    def __init__(self, provider: Any) -> None:
        """中文说明：初始化对象配置、依赖和运行状态。"""
        self.provider = provider
        self.trainable = False

    def train(self, mode: bool = True) -> Any:
        """中文说明：把 Provider 切换到训练或固定推理模式。"""
        if hasattr(self.provider, "eval"):
            self.provider.eval()
        return self.provider

    def eval(self) -> Any:
        """中文说明：把 Provider 切换到固定推理模式。"""
        if hasattr(self.provider, "eval"):
            self.provider.eval()
        return self.provider

    def reset(self) -> None:
        """中文说明：清理本模块的运行状态和循环记忆。"""
        reset = getattr(self.provider, "reset", None)
        if reset is not None:
            reset()

    def __getattr__(self, name: str) -> Any:
        """中文说明：执行 getattr__ 所需的内部辅助逻辑。"""
        return getattr(self.provider, name)

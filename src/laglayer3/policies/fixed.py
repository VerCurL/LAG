"""Fixed Provider wrapper used for frozen rules and model checkpoints."""

from typing import Any, Callable, Mapping, Optional


class FixedProvider:
    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.trainable = False

    def train(self, mode: bool = True) -> Any:
        if hasattr(self.provider, "eval"):
            self.provider.eval()
        return self.provider

    def eval(self) -> Any:
        if hasattr(self.provider, "eval"):
            self.provider.eval()
        return self.provider

    def reset(self) -> None:
        reset = getattr(self.provider, "reset", None)
        if reset is not None:
            reset()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

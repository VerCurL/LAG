"""Independent 5 Hz/EAS contracts; importing this package needs no NumPy."""

from .contracts import EASState, EASReference, EASInnerReference, EASTrim

__all__ = ["EASState", "EASReference", "EASInnerReference", "EASTrim"]

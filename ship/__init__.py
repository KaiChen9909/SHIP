"""SHIP: Structured and Hierarchical Insight with Privacy."""

from .config import Attribute, SHIPConfig
from .pipeline import SHIP, SHIPResult

__all__ = ["SHIP", "SHIPConfig", "SHIPResult", "Attribute"]

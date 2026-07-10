"""AI Crypto Trading System.

Autonomous cryptocurrency trading: classical strategies + risk management,
with pattern-discovery and sentiment as supporting signals.

Subpackages are imported lazily by consumers — nothing is imported eagerly
here so that lightweight tools (dashboard, scripts) don't pay for heavy
dependencies they don't use.
"""

__version__ = '2.0.0'

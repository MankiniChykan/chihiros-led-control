# custom_components/chihiros/chihiros_led_control/const.py
"""Chihiros LED-control constants.

This module mirrors the top-level integration UUIDs so there is a single
source of truth. We normalize to UPPERCASE here to preserve any existing
comparisons/expectations in the LED stack.
"""

from __future__ import annotations

# Prefer the integration-wide UUIDs; fall back to literals if needed
try:
    from ...const import UART_SERVICE_UUID as _UART_SERVICE_UUID
    from ...const import UART_RX_CHAR_UUID as _UART_RX_CHAR_UUID
    from ...const import UART_TX_CHAR_UUID as _UART_TX_CHAR_UUID
except Exception:
    # Fallback (should not be used in Home Assistant; kept for safety/tests)
    _UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
    _UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
    _UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

# Normalize to uppercase to match previous behavior in this subpackage
UART_SERVICE_UUID: str = _UART_SERVICE_UUID.upper()
UART_RX_CHAR_UUID: str = _UART_RX_CHAR_UUID.upper()
UART_TX_CHAR_UUID: str = _UART_TX_CHAR_UUID.upper()

__all__ = ["UART_SERVICE_UUID", "UART_RX_CHAR_UUID", "UART_TX_CHAR_UUID"]

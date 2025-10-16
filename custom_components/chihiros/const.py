# custom_components/chihiros/const.py
"""Constants for the Chihiros integration."""

from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────────
# Core identifiers
# ────────────────────────────────────────────────────────────────────────────────
DOMAIN: str = "chihiros"
MANUFACTURER: str = "Chihiros"

# ────────────────────────────────────────────────────────────────────────────────
# Common dispatcher signal templates
# ────────────────────────────────────────────────────────────────────────────────
# Fired after a manual dose completes, to request sensor refresh
SIGNAL_REFRESH_TOTALS_ENTRY = f"{DOMAIN}_{{entry_id}}_refresh_totals"
# Fired after a dose (or totals probe) for a specific address
SIGNAL_REFRESH_TOTALS_ADDR = f"{DOMAIN}_refresh_totals_{{address}}"
# Fired by the persistent BLE listener when a 0x5B totals frame is decoded
SIGNAL_PUSH_TOTALS = f"{DOMAIN}_push_totals_{{address}}"

# ────────────────────────────────────────────────────────────────────────────────
# BLE protocol details (shared between LED + DOSER)
# ────────────────────────────────────────────────────────────────────────────────
# UUIDs for the UART RX/TX characteristics used across Chihiros devices.
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # write
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify

# ────────────────────────────────────────────────────────────────────────────────
# Default settings / platform grouping
# ────────────────────────────────────────────────────────────────────────────────
LED_PLATFORMS = ["light", "switch", "sensor"]
DOSER_PLATFORMS = ["button", "number", "sensor"]

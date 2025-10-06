"""Doser device model for discovery table."""

from __future__ import annotations
from ...chihiros_led_control.main.base_device import BaseDevice


class Doser(BaseDevice):
    """Chihiros 4-channel dosing pump."""

    _model_name = "Doser"
    # Seen in your scans: e.g. "DYDOSED203E0FEFCBC"
    _model_codes = ["DYDOSED", "DYDOSE", "DOSER"]
    _colors: dict[str, int] = {}

# custom_components/chihiros/chihiros_doser_control/device/doser.py
"""Doser device model for discovery table.

This class enables the doser to participate in the shared model registry
used by `get_model_class_from_name`. It inherits the BLE/session machinery
from the LED `BaseDevice` but exposes no color channels.
"""

from __future__ import annotations

from ...chihiros_led_control.device.base_device import BaseDevice


class Doser(BaseDevice):
    """Chihiros 4-channel dosing pump."""

    _model_name = "Doser"
    # Seen in scans: e.g. "DYDOSED203E0FEFCBC"
    _model_codes = ["DYDOSED", "DYDOSE", "DOSER"]
    # No light channels for a doser
    _colors: dict[str, int] = {}

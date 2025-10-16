# custom_components/chihiros/chihiros_led_control/device/__init__.py
"""Model registry and helpers for Chihiros devices.

- Keeps imports light at module import time (no bleak / HA deps).
- Builds a CODE -> Class registry lazily, including optional doser model if present.
- Provides helpers to pick a device class from an advertised BLE name, or to
  instantiate from a MAC address when running outside HA.
"""
from __future__ import annotations

import inspect
from typing import Dict, Type

from .base_device import BaseDevice  # base class only (no bleak import here)

# Lazy-built registry of MODEL_CODE -> class
_CODE2MODEL: Dict[str, Type[BaseDevice]] | None = None


def _safe_import(path: str):
    """Import a module safely, returning the module or None on ImportError."""
    try:
        return __import__(path, fromlist=["*"])
    except ImportError:
        return None


def _ensure_registry() -> Dict[str, Type[BaseDevice]]:
    """Build the CODE2MODEL registry lazily by importing model modules only once."""
    global _CODE2MODEL
    if _CODE2MODEL is not None:
        return _CODE2MODEL

    # Current module path:
    #   custom_components.chihiros.chihiros_led_control.device
    root = __name__.split(".chihiros_led_control.device")[0]

    modules_to_try = [
        # LED families
        __name__ + ".a2",
        __name__ + ".c2",
        __name__ + ".c2rgb",
        __name__ + ".commander1",
        __name__ + ".commander4",
        __name__ + ".fallback",
        __name__ + ".generic_rgb",
        __name__ + ".generic_white",
        __name__ + ".generic_wrgb",
        __name__ + ".tiny_terrarium_egg",
        __name__ + ".universal_wrgb",
        __name__ + ".wrgb2",
        __name__ + ".wrgb2_pro",
        __name__ + ".wrgb2_slim",
        __name__ + ".z_light_tiny",
        # Optional: doser model (present only if the doser package is installed)
        f"{root}.chihiros_doser_control.device.doser",
    ]

    code2model: Dict[str, Type[BaseDevice]] = {}
    for mod_path in modules_to_try:
        mod = _safe_import(mod_path)
        if not mod:
            continue
        for _name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and issubclass(obj, BaseDevice):
                codes = getattr(obj, "_model_codes", [])
                if not isinstance(codes, (list, tuple)):
                    continue
                for code in codes:
                    if isinstance(code, str) and code:
                        code2model[code.upper()] = obj

    _CODE2MODEL = code2model
    return _CODE2MODEL


def get_model_class_from_name(device_name: str) -> Type[BaseDevice]:
    """Return the device class for a given BLE advertised name.

    Matches by prefix so names like 'DYDOSED203E0FEFCBC' resolve with codes
    ['DYDOSED2', 'DYDOSED', 'DYDOSE'].
    """
    from .fallback import Fallback  # cheap and safe at import-time

    if not device_name:
        return Fallback

    up = device_name.upper()
    registry = _ensure_registry()

    # Exact match first
    if up in registry:
        return registry[up]

    # Prefix match: prefer the longest matching code
    best_cls: Type[BaseDevice] | None = None
    best_len = -1
    for code, cls in registry.items():
        if up.startswith(code) and len(code) > best_len:
            best_cls = cls
            best_len = len(code)
    return best_cls or Fallback


async def get_device_from_address(device_address: str) -> BaseDevice:
    """Instantiate the correct device class from a MAC address (lazy bleak import).

    Note: This helper is intended for CLI/testing contexts. Inside Home Assistant,
    we resolve the BLEDevice via the HA bluetooth integration and create the device
    class from that object in __init__.py.
    """
    from bleak import BleakScanner  # lazy import

    ble_dev = await BleakScanner.find_device_by_address(device_address)
    if ble_dev and ble_dev.name:
        model_class = get_model_class_from_name(ble_dev.name)
        return model_class(ble_dev)

    from ..exception import DeviceNotFound  # lightweight local import

    raise DeviceNotFound


__all__ = [
    "BaseDevice",
    "get_device_from_address",
    "get_model_class_from_name",
]

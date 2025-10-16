# custom_components/chihiros/models.py
"""The chihiros integration models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Importing these is safe: coordinator.py provides a fake parent class
# when Home Assistant libs are not installed, so imports won't explode.
from .coordinator import ChihirosDataUpdateCoordinator

if TYPE_CHECKING:  # pragma: no cover
    # Keep the BaseDevice import type-only to avoid any heavy imports at runtime.
    from .chihiros_led_control.device import BaseDevice
else:  # Fallback runtime type to avoid importing the LED stack here.
    BaseDevice = object  # type: ignore[misc,assignment]


@dataclass
class ChihirosData:
    """Root data container stored under hass.data[DOMAIN][entry_id].

    Attributes:
        title:         Entry title shown in HA.
        device:        A concrete BaseDevice (LED or DOSER) instance.
        coordinator:   The central coordinator (BLE event bridge / dispatcher).
                       For dosers, __init__.py attaches:
                         - coordinator.device_type = "doser"
                         - coordinator.address
                         - coordinator.enabled_channels / channel_count
                         - coordinator.pinned_session (persistent BLE session)
                         - coordinator.write_gate (asyncio.Lock for serialized writes)
    """

    title: str
    device: BaseDevice
    coordinator: ChihirosDataUpdateCoordinator

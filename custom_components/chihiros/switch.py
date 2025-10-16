# custom_components/chihiros/switch.py
"""Switch platform for Chihiros LED Control to toggle auto/manual mode."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.helpers.entity import DeviceInfo

from .const import DOMAIN, MANUFACTURER
from .models import ChihirosData

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up the switch platform for Chihiros LED Control."""
    chihiros_data: ChihirosData = hass.data[DOMAIN][entry.entry_id]

    # Do not create any switches for doser devices
    if getattr(chihiros_data.coordinator, "device_type", "led") != "led":
        return

    async_add_entities(
        [ChihirosAutoManualSwitch(chihiros_data.coordinator, chihiros_data.device, entry)]
    )


class ChihirosAutoManualSwitch(SwitchEntity):
    """Switch to toggle between auto and manual mode."""

    _attr_should_poll = False

    def __init__(self, coordinator, device, config_entry) -> None:
        """Initialize the switch."""
        self._device = device
        self._coordinator = coordinator
        self._entry = config_entry

        address = getattr(coordinator, "address", None) or "unknown"
        name = getattr(device, "name", None) or "Chihiros LED"

        self._attr_name = f"{name} Auto Mode"
        self._attr_unique_id = f"{address}_auto_mode"
        self._attr_is_on = False  # assume manual until changed

        # Prefer connections if we have a BLE address; always include identifiers
        connections = {("bluetooth", address)} if address != "unknown" else None
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            connections=connections,
            manufacturer=MANUFACTURER,
            model=getattr(device, "model_name", "LED"),
            name=name,
        )

    @property
    def is_on(self) -> bool:
        """Return True if the switch is in auto mode."""
        return self._attr_is_on

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Auto mode: set brightness to auto level and enable auto mode."""
        try:
            await self._device.enable_auto_mode()
            self._attr_is_on = True
            self.async_write_ha_state()
            _LOGGER.debug("Switched to auto mode for %s", getattr(self._device, "name", "LED"))
        except Exception as e:
            _LOGGER.warning("Failed to enable auto mode: %s", e)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Manual mode: set brightness to last known or default value."""
        try:
            # Some models expose async method; fall back to sync if needed
            if hasattr(self._device, "async_set_manual_mode"):
                await self._device.async_set_manual_mode()
            else:
                await self.hass.async_add_executor_job(self._device.set_manual_mode)  # type: ignore[attr-defined]
            self._attr_is_on = False
            self.async_write_ha_state()
            _LOGGER.debug("Switched to manual mode for %s", getattr(self._device, "name", "LED"))
        except Exception as e:
            _LOGGER.warning("Failed to set manual mode: %s", e)

# custom_components/chihiros/button.py
"""
Chihiros Doser — “Dose Now” buttons (central-coordinator friendly)

- One button per enabled channel.
- Calls the integration's `dose_ml` service (which uses the pinned BLE session).
- After dosing, nudges totals refresh via the same dispatcher signals
  consumed by sensor.py (push-driven totals).

No direct BLE work happens here; all I/O is delegated to services that
share the persistent connection maintained by the central coordinator.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from homeassistant.components.button import ButtonEntity
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import DOMAIN

# Small post-dose dwell so the device updates its internal totals
_REFRESH_DELAY_S = 0.30


async def async_setup_entry(hass, entry, async_add_entities):
    data = hass.data[DOMAIN][entry.entry_id]
    coord = data.coordinator

    # Only for doser devices
    if getattr(coord, "device_type", "led") != "doser":
        return

    # Build from explicit enabled channels (Options) or fall back to 1..channel_count
    channels = list(getattr(coord, "enabled_channels", []))
    if not channels:
        count = int(getattr(coord, "channel_count", 4))
        channels = list(range(1, count + 1))

    entities = [DoserDoseNowButton(hass, entry, coord, ch) for ch in channels]
    async_add_entities(entities)


class DoserDoseNowButton(ButtonEntity):
    """Per-channel 'Dose now' button."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:play-circle"

    def __init__(self, hass, entry, coord, ch: int) -> None:
        self._hass = hass
        self._entry = entry
        self._coord = coord
        self._ch = ch

        self._attr_name = f"Ch {ch} Dose Now"
        self._attr_unique_id = f"{getattr(coord, 'address', 'unknown')}-ch{ch}-dose-now"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            manufacturer="Chihiros",
            model="Doser",
            name=(self._entry.title or "Chihiros Doser"),
        )

    async def async_press(self) -> None:
        address = getattr(self._coord, "address", None)
        # Amount map is expected to be {1: ml, 2: ml, ...}; default 1.0 mL if missing
        amount = float(getattr(self._coord, "doser_amounts", {}).get(self._ch, 1.0))

        # If the coordinator exposes a write gate (from the pinned session), use it
        gate = getattr(self._coord, "write_gate", None)
        if isinstance(gate, asyncio.Lock):
            async with gate:
                await self._call_dose_service(address, amount)
        else:
            await self._call_dose_service(address, amount)

        # Give the device a beat to update its counters, then nudge totals refresh
        await asyncio.sleep(_REFRESH_DELAY_S)
        async_dispatcher_send(self._hass, f"{DOMAIN}_{self._entry.entry_id}_refresh_totals")
        if address:
            async_dispatcher_send(self._hass, f"{DOMAIN}_refresh_totals_{address.lower()}")

    async def _call_dose_service(self, address: Optional[str], amount: float) -> None:
        # Channel is 1-based on the user/API layer
        await self._hass.services.async_call(
            DOMAIN,
            "dose_ml",
            {"address": address, "channel": self._ch, "ml": amount},
            blocking=True,
        )

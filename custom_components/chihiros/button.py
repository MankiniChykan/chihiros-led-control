# custom_components/chihiros/button.py
"""
Chihiros Doser — “Dose Now” buttons

Purpose:
- Per-channel button that triggers the integration's `dose_ml` service.
- After sending the dose, it nudges the daily-totals sensors to refresh by
  dispatching the same signals `sensor.py` subscribes to.

Matches current behavior in:
- protocol.py (dose is 1-based in the service layer, converted on-wire internally)
- sensor.py (listens to both per-entry and per-address refresh signals)

Nothing else is added to avoid diverging from device-confirmed behavior.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from homeassistant.components.button import ButtonEntity
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import DOMAIN

# Small, post-dose dwell so the device updates its internal totals
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
        # Amount map is expected to be {1: ml, 2: ml, ...}; default 1.0 mL if missing
        amount = float(getattr(self._coord, "doser_amounts", {}).get(self._ch, 1.0))
        address = getattr(self._coord, "address", None)

        # Send the dose via the integration's service (channel is 1-based)
        await self._hass.services.async_call(
            DOMAIN,
            "dose_ml",
            {"address": address, "channel": self._ch, "ml": amount},
            blocking=True,
        )

        # Give the device a beat to update its internal counters, then refresh sensors
        await asyncio.sleep(_REFRESH_DELAY_S)
        async_dispatcher_send(self._hass, f"{DOMAIN}_{self._entry.entry_id}_refresh_totals")
        if address:
            async_dispatcher_send(self._hass, f"{DOMAIN}_refresh_totals_{address.lower()}")

# custom_components/chihiros/sensor.py
"""
Chihiros Doser — Daily Totals sensors (LED 0x5B totals via central coordinator).

This platform no longer opens its own BLE connections. Instead, it relies on the
central ChihirosDataUpdateCoordinator to keep a persistent connection and push
decoded totals (ml) via the dispatcher signal:

    f"{DOMAIN}_push_totals_{address.lower()}"

We keep a tiny DataUpdateCoordinator here only as a cache + bridge for sensors.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any, Optional

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# Coordinator “polling” cadence — this coordinator does not do BLE I/O,
# but we keep a periodic tick (optional) in case someone wants scheduled refresh nudges.
UPDATE_EVERY = timedelta(minutes=15)

# Enable/disable totals sensors if you need to troubleshoot
DISABLE_DOSER_TOTAL_SENSORS: bool = False


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities
) -> None:
    """Set up doser daily total sensors per config entry (push-driven)."""
    if DISABLE_DOSER_TOTAL_SENSORS:
        _LOGGER.info(
            "chihiros.sensor: totals sensors DISABLED; skipping setup for %s",
            entry.entry_id,
        )
        return

    data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    address: Optional[str] = getattr(getattr(data, "coordinator", None), "address", None)
    _LOGGER.debug("chihiros.sensor: setup entry=%s addr=%s", entry.entry_id, address)

    coordinator = DoserTotalsProxyCoordinator(hass, address, entry)

    # Non-blocking initial refresh (just publishes the cache)
    hass.async_create_task(coordinator.async_request_refresh())

    # Per-entry refresh “nudge” (e.g., after a service call)
    # This asks whoever maintains the BLE session to try probing totals;
    # the actual values still arrive via the push signal.
    signal_entry_refresh = f"{DOMAIN}_{entry.entry_id}_refresh_totals"

    def _signal_refresh_entry() -> None:
        asyncio.run_coroutine_threadsafe(
            coordinator.async_request_refresh(), hass.loop
        )

    unsub_entry = async_dispatcher_connect(hass, signal_entry_refresh, _signal_refresh_entry)
    entry.async_on_unload(unsub_entry)

    # Per-address refresh “nudge”
    if address:
        sig_addr = f"{DOMAIN}_refresh_totals_{address.lower()}"

        def _signal_refresh_addr() -> None:
            asyncio.run_coroutine_threadsafe(
                coordinator.async_request_refresh(), hass.loop
            )

        unsub_addr = async_dispatcher_connect(hass, sig_addr, _signal_refresh_addr)
        entry.async_on_unload(unsub_addr)

    # Push path — the central coordinator decodes totals and dispatches them; we adopt directly.
    if address:
        push_sig = f"{DOMAIN}_push_totals_{address.lower()}"

        def _on_push(data: dict[str, Any]) -> None:
            # Expect {"ml":[...], "raw": bytes/bytearray}
            coordinator.async_set_updated_data(
                {
                    "ml": list((data.get("ml") or [None, None, None, None]))[:4],
                    "raw": (bytes(data.get("raw")) if isinstance(data.get("raw"), (bytes, bytearray)) else None),
                }
            )

        unsub_push = async_dispatcher_connect(hass, push_sig, _on_push)
        entry.async_on_unload(unsub_push)

    sensors = [ChDoserDailyTotalSensor(coordinator, entry, ch) for ch in range(4)]
    async_add_entities(sensors, update_before_add=False)


class DoserTotalsProxyCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """
    A very light coordinator that only caches the most recent totals pushed
    by the central BLE session. On refresh, it can optionally nudge the session
    to probe totals (no direct BLE I/O here).
    """

    def __init__(
        self, hass: HomeAssistant, address: Optional[str], entry: ConfigEntry
    ) -> None:
        super().__init__(
            hass,
            logger=_LOGGER,
            name=f"{DOMAIN}-doser-totals-proxy",
            update_interval=UPDATE_EVERY,  # harmless; all updates are push-driven
        )
        self.address = (address or "").upper() or None
        self.entry = entry
        self._last: dict[str, Any] = {"ml": [None, None, None, None], "raw": None}
        self._lock = asyncio.Lock()

    async def _async_update_data(self) -> dict[str, Any]:
        """Return the last known data; optionally ask the central session to probe."""
        if DISABLE_DOSER_TOTAL_SENSORS:
            _LOGGER.debug("sensor: proxy update skipped (disabled)")
            return self._last

        # Nudge the central coordinator to probe (if it listens to this signal).
        # This does not guarantee an immediate update; actual totals arrive via PUSH.
        if self.address:
            try:
                async_dispatcher_send(
                    self.hass, f"{DOMAIN}_refresh_totals_{self.address.lower()}"
                )
            except Exception:
                pass

        # Just return cache. The push handler will call async_set_updated_data() when values arrive.
        return self._last


class ChDoserDailyTotalSensor(
    CoordinatorEntity[DoserTotalsProxyCoordinator], SensorEntity
):
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "mL"
    _attr_has_entity_name = True

    def __init__(
        self, coordinator: DoserTotalsProxyCoordinator, entry: ConfigEntry, ch: int
    ) -> None:
        super().__init__(coordinator)
        self._ch = ch
        self._attr_name = f"Ch {ch + 1} Daily Dose"
        self._attr_unique_id = (
            f"{entry.entry_id}-doser-ch{ch + 1}-daily_total_ml"
        )
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)}
        )

    @property
    def native_value(self) -> Optional[float]:
        ml = (self.coordinator.data or {}).get("ml") or [
            None,
            None,
            None,
            None,
        ]
        return ml[self._ch] if self._ch < len(ml) else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        raw = (self.coordinator.data or {}).get("raw")
        return {
            "raw_frame": raw.hex(" ").upper()
            if isinstance(raw, (bytes, bytearray))
            else None
        }

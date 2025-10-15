# custom_components/chihiros/sensor.py
"""
Chihiros Doser — Daily Totals sensors (LED 0x5B probe, tolerant decode).

This sensor platform passively listens for the device's daily totals frame
(LED/report side, CMD=0x5B) and exposes four sensors — one per channel —
returning mL values as decoded by protocol.parse_totals_frame().

How it works (aligned with protocol.py):
• On each scheduled refresh, we connect over BLE (via HA's connector),
  subscribe to UART_TX notifications, and send a tiny set of LED-side probes
  built by protocol.build_totals_probes() (0x34, then 0x22).
• The first notify that protocol.parse_totals_frame(...) can decode into four
  channel values wins; we cache and expose it via DataUpdateCoordinator.
• The coordinator also accepts “push” updates via the dispatcher signal
  f"{DOMAIN}_push_totals_{address}", for cases where you already listen to
  notifications elsewhere and want to feed decoded results directly.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any, Optional

from homeassistant.components import bluetooth
from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# Seconds to wait for a single totals notify (0x5B …)
SCAN_TIMEOUT = 12.0
# Coordinator polling cadence (you can adjust in the future)
UPDATE_EVERY = timedelta(minutes=15)

# Enable/disable totals sensors if you need to troubleshoot
DISABLE_DOSER_TOTAL_SENSORS: bool = False


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities
) -> None:
    """Set up doser daily total sensors per config entry."""
    if DISABLE_DOSER_TOTAL_SENSORS:
        _LOGGER.info(
            "chihiros.sensor: totals sensors DISABLED; skipping setup for %s",
            entry.entry_id,
        )
        return

    data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    address: Optional[str] = getattr(getattr(data, "coordinator", None), "address", None)
    _LOGGER.debug("chihiros.sensor: setup entry=%s addr=%s", entry.entry_id, address)

    coordinator = DoserTotalsCoordinator(hass, address, entry)

    # Non-blocking initial refresh
    hass.async_create_task(coordinator.async_request_refresh())

    # Per-entry refresh signal (e.g., after “Dose Now” service)
    signal = f"{DOMAIN}_{entry.entry_id}_refresh_totals"

    def _signal_refresh_entry() -> None:
        asyncio.run_coroutine_threadsafe(
            coordinator.async_request_refresh(), hass.loop
        )

    unsub = async_dispatcher_connect(hass, signal, _signal_refresh_entry)
    entry.async_on_unload(unsub)

    # Also allow per-address refresh signal
    if address:
        sig_addr = f"{DOMAIN}_refresh_totals_{address.lower()}"

        def _signal_refresh_addr() -> None:
            asyncio.run_coroutine_threadsafe(
                coordinator.async_request_refresh(), hass.loop
            )

        unsub2 = async_dispatcher_connect(hass, sig_addr, _signal_refresh_addr)
        entry.async_on_unload(unsub2)

    # Push path — services can decode totals and dispatch them; we adopt directly
    if address:
        push_sig = f"{DOMAIN}_push_totals_{address.lower()}"

        def _on_push(data: dict[str, Any]) -> None:
            # Expect {"ml":[...], "raw": bytes/bytearray}
            coordinator.async_set_updated_data(data)

        unsub_push = async_dispatcher_connect(hass, push_sig, _on_push)
        entry.async_on_unload(unsub_push)

    sensors = [ChDoserDailyTotalSensor(coordinator, entry, ch) for ch in range(4)]
    async_add_entities(sensors, update_before_add=False)


class DoserTotalsCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Listen briefly for the 0x5B totals notify and decode ml pairs."""

    def __init__(
        self, hass: HomeAssistant, address: Optional[str], entry: ConfigEntry
    ) -> None:
        super().__init__(
            hass,
            logger=_LOGGER,
            name=f"{DOMAIN}-doser-totals",
            update_interval=UPDATE_EVERY,
        )
        self.address = address
        self.entry = entry
        self._last: dict[str, Any] = {"ml": [None, None, None, None], "raw": None}
        self._lock = asyncio.Lock()  # avoid overlapping BLE connects

    async def _async_update_data(self) -> dict[str, Any]:
        if DISABLE_DOSER_TOTAL_SENSORS:
            _LOGGER.debug("sensor: coordinator update skipped (disabled)")
            return self._last

        # Lazy-import runtime-only deps so importing this file won’t require them
        from bleak_retry_connector import (  # type: ignore
            BleakClientWithServiceCache,
            BLEAK_RETRY_EXCEPTIONS as BLEAK_EXC,
            establish_connection,
        )
        from .chihiros_doser_control.protocol import (  # type: ignore
            UART_TX,
            UART_RX,
        )
        from .chihiros_doser_control import protocol as dp  # type: ignore

        async with self._lock:
            if not self.address:
                _LOGGER.debug("sensor: no BLE address; keeping last values")
                return self._last

            # HA stores addresses uppercase
            ble_dev = bluetooth.async_ble_device_from_address(
                self.hass, self.address.upper(), True
            )
            if not ble_dev:
                _LOGGER.debug(
                    "sensor: no BLEDevice for %s; keeping last", self.address
                )
                return self._last

            got: asyncio.Future[dict[str, Any]] = (
                asyncio.get_running_loop().create_future()
            )

            def _cb(_char, payload: bytearray) -> None:
                """Notify callback — strict LED totals (0x5B) only."""
                try:
                    if not payload:
                        return
                    values = dp.parse_totals_frame(payload)
                    if values is not None and not got.done():
                        got.set_result({"ml": values[:4], "raw": bytes(payload)})
                except Exception:  # pragma: no cover
                    _LOGGER.exception("sensor: notify parse error")

            client = None
            try:
                # Use HA-friendly connector; it queues if a slot isn’t available
                client = await establish_connection(  # type: ignore
                    BleakClientWithServiceCache, ble_dev, f"{DOMAIN}-totals"
                )
                await client.start_notify(UART_TX, _cb)  # type: ignore

                # Build the minimal set of probes supported by the device/firmware
                try:
                    frames: list[bytes] = list(dp.build_totals_probes())
                except Exception:
                    frames = []

                # Fallback: try common LED-side modes
                if not frames:
                    try:
                        frames.extend(
                            [
                                dp.encode_5b(0x34, []),
                                dp.encode_5b(0x22, []),
                            ]
                        )
                    except Exception:
                        pass

                # Write probes; most devices accept on UART_RX, a few on TX (so guard)
                for idx, frame in enumerate(frames):
                    try:
                        await client.write_gatt_char(UART_RX, frame, response=True)  # type: ignore
                    except Exception:
                        try:
                            await client.write_gatt_char(UART_TX, frame, response=True)  # type: ignore
                        except Exception:
                            _LOGGER.debug(
                                "sensor: probe write failed (idx=%d)", idx, exc_info=True
                            )
                    await asyncio.sleep(0.08)  # keep under SCAN_TIMEOUT budget

                try:
                    res = await asyncio.wait_for(got, timeout=SCAN_TIMEOUT)
                    self._last = res
                except asyncio.TimeoutError:
                    _LOGGER.debug(
                        "sensor: no totals frame within %.1fs; keeping last",
                        SCAN_TIMEOUT,
                    )
                finally:
                    try:
                        await client.stop_notify(UART_TX)  # type: ignore
                    except Exception:
                        pass
            except BLEAK_EXC as e:  # type: ignore
                _LOGGER.debug("sensor: BLE/slot error: %s; keeping last", e)
            except Exception as e:
                _LOGGER.warning("sensor: BLE error: %s", e)
            finally:
                if client:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

            return self._last


class ChDoserDailyTotalSensor(
    CoordinatorEntity[DoserTotalsCoordinator], SensorEntity
):
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "mL"
    _attr_has_entity_name = True

    def __init__(
        self, coordinator: DoserTotalsCoordinator, entry: ConfigEntry, ch: int
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

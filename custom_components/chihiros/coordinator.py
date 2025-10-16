# custom_components/chihiros/coordinator.py
"""Central BLE coordinator for Chihiros devices in Home Assistant.

This module provides a single place to:
- Own a long-lived BLE connection (via your BaseDevice client)
- Keep the link warm with a lightweight keep-alive
- Bridge HA Bluetooth events to the client when needed
- Offer simple helpers to subscribe/unsubscribe extra notify callbacks

It intentionally avoids protocol specifics — those live in your control
modules (e.g. chihiros_doser_control/*). The coordinator just makes sure
there *is* a stable BLE session for them to use.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

try:
    from homeassistant.components import bluetooth
    from homeassistant.components.bluetooth.passive_update_coordinator import (
        PassiveBluetoothDataUpdateCoordinator,
    )
    from homeassistant.core import HomeAssistant, callback

    CoordinatorBase: type[PassiveBluetoothDataUpdateCoordinator | _FakeBase]
    CoordinatorBase = PassiveBluetoothDataUpdateCoordinator  # type: ignore[assignment]
except ModuleNotFoundError:
    # Minimal stubs so the file can be imported outside HA (tests, linters, etc.)
    class _FakeBase:  # pragma: no cover
        def __init__(self, *a, **kw): ...
    def callback(fn):  # pragma: no cover
        return fn
    class bluetooth:  # pragma: no cover
        class BluetoothScanningMode:
            ACTIVE = "active"
    class HomeAssistant:  # pragma: no cover
        loop: asyncio.AbstractEventLoop

    CoordinatorBase = _FakeBase  # type: ignore

from .chihiros_led_control.device.base_device import BaseDevice

if TYPE_CHECKING:
    from bleak.backends.device import BLEDevice
    from bleak.backends.service import BleakGATTCharacteristic

_LOGGER: logging.Logger = logging.getLogger(__name__)

NotifyCallback = Callable[[BleakGATTCharacteristic, bytearray], None]


class ChihirosDataUpdateCoordinator(CoordinatorBase):  # type: ignore[misc]
    """Manage the BLE session and keep it persistent for the integration."""

    # How often to poke the client to keep the BaseDevice auto-disconnect timer from firing.
    # BaseDevice currently schedules a disconnect ~120s after last activity; we refresh well before that.
    KEEPALIVE_SECONDS: int = 60

    def __init__(self, hass: HomeAssistant, client: BaseDevice, ble_device: BLEDevice) -> None:
        """Initialize the coordinator."""
        self.hass = hass
        self.api: BaseDevice = client
        self.ble_device = ble_device
        self.address: str = getattr(ble_device, "address", "").upper()

        # Data for CoordinatorEntity subclasses (if any want to read something here)
        self.data: dict[str, Any] = {}

        # Internal state
        self._connect_lock = asyncio.Lock()
        self._keepalive_task: Optional[asyncio.Task] = None
        self._started: bool = False

        super().__init__(
            hass,
            _LOGGER,
            self.address or "chihiros-ble",
            bluetooth.BluetoothScanningMode.ACTIVE,  # keeps HA radio engaged for this device
        )

    # ────────────────────────────────────────────────────────────────
    # HA lifecycle
    # ────────────────────────────────────────────────────────────────
    async def async_start(self) -> None:
        """Begin the persistent session and start keepalive."""
        if self._started:
            return
        self._started = True
        await self._ensure_connected()
        self._start_keepalive()

    async def async_stop(self) -> None:
        """Tear down the session and stop keepalive."""
        self._started = False
        self._stop_keepalive()
        try:
            await self.api.disconnect()
        except Exception:  # pragma: no cover
            _LOGGER.debug("coordinator: disconnect raised", exc_info=True)

    # ────────────────────────────────────────────────────────────────
    # Keepalive (prevents BaseDevice timed auto-disconnect)
    # ────────────────────────────────────────────────────────────────
    def _start_keepalive(self) -> None:
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = self.hass.loop.create_task(self._run_keepalive())

    def _stop_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task and not task.done():
            task.cancel()

    async def _run_keepalive(self) -> None:
        """Periodically call connect() which resets BaseDevice's disconnect timer."""
        try:
            while self._started:
                try:
                    await self._ensure_connected()
                except Exception as e:
                    _LOGGER.debug("coordinator: keepalive connect failed: %s", e, exc_info=True)
                await asyncio.sleep(self.KEEPALIVE_SECONDS)
        except asyncio.CancelledError:  # normal shutdown path
            pass

    # ────────────────────────────────────────────────────────────────
    # Connection plumbing
    # ────────────────────────────────────────────────────────────────
    async def _ensure_connected(self) -> None:
        """Ensure BaseDevice is connected and notifications are active."""
        async with self._connect_lock:
            try:
                client = await self.api.connect()
                # BaseDevice.connect() already subscribes to TX notifications in _ensure_connected()
                # Nothing else to do here unless you want to add extra notify callbacks.
                _LOGGER.debug("%s: coordinator connected", self.address)
            except Exception as e:
                _LOGGER.debug("%s: coordinator connect error: %s", self.address, e, exc_info=True)
                raise

    # Public helper for other parts of the integration that need the client
    async def async_with_client(self, fn: Callable[[Any], "asyncio.Future[Any] | Any"]) -> Any:
        """Run a coroutine/callable with an ensured connection."""
        await self._ensure_connected()
        return await fn(self.api.client)  # BaseDevice.client is the Bleak client

    # ────────────────────────────────────────────────────────────────
    # Notify callback registration (fan-out owned centrally)
    # ────────────────────────────────────────────────────────────────
    def add_notify_callback(self, cb: NotifyCallback) -> None:
        """Register an extra notification callback on the BaseDevice."""
        try:
            self.api.add_notify_callback(cb)
        except Exception:  # pragma: no cover
            _LOGGER.debug("coordinator: add_notify_callback failed", exc_info=True)

    def remove_notify_callback(self, cb: NotifyCallback) -> None:
        """Unregister a notification callback on the BaseDevice."""
        try:
            self.api.remove_notify_callback(cb)
        except Exception:  # pragma: no cover
            _LOGGER.debug("coordinator: remove_notify_callback failed", exc_info=True)

    # ────────────────────────────────────────────────────────────────
    # Bluetooth event hooks (optional but nice for diagnostics)
    # ────────────────────────────────────────────────────────────────
    @callback
    def _async_handle_bluetooth_event(
        self,
        service_info: "bluetooth.BluetoothServiceInfoBleak",
        change: "bluetooth.BluetoothChange",
    ) -> None:
        """HA Bluetooth stack says something changed for our device."""
        _LOGGER.debug("%s: BT event: %s", self.address, change)
        super()._async_handle_bluetooth_event(service_info, change)
        # If you want to trigger a reconnect on certain changes, you could:
        # self.hass.loop.create_task(self._ensure_connected())

    @callback
    def _async_handle_unavailable(
        self, service_info: "bluetooth.BluetoothServiceInfoBleak"
    ) -> None:
        """Handle the device going temporarily unavailable (out of range, etc.)."""
        _LOGGER.debug("%s: device unavailable", self.address)
        super()._async_handle_unavailable(service_info)
        # Keepalive will continue to attempt re-connection in the background.


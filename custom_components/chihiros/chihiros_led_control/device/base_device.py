# custom_components/chihiros/chihiros_led_control/device/base_device.py
"""Base class for Chihiros LED devices.

Notes:
- No Home Assistant imports at module import time (lazy bleak + HA bits).
- Uses bleak-retry-connector to share HA's Bluetooth proxy and retry logic.
- Char UUIDs are sourced from the integration-level constants module.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, ABCMeta
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Union, Sequence

# Editor-only types; avoid importing bleak at runtime unless needed.
if TYPE_CHECKING:  # pragma: no cover
    from bleak.backends.device import BLEDevice
    from bleak.backends.scanner import AdvertisementData
    from bleak.backends.service import (
        BleakGATTCharacteristic,
        BleakGATTServiceCollection,
    )
else:
    BLEDevice = Any
    AdvertisementData = Any
    BleakGATTCharacteristic = Any
    BleakGATTServiceCollection = Any

from .. import commands
from ..const import UART_RX_CHAR_UUID, UART_TX_CHAR_UUID
from ..exception import CharacteristicMissingError
from ..weekday_encoding import WeekdaySelect, encode_selected_weekdays

DEFAULT_ATTEMPTS = 3
DISCONNECT_DELAY = 120
BLEAK_BACKOFF_TIME = 0.25


class _classproperty(property):
    def __get__(self, owner_self: object, owner_cls: ABCMeta) -> str:  # type: ignore[override]
        ret: str = self.fget(owner_cls)  # type: ignore[misc]
        return ret


def _is_ha_runtime() -> bool:
    try:
        import homeassistant  # type: ignore
        return True
    except Exception:
        return False


def _mk_ble_device(addr_or_ble: Union[BLEDevice, str]) -> BLEDevice:
    """Return a BLEDevice, fabricating one in CLI mode if only a MAC is given."""
    from bleak.backends.device import BLEDevice as _BLEDevice  # lazy

    if isinstance(addr_or_ble, _BLEDevice):
        return addr_or_ble
    if _is_ha_runtime():
        # Inside HA we should always be passed a BLEDevice (keeps proxy path correct).
        raise RuntimeError("In Home Assistant, pass a real BLEDevice (not a MAC string).")
    mac = str(addr_or_ble).upper()
    return _BLEDevice(mac, None, 0)


def _import_bleak_retry():
    """Lazy import bleak-retry-connector items when actually needed."""
    from bleak_retry_connector import (
        BLEAK_RETRY_EXCEPTIONS as BLEAK_EXCEPTIONS,
        BleakError,
        BleakClientWithServiceCache,
        BleakNotFoundError,
        establish_connection,
        retry_bluetooth_connection_error,
    )
    return {
        "BLEAK_EXCEPTIONS": BLEAK_EXCEPTIONS,
        "BleakError": BleakError,
        "BleakClientWithServiceCache": BleakClientWithServiceCache,
        "BleakNotFoundError": BleakNotFoundError,
        "establish_connection": establish_connection,
        "retry_bluetooth_connection_error": retry_bluetooth_connection_error,
    }


def _import_bleak_dbus_error():
    """Return BleakDBusError (or a stub) lazily for non-DBus backends."""
    try:
        from bleak.exc import BleakDBusError  # type: ignore
    except Exception:

        class BleakDBusError(Exception):  # type: ignore
            pass

    return BleakDBusError


class BaseDevice(ABC):
    """Base device class used by device classes."""

    _model_name: str | None = None
    _model_codes: list[str] = []
    _colors: dict[str, int] = {}
    _msg_id = commands.next_message_id()
    _logger: logging.Logger

    def __init__(
        self,
        ble_device: Union[BLEDevice, str],
        advertisement_data: AdvertisementData | None = None,
    ) -> None:
        self._ble_device = _mk_ble_device(ble_device)
        self._logger = logging.getLogger(self._ble_device.address.replace(":", "-"))
        self._advertisement_data = advertisement_data
        self._client: Any = None  # BleakClientWithServiceCache | None
        self._disconnect_timer: asyncio.TimerHandle | None = None
        self._operation_lock: asyncio.Lock = asyncio.Lock()
        self._read_char: BleakGATTCharacteristic | None = None
        self._write_char: BleakGATTCharacteristic | None = None
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        self._expected_disconnect = False
        self.loop = asyncio.get_running_loop()
        self._extra_notify_callbacks: list[
            Callable[[BleakGATTCharacteristic, bytearray], None]
        ] = []
        assert self._model_name is not None

    # ---- properties ----
    @property
    def current_msg_id(self) -> tuple[int, int]:
        return self._msg_id

    def get_next_msg_id(self) -> tuple[int, int]:
        self._msg_id = commands.next_message_id(self._msg_id)
        return self._msg_id

    @property
    def colors(self) -> dict[str, int]:
        return self._colors

    @property
    def address(self) -> str:
        return self._ble_device.address

    @property
    def name(self) -> str:
        if hasattr(self._ble_device, "name"):
            return self._ble_device.name or self._ble_device.address
        return self._ble_device.address

    @property
    def rssi(self) -> int | None:
        if self._advertisement_data:
            return self._advertisement_data.rssi
        return None

    @_classproperty
    def model_name(cls) -> str | None:  # type: ignore[override]
        return cls._model_name

    @_classproperty
    def model_codes(cls) -> list[str]:  # type: ignore[override]
        return cls._model_codes

    # ---- commands (internals) ----
    async def async_set_color_brightness(self, brightness: int, color: str | int = 0) -> None:
        color_id: int | None = None
        if isinstance(color, int) and color in self._colors.values():
            color_id = color
        elif isinstance(color, str) and color in self._colors:
            color_id = self._colors.get(color)
        if color_id is None:
            self._logger.warning("Color not supported: `%s`", color)
            return
        cmd = commands.create_manual_setting_command(
            self.get_next_msg_id(), color_id, int(brightness)
        )
        await self._send_command(cmd, 3)

    async def async_set_brightness(self, brightness: int) -> None:
        await self.async_set_color_brightness(brightness=brightness, color=0)

    async def async_set_manual_mode(self) -> None:
        for color_name in self._colors:
            await self.async_set_color_brightness(100, color_name)

    # Back-compat shim for switch platform (expects enable_auto_mode)
    async def enable_auto_mode(self) -> None:
        await self.get_enable_auto_mode()

    # ---- commands (control helpers) ----
    async def get_turn_on(self) -> None:
        for color_name in self._colors:
            await self.async_set_color_brightness(100, color_name)

    async def get_turn_off(self) -> None:
        for color_name in self._colors:
            await self.async_set_color_brightness(0, color_name)

    async def get_add_setting(
        self,
        sunrise: datetime,
        sunset: datetime,
        max_brightness: int = 100,
        ramp_up_in_minutes: int = 0,
        weekdays: Sequence[WeekdaySelect] | None = None,
    ) -> None:
        cmd = commands.create_add_auto_setting_command(
            self.get_next_msg_id(),
            sunrise.time(),
            sunset.time(),
            (max_brightness, 255, 255),
            ramp_up_in_minutes,
            encode_selected_weekdays(list(weekdays or [WeekdaySelect.everyday])),
        )
        await self._send_command(cmd, 3)

    async def get_rgb_brightness(self, brightness: Sequence[int]) -> None:
        vals = [int(v) for v in brightness]
        if len(vals) == 1:
            vals = [vals[0], vals[0], vals[0]]
        elif len(vals) not in (3, 4):
            raise ValueError("Provide either 1 value, or 3 (R G B), or 4 (R G B W).")
        for v in vals:
            if not (0 <= v <= 140):
                raise ValueError("Each channel must be between 0 and 140.")
        total = sum(vals)
        if len(vals) == 3 and total > 300:
            raise ValueError("The values of RGB (R+G+B) must not exceed 300%.")
        if len(vals) == 4 and total > 400:
            raise ValueError("The values of RGBW (R+G+B+W) must not exceed 400%.")
        for chan_index, chan_value in enumerate(vals):
            await self.async_set_color_brightness(
                brightness=chan_value, color=chan_index
            )

    async def get_add_rgb_setting(
        self,
        sunrise: datetime,
        sunset: datetime,
        max_brightness: tuple[int, int, int] = (100, 100, 100),
        ramp_up_in_minutes: int = 0,
        weekdays: Sequence[WeekdaySelect] | None = None,
    ) -> None:
        cmd = commands.create_add_auto_setting_command(
            self.get_next_msg_id(),
            sunrise.time(),
            sunset.time(),
            max_brightness,
            ramp_up_in_minutes,
            encode_selected_weekdays(list(weekdays or [WeekdaySelect.everyday])),
        )
        await self._send_command(cmd, 3)

    async def get_remove_setting(
        self,
        sunrise: datetime,
        sunset: datetime,
        ramp_up_in_minutes: int = 0,
        weekdays: Sequence[WeekdaySelect] | None = None,
    ) -> None:
        cmd = commands.create_delete_auto_setting_command(
            self.get_next_msg_id(),
            sunrise.time(),
            sunset.time(),
            ramp_up_in_minutes,
            encode_selected_weekdays(list(weekdays or [WeekdaySelect.everyday])),
        )
        await self._send_command(cmd, 3)

    async def get_reset_settings(self) -> None:
        cmd = commands.create_reset_auto_settings_command(self.get_next_msg_id())
        await self._send_command(cmd, 3)

    async def get_enable_auto_mode(self) -> None:
        switch_cmd = commands.create_switch_to_auto_mode_command(self.get_next_msg_id())
        time_cmd = commands.create_set_time_command(self.get_next_msg_id())
        await self._send_command(switch_cmd, 3)
        await self._send_command(time_cmd, 3)

    # ---- Bluetooth plumbing ----
    async def _send_command(
        self, commands_bytes: list[bytes] | bytes | bytearray, retry: int | None = None
    ) -> None:
        await self._ensure_connected()
        if not isinstance(commands_bytes, list):
            commands_bytes = [commands_bytes]
        await self._send_command_while_connected(commands_bytes, retry)

    async def _send_command_while_connected(
        self, commands_bytes: list[bytes], retry: int | None = None
    ) -> None:
        bits = _import_bleak_retry()
        BleakNotFoundError = bits["BleakNotFoundError"]
        BLEAK_EXCEPTIONS = bits["BLEAK_EXCEPTIONS"]

        self._logger.debug(
            "%s: Sending commands %s",
            self.name,
            [command.hex() for command in commands_bytes],
        )
        if self._operation_lock.locked():
            self._logger.debug(
                "%s: Operation already in progress, waiting for it to complete; RSSI: %s",
                self.name,
                self.rssi,
            )
        async with self._operation_lock:
            try:
                await self._send_command_locked(commands_bytes)
                return
            except BleakNotFoundError:
                self._logger.error(
                    "%s: device not found, no longer in range, or poor RSSI: %s",
                    self.name,
                    self.rssi,
                    exc_info=True,
                )
                raise
            except CharacteristicMissingError as ex:
                self._logger.debug(
                    "%s: characteristic missing: %s; RSSI: %s",
                    self.name,
                    ex,
                    self.rssi,
                    exc_info=True,
                )
                raise
            except BLEAK_EXCEPTIONS:
                self._logger.debug("%s: communication failed", self.name, exc_info=True)
                raise
        raise RuntimeError("Unreachable")

    async def _send_command_locked(self, commands_bytes: list[bytes]) -> None:
        """Retry-wrapped by bleak_retry_connector; do not call directly."""
        BleakDBusError = _import_bleak_dbus_error()
        BleakError = _import_bleak_retry()["BleakError"]

        try:
            await self._execute_command_locked(commands_bytes)
        except BleakDBusError as ex:
            await asyncio.sleep(BLEAK_BACKOFF_TIME)
            self._logger.debug(
                "%s: RSSI: %s; Backing off %ss; Disconnecting due to error: %s",
                self.name,
                self.rssi,
                BLEAK_BACKOFF_TIME,
                ex,
            )
            await self._execute_disconnect()
            raise
        except BleakError as ex:
            self._logger.debug(
                "%s: RSSI: %s; Disconnecting due to error: %s", self.name, self.rssi, ex
            )
            await self._execute_disconnect()
            raise

    async def _execute_command_locked(self, commands_bytes: list[bytes]) -> None:
        if self._client is None:
            raise RuntimeError("Client not connected")
        if not self._read_char:
            raise CharacteristicMissingError("Read characteristic missing")
        if not self._write_char:
            raise CharacteristicMissingError("Write characteristic missing")
        for command in commands_bytes:
            await self._client.write_gatt_char(self._write_char, command, False)

    def _notification_handler(
        self, _sender: BleakGATTCharacteristic, data: bytearray
    ) -> None:
        self._logger.debug(
            "%s: Notification received: %s", self.name, data.hex(" ").upper()
        )
        for cb in tuple(getattr(self, "_extra_notify_callbacks", [])):
            try:
                cb(_sender, data)
            except Exception:
                self._logger.debug(
                    "%s: notify callback raised", self.name, exc_info=True
                )

    def add_notify_callback(
        self, cb: Callable[[BleakGATTCharacteristic, bytearray], None]
    ) -> None:
        if cb not in getattr(self, "_extra_notify_callbacks", []):
            self._extra_notify_callbacks.append(cb)

    def remove_notify_callback(
        self, cb: Callable[[BleakGATTCharacteristic, bytearray], None]
    ) -> None:
        try:
            self._extra_notify_callbacks.remove(cb)
        except ValueError:
            pass

    def _disconnected(self, client: Any) -> None:
        if self._expected_disconnect:
            self._logger.debug("%s: Disconnected from device; RSSI: %s", self.name, self.rssi)
            return
        self._logger.warning(
            "%s: Device unexpectedly disconnected; RSSI: %s", self.name, self.rssi
        )

    def _resolve_characteristics(self, services: BleakGATTServiceCollection) -> bool:
        # Read: TX (device → host), Write: RX (host → device)
        for characteristic in [UART_TX_CHAR_UUID]:
            if char := services.get_characteristic(characteristic):
                self._read_char = char
                break
        for characteristic in [UART_RX_CHAR_UUID]:
            if char := services.get_characteristic(characteristic):
                self._write_char = char
                break
        return bool(self._read_char and self._write_char)

    async def _ensure_connected(self) -> None:
        bits = _import_bleak_retry()
        BleakClientWithServiceCache = bits["BleakClientWithServiceCache"]
        establish_connection = bits["establish_connection"]

        if self._connect_lock.locked():
            self._logger.debug(
                "%s: Connection already in progress, waiting for it to complete; RSSI: %s",
                self.name,
                self.rssi,
            )
        if self._client and getattr(self._client, "is_connected", False):
            self._reset_disconnect_timer()
            return

        async with self._connect_lock:
            if self._client and getattr(self._client, "is_connected", False):
                self._reset_disconnect_timer()
                return

            self._logger.debug("%s: Connecting; RSSI: %s", self.name, self.rssi)

            if isinstance(self._ble_device, object) and hasattr(self._ble_device, "address"):
                device_arg: Union[str, BLEDevice] = self._ble_device
                kwargs = {
                    "use_services_cache": True,
                    "ble_device_callback": lambda: self._ble_device,  # type: ignore[return-value]
                }
            else:
                device_arg = self._ble_device.address  # type: ignore[attr-defined]
                kwargs = {"use_services_cache": True}

            client = await establish_connection(
                BleakClientWithServiceCache,
                device_arg,
                self.name,
                self._disconnected,
                **kwargs,
            )

            self._logger.debug("%s: Connected; RSSI: %s", self.name, self.rssi)
            services = client.services or await client.get_services()
            resolved = self._resolve_characteristics(services)

            self._client = client
            self._reset_disconnect_timer()

            if resolved and self._read_char:
                self._logger.debug(
                    "%s: Subscribe to notifications; RSSI: %s", self.name, self.rssi
                )
                await client.start_notify(self._read_char, self._notification_handler)  # type: ignore[arg-type]
            else:
                raise CharacteristicMissingError("Failed to resolve UART characteristics")

    # public helpers
    async def connect(self) -> Any:
        await self._ensure_connected()
        assert self._client is not None  # nosec
        return self._client

    @property
    def client(self) -> Any:
        return self._client

    def _reset_disconnect_timer(self) -> None:
        if self._disconnect_timer:
            self._disconnect_timer.cancel()
        self._expected_disconnect = False
        self._disconnect_timer = self.loop.call_later(
            DISCONNECT_DELAY, self._disconnect
        )

    async def disconnect(self) -> None:
        self._logger.debug("%s: Disconnecting", self.name)
        await self._execute_disconnect()

    async def _execute_disconnect(self) -> None:
        async with self._connect_lock:
            read_char = self._read_char
            client = self._client
            self._expected_disconnect = True
            self._client = None
            self._read_char = None
            self._write_char = None
            self._extra_notify_callbacks = []

            if client and getattr(client, "is_connected", False):
                if read_char:
                    try:
                        await client.stop_notify(read_char)
                    except Exception:
                        self._logger.debug(
                            "%s: stop_notify failed (already stopped?)",
                            self.name,
                            exc_info=True,
                        )
                await client.disconnect()

    def _disconnect(self) -> None:
        self._disconnect_timer = None
        asyncio.create_task(self._execute_timed_disconnect())

    async def _execute_timed_disconnect(self) -> None:
        self._logger.debug(
            "%s: Disconnecting after timeout of %s", self.name, DISCONNECT_DELAY
        )
        await self._execute_disconnect()


# Decorate retry AFTER class is defined (keeps import-time light)
_retry_bits = _import_bleak_retry()
retry_bluetooth_connection_error = _retry_bits["retry_bluetooth_connection_error"]

BaseDevice._send_command_locked = retry_bluetooth_connection_error(DEFAULT_ATTEMPTS)(  # type: ignore[attr-defined]
    BaseDevice._send_command_locked  # type: ignore
)

# custom_components/chihiros/chihiros_led_control/device/base_device.py
"""Module defining a base device class."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, ABCMeta
from datetime import datetime
from typing import Callable, Optional, Union, Sequence

import typer
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from bleak.backends.service import BleakGATTCharacteristic  # type: ignore
from bleak.backends.service import BleakGATTServiceCollection
from bleak.exc import BleakDBusError
from bleak_retry_connector import BLEAK_RETRY_EXCEPTIONS as BLEAK_EXCEPTIONS
from bleak_retry_connector import BleakError  # type: ignore
from bleak_retry_connector import (
    BleakClientWithServiceCache,
    BleakNotFoundError,
    establish_connection,
    retry_bluetooth_connection_error,
)
from typing_extensions import Annotated

from .. import commands
from ..const import UART_RX_CHAR_UUID, UART_TX_CHAR_UUID
from ..exception import CharacteristicMissingError
from ..weekday_encoding import WeekdaySelect, encode_selected_weekdays

DEFAULT_ATTEMPTS = 3

DISCONNECT_DELAY = 120
BLEAK_BACKOFF_TIME = 0.25


class _classproperty(property):
    def __get__(self, owner_self: object, owner_cls: ABCMeta) -> str:  # type: ignore
        ret: str = self.fget(owner_cls)  # type: ignore
        return ret


def _is_ha_runtime() -> bool:
    """Best-effort check: are we running under Home Assistant?"""
    try:
        import homeassistant  # type: ignore
        return True
    except Exception:
        return False


def _mk_ble_device(addr_or_ble: Union[BLEDevice, str]) -> BLEDevice:
    """
    Return a BLEDevice.

    - If a BLEDevice is provided, return it.
    - If a MAC string is provided:
        * In Home Assistant: refuse (HA should pass a real BLEDevice).
        * In CLI/dev: fabricate a minimal BLEDevice (name=None, details=0/None).
    """
    if isinstance(addr_or_ble, BLEDevice):
        return addr_or_ble

    if _is_ha_runtime():
        raise RuntimeError(
            "In Home Assistant, pass a real BLEDevice (not a MAC string)."
        )

    mac = str(addr_or_ble).upper()
    # Bleak 1.x ctor signature on all backends we support: BLEDevice(address, name, details)
    # Keep 'name=None' so Bleak doesn't think we have an advertised name.
    return BLEDevice(mac, None, 0)  # or BLEDevice(mac, None, None)


# ────────────────────────────────────────────────────────────────
# Brightness helpers
# ────────────────────────────────────────────────────────────────

def _check_rgb_totals(vals: Sequence[int]) -> None:
    """Enforce total-power safety: RGB ≤ 300, RGBW ≤ 400."""
    n = len(vals)
    total = sum(int(v) for v in vals)
    if n == 3 and total > 300:
        raise ValueError("The values of RGB (R+G+B) must not exceed 300%.")
    if n == 4 and total > 400:
        raise ValueError("The values of RGBW (R+G+B+W) must not exceed 400%.")


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
        """Create a new device."""
        self._ble_device = _mk_ble_device(ble_device)
        self._logger = logging.getLogger(self._ble_device.address.replace(":", "-"))
        self._advertisement_data = advertisement_data
        self._client: BleakClientWithServiceCache | None = None
        self._disconnect_timer: asyncio.TimerHandle | None = None
        self._operation_lock: asyncio.Lock = asyncio.Lock()
        self._read_char: BleakGATTCharacteristic | None = None
        self._write_char: BleakGATTCharacteristic | None = None
        self._connect_lock: asyncio.Lock = asyncio.Lock()
        self._expected_disconnect = False
        self.loop = asyncio.get_running_loop()
        # Fan-out list for temporary notify subscribers (e.g., CLI probes)
        self._extra_notify_callbacks: list[
            Callable[[BleakGATTCharacteristic, bytearray], None]
        ] = []
        assert self._model_name is not None

    # Base methods

    def set_log_level(self, level: int | str) -> None:
        """Set log level."""
        if isinstance(level, str):
            # default INFO
            level = logging._nameToLevel.get(level, 20)
        self._logger.setLevel(level)

    def set_ble_device_and_advertisement_data(
        self, ble_device: BLEDevice, advertisement_data: AdvertisementData
    ) -> None:
        """Set the ble device."""
        self._ble_device = ble_device
        self._advertisement_data = advertisement_data

# ────────────────────────────────────────────────────────────────
# class BaseDevice
# @property
# ────────────────────────────────────────────────────────────────
   
    @property
    def current_msg_id(self) -> tuple[int, int]:
        """Get current message id."""
        return self._msg_id

    def get_next_msg_id(self) -> tuple[int, int]:
        """Get next message id."""
        self._msg_id = commands.next_message_id(self._msg_id)
        return self._msg_id
    
    @property
    def colors(self) -> dict[str, int]:
        """Return the colors."""
        return self._colors

    @property
    def address(self) -> str:
        """Return the address."""
        return self._ble_device.address

    @property
    def name(self) -> str:
        """Get the name of the device."""
        if hasattr(self._ble_device, "name"):
            return self._ble_device.name or self._ble_device.address
        return self._ble_device.address

    @property
    def rssi(self) -> int | None:
        """Get the rssi of the device."""
        if self._advertisement_data:
            return self._advertisement_data.rssi
        return None

# ────────────────────────────────────────────────────────────────
# class BaseDevice
# @_classproperty
# ────────────────────────────────────────────────────────────────

    @_classproperty
    def model_name(cls) -> str | None:  # type: ignore[override]
        """Get the model of the device."""
        return cls._model_name

    @_classproperty
    def model_codes(cls) -> list[str]:  # type: ignore[override]
        """Return the model codes."""
        return cls._model_codes
    
    
# ────────────────────────────────────────────────────────────────
# Command methods intern
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# set_color_brightness to async_set_color_brightness  in class BaseDevice
# ────────────────────────────────────────────────────────────────   

    async def async_set_color_brightness(
        self,
        brightness: Annotated[int, typer.Argument(min=0, max=140)],
        color: str | int = 0,
    ) -> None:
        """Set brightness of a color."""
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

# ────────────────────────────────────────────────────────────────
# set_brightnes to async_set_brightnes in class BaseDevice
# ────────────────────────────────────────────────────────────────   

    async def async_set_brightness(
        self, brightness: Annotated[int, typer.Argument(min=0, max=140)]
    ) -> None:
        """Set overall brightness (maps to the default color channel)."""
        await self.async_set_color_brightness(brightness=brightness, color=0)

# ────────────────────────────────────────────────────────────────
# set_manual_mode to async_set_manual_mode in class BaseDevice
# ────────────────────────────────────────────────────────────────   

    async def async_set_manual_mode(self) -> None:
        """Switch to manual mode by sending a manual mode command."""
        for color_name in self._colors:
            await self.async_set_color_brightness(100, color_name)
 



# ────────────────────────────────────────────────────────────────
# Command methods ctl
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# turn_on to get_turn_on in class BaseDevice
# ────────────────────────────────────────────────────────────────   

    async def get_turn_on(self) -> None:
        """Turn on light."""
        for color_name in self._colors:
            await self.async_set_color_brightness(100, color_name)

# ────────────────────────────────────────────────────────────────
# turn_off to gget_turn_off in class BaseDevice
# ────────────────────────────────────────────────────────────────

    async def get_turn_off(self) -> None:
        """Turn off light."""
        for color_name in self._colors:
            await self.async_set_color_brightness(0, color_name)

# ────────────────────────────────────────────────────────────────
# add_setting to get_add_setting in class BaseDevice
# ────────────────────────────────────────────────────────────────

    async def get_add_setting(
        self,
        sunrise: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
        sunset: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
        max_brightness: Annotated[int, typer.Option(max=100, min=0)] = 100,
        ramp_up_in_minutes: Annotated[int, typer.Option(min=0, max=150)] = 0,
        weekdays: Annotated[list[WeekdaySelect], typer.Option()] = [
            WeekdaySelect.everyday
        ],
    ) -> None:
        """Add an automation setting to the light."""
        cmd = commands.create_add_auto_setting_command(
            self.get_next_msg_id(),
            sunrise.time(),
            sunset.time(),
            (max_brightness, 255, 255),
            ramp_up_in_minutes,
            encode_selected_weekdays(weekdays),
        )
        await self._send_command(cmd, 3)

# ────────────────────────────────────────────────────────────────
# rgb_brightness to get_rgb_brightness in class BaseDevice
# ────────────────────────────────────────────────────────────────

    async def get_rgb_brightness(
        self, brightness: Annotated[Sequence[int], typer.Argument()]
    ) -> None:
        """
        Set per-channel brightness. Accepts 1, 3, or 4 values:
          • 1 → broadcast to R,G,B
          • 3 → R, G, B
          • 4 → R, G, B, W
        Enforces per-channel range (0..140) and total caps (RGB≤300, RGBW≤400).
        """
        vals = [int(v) for v in brightness]

        if len(vals) == 1:
            # replicate across RGB; to include W on RGBW models, change to: [v,v,v,v]
            vals = [vals[0], vals[0], vals[0]]
        elif len(vals) in (3, 4):
            pass
        else:
            raise ValueError("Provide either 1 value, or 3 (R G B), or 4 (R G B W).")

        # per-channel bounds
        for v in vals:
            if not (0 <= v <= 140):
                raise ValueError("Each channel must be between 0 and 140.")

        # total-power safety
        _check_rgb_totals(vals)

        # send values to channels in order (0=R,1=G,2=B,3=W)
        for chan_index, chan_value in enumerate(vals):
            await self.async_set_color_brightness(brightness=chan_value, color=chan_index)

# ────────────────────────────────────────────────────────────────
# add_rgb_setting to get_add_rgb_setting in class BaseDevice
# ────────────────────────────────────────────────────────────────

    async def get_add_rgb_setting(
        self,
        sunrise: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
        sunset: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
        max_brightness: Annotated[tuple[int, int, int], typer.Option()] = (
            100,
            100,
            100,
        ),
        ramp_up_in_minutes: Annotated[int, typer.Option(min=0, max=150)] = 0,
        weekdays: Annotated[list[WeekdaySelect], typer.Option()] = [
            WeekdaySelect.everyday
        ],
    ) -> None:
        """Add an automation setting to the RGB light."""
        cmd = commands.create_add_auto_setting_command(
            self.get_next_msg_id(),
            sunrise.time(),
            sunset.time(),
            max_brightness,
            ramp_up_in_minutes,
            encode_selected_weekdays(weekdays),
        )
        await self._send_command(cmd, 3)

# ────────────────────────────────────────────────────────────────
# remove_setting to get_remove_setting in class BaseDevice
# ────────────────────────────────────────────────────────────────

    async def get_remove_setting(
        self,
        sunrise: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
        sunset: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
        ramp_up_in_minutes: Annotated[int, typer.Option(min=0, max=150)] = 0,
        weekdays: Annotated[list[WeekdaySelect], typer.Option()] = [
            WeekdaySelect.everyday
        ],
    ) -> None:
        """Remove an automation setting from the light."""
        cmd = commands.create_delete_auto_setting_command(
            self.get_next_msg_id(),
            sunrise.time(),
            sunset.time(),
            ramp_up_in_minutes,
            encode_selected_weekdays(weekdays),
        )
        await self._send_command(cmd, 3)

# ────────────────────────────────────────────────────────────────
# reset_settings to get_reset_settings in class BaseDevice
# ────────────────────────────────────────────────────────────────

    async def get_reset_settings(self) -> None:
        """Remove all automation settings from the light."""
        cmd = commands.create_reset_auto_settings_command(self.get_next_msg_id())
        await self._send_command(cmd, 3)

# ────────────────────────────────────────────────────────────────
# enable_auto_mode to get_enable_auto_mode in class BaseDevice
# ────────────────────────────────────────────────────────────────

    async def get_enable_auto_mode(self) -> None:
        """Enable auto mode of the light."""
        switch_cmd = commands.create_switch_to_auto_mode_command(self.get_next_msg_id())
        time_cmd = commands.create_set_time_command(self.get_next_msg_id())
        await self._send_command(switch_cmd, 3)
        await self._send_command(time_cmd, 3)

# ────────────────────────────────────────────────────────────────
# # Bluetooth methods
# ────────────────────────────────────────────────────────────────  

    async def _send_command(
        self, commands: list[bytes] | bytes | bytearray, retry: int | None = None
    ) -> None:
        """Send command to device and read response."""
        await self._ensure_connected()
        if not isinstance(commands, list):
            commands = [commands]
        await self._send_command_while_connected(commands, retry)

    async def _send_command_while_connected(
        self, commands: list[bytes], retry: int | None = None
    ) -> None:
        """Send command to device and read response."""
        self._logger.debug(
            "%s: Sending commands %s",
            self.name,
            [command.hex() for command in commands],
        )
        if self._operation_lock.locked():
            self._logger.debug(
                "%s: Operation already in progress, waiting for it to complete; RSSI: %s",
                self.name,
                self.rssi,
            )
        async with self._operation_lock:
            try:
                await self._send_command_locked(commands)
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

    @retry_bluetooth_connection_error(DEFAULT_ATTEMPTS)
    async def _send_command_locked(self, commands: list[bytes]) -> None:
        """Send command to device and read response."""
        try:
            await self._execute_command_locked(commands)
        except BleakDBusError as ex:
            # Disconnect so we can reset state and try again
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
            # Disconnect so we can reset state and try again
            self._logger.debug(
                "%s: RSSI: %s; Disconnecting due to error: %s", self.name, self.rssi, ex
            )
            await self._execute_disconnect()
            raise

    async def _execute_command_locked(self, commands: list[bytes]) -> None:
        """Execute command and read response."""
        assert self._client is not None  # nosec
        if not self._read_char:
            raise CharacteristicMissingError("Read characteristic missing")
        if not self._write_char:
            raise CharacteristicMissingError("Write characteristic missing")
        for command in commands:
            await self._client.write_gatt_char(self._write_char, command, False)

    def _notification_handler(
        self, _sender: BleakGATTCharacteristic, data: bytearray
    ) -> None:
        """Handle notification responses."""
        # Keep debug-level to avoid log spam.
        self._logger.debug(
            "%s: Notification received: %s", self.name, data.hex(" ").upper()
        )

        # Fan-out to any temporary listeners (e.g. CLI probes).
        for cb in tuple(self._extra_notify_callbacks):
            try:
                cb(_sender, data)
            except Exception:
                self._logger.debug(
                    "%s: notify callback raised", self.name, exc_info=True
                )

    # Temporary notify listener management
    def add_notify_callback(
        self, cb: Callable[[BleakGATTCharacteristic, bytearray], None]
    ) -> None:
        if cb not in self._extra_notify_callbacks:
            self._extra_notify_callbacks.append(cb)

    def remove_notify_callback(
        self, cb: Callable[[BleakGATTCharacteristic, bytearray], None]
    ) -> None:
        try:
            self._extra_notify_callbacks.remove(cb)
        except ValueError:
            pass

    def _disconnected(self, client: BleakClientWithServiceCache) -> None:
        """Disconnected callback."""
        if self._expected_disconnect:
            self._logger.debug(
                "%s: Disconnected from device; RSSI: %s", self.name, self.rssi
            )
            return
        self._logger.warning(
            "%s: Device unexpectedly disconnected; RSSI: %s",
            self.name,
            self.rssi,
        )

    def _resolve_characteristics(self, services: BleakGATTServiceCollection) -> bool:
        """Resolve characteristics."""
        # TX (notify/read)
        for characteristic in [UART_TX_CHAR_UUID]:
            if char := services.get_characteristic(characteristic):
                self._read_char = char
                break
        # RX (write)
        for characteristic in [UART_RX_CHAR_UUID]:
            if char := services.get_characteristic(characteristic):
                self._write_char = char
                break
        return bool(self._read_char and self._write_char)

    async def _ensure_connected(self) -> None:
        """Ensure connection to device is established."""
        if self._connect_lock.locked():
            self._logger.debug(
                "%s: Connection already in progress, waiting for it to complete; RSSI: %s",
                self.name,
                self.rssi,
            )
        if self._client and self._client.is_connected:
            self._reset_disconnect_timer()
            return
        async with self._connect_lock:
            # Check again while holding the lock
            if self._client and self._client.is_connected:
                self._reset_disconnect_timer()
                return
            self._logger.debug("%s: Connecting; RSSI: %s", self.name, self.rssi)

            # Prefer passing a BLEDevice so the connector can safely reference .address.
            # Fall back to the address string only if we don't actually have a BLEDevice.
            if isinstance(self._ble_device, BLEDevice):
                device_arg: Union[str, BLEDevice] = self._ble_device
                kwargs = {
                    "use_services_cache": True,
                    "ble_device_callback": lambda: self._ble_device,  # type: ignore[return-value]
                }
            else:
                device_arg = self._ble_device.address  # string MAC
                kwargs = {
                    "use_services_cache": True,
                }

            client = await establish_connection(
                BleakClientWithServiceCache,
                device_arg,
                self.name,
                self._disconnected,
                **kwargs,
            )

            self._logger.debug("%s: Connected; RSSI: %s", self.name, self.rssi)
            # services may not be pre-populated
            services = client.services or await client.get_services()
            resolved = self._resolve_characteristics(services)

            self._client = client
            self._reset_disconnect_timer()

            if resolved and self._read_char:
                self._logger.debug(
                    "%s: Subscribe to notifications; RSSI: %s", self.name, self.rssi
                )
                await client.start_notify(self._read_char, self._notification_handler)  # type: ignore
            else:
                raise CharacteristicMissingError("Failed to resolve UART characteristics")

    # Public helpers

    async def connect(self) -> BleakClientWithServiceCache:
        """Establish a connection (idempotent) and return the Bleak client."""
        await self._ensure_connected()
        assert self._client is not None  # nosec
        return self._client

    @property
    def client(self) -> BleakClientWithServiceCache | None:
        """Return the current Bleak client (None if not connected)."""
        return self._client

    def _reset_disconnect_timer(self) -> None:
        """Reset disconnect timer."""
        if self._disconnect_timer:
            self._disconnect_timer.cancel()
        self._expected_disconnect = False
        self._disconnect_timer = self.loop.call_later(
            DISCONNECT_DELAY, self._disconnect
        )

    async def disconnect(self) -> None:
        """Disconnect."""
        self._logger.debug("%s: Disconnecting", self.name)
        await self._execute_disconnect()

    async def _execute_disconnect(self) -> None:
        """Execute disconnection."""
        async with self._connect_lock:
            read_char = self._read_char
            client = self._client
            self._expected_disconnect = True
            self._client = None
            self._read_char = None
            self._write_char = None

            # Clear any temporary listeners; they should not linger across sessions.
            self._extra_notify_callbacks.clear()

            if client and client.is_connected:
                if read_char:
                    # On Windows/WinRT, stop_notify can raise KeyError if someone else
                    # (e.g., a CLI probe) already removed the callback token. Be lenient.
                    try:
                        await client.stop_notify(read_char)
                    except Exception:
                        self._logger.debug(
                            "%s: stop_notify failed (already stopped?)", self.name, exc_info=True
                        )
                await client.disconnect()

    def _disconnect(self) -> None:
        """Disconnect from device."""
        self._disconnect_timer = None
        asyncio.create_task(self._execute_timed_disconnect())

    async def _execute_timed_disconnect(self) -> None:
        """Execute timed disconnection."""
        self._logger.debug(
            "%s: Disconnecting after timeout of %s",
            self.name,
            DISCONNECT_DELAY,
        )
        await self._execute_disconnect()

"""Expose the doser Typer app for mounting under chihirosctl.

This file re-exports the Typer `app` defined in
`custom_components.chihiros.chihiros_doser_control/device/doser_device.py`
and *extends the same app instance* with a few extra helper commands.
"""

from __future__ import annotations


import asyncio
from typing import List

import typer
from typing_extensions import Annotated
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakDeviceNotFoundError, BleakError

# Import the existing doser CLI app and (optionally) the device class
from .doser_device import app as app
from .doser_device import DoserDevice, _resolve_ble_or_fail # used by the extra helpers below
from datetime import datetime, time, timedelta

from ....helper.weekday_encoding import (
    WeekdaySelect,
    encode_selected_weekdays,
)

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _parse_params_tokens(tokens: List[str]) -> List[int]:
    """
    Accept decimal or hex tokens:
      - decimal: 10, 255
      - hex: 0x0A, 0Ah
    """
    out: List[int] = []
    for t in tokens:
        s = t.strip().lower()
        if s.startswith("0x"):
            v = int(s, 16)
        elif s.endswith("h") and all(c in "0123456789abcdef" for c in s[:-1]):
            v = int(s[:-1], 16)
        else:
            v = int(s, 10)
        if not (0 <= v <= 255):
            raise typer.BadParameter(f"Parameter byte out of range 0..255: {t}")
        out.append(v)
    return out

# ────────────────────────────────────────────────────────────────
# Simple READ helpers (call into the DoserDevice class)
# ────────────────────────────────────────────────────────────────

@app.command(name="read-dosing-auto")
def read_dosing_auto(
    device_address: Annotated[str, typer.Argument(help="BLE MAC, e.g. AA:BB:CC:DD:EE:FF")],
    ch_id: Annotated[int | None, typer.Option(help="Channel 0..3; omit for all")] = None,
    timeout_s: Annotated[float, typer.Option(help="Timeout seconds", min=0.1)] = 2.0,
) -> None:
    print(f"Connect to device {device_address} ....")
    async def run():
        dd = DoserDevice(device_address)
        try:
            await dd.read_dosing_pump_auto_settings(ch_id=ch_id, timeout_s=timeout_s)
        finally:
            await dd.disconnect()
    import asyncio as _asyncio
    _asyncio.run(run())


@app.command(name="read-dosing-container")
def read_dosing_container(
    device_address: Annotated[str, typer.Argument(help="BLE MAC, e.g. AA:BB:CC:DD:EE:FF")],
    ch_id: Annotated[int | None, typer.Option(help="Channel 0..3; omit for all")] = None,
    timeout_s: Annotated[float, typer.Option(help="Timeout seconds", min=0.1)] = 2.0,
) -> None:
    print(f"Connect to device {device_address} ....")
    async def run():
        dd = DoserDevice(device_address)
        try:
            await dd.read_dosing_container_status(ch_id=ch_id, timeout_s=timeout_s)
        finally:
            await dd.disconnect()
    import asyncio as _asyncio
    _asyncio.run(run())

@app.command("set-dosing-pump-manuell-ml")
def cli_set_dosing_pump_manuell_ml(
    device_address: Annotated[str, typer.Argument(help="BLE MAC, e.g. AA:BB:CC:DD:EE:FF")],
    ch_id: Annotated[int, typer.Option("--ch-id", help="Channel 0..3", min=0, max=3)],
    ch_ml: Annotated[float, typer.Option("--ch-ml", help="Dose (mL)", min=0.2, max=999.9)],
):
    """Immediate one-shot dose."""
    print(f"Connect to device {device_address} ....")
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            await dd.set_dosing_pump_manuell_ml(ch_id, ch_ml)
        except (BleakDeviceNotFoundError, BleakError, OSError) as ex:
            _handle_connect_errors(ex)
        finally:
            if dd:
                await dd.disconnect()
    asyncio.run(run())


@app.command("enable-auto-mode-dosing-pump")
def cli_enable_auto_mode_dosing_pump(
    device_address: Annotated[str, typer.Argument(help="BLE MAC")],
    ch_id: Annotated[int, typer.Option("--ch-id", help="Channel 0..3", min=0, max=3)] = 0,
):
    """Explicitly switch the doser channel to auto mode and sync time."""
    print(f"Connect to device {device_address} ....")
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            await dd.enable_auto_mode_dosing_pump(ch_id)
        except (BleakDeviceNotFoundError, BleakError, OSError) as ex:
            _handle_connect_errors(ex)
        finally:
            if dd:
                await dd.disconnect()
    asyncio.run(run())


@app.command("add-setting-dosing-pump")
def cli_add_setting_dosing_pump(
    device_address: Annotated[str, typer.Argument(help="BLE MAC")],
    performance_time: Annotated[datetime, typer.Argument(formats=["%H:%M"], help="HH:MM")],
    ch_id: Annotated[int, typer.Option("--ch-id", help="Channel 0..3", min=0, max=3)],
    ch_ml: Annotated[float, typer.Option("--ch-ml", help="Daily dose mL", min=0.2, max=999.9)],
    weekdays: Annotated[List[WeekdaySelect], typer.Option(
        "--weekdays", "-w", help="Repeat days; can be passed multiple times", case_sensitive=False
    )] = [WeekdaySelect.everyday],
):
    """Add a 24h schedule entry at time with amount, on selected weekdays."""
    print(f"Connect to device {device_address} ....")
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            mask = encode_selected_weekdays(weekdays)
            tenths = int(round(ch_ml * 10))
            await dd.add_setting_dosing_pump(performance_time.time(), ch_id, mask, tenths)
        except (BleakDeviceNotFoundError, BleakError, OSError) as ex:
            _handle_connect_errors(ex)
        finally:
            if dd:
                await dd.disconnect()
    asyncio.run(run())


@app.command("raw-dosing-pump")
def cli_raw_dosing_pump(
    device_address: Annotated[str, typer.Argument(help="BLE MAC")],
    cmd_id: Annotated[int, typer.Option("--cmd-id", help="Command (e.g. 165)")],
    mode: Annotated[int, typer.Option("--mode", help="Mode (e.g. 27)")],
    # positional params come before any defaulted option to avoid Typer errors
    params: Annotated[List[int], typer.Argument(help="Parameter list, e.g. 0 0 14 2 0 0")],
    repeats: Annotated[int, typer.Option("--repeats", help="Send frame N times", min=1)] = 3,
):
    """Send a raw A5 frame: [cmd, 1, len, msg_hi, msg_lo, mode, *params, checksum]."""
    print(f"Connect to device {device_address} ....")
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            await dd.raw_dosing_pump(cmd_id, mode, params, repeats)
        except (BleakDeviceNotFoundError, BleakError, OSError) as ex:
            _handle_connect_errors(ex)
        finally:
            if dd:
                await dd.disconnect()
    asyncio.run(run())
 
if __name__ == "__main__":
    try:
        app()
    except asyncio.CancelledError:
        pass
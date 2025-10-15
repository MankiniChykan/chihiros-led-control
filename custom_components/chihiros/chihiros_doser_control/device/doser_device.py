# custom_components/chihiros/chihiros_doser_control/device/doser_device.py
"""Chihiros Doser BLE device wrapper + minimal Typer app.

This module provides:
- DoserDevice: a thin BLE helper around the Nordic UART service
  used by the Chihiros dosing pump family.
- A tiny Typer `app` (the parent CLI) which chihirosdoserctl.py
  mounts and extends with more commands.
- WeekdaySelect + encode_selected_weekdays utilities (kept local
  to avoid cross-package deps).

It is designed to match the on-wire behaviour observed in logs:
  - A5/165 family for dosing writes:
      * mode 0x1B (27): manual dose & weekly row
      * mode 0x20 (32): activate/flags (auto mode switch)
      * mode 0x15 (21): time-of-day helper seen alongside weekly
      * mode 0x16 (22): interval/repeats (optional)
  - 90/09: time set (YY,MM,DOW,HH,MM,SS)
  - 5B/LED family for daily totals (mL×10 or ×100 auto-detected)

All frames avoid 0x5A in payload bytes and rotate message IDs
away from 0x5A in either byte, with XOR checksum over the body.

The public methods here are intentionally small and opinionated:
  - connect()/disconnect()
  - start/stop notify on TX; add/remove notify callbacks
  - set_dosing_pump_manuell_ml()
  - enable_auto_mode_dosing_pump()
  - read_dosing_pump_auto_settings()   (passive decode printer)
  - read_dosing_container_status()     (passive decode printer)
  - raw_dosing_pump()                  (raw A5 sender)

The high-reliability write cadence (mode32 → mode27 → mode22 with
ACK/heartbeat waits) lives in protocol.program_channel_weekly_with_interval()
and is used by the CLI command in chihirosdoserctl.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime  # kept for external callers / parity
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple

# Windows CLI nicety; harmless elsewhere
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

# Typing-time only; avoid importing bleak / typer at runtime unless needed
if TYPE_CHECKING:
    from bleak import BleakClient  # type: ignore
else:
    BleakClient = Any

_LOG = logging.getLogger("chihiros.doser")
P = print  # replaced by typer.echo when CLI wiring is active


# ────────────────────────────────────────────────────────────────
# Helpers to detect HA and lazy-import heavy deps
# ────────────────────────────────────────────────────────────────
def _is_ha_runtime() -> bool:
    try:
        import homeassistant  # type: ignore
        return True
    except Exception:
        return False


def _bleak():
    from bleak import BleakClient, BleakScanner  # type: ignore
    return BleakClient, BleakScanner


def _protocol():
    from ..protocol import (  # type: ignore
        UART_SERVICE, UART_RX, UART_TX,
        CMD_MANUAL_DOSE, MODE_MANUAL_DOSE,
        dose_ml,
        build_totals_query_5b, parse_totals_frame,
        parse_log_blob, decode_records, build_device_state, to_ctl_lines,
        send_and_await_ack,
    )
    return {
        "UART_SERVICE": UART_SERVICE,
        "UART_RX": UART_RX,
        "UART_TX": UART_TX,
        "CMD_MANUAL_DOSE": CMD_MANUAL_DOSE,
        "MODE_MANUAL_DOSE": MODE_MANUAL_DOSE,
        "dose_ml": dose_ml,
        "build_totals_query_5b": build_totals_query_5b,
        "parse_totals_frame": parse_totals_frame,
        "parse_log_blob": parse_log_blob,
        "decode_records": decode_records,
        "build_device_state": build_device_state,
        "to_ctl_lines": to_ctl_lines,
        "send_and_await_ack": send_and_await_ack,
    }


def _dosingcommands():
    from ..dosingcommands import (  # type: ignore
        create_set_time_command,
        create_switch_to_auto_mode_dosing_pump_command,
        create_command_encoding_dosing_pump,
    )
    return {
        "create_set_time_command": create_set_time_command,
        "create_switch_to_auto_mode_dosing_pump_command": create_switch_to_auto_mode_dosing_pump_command,
        "create_command_encoding_dosing_pump": create_command_encoding_dosing_pump,
    }


# ────────────────────────────────────────────────────────────────
# Weekday helpers (local copy to avoid LED package dependency)
# ────────────────────────────────────────────────────────────────
class WeekdaySelect:
    name = "weekday"

    _aliases = {
        "mon": 64, "monday": 64,
        "tue": 32, "tuesday": 32,
        "wed": 16, "wednesday": 16,
        "thu": 8,  "thursday": 8,
        "fri": 4,  "friday": 4,
        "sat": 2,  "saturday": 2,
        "sun": 1,  "sunday": 1,
        "everyday": 127, "daily": 127, "all": 127
    }

    def convert(self, value: str, *_args):
        v = (value or "").strip().lower()
        if v in self._aliases:
            return self._aliases[v]
        raise ValueError(f"Invalid weekday value: {value}")


def encode_selected_weekdays(values: Iterable[str | int]) -> int:
    """Return a 7-bit mask. Accepts strings (mon..sun,everyday) or ints."""
    mask = 0
    ws = WeekdaySelect()
    for v in values:
        if isinstance(v, int):
            mask |= (v & 0x7F)
        else:
            mask |= ws.convert(str(v)) & 0x7F
    return mask or 127  # default everyday


# ────────────────────────────────────────────────────────────────
# BLE helper
# ────────────────────────────────────────────────────────────────
async def _resolve_ble_or_fail(address_or_name: str) -> "BleakClient":
    """Resolve a BLE device by MAC/name and return a BleakClient."""
    BleakClient, BleakScanner = _bleak()
    s = (address_or_name or "").strip()
    # If it looks like a MAC, try it directly
    if ":" in s and len(s) >= 17:
        return BleakClient(s)
    # Else scan by name prefix then exact address
    devices = await BleakScanner.discover(timeout=6.0)
    for d in devices:
        if getattr(d, "name", None) and s.lower() in d.name.lower():
            return BleakClient(d)
    for d in devices:
        if s.lower() == (getattr(d, "address", "") or "").lower():
            return BleakClient(d)
    # Prefer not to import typer in HA just to raise an error
    if _is_ha_runtime():
        raise RuntimeError(f"BLE device not found: {address_or_name}")
    try:
        import typer  # type: ignore
        raise typer.BadParameter(f"BLE device not found: {address_or_name}")
    except Exception:
        raise RuntimeError(f"BLE device not found: {address_or_name}")


NotifyCallback = Callable[[str, bytearray], None]


@dataclass
class _Callbacks:
    notify: Set[NotifyCallback]


class DoserDevice:
    """Minimal BLE client for the doser using Nordic UART."""

    def __init__(self, client: "BleakClient"):
        self._client = client
        self._callbacks = _Callbacks(notify=set())
        self._log = _LOG

    # Public: logging
    def set_log_level(self, level: str = "INFO") -> None:
        self._log.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Public: connect / disconnect
    async def connect(self) -> "BleakClient":
        prot = _protocol()
        UART_SERVICE = prot["UART_SERVICE"]
        if not self._client.is_connected:
            await self._client.connect()
            # Confirm UART service is present (best-effort)
            try:
                svcs = await self._client.get_services()
                if UART_SERVICE not in {s.uuid.upper() for s in svcs.services}:
                    self._log.debug("UART service not listed, continuing anyway.")
            except Exception:
                pass
        return self._client

    async def disconnect(self) -> None:
        try:
            if self._client.is_connected:
                await self._client.disconnect()
        except Exception:
            pass

    # Public: notifications on TX
    async def start_notify_tx(self) -> None:
        UART_TX = _protocol()["UART_TX"]

        def _cb(handle: int, data: bytearray):
            for fn in list(self._callbacks.notify):
                try:
                    fn("tx", data)
                except Exception:
                    self._log.exception("notify-callback error")

        await self._client.start_notify(UART_TX, _cb)

    async def stop_notify_tx(self) -> None:
        UART_TX = _protocol()["UART_TX"]
        try:
            await self._client.stop_notify(UART_TX)
        except Exception:
            pass

    def add_notify_callback(self, cb: NotifyCallback) -> None:
        self._callbacks.notify.add(cb)

    def remove_notify_callback(self, cb: NotifyCallback) -> None:
        self._callbacks.notify.discard(cb)

    # Public: writes
    async def set_dosing_pump_manuell_ml(self, ch_id: int, ml: float | int | str) -> None:
        """Immediate one-shot dose using 25.6+0.1 encoding (A5/27)."""
        dose_ml = _protocol()["dose_ml"]
        client = await self.connect()
        await dose_ml(client, channel_1based=(int(ch_id) + 1), ml=ml)

    async def enable_auto_mode_dosing_pump(self, ch_id: int) -> None:
        """
        Switch channel to auto mode and sync device time.
        Sequence:
          - 90/09 (set time)
          - 165/32 [ch, catch_up=0, active=1] (await ACK 0x07)
        """
        prot = _protocol()
        cmds = _dosingcommands()
        UART_RX = prot["UART_RX"]
        CMD_MANUAL_DOSE = prot["CMD_MANUAL_DOSE"]
        send_and_await_ack = prot["send_and_await_ack"]
        create_set_time_command = cmds["create_set_time_command"]
        create_switch_to_auto_mode_dosing_pump_command = cmds["create_switch_to_auto_mode_dosing_pump_command"]

        client = await self.connect()

        # Set time
        msg_hi, msg_lo = 0, 0
        frame_time = create_set_time_command((msg_hi, msg_lo))
        await client.write_gatt_char(UART_RX, frame_time, response=True)

        # Switch to auto
        msg_hi, msg_lo = 0, 1
        frame_auto = create_switch_to_auto_mode_dosing_pump_command((msg_hi, msg_lo), int(ch_id), 0, 1)
        await send_and_await_ack(
            client,
            bytes(frame_auto),
            expect_cmd=CMD_MANUAL_DOSE,
            expect_ack_mode=0x07,
            ack_timeout_ms=400,
            heartbeat_min_ms=300,
            heartbeat_max_ms=900,
            retries=1,
        )

    async def raw_dosing_pump(self, cmd_id: int, mode: int, params: List[int], repeats: int = 1) -> None:
        """Send a raw A5 frame."""
        UART_RX = _protocol()["UART_RX"]
        create_command = _dosingcommands()["create_command_encoding_dosing_pump"]
        client = await self.connect()
        msg_hi, msg_lo = 0, 0
        for _ in range(int(repeats)):
            frame = create_command(cmd_id, mode, (msg_hi, msg_lo), [int(p) & 0xFF for p in params])
            await client.write_gatt_char(UART_RX, frame, response=True)
            msg_lo = (msg_lo + 1) & 0xFF

    # Public: “read” helpers (passive parse/printer)
    async def read_dosing_pump_auto_settings(self, *, ch_id: int | None = None, timeout_s: float = 2.0) -> None:
        """
        Passive listener that prints any A5 frames we can interpret into a CTL-style
        summary (enable flags, time HH:MM, weekdays, and amount mL).
        """
        prot = _protocol()
        parse_log_blob = prot["parse_log_blob"]
        decode_records = prot["decode_records"]
        build_device_state = prot["build_device_state"]
        to_ctl_lines = prot["to_ctl_lines"]

        client = await self.connect()
        buf: list[str] = []

        def _cb(_kind: str, payload: bytearray) -> None:
            try:
                b = bytes(payload)
                buf.append(f'["{b[0] if b else 0}", "{b[5] if len(b)>5 else 0}", {list(b[6:-1])}]')
            except Exception:
                pass

        await self.start_notify_tx()
        self.add_notify_callback(_cb)
        try:
            await asyncio.sleep(max(0.3, float(timeout_s)))
            text = "\n".join(buf)
            recs = parse_log_blob(text)
            dec = decode_records(recs)
            state = build_device_state(dec)
            lines = to_ctl_lines(state)
            if ch_id is not None:
                prefix = f"ch{int(ch_id)}."
                lines = [ln for ln in lines if ln.startswith(prefix)]
            if lines:
                for ln in lines:
                    P(ln)
            else:
                P("No auto schedule information observed in the window.")
        finally:
            self.remove_notify_callback(_cb)
            await self.stop_notify_tx()

    async def read_dosing_container_status(self, *, ch_id: int | None = None, timeout_s: float = 2.0) -> None:
        """
        Passive printer. If firmware emits container / tank status frames on TX,
        this function will show them as raw hex. (We keep this generic until
        we have stable decode samples across models.)
        """
        client = await self.connect()
        got: list[str] = []

        def _cb(_kind: str, payload: bytearray) -> None:
            got.append(bytes(payload).hex(" ").upper())

        await self.start_notify_tx()
        self.add_notify_callback(_cb)
        try:
            await asyncio.sleep(max(0.3, float(timeout_s)))
            if got:
                P("Container/Tank notifications (raw):")
                for line in got:
                    P(f"  {line}")
            else:
                P("No container/tank notifications observed.")
        finally:
            self.remove_notify_callback(_cb)
            await self.stop_notify_tx()


# ────────────────────────────────────────────────────────────────
# Optional Typer app (CLI only; never during Home Assistant runtime)
# ────────────────────────────────────────────────────────────────
if not _is_ha_runtime():
    try:
        import typer  # type: ignore

        app = typer.Typer(help="Chihiros Doser (base app)")

        @app.callback()
        def _root():
            """Base app for chihirosdoserctl.py to extend."""
            pass

        P = typer.echo  # nicer CLI printing
    except Exception:
        app = None  # type: ignore
else:
    app = None  # type: ignore

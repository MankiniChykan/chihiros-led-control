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

Public methods (opinionated but small):
  - connect()/disconnect()
  - start/stop notify on TX; add/remove notify callbacks
  - read_burst()                           <-- NEW
  - query_totals_with_burst()              <-- NEW
  - dose_then_capture_totals()             <-- NEW
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
import time
from dataclasses import dataclass, field
from datetime import datetime  # kept for external callers / parity
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple, Dict

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
# Burst capture primitives
# ────────────────────────────────────────────────────────────────
@dataclass
class BurstResult:
    """Result of a single UART_TX notification burst."""
    started_at: float
    ended_at: float
    frames: List[bytes] = field(default_factory=list)
    total_bytes: int = 0


class _BurstCollector:
    """Collect UART_TX notify frames until idle or max duration reached."""

    def __init__(self, idle_ms: int = 450, max_ms: int = 5500, log: Optional[logging.Logger] = None):
        self.idle_ms = int(idle_ms)
        self.max_ms = int(max_ms)
        self._log = log or _LOG
        self._started = 0.0
        self._last = 0.0
        self._frames: List[bytes] = []
        self._total = 0
        loop = asyncio.get_event_loop()
        self._done: asyncio.Future[BurstResult] = loop.create_future()
        self._watch: Optional[asyncio.Task] = None

    def feed(self, frame: bytes) -> None:
        now = time.monotonic()
        if not self._frames:
            self._started = now
            # arm watchdog on first frame
            self._watch = asyncio.create_task(self._watchdog())
        self._last = now
        self._frames.append(frame)
        self._total += len(frame)

    async def _watchdog(self) -> None:
        start = self._started or time.monotonic()
        # periodic check for idle or max window
        while True:
            await asyncio.sleep(self.idle_ms / 1000.0)
            now = time.monotonic()
            if self._frames and (now - self._last) * 1000.0 >= self.idle_ms:
                break
            if (now - start) * 1000.0 >= self.max_ms:
                break
        if not self._done.done():
            self._done.set_result(BurstResult(
                started_at=self._started or start,
                ended_at=time.monotonic(),
                frames=self._frames[:],
                total_bytes=self._total,
            ))

    async def wait(self) -> BurstResult:
        return await self._done


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
    """Minimal BLE client for the doser using Nordic UART with burst capture."""

    def __init__(self, client: "BleakClient"):
        self._client = client
        self._callbacks = _Callbacks(notify=set())
        self._log = _LOG
        self._active_burst: Optional[_BurstCollector] = None

    # Public: logging
    def set_log_level(self, level: str = "INFO") -> None:
        self._log.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Internal: ensure services (refresh=True to bust stale cache after proxy swaps)
    async def _ensure_services(self, refresh: bool = True) -> None:
        prot = _protocol()
        UART_SERVICE = prot["UART_SERVICE"]
        svcs = await self._client.get_services(refresh)
        have = {s.uuid.upper() for s in svcs.services}
        if UART_SERVICE not in have:
            raise RuntimeError("UART service not present on device (service cache stale or wrong peripheral)")

    # Public: connect / disconnect
    async def connect(self) -> "BleakClient":
        if not self._client.is_connected:
            await self._client.connect()
        # force a fresh discovery to avoid "characteristic not found" after proxy handoffs
        await self._ensure_services(refresh=True)
        return self._client

    async def disconnect(self) -> None:
        try:
            if self._client.is_connected:
                await self._client.disconnect()
        except Exception:
            pass

    # Public: notifications on TX (notify-first)
    async def start_notify_tx(self) -> None:
        UART_TX = _protocol()["UART_TX"]

        def _cb(handle: int, data: bytearray):
            # Feed burst collector if running
            bc = self._active_burst
            if bc:
                try:
                    bc.feed(bytes(data))
                except Exception:
                    # never raise from callback
                    pass
            # Fan out to user callbacks
            for fn in list(self._callbacks.notify):
                try:
                    fn("tx", data)
                except Exception:
                    self._log.exception("notify-callback error")

        # (re)ensure services in case notify is called after reconnect
        await self._ensure_services(refresh=False)
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

    # ────────────────────────────────────────────────────────────────
    # Burst capture API
    # ────────────────────────────────────────────────────────────────
    async def read_burst(self, *, idle_ms: int = 450, max_ms: int = 5500, timeout_pad_s: float = 2.0) -> BurstResult:
        """
        Collect all UART_TX notifications until the link is idle for `idle_ms`
        or until `max_ms` passes. Returns the frames and simple diagnostics.
        """
        # Arm collector
        self._active_burst = _BurstCollector(idle_ms=idle_ms, max_ms=max_ms, log=self._log)
        try:
            # Add a small timeout pad to avoid hanging if no frames arrive
            burst = await asyncio.wait_for(self._active_burst.wait(), timeout=(max_ms / 1000.0) + float(timeout_pad_s))
            return burst
        finally:
            self._active_burst = None

    # ────────────────────────────────────────────────────────────────
    # High-level helpers using burst capture
    # ────────────────────────────────────────────────────────────────
    async def query_totals_with_burst(self, *, idle_ms: int = 450, max_ms: int = 5500) -> Tuple[Optional[Dict[str, Any]], BurstResult, Dict[str, Any]]:
        """
        Start notify FIRST, send 0x5B totals probes, capture burst, parse last totals.

        Returns (totals_dict_or_none, burst, diag)
        """
        prot = _protocol()
        UART_RX = prot["UART_RX"]
        build_totals_query_5b = prot["build_totals_query_5b"]
        parse_totals_frame = prot["parse_totals_frame"]

        client = await self.connect()
        await self.start_notify_tx()
        # let CCCD settle to avoid missing the first notify on some stacks
        await asyncio.sleep(0.15)

        # Send a couple of known-good totals probes (as you already do in HA)
        probes = [
            build_totals_query_5b(seq=0x06),  # 1E probe
            build_totals_query_5b(seq=0x07),  # 22 probe
        ]
        for frame in probes:
            try:
                await client.write_gatt_char(UART_RX, frame, response=True)
                await asyncio.sleep(0.12)  # small pacing between writes
            except Exception:
                # Try one service refresh and retry once
                await self._ensure_services(refresh=True)
                await client.write_gatt_char(UART_RX, frame, response=True)
                await asyncio.sleep(0.12)

        # Capture everything the device says
        burst = await self.read_burst(idle_ms=idle_ms, max_ms=max_ms)

        # Parse: pick the last valid 0x5B totals in the burst
        best = None
        seen_cmds: List[int] = []
        for raw in burst.frames:
            if not raw:
                continue
            seen_cmds.append(raw[0])
            try:
                dec = parse_totals_frame(raw)
                if dec:
                    best = dec
            except Exception:
                continue

        diag = {
            "frames_in_burst": len(burst.frames),
            "seen_cmds": seen_cmds,
            "had_totals": best is not None,
            "burst_duration_ms": int((burst.ended_at - burst.started_at) * 1000),
            "burst_total_bytes": burst.total_bytes,
        }
        return best, burst, diag

    async def dose_then_capture_totals(
        self,
        *,
        channel_1based: int,
        ml: float | int | str,
        idle_ms: int = 450,
        max_ms: int = 6000,
        send_extra_probes: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], BurstResult, Dict[str, Any]]:
        """
        Perform a manual dose and capture the post-dose UART burst on the SAME connection.

        Returns (totals_dict_or_none, burst, diag)
        """
        prot = _protocol()
        UART_RX = prot["UART_RX"]
        dose_ml_fn = prot["dose_ml"]
        build_totals_query_5b = prot["build_totals_query_5b"]
        parse_totals_frame = prot["parse_totals_frame"]

        client = await self.connect()
        await self.start_notify_tx()
        await asyncio.sleep(0.15)

        # (optional) prime totals queries before and after the dose; some FW replies slower
        pre_probes = [build_totals_query_5b(seq=0x0D), build_totals_query_5b(seq=0x0E)]
        for frame in pre_probes:
            try:
                await client.write_gatt_char(UART_RX, frame, response=True)
                await asyncio.sleep(0.10)
            except Exception:
                await self._ensure_services(refresh=True)
                await client.write_gatt_char(UART_RX, frame, response=True)
                await asyncio.sleep(0.10)

        # Write the dose frame on the same session
        await dose_ml_fn(client, channel_1based=int(channel_1based), ml=ml)

        if send_extra_probes:
            post_probes = [build_totals_query_5b(seq=0x12), build_totals_query_5b(seq=0x13)]
            for frame in post_probes:
                try:
                    await client.write_gatt_char(UART_RX, frame, response=True)
                    await asyncio.sleep(0.12)
                except Exception:
                    await self._ensure_services(refresh=True)
                    await client.write_gatt_char(UART_RX, frame, response=True)
                    await asyncio.sleep(0.12)

        # Capture all UART_TX chatter after dose
        burst = await self.read_burst(idle_ms=idle_ms, max_ms=max_ms)

        # Parse: prefer the last totals if present
        best = None
        seen_cmds: List[int] = []
        for raw in burst.frames:
            if not raw:
                continue
            seen_cmds.append(raw[0])
            try:
                dec = parse_totals_frame(raw)
                if dec:
                    best = dec
            except Exception:
                continue

        diag = {
            "frames_in_burst": len(burst.frames),
            "seen_cmds": seen_cmds,
            "had_totals": best is not None,
            "burst_duration_ms": int((burst.ended_at - burst.started_at) * 1000),
            "burst_total_bytes": burst.total_bytes,
            "channel_1based": int(channel_1based),
            "ml": float(ml) if isinstance(ml, (int, float, str)) and str(ml).replace('.', '', 1).isdigit() else ml,
        }
        return best, burst, diag

    # ────────────────────────────────────────────────────────────────
    # Basic writes (kept for compatibility)
    # ────────────────────────────────────────────────────────────────
    async def set_dosing_pump_manuell_ml(self, ch_id: int, ml: float | int | str) -> None:
        """Immediate one-shot dose using 25.6+0.1 encoding (A5/27)."""
        dose_ml = _protocol()["dose_ml"]
        client = await self.connect()
        await self.start_notify_tx()  # harmless if caller forgets; keeps us ready
        await asyncio.sleep(0.05)
        # NOTE: upstream `dose_ml` expects 1-based channel index
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
        await self.start_notify_tx()
        await asyncio.sleep(0.05)

        # Set time
        msg_hi, msg_lo = 0, 0
        frame_time = create_set_time_command((msg_hi, msg_lo))
        await client.write_gatt_char(UART_RX, frame_time, response=True)
        await asyncio.sleep(0.10)

        # Switch to auto
        msg_hi, msg_lo = 0, 1
        frame_auto = create_switch_to_auto_mode_dosing_pump_command((msg_hi, msg_lo), int(ch_id), 0, 1)
        await send_and_await_ack(
            client,
            bytes(frame_auto),
            expect_cmd=CMD_MANUAL_DOSE,
            expect_ack_mode=0x07,
            ack_timeout_ms=600,          # slightly relaxed for proxies
            heartbeat_min_ms=250,
            heartbeat_max_ms=900,
            retries=1,
        )

    async def raw_dosing_pump(self, cmd_id: int, mode: int, params: List[int], repeats: int = 1) -> None:
        """Send a raw A5 frame."""
        UART_RX = _protocol()["UART_RX"]
        create_command = _dosingcommands()["create_command_encoding_dosing_pump"]
        client = await self.connect()
        await self.start_notify_tx()
        await asyncio.sleep(0.05)
        msg_hi, msg_lo = 0, 0
        for _ in range(int(repeats)):
            frame = create_command(cmd_id, mode, (msg_hi, msg_lo), [int(p) & 0xFF for p in params])
            await client.write_gatt_char(UART_RX, frame, response=True)
            await asyncio.sleep(0.12)
            msg_lo = (msg_lo + 1) & 0xFF

    # ────────────────────────────────────────────────────────────────
    # Passive “reads” (printers)
    # ────────────────────────────────────────────────────────────────
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
        await self.connect()
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

        app = typer.Typer(help="Chihiros Doser (base app with burst capture)")

        @app.callback()
        def _root():
            """Base app for chihirosdoserctl.py to extend."""
            pass

        @app.command("totals-burst")
        def _totals_burst(addr: str, idle_ms: int = 450, max_ms: int = 5500):
            """Query totals using 0x5B probes and burst-capture the reply."""
            async def _run():
                BleakClient, _ = _bleak()
                dev = DoserDevice(BleakClient(addr))
                totals, burst, diag = await dev.query_totals_with_burst(idle_ms=idle_ms, max_ms=max_ms)
                if totals:
                    P(f"Totals: {totals}")
                else:
                    P("No totals detected in burst.")
                P(f"Diag: {diag}")
                await dev.disconnect()
            asyncio.run(_run())

        @app.command("dose-and-capture")
        def _dose_and_capture(addr: str, ch: int, ml: float, idle_ms: int = 450, max_ms: int = 6000):
            """Send manual dose and capture totals on the same connection."""
            async def _run():
                BleakClient, _ = _bleak()
                dev = DoserDevice(BleakClient(addr))
                totals, burst, diag = await dev.dose_then_capture_totals(
                    channel_1based=int(ch), ml=ml, idle_ms=idle_ms, max_ms=max_ms
                )
                if totals:
                    P(f"Totals: {totals}")
                else:
                    P("No totals detected in post-dose burst.")
                P(f"Diag: {diag}")
                await dev.disconnect()
            asyncio.run(_run())

        P = typer.echo  # nicer CLI printing
    except Exception:
        app = None  # type: ignore
else:
    app = None  # type: ignore

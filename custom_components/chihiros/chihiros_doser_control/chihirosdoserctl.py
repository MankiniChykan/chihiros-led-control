# custom_components/chihiros/chihiros_doser_control/chihirosdoserctl.py
"""
Chihiros Doser — user-facing CLI (Typer app mount).

This module *mounts and extends* the base Typer app defined in
`device/doser_device.py`, adding CLI commands that are convenient during
development and maintenance — while staying strictly within the device’s
observed behavior.

What this file provides:
• Global `--debug` option with rich logging.
• BYTES ENCODE/DECODE helpers:
    - Pretty-print arbitrary payloads and decode LED totals if applicable.
    - Parse text/JSON logs into normalized events, then flatten to CTL lines.
• READ helpers:
    - read-dosing-auto: passive listen + decode (enable/time/weekday/amount).
    - read-dosing-container: passive printer for container/tank frames (raw).
• WRITE helpers:
    - set-dosing-pump-manuell-ml: one-shot manual dose using Mode 27.
    - enable-auto-mode-dosing-pump: set time (90/09) and switch to auto (32).
    - add-setting-dosing-pump: uses the proven cadence (32 → 27 → 22) so the
      schedule “latches” exactly as the phone app expects.
• Probe: LED totals (0x5B 0x34 / 0x22).
• Wireshark JSON → JSONL extractor for ATT payloads.

Notes:
- This CLI is *standalone* and does **not** depend on Home Assistant or the
  central coordinator. It talks directly to the device using bleak.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List

import typer
from typer import Context
from typing_extensions import Annotated
from bleak.exc import BleakDeviceNotFoundError, BleakError

# Import the existing doser CLI app, the device class, and local weekday helpers
from .device.doser_device import app as app
from .device.doser_device import (
    DoserDevice,
    _resolve_ble_or_fail,
    WeekdaySelect,
    encode_selected_weekdays,
)

# Protocol bits & orchestration
from .protocol import (
    UART_TX,
    UART_RX,
    build_totals_query_5b,
    parse_totals_frame,
    parse_log_blob,
    decode_records,
    build_device_state,
    to_ctl_lines,
    program_channel_weekly_with_interval,  # ensures latching
)

# Wireshark parser deps
import json, base64
from pathlib import Path


# ────────────────────────────────────────────────────────────────
# Global options
# ────────────────────────────────────────────────────────────────
@app.callback(invoke_without_command=True)
def _global_options(
    ctx: Context,
    debug: Annotated[
        bool,
        typer.Option("--debug/--no-debug", help="Enable verbose debug logging"),
    ] = False,
):
    ctx.obj = ctx.obj or {}
    ctx.obj["debug"] = bool(debug)

    if debug:
        try:
            from rich.logging import RichHandler  # type: ignore
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_level=True)],
            )
        except Exception:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        logging.getLogger("bleak").setLevel(logging.DEBUG)
        logging.getLogger("chihiros").setLevel(logging.DEBUG)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def _parse_hex_blob(blob: str) -> bytes:
    s = "".join(blob.strip().split())
    if len(s) % 2 != 0:
        raise typer.BadParameter("Hex length must be even.")
    try:
        return bytes.fromhex(s)
    except ValueError as e:
        raise typer.BadParameter("Invalid hex characters in payload.") from e


def _int0(val: str | int) -> int:
    if isinstance(val, int):
        return val
    try:
        return int(str(val), 0)
    except ValueError as e:
        raise typer.BadParameter(f"Invalid integer/hex value: {val}") from e


def _set_dd_debug_if_needed(ctx: Context, dd: DoserDevice) -> None:
    if ctx.obj and ctx.obj.get("debug"):
        dd.set_log_level("DEBUG")


def _bhex(b: bytes | bytearray) -> str:
    return bytes(b).hex(" ").upper()


# ────────────────────────────────────────────────────────────────
# BYTES ENCODE — pretty-print & totals decode
# ────────────────────────────────────────────────────────────────
@app.command(name="bytes-encode")
def bytes_encode(
    ctx: Context,
    payloads: Annotated[
        List[str],
        typer.Argument(help="One or more hex payloads (with or without spaces)."),
    ],
    table: Annotated[bool, typer.Option("--table/--no-table")] = True,
) -> None:
    try:
        from prettytable import PrettyTable, SINGLE_BORDER  # type: ignore
        PT_AVAILABLE = True
    except Exception:
        PT_AVAILABLE = False

    for idx, params in enumerate(payloads, start=1):
        value_bytes = _parse_hex_blob(params)
        if not table:
            norm = " ".join(f"{b:02x}" for b in value_bytes)
            typer.echo(f"[{idx}] {norm}   (len={len(value_bytes)})")
        else:
            cmd_id = value_bytes[0] if len(value_bytes) >= 1 else "????"
            proto_ver = value_bytes[1] if len(value_bytes) >= 2 else "????"
            length_fld = value_bytes[2] if len(value_bytes) >= 3 else None
            msg_hi = value_bytes[3] if len(value_bytes) >= 4 else "????"
            msg_lo = value_bytes[4] if len(value_bytes) >= 5 else "????"
            mode = value_bytes[5] if len(value_bytes) >= 6 else "????"

            total_after_header = max(0, len(value_bytes) - 7)
            if isinstance(length_fld, int):
                if cmd_id in (0x5B, 91):
                    param_len = max(0, length_fld - 2)
                else:
                    param_len = max(0, length_fld - 5)
            else:
                param_len = total_after_header
            if param_len != total_after_header:
                param_len = total_after_header

            params_start = 6
            params_end = min(len(value_bytes) - 1, params_start + param_len)
            params_list = [int(b) for b in value_bytes[params_start:params_end]]
            checksum = value_bytes[-1] if len(value_bytes) >= 1 else "????"

            if PT_AVAILABLE:
                table_obj = PrettyTable()
                table_obj.set_style(SINGLE_BORDER)
                table_obj.title = f"Encode Message #{idx}"
                table_obj.field_names = [
                    "Command Print",
                    "Command ID",
                    "Version",
                    "Command Length",
                    "Message ID High",
                    "Message ID Low",
                    "Mode",
                    "Parameters",
                    "Checksum",
                ]
                table_obj.add_row(
                    [
                        str([int(b) for b in value_bytes]),
                        str(cmd_id),
                        str(proto_ver),
                        str(length_fld if length_fld is not None else "????"),
                        str(msg_hi),
                        str(msg_lo),
                        str(mode),
                        str(params_list),
                        str(checksum),
                    ]
                )
                print(table_obj)
            else:
                typer.echo(f"[#{idx}] Encode Message")
                typer.echo(f"  Command ID       : {cmd_id}")
                typer.echo(f"  Version          : {proto_ver}")
                typer.echo(f"  Command Length   : {length_fld if length_fld is not None else '????'}")
                typer.echo(f"  Message ID High  : {msg_hi}")
                typer.echo(f"  Message ID Low   : {msg_lo}")
                typer.echo(f"  Mode             : {mode}")
                typer.echo(f"  Parameters       : {params_list}")
                typer.echo(f"  Checksum         : {checksum}")

            if cmd_id in (0x5B, 91):
                vals = parse_totals_frame(value_bytes)
                if vals:
                    typer.echo(
                        "Decoded totals (mL): "
                        f"CH1={vals[0]:.2f}, CH2={vals[1]:.2f}, CH3={vals[2]:.2f}, CH4={vals[3]:.2f}"
                    )


# ────────────────────────────────────────────────────────────────
# BYTES DECODE — parse text logs to CTL lines
# ────────────────────────────────────────────────────────────────
@app.command(name="bytes-decode")
def bytes_decode_to_ctl(
    ctx: Context,
    file_path: Annotated[str, typer.Argument(help="Path to a text file with 'Encode Message …' blocks or JSON lines")],
    print_raw: Annotated[bool, typer.Option("--raw/--no-raw", help="Also print decoded JSON rows")] = False,
) -> None:
    try:
        text = open(file_path, "r", encoding="utf-8").read()
    except OSError as e:
        raise typer.BadParameter(f"Could not read file: {e}") from e
    print(text)
    recs = parse_log_blob(text)
    if not recs:
        typer.echo("No records parsed.")
        raise typer.Exit(2)

    dec = decode_records(recs)
    if print_raw:
        import json as _json

        typer.echo(_json.dumps(dec, indent=2))

    state = build_device_state(dec)
    lines = to_ctl_lines(state)
    if not lines:
        typer.echo("No CTL lines produced.")
        raise typer.Exit(1)

    for ln in lines:
        typer.echo(ln)


# ────────────────────────────────────────────────────────────────
# Simple READ helpers
# ────────────────────────────────────────────────────────────────
@app.command(name="read-dosing-auto")
def read_dosing_auto(
    ctx: Context,
    device_address: Annotated[str, typer.Argument(help="BLE MAC, e.g. AA:BB:CC:DD:EE:FF")],
    ch_id: Annotated[int | None, typer.Option(help="Channel 0..3; omit for all")] = None,
    timeout_s: Annotated[float, typer.Option(help="Timeout seconds", min=0.1)] = 2.0,
) -> None:
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            _set_dd_debug_if_needed(ctx, dd)
            await dd.read_dosing_pump_auto_settings(ch_id=ch_id, timeout_s=timeout_s)
        finally:
            if dd:
                await dd.disconnect()

    asyncio.run(run())


@app.command(name="read-dosing-container")
def read_dosing_container(
    ctx: Context,
    device_address: Annotated[str, typer.Argument(help="BLE MAC, e.g. AA:BB:CC:DD:EE:FF")],
    ch_id: Annotated[int | None, typer.Option(help="Channel 0..3; omit for all")] = None,
    timeout_s: Annotated[float, typer.Option(help="Timeout seconds", min=0.1)] = 2.0,
) -> None:
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            _set_dd_debug_if_needed(ctx, dd)
            await dd.read_dosing_container_status(ch_id=ch_id, timeout_s=timeout_s)
        finally:
            if dd:
                await dd.disconnect()

    asyncio.run(run())


# ────────────────────────────────────────────────────────────────
# Write helpers (now using cadence that makes schedules latch)
# ────────────────────────────────────────────────────────────────
@app.command("set-dosing-pump-manuell-ml")
def cli_set_dosing_pump_manuell_ml(
    ctx: Context,
    device_address: Annotated[str, typer.Argument(help="BLE MAC, e.g. AA:BB:CC:DD:EE:FF")],
    ch_id: Annotated[int, typer.Option("--ch-id", help="Channel 0..3 (0≙CH1)", min=0, max=3)],
    ch_ml: Annotated[float, typer.Option("--ch-ml", help="Dose (mL)", min=0.2, max=999.9)],
):
    """Immediate one-shot dose."""
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            _set_dd_debug_if_needed(ctx, dd)
            await dd.set_dosing_pump_manuell_ml(ch_id, ch_ml)
        except (BleakDeviceNotFoundError, BleakError, OSError):
            raise
        finally:
            if dd:
                await dd.disconnect()

    asyncio.run(run())


@app.command("enable-auto-mode-dosing-pump")
def cli_enable_auto_mode_dosing_pump(
    ctx: Context,
    device_address: Annotated[str, typer.Argument(help="BLE MAC")],
    ch_id: Annotated[int, typer.Option("--ch-id", help="Channel 0..3 (0≙CH1)", min=0, max=3)] = 0,
):
    """Explicitly switch the doser channel to auto mode and sync time."""
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            _set_dd_debug_if_needed(ctx, dd)
            await dd.enable_auto_mode_dosing_pump(ch_id)
        except (BleakDeviceNotFoundError, BleakError, OSError):
            raise
        finally:
            if dd:
                await dd.disconnect()

    asyncio.run(run())


@app.command("add-setting-dosing-pump")
def cli_add_setting_dosing_pump(
    ctx: Context,
    device_address: Annotated[str, typer.Argument(help="BLE MAC")],
    performance_time: Annotated[datetime, typer.Argument(formats=["%H:%M"], help="HH:MM")],
    ch_id: Annotated[int, typer.Option("--ch-id", help="Channel 0..3 (0≙CH1)", min=0, max=3)],
    ch_ml: Annotated[float, typer.Option("--ch-ml", help="Daily dose mL", min=0.2, max=999.9)],
    weekdays: Annotated[
        List[str],
        typer.Option("--weekdays", "-w", help="Repeat days; can be passed multiple times", case_sensitive=False),
    ] = ["everyday"],
    interval_min: Annotated[int, typer.Option("--interval-min", help="Interval minutes (Mode 22)", min=0, max=255)] = 0,
    times_per_day: Annotated[int, typer.Option("--times-per-day", help="Times/day (Mode 22)", min=0, max=255)] = 0,
    verify: Annotated[bool, typer.Option("--verify/--no-verify", help="Read back after write")] = True,
):
    """
    Programs with the proven cadence so changes appear in the phone app:
      165/32 (activate) → 165/27 (weekly row) → 165/22 (interval, optional).
    """
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            _set_dd_debug_if_needed(ctx, dd)
            client = await dd.connect()

            mask = encode_selected_weekdays(weekdays)
            tenths = int(round(ch_ml * 10))

            await program_channel_weekly_with_interval(
                client,
                channel=ch_id,
                weekdays_mask=mask,
                enabled=True,
                hh=performance_time.hour,
                mm=performance_time.minute,
                dose_tenths=tenths,
                interval_min=interval_min,
                times_per_day=times_per_day,
            )
            typer.echo("OK: wrote schedule (mode32→mode27→mode22).")
            if verify:
                await dd.read_dosing_pump_auto_settings(ch_id=ch_id, timeout_s=2.0)
        except (BleakDeviceNotFoundError, BleakError, OSError):
            raise
        finally:
            if dd:
                await dd.disconnect()

    asyncio.run(run())


# ────────────────────────────────────────────────────────────────
# Probe: LED totals
# ────────────────────────────────────────────────────────────────
@app.command("probe-totals")
def cli_probe_totals(
    ctx: Context,
    device_address: Annotated[str, typer.Argument(help="BLE MAC, e.g. AA:BB:CC:DD:EE:FF")],
    timeout_s: Annotated[float, typer.Option(help="Listen timeout seconds", min=0.5)] = 6.0,
    mode_5b: Annotated[
        List[str],
        typer.Option("--mode-5b", help="One or more 0x5B modes to try (e.g. 0x34 0x22). Tries in order."),
    ] = ["0x34", "0x22"],
    json_out: Annotated[
        bool,
        typer.Option("--json/--no-json", help="Emit machine-readable JSON instead of text output"),
    ] = False,
) -> None:
    async def run():
        dd: DoserDevice | None = None
        got: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

        def _on_notify(_char, payload: bytearray) -> None:
            vals = parse_totals_frame(payload)
            if vals and not got.done():
                got.set_result(bytes(payload))

        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            _set_dd_debug_if_needed(ctx, dd)
            client = await dd.connect()

            await dd.start_notify_tx()
            dd.add_notify_callback(_on_notify)

            def _int0_local(v: str | int) -> int:  # local to command
                if isinstance(v, int):
                    return v
                return int(str(v), 0)

            modes = [_int0_local(m) for m in (mode_5b or ["0x34", "0x22"])]
            for m in modes:
                frame = build_totals_query_5b(m)
                try:
                    await client.write_gatt_char(UART_RX, frame, response=True)
                except Exception:
                    await client.write_gatt_char(UART_TX, frame, response=True)
                await asyncio.sleep(0.08)

            try:
                payload = await asyncio.wait_for(got, timeout=timeout_s)
                vals = parse_totals_frame(payload) or []
                if json_out:
                    import json as _json

                    out = {
                        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "address": device_address,
                        "modes_tried": [f"0x{m:02X}" for m in modes],
                        "totals_ml": vals[:4],
                        "raw_hex": _bhex(payload),
                    }
                    typer.echo(_json.dumps(out, ensure_ascii=False))
                else:
                    if len(vals) >= 4:
                        typer.echo(
                            f"Totals (ml): CH1={vals[0]:.2f}, CH2={vals[1]:.2f}, "
                            f"CH3={vals[2]:.2f}, CH4={vals[3]:.2f}"
                        )
                        typer.echo(f"Raw: {_bhex(payload)}")
                    else:
                        typer.echo("Totals received but could not parse 4 channels.")
            except asyncio.TimeoutError:
                msg = "No totals frame received within timeout."
                if json_out:
                    import json as _json

                    typer.echo(
                        _json.dumps(
                            {
                                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                "address": device_address,
                                "modes_tried": [f"0x{m:02X}" for m in modes],
                                "error": msg,
                            },
                            ensure_ascii=False,
                        )
                    )
                else:
                    typer.echo(msg)
        except (BleakDeviceNotFoundError, BleakError, OSError):
            raise
        finally:
            if dd:
                try:
                    dd.remove_notify_callback(_on_notify)
                    await dd.stop_notify_tx()
                finally:
                    await dd.disconnect()

    asyncio.run(run())


# ────────────────────────────────────────────────────────────────
# Raw A5 frame sender
# ────────────────────────────────────────────────────────────────
@app.command("raw-dosing-pump")
def cli_raw_dosing_pump(
    ctx: Context,
    device_address: Annotated[str, typer.Argument(help="BLE MAC")],
    cmd_id: Annotated[int, typer.Option("--cmd-id", help="Command (e.g. 165)")],
    mode: Annotated[int, typer.Option("--mode", help="Mode (e.g. 27)")],
    params: Annotated[List[int], typer.Argument(help="Parameter list, e.g. 0 0 14 2 0 0")],
    repeats: Annotated[int, typer.Option("--repeats", help="Send frame N times", min=1)] = 3,
):
    """Send a raw A5 frame: [cmd, 1, len, msg_hi, msg_lo, mode, *params, checksum]."""
    async def run():
        dd: DoserDevice | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            _set_dd_debug_if_needed(ctx, dd)
            await dd.raw_dosing_pump(cmd_id, mode, params, repeats)
            typer.echo(f"Sent raw frame cmd={cmd_id} mode={mode} params={params} x{repeats}")
        except (BleakDeviceNotFoundError, BleakError, OSError):
            raise
        finally:
            if dd:
                await dd.disconnect()

    asyncio.run(run())


# ────────────────────────────────────────────────────────────────
# Wireshark BLE to JSONL Parser
# ────────────────────────────────────────────────────────────────
@app.command("wireshark-parse")
def wireshark_parse(
    input_path: Annotated[Path, typer.Argument(help="Wireshark JSON export (array or NDJSON)")],
    handle: Annotated[str, typer.Option("--handle", help="ATT handle to match (default 0x0010 for Nordic UART TX)")] = "0x0010",
    op: Annotated[str, typer.Option("--op", help="ATT op to extract: write|notify|any")] = "write",
    rx: Annotated[str, typer.Option("--rx", help="Include notifications (RX): no|also|only")] = "no",
):
    """
    Convert a Wireshark JSON export into JSON Lines with ATT payloads.
    """
    def iter_records(stream):
        first = stream.read(1)
        if not first:
            return
        if first == "[":
            buf = "[" + stream.read()
            for x in json.loads(buf):
                yield x
        else:
            line = first + stream.readline()
            if line.strip():
                yield json.loads(line)
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    def layers_of(rec):
        src = rec.get("_source", rec)
        layers = src.get("layers", src.get("_source.layers")) or {}
        if not isinstance(layers, dict):
            return {}
        out = {}
        for k, v in layers.items():
            out[k] = v[0] if isinstance(v, list) and len(v) == 1 else v
        return out

    def get(d, *path, default=None):
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def btatt_value_to_bytes(v: str) -> bytes:
        hexstr = v.replace(":", "").replace(" ", "").strip()
        return bytes.fromhex(hexstr)

    with input_path.open("r", encoding="utf-8") as f:
        for rec in iter_records(f):
            layers = layers_of(rec)
            frame = layers.get("frame", {})
            btatt = layers.get("btatt", {})
            if not isinstance(btatt, dict):
                continue

            method = get(btatt, "btatt.opcode.method") or get(btatt, "btatt.opcode")
            handle_val = get(btatt, "btatt.handle")
            value = btatt.get("btatt.value")
            if not value and isinstance(btatt.get("btatt.value_tree"), dict):
                value = btatt["btatt.value_tree"].get("btatt.value")

            if isinstance(method, list):
                method = method[0]
            if isinstance(handle_val, list):
                handle_val = handle_val[0]
            if isinstance(value, list):
                value = value[0]

            is_write = method == "0x12"
            is_notify = method == "0x1b"

            if op == "write" and not is_write:
                continue
            if op == "notify" and not is_notify:
                continue
            if op == "any" and not (is_write or is_notify):
                continue

            def _norm_handle(h: str) -> str:
                try:
                    return f"0x{int(h, 16):x}"
                except Exception:
                    return (h or "").lower()

            if handle_val and _norm_handle(handle_val) != _norm_handle(handle):
                if not (rx in ("only", "also") and is_notify):
                    continue

            if not value:
                continue

            data = btatt_value_to_bytes(value)
            out = {
                "ts": get(frame, "frame.time") or get(frame, "frame.time_epoch"),
                "att_op": "Write Request" if is_write else ("Handle Value Notification" if is_notify else method),
                "att_handle": handle_val,
                "bytes_hex": data.hex(),
                "bytes_b64": base64.b64encode(data).decode("ascii"),
                "len": len(data),
            }
            typer.echo(json.dumps(out, separators=(",", ":")))

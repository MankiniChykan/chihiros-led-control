from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta
from typing import List, Optional

import typer
from typing_extensions import Annotated
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakDeviceNotFoundError, BleakError

# 👈 go up to the common LED package (shared BaseDevice, time sync, weekday utils)
from ...chihiros_led_control.main.base_device import BaseDevice
from ...chihiros_led_control.main import msg_command as msg_cmd
from ...chihiros_led_control.main import ctl_command as ctl_cmd
from ....helper.weekday_encoding import (
    WeekdaySelect,
    encode_selected_weekdays,
)

app = typer.Typer(help="Chihiros doser control")


def _parse_hex_blob(blob: str) -> bytes:
    s = "".join(blob.strip().split())
    if len(s) % 2 != 0:
        raise typer.BadParameter("Hex length must be even.")
    try:
        return bytes.fromhex(s)
    except ValueError as e:
        raise typer.BadParameter("Invalid hex characters in payload.") from e
    
 # ────────────────────────────────────────────────────────────────
# BYTES ENCODE — pretty-print + decode helpers
# (Extends the same `app` imported from doser_device)
# ────────────────────────────────────────────────────────────────

@app.command(name="bytes-encode")
def bytes_encode(
    params: Annotated[str, typer.Argument(help="Hex string with/without spaces")],
    table: Annotated[bool, typer.Option("--table/--no-table")] = True,
) -> None:
    """
    Pretty-print an A5/5B frame and (if 8 params) decode 4 channels of daily totals
    using the 25.6 bucket + 0.1 mL scheme.
    """
    value_bytes = _parse_hex_blob(params)

    if not table:
        norm = " ".join(f"{b:02x}" for b in value_bytes)
        typer.echo(f"{norm}   (len={len(value_bytes)})")
        return

    # Safe extraction with placeholders
    cmd_id      = value_bytes[0] if len(value_bytes) >= 1 else "????"
    proto_ver   = value_bytes[1] if len(value_bytes) >= 2 else "????"
    length_fld  = value_bytes[2] if len(value_bytes) >= 3 else None
    msg_hi      = value_bytes[3] if len(value_bytes) >= 4 else "????"
    msg_lo      = value_bytes[4] if len(value_bytes) >= 5 else "????"
    mode        = value_bytes[5] if len(value_bytes) >= 6 else "????"

    # Determine param length per protocol family
    total_after_header = max(0, len(value_bytes) - 7)
    if isinstance(length_fld, int):
        if cmd_id in (0x5B, 91):         # LED-style
            param_len = max(0, length_fld - 2)
        else:                             # A5-style (doser)
            param_len = max(0, length_fld - 5)
    else:
        param_len = total_after_header

    if param_len != total_after_header:
        param_len = total_after_header

    params_start = 6
    params_end   = min(len(value_bytes) - 1, params_start + param_len)
    params_list  = [int(b) for b in value_bytes[params_start:params_end]]
    checksum     = value_bytes[-1] if len(value_bytes) >= 1 else "????"

    try:
        # PrettyTable is optional; fall back to plain output if missing
        from prettytable import PrettyTable, SINGLE_BORDER  # type: ignore
        table_obj = PrettyTable()
        table_obj.set_style(SINGLE_BORDER)
        table_obj.title = "Encode Message"
        table_obj.field_names = [
            "Command Print", "Command ID", "Version", "Command Length",
            "Message ID High", "Message ID Low", "Mode", "Parameters", "Checksum",
        ]
        table_obj.add_row([
            str([int(b) for b in value_bytes]),
            str(cmd_id),
            str(proto_ver),
            str(length_fld if length_fld is not None else "????"),
            str(msg_hi),
            str(msg_lo),
            str(mode),
            str(params_list),
            str(checksum),
        ])
        print(table_obj)  # rich.print
    except Exception:
        # Fallback (no prettytable)
        typer.echo("Encode Message")
        typer.echo(f"  Command ID       : {cmd_id}")
        typer.echo(f"  Version          : {proto_ver}")
        typer.echo(f"  Command Length   : {length_fld if length_fld is not None else '????'}")
        typer.echo(f"  Message ID High  : {msg_hi}")
        typer.echo(f"  Message ID Low   : {msg_lo}")
        typer.echo(f"  Mode             : {mode}")
        typer.echo(f"  Parameters       : {params_list}")
        typer.echo(f"  Checksum         : {checksum}")

    # Optional: decode daily totals for 4 channels using the 25.6 scheme
    if len(params_list) == 8:
        def decode_ml_25_6(hi: int, lo: int) -> float:
            # hi*25.6 + lo*0.1
            return round(hi * 25.6 + lo / 10.0, 1)

        mls = [decode_ml_25_6(params_list[i], params_list[i + 1]) for i in range(0, 8, 2)]
        typer.echo(
            f"Decoded daily totals (ml): ch0={mls[0]:.2f}, ch1={mls[1]:.2f}, "
            f"ch2={mls[2]:.2f}, ch3={mls[3]:.2f}"
        )

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
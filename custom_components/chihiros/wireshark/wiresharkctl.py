# custom_components/chihiros/wireshark/wiresharkctl.py
"""Wireshark-related CLI for chihirosctl.

All heavy helpers live outside Home Assistant in /tools:
  - tools/wireshark_core.py
  - tools/btsnoop_to_jsonl.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List
from datetime import datetime

import typer
from typing_extensions import Annotated

# Ensure /tools is importable even when running under HA
TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Try to import helper modules from /tools
try:
    from .wireshark_core import parse_wireshark_stream, write_jsonl  # type: ignore
   
except Exception:
    parse_wireshark_stream = None  # type: ignore
    write_jsonl = None  # type: ignore

try:
    from btsnoop_to_jsonl import iter_btsnoop_records, write_jsonl as write_jsonl_btsnoop  # type: ignore
except Exception:
    iter_btsnoop_records = None  # type: ignore
    write_jsonl_btsnoop = None  # type: ignore

# Optional: doser protocol helpers for decode/encode & raw TX
try:
    from ..chihiros_doser_control.protocol import (
        parse_log_blob,
        decode_records,
        build_device_state,
        to_ctl_lines,
    )
    from ..chihiros_doser_control.device.doser_device import (  # type: ignore
        DoserDevice,
        _resolve_ble_or_fail,
    )
except Exception:
    parse_log_blob = decode_records = build_device_state = to_ctl_lines = None  # type: ignore
    DoserDevice = None  # type: ignore
    _resolve_ble_or_fail = None  # type: ignore

app = typer.Typer(help="Wireshark helpers (parse/peek/encode/decode/tx)")

def _require_ws():
    if parse_wireshark_stream is None or write_jsonl is None:
        typer.secho("Wireshark core helpers not available (tools/wireshark_core.py).", fg=typer.colors.RED)
        raise typer.Exit(code=2)

def _require_btsnoop():
    if iter_btsnoop_records is None or write_jsonl_btsnoop is None:
        typer.secho("BTSnoop helpers not available (tools/btsnoop_to_jsonl.py).", fg=typer.colors.RED)
        raise typer.Exit(code=2)

# ── parse ───────────────────────────────────────────────────────

@app.command("parse")
def wireshark_parse(
    infile: Annotated[Path, typer.Argument(exists=True, readable=True, help="Wireshark export (JSON array or NDJSON)")],
    outfile: Annotated[Path, typer.Option("--out", "-o", help="Output JSONL path (use '-' for stdout)")] = Path("-"),
    handle: Annotated[str, typer.Option(help="ATT handle to match (default Nordic UART TX 0x0010)")] = "0x0010",
    op: Annotated[str, typer.Option(help="ATT op filter: write|notify|any")] = "write",
    rx: Annotated[str, typer.Option(help="Include notifications: no|also|only")] = "no",
    pretty: Annotated[bool, typer.Option("--pretty/--no-pretty", help="Pretty JSONL (indented)")] = False,
) -> None:
    """Convert a Wireshark JSON export into JSON Lines of BLE ATT payloads."""
    _require_ws()
    try:
        with infile.open("r", encoding="utf-8") as f:
            rows = parse_wireshark_stream(f, handle=handle, op=op, rx=rx)  # type: ignore
           
            if str(outfile) == "-":
                import sys as _sys
                write_jsonl(rows, _sys.stdout, pretty=pretty)  # type: ignore
            else:
                outfile.parent.mkdir(parents=True, exist_ok=True)
                with outfile.open("w", encoding="utf-8") as out:
                    write_jsonl(rows, out, pretty=pretty)  # type: ignore
    except Exception as e:
        raise typer.BadParameter(f"Parse failed: {e}") from e

# ── peek ────────────────────────────────────────────────────────

@app.command("peek")
def wireshark_peek(
    infile: Annotated[Path, typer.Argument(exists=True, readable=True, help="Wireshark export (JSON array or NDJSON)")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of frames to show", min=1)] = 12,
    handle: Annotated[str, typer.Option(help="ATT handle match (default 0x0010)")] = "0x0010",
    op: Annotated[str, typer.Option(help="ATT op filter: write|notify|any")] = "any",
    rx: Annotated[str, typer.Option(help="Include notifications: no|also|only")] = "also",
) -> None:
    """Show the first few normalized frames (ts, op, handle, len, hex…)."""
    _require_ws()
    try:
        from rich.table import Table
        from rich.console import Console
        has_rich = True
    except Exception:
        has_rich = False

    try:
        with infile.open("r", encoding="utf-8") as f:
            rows = parse_wireshark_stream(f, handle=handle, op=op, rx=rx)  # type: ignore
            shown = 0
            if has_rich:
                table = Table("idx", "time", "op", "handle", "len", "hex")
                for rec in rows:
                    shown += 1
                    if shown > limit:
                        break
                    table.add_row(
                        str(shown),
                        str(rec.get("ts", ""))[:23],
                        str(rec.get("att_op", "")),
                        str(rec.get("att_handle", "")),
                        str(rec.get("len", "")),
                        (rec.get("bytes_hex", "")[:64] + ("…" if rec.get("len", 0) > 32 else "")),
                    )
                Console().print(table)
            else:
                for rec in rows:
                    shown += 1
                    if shown > limit:
                        break
                    ts = str(rec.get("ts", ""))
                    opv = rec.get("att_op", "")
                    h = rec.get("att_handle", "")
                    ln = rec.get("len", "")
                    hx = rec.get("bytes_hex", "")
                    print(f"[{shown:02d}] {ts}  {opv}  handle={h}  len={ln}  hex={hx[:64]}{'…' if ln and ln>32 else ''}")
            if shown == 0:
                typer.secho("No matching frames.", fg=typer.colors.YELLOW)
    except Exception as e:
        raise typer.BadParameter(f"Peek failed: {e}") from e

# ── bytes-encode / bytes-decode ─────────────────────────────────

@app.command(name="bytes-encode")
def wireshark_bytes_encode(
    payloads: Annotated[List[str], typer.Argument(help="One or more hex payloads (with or without spaces).")],
    table: Annotated[bool, typer.Option("--table/--no-table")] = True,
) -> None:
    """
    Pretty-print one or more A5/5B frames and, when applicable, decode 4-channel totals.
    """
    try:
        from prettytable import PrettyTable, SINGLE_BORDER  # type: ignore
        PT_AVAILABLE = True
    except Exception:
        PT_AVAILABLE = False

    def _parse_hex_blob(blob: str) -> bytes:
        s = "".join(blob.strip().split())
        if len(s) % 2 != 0:
            raise typer.BadParameter("Hex length must be even.")
        try:
            return bytes.fromhex(s)
        except ValueError as e:
            raise typer.BadParameter("Invalid hex characters in payload.") from e

    def _decode_if_totals(params: list[int]) -> str | None:
        if len(params) != 8:
            return None
        vals = [round(params[i] * 25.6 + params[i + 1] / 10.0, 1) for i in range(0, 8, 2)]
        return f"Decoded daily totals (ml): CH1={vals[0]:.2f}, CH2={vals[1]:.2f}, CH3={vals[2]:.2f}, CH4={vals[3]:.2f}"

    for idx, p in enumerate(payloads, start=1):
        b = _parse_hex_blob(p)
        cmd_id     = b[0] if len(b) > 0 else None
        version    = b[1] if len(b) > 1 else None
        length_fld = b[2] if len(b) > 2 else None
        msg_hi     = b[3] if len(b) > 3 else None
        msg_lo     = b[4] if len(b) > 4 else None
        mode       = b[5] if len(b) > 5 else None
        params     = [int(x) for x in b[6:-1]] if len(b) >= 8 else []
        checksum   = b[-1] if len(b) >= 1 else None

        if not table:
            typer.echo(f"[{idx}] {b.hex(' ')}   (len={len(b)})")
            if (text := _decode_if_totals(params)):
                typer.echo(text)
            continue

        if PT_AVAILABLE:
            pt = PrettyTable()
            pt.set_style(SINGLE_BORDER)
            pt.title = f"Encode Message #{idx}"
            pt.field_names = ["Command Print", "Command ID", "Version", "Command Length",
                              "Message ID High", "Message ID Low", "Mode", "Parameters", "Checksum"]
            pt.add_row([
                str([int(x) for x in b]),
                str(cmd_id),
                str(version),
                str(length_fld),
                str(msg_hi),
                str(msg_lo),
                str(mode),
                str(params),
                str(checksum),
            ])
            print(pt)
            if (text := _decode_if_totals(params)):
                typer.echo(text)
        else:
            typer.echo(f"[#{idx}] Encode Message")
            typer.echo(f"  Command ID       : {cmd_id}")
            typer.echo(f"  Version          : {version}")
            typer.echo(f"  Command Length   : {length_fld}")
            typer.echo(f"  Message ID High  : {msg_hi}")
            typer.echo(f"  Message ID Low   : {msg_lo}")
            typer.echo(f"  Mode             : {mode}")
            typer.echo(f"  Parameters       : {params}")
            typer.echo(f"  Checksum         : {checksum}")
            if (text := _decode_if_totals(params)):
                typer.echo(text)

@app.command(name="bytes-decode")
def wireshark_bytes_decode_to_ctl(
    file_path: Annotated[str, typer.Argument(help="Path to a text file with 'Encode Message …' blocks or JSON lines")],
    raw: Annotated[bool, typer.Option("--raw/--no-raw", help="Also print decoded JSON rows")] = False,
) -> None:
    """Decode captured ‘Encode Message …’ blocks into normalized CTL lines."""
    
    if parse_log_blob is None:
        raise typer.BadParameter("Protocol helpers not available in this environment.")
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except OSError as e:
        raise typer.BadParameter(f"Could not read file: {e}") from e
    
    recs = parse_log_blob(text)
    if not recs:
        typer.echo("No records parsed.")
        raise typer.Exit(2)

    dec = decode_records(recs)
    if raw:
        import json as _json
        typer.echo(_json.dumps(dec, indent=2))

    state = build_device_state(dec)
    lines = to_ctl_lines(state)
    if not lines:
        typer.echo("No CTL lines produced.")
        raise typer.Exit(1)
    for ln in lines:
        typer.echo(ln)

# ── raw-command ─────────────────────────────────────────────────

@app.command("raw-command")
def wireshark_raw_command(
    device_address: Annotated[str, typer.Argument(help="BLE MAC")],
    cmd_id: Annotated[int, typer.Option("--cmd-id", help="Command (e.g. 165)")],
    mode: Annotated[int, typer.Option("--mode", help="Mode (e.g. 27)")],
    params: Annotated[List[int], typer.Argument(help="Parameter list, e.g. 0 0 14 2 0 0")],
    repeats: Annotated[int, typer.Option("--repeats", help="Send frame N times", min=1)] = 3,
):
    """Send a raw A5/0x5B frame to the device (for reverse-engineering/debug)."""
    if DoserDevice is None or _resolve_ble_or_fail is None:
        raise typer.BadParameter("Raw command sender is unavailable in this environment.")

    async def run():
        dd: Any | None = None
        try:
            ble = await _resolve_ble_or_fail(device_address)
            dd = DoserDevice(ble)
            await dd.raw_dosing_pump(cmd_id, mode, params, repeats)
        finally:
            if dd:
                await dd.disconnect()

    import asyncio
    asyncio.run(run())

# ── btsnoop-to-jsonl ───────────────────────────────────────────

@app.command("btsnoop-to-jsonl")
def wireshark_btsnoop_to_jsonl(
    infile: Annotated[Path, typer.Argument(exists=True, readable=True, help="Android btsnoop_hci.log (binary)")],
    outfile: Annotated[Path, typer.Option("--out", "-o", help="Output JSONL path (use '-' for stdout)")] = Path("-"),
    pretty: Annotated[bool, typer.Option("--pretty/--no-pretty", help="Pretty JSONL (indented)")] = False,
) -> None:
    """Convert Android btsnoop_hci.log to JSONL with ISO-UTC timestamps."""
    _require_btsnoop()
    try:
        if str(outfile) == "-":
            import sys as _sys
            write_jsonl_btsnoop(iter_btsnoop_records(infile), _sys.stdout, pretty=pretty)  # type: ignore
        else:
            outfile.parent.mkdir(parents=True, exist_ok=True)
            with outfile.open("w", encoding="utf-8") as out:
                write_jsonl_btsnoop(iter_btsnoop_records(infile), out, pretty=pretty)  # type: ignore
    except Exception as e:
        raise typer.BadParameter(f"btsnoop conversion failed: {e}") from e
    
@app.command("extract-frames")
def wireshark_extract_frames(
    infile: Annotated[Path, typer.Argument(exists=True, readable=True, help="JSONL created by btsnoop-to-jsonl")],
    outfile: Annotated[Path, typer.Option("--out", "-o", help="Output file with JSON lines {cmd,mode,params}")]=Path("-"),
    also_hex: Annotated[bool, typer.Option("--also-hex/--no-also-hex", help="Also write a .hex file with raw frames")]=False,
) -> None:
    """
    Scan btsnoop JSONL (raw HCI) for embedded A5/5B frames and export them as JSON lines
    that `bytes-decode` understands: {"cmd":165,"mode":27,"params":[...]}.

    This is a heuristic: we look for 0xA5/0x5B, verify XOR checksum and slice cmd/mode/params.
    """
    import json

    def _xor_checksum(buf: bytes) -> int:
        # same as protocol.py: XOR from index 1..end
        c = buf[1]
        for b in buf[2:]:
            c ^= b
        return c & 0xFF

    def _find_frames(payload: bytes) -> list[bytes]:
        out: list[bytes] = []
        n = len(payload)
        # try every starting position; frames are typically short (< 64), but we allow up to ~128
        for i in range(n):
            first = payload[i]
            if first not in (0xA5, 0x5B):
                continue
            # minimal frame len is 8 (cmd,01,len,hi,lo,mode,p0,chk)
            for L in range(8, min(128, n - i) + 1):
                s = payload[i:i + L]
                if len(s) < 8:
                    continue
                # basic structure: s[1] should be 0x01, checksum must match
                if s[1] != 0x01:
                    continue
                if _xor_checksum(s[:-1]) != s[-1]:
                    continue
                # length field sanity: s[2] = params_len + (fixed fields after len)
                # For our A5/5B encoders len = len(params)+5 and total frame is 3 + (len)+1
                params_len = s[2] - 5
                expected_total = 3 + s[2] + 1
                if params_len < 0 or expected_total != len(s):
                    continue
                out.append(bytes(s))
                break  # don’t report overlapping larger slices starting at same i
        return out

    def _frame_to_jsonline(frm: bytes) -> str | None:
        if len(frm) < 8:
            return None
        cmd = frm[0]
        mode = frm[5]
        params = list(int(x) for x in frm[6:-1])
        return json.dumps({"cmd": int(cmd), "mode": int(mode), "params": params}, ensure_ascii=False)

    # read JSONL
    frames_jsonl: list[str] = []
    frames_hex: list[str] = []
    with infile.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            hx = obj.get("bytes_hex")
            if not isinstance(hx, str) or len(hx) < 4:
                continue
            try:
                raw = bytes.fromhex(hx)
            except ValueError:
                continue
            for frm in _find_frames(raw):
                j = _frame_to_jsonline(frm)
                if j:
                    frames_jsonl.append(j)
                    frames_hex.append(frm.hex(" "))

    if not frames_jsonl:
        typer.secho("No A5/5B frames found.", fg=typer.colors.YELLOW)
        raise typer.Exit(2)

    # write outputs
    if str(outfile) == "-":
        import sys
        for j in frames_jsonl:
            sys.stdout.write(j + "\n")
    else:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_text("\n".join(frames_jsonl) + "\n", encoding="utf-8")

    if also_hex:
        hex_path = outfile.with_suffix(".hex") if str(outfile) != "-" else None
        if hex_path:
            hex_path.write_text("\n".join(frames_hex) + "\n", encoding="utf-8")

    typer.secho(f"Extracted {len(frames_jsonl)} frame(s).", fg=typer.colors.GREEN)    
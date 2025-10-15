# tools/wireshark/wireshark_ble_to_jsonl.py
from __future__ import annotations

import argparse
import base64
import json
from typing import Any, Dict, Iterable, Iterator, Optional, TextIO


def _iter_input(stream: TextIO) -> Iterator[Dict[str, Any]]:
    """
    Accept either a single JSON array or NDJSON (one JSON object per line).
    Yields raw records (dicts).
    """
    first = stream.read(1)
    if not first:
        return
    if first == "[":
        buf = "[" + stream.read()
        arr = json.loads(buf)
        for x in arr:
            if isinstance(x, dict):
                yield x
        return
    # NDJSON path
    line = first + stream.readline()
    if line.strip():
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        except Exception:
            pass
    for line in stream:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                yield obj
        except Exception:
            continue


def _layers_of(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wireshark JSON puts decoded fields under _source.layers and frequently
    wraps values in single-item lists. Normalize that to plain dict/scalars.
    """
    src = rec.get("_source", rec)
    layers = src.get("layers", src.get("_source.layers")) or {}
    if not isinstance(layers, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in layers.items():
        out[k] = v[0] if isinstance(v, list) and len(v) == 1 else v
    return out


def _get(d: Dict[str, Any], *path: str, default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _norm_handle(h: Optional[str]) -> Optional[str]:
    if not h:
        return h
    s = str(h).lower().strip()
    if s.startswith("0x"):
        try:
            return f"0x{int(s, 16):x}"
        except Exception:
            return s
    try:
        return f"0x{int(s, 16):x}"
    except Exception:
        return s


def _btatt_value_to_bytes(v: str) -> bytes:
    # Wireshark may use "aa:bb:cc" or "aabbcc"
    hexstr = v.replace(":", "").replace(" ", "").strip()
    return bytes.fromhex(hexstr)


def parse_ble_stream(
    stream: TextIO,
    *,
    handle: str = "0x0010",
    op: str = "write",     # write|notify|any
    rx: str = "no",        # no|also|only
) -> Iterator[Dict[str, Any]]:
    """
    Yield normalized dicts for ATT payloads from a Wireshark JSON export stream.
    Each dict has: ts, att_op, att_handle, bytes_hex, bytes_b64, len
    """
    want = {"write", "notify", "any"}
    if op not in want:
        raise ValueError(f"op must be one of {sorted(want)}")
    allow_rx = rx in ("also", "only")
    only_rx  = rx == "only"

    target = _norm_handle(handle)

    for rec in _iter_input(stream):
        layers = _layers_of(rec)
        frame = layers.get("frame", {})
        btatt = layers.get("btatt", {})
        if not isinstance(btatt, dict):
            continue

        method = _get(btatt, "btatt.opcode.method") or _get(btatt, "btatt.opcode")
        hval   = _get(btatt, "btatt.handle")
        # value can live directly or inside value_tree
        value = btatt.get("btatt.value")
        if not value and isinstance(btatt.get("btatt.value_tree"), dict):
            value = btatt["btatt.value_tree"].get("btatt.value")

        # Normalize possible list wrappers
        if isinstance(method, list): method = method[0]
        if isinstance(hval, list):   hval   = hval[0]
        if isinstance(value, list):  value  = value[0]

        # 0x12 = Write Request; optionally include 0x52 (Write Command) if you like
        is_write  = (method == "0x12")
        # 0x1b = Handle Value Notification
        is_notify = (method == "0x1b")

        if op == "write" and not is_write:
            continue
        if op == "notify" and not is_notify:
            continue
        if op == "any" and not (is_write or is_notify):
            continue

        nh = _norm_handle(hval)
        if only_rx:
            # only notifications regardless of handle
            if not is_notify:
                continue
        else:
            # for write or any modes, enforce handle unless we're allowing RX notify mismatch
            if nh and nh != target:
                if not (allow_rx and is_notify):
                    continue

        if not value:
            continue

        try:
            b = _btatt_value_to_bytes(value)
        except Exception:
            continue

        yield {
            "ts": _get(frame, "frame.time") or _get(frame, "frame.time_epoch"),
            "att_op": "Write Request" if is_write else ("Handle Value Notification" if is_notify else method),
            "att_handle": _norm_handle(hval),
            "bytes_hex": b.hex(),
            "bytes_b64": base64.b64encode(b).decode("ascii"),
            "len": len(b),
        }


def write_jsonl(
    rows: Iterable[Dict[str, Any]],
    out: TextIO,
    *,
    pretty: bool = False,
) -> None:
    """Write rows to out in JSON Lines; pretty=True uses indent=2 (still 1 per line)."""
    if pretty:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False, indent=2))
            out.write("\n")
    else:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            out.write("\n")


def _main() -> int:
    p = argparse.ArgumentParser(
        description="Convert Wireshark BLE JSON export to JSONL of ATT payloads."
    )
    p.add_argument("infile", help="Wireshark export (JSON array or NDJSON)")
    p.add_argument("-o", "--out", dest="outfile", default="-", help="Output JSONL path (default: stdout)")
    p.add_argument("--handle", default="0x0010", help="ATT handle to match (default: 0x0010)")
    p.add_argument("--op", choices=["write", "notify", "any"], default="write", help="Filter by ATT op")
    p.add_argument("--rx", choices=["no", "also", "only"], default="no",
                   help="Include notifications: no|also|only (default: no)")
    p.add_argument("--pretty", action="store_true", help="Pretty JSON (indented) per line")
    args = p.parse_args()

    try:
        with open(args.infile, "r", encoding="utf-8") as f:
            rows = parse_ble_stream(f, handle=args.handle, op=args.op, rx=args.rx)
            if args.outfile == "-":
                import sys as _sys
                write_jsonl(rows, _sys.stdout, pretty=args.pretty)
            else:
                from pathlib import Path
                outp = Path(args.outfile)
                outp.parent.mkdir(parents=True, exist_ok=True)
                with outp.open("w", encoding="utf-8") as out:
                    write_jsonl(rows, out, pretty=args.pretty)
    except Exception as e:
        print(f"Error: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

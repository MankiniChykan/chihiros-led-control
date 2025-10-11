# tools/wireshark_core.py
from __future__ import annotations

import json
from typing import Dict, Any, Iterable, Iterator, TextIO
from datetime import datetime

def _iter_input(stream: TextIO) -> Iterator[Dict[str, Any]]:
    """
    Accepts either a single JSON array or NDJSON (one JSON object per line).
    Yields raw records (dicts).
    """
    # Peek the first non-whitespace char
    pos = stream.tell()
    head = stream.read(1)
    while head and head.isspace():
        head = stream.read(1)
    stream.seek(pos)

    if head == "[":
        data = json.load(stream)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        return

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
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _norm_handle(h: str | None) -> str:
    if not h:
        return ""
    try:
        return f"0x{int(str(h), 16):x}"
    except Exception:
        return str(h).lower()


def parse_wireshark_stream(
    stream: TextIO,
    handle: str = "0x0010",
    op: str = "write",      # write|notify|any
    rx: str = "no",         # no|also|only
) -> Iterable[Dict[str, Any]]:
    """
    Yields normalized rows from a Wireshark JSON/NDJSON export:

      {
        "ts": "2024-01-02T03:04:05.123456Z",
        "att_op": "Write Request" | "Handle Value Notification" | "<opcode>",
        "att_handle": "0x0010",
        "len": 20,
        "bytes_hex": "a5010d....",
      }

    Filters by ATT handle/op according to (handle, op, rx).
    """
    want = _norm_handle(handle)

    for rec in _iter_input(stream):
        layers = _layers_of(rec)
        frame = layers.get("frame", {})
        btatt = layers.get("btatt", {})
        if not isinstance(btatt, dict):
            continue

        method = _get(btatt, "btatt.opcode.method") or _get(btatt, "btatt.opcode")
        handle_val = btatt.get("btatt.handle")
        value = btatt.get("btatt.value")

        # Sometimes the payload lives under value_tree
        if not value and isinstance(btatt.get("btatt.value_tree"), dict):
            value = btatt["btatt.value_tree"].get("btatt.value")

        if isinstance(method, list): method = method[0]
        if isinstance(handle_val, list): handle_val = handle_val[0]
        if isinstance(value, list): value = value[0]

        is_write = (method == "0x12")
        is_notify = (method == "0x1b")

        # opcode filter
        if op == "write" and not is_write:
            continue
        if op == "notify" and not is_notify:
            continue
        if op == "any" and not (is_write or is_notify):
            continue

        # handle filter
        if handle_val:
            if _norm_handle(handle_val) != want:
                # allow notifications when rx says "also" or "only"
                if not (rx in ("only", "also") and is_notify):
                    continue
        else:
            # no handle; keep only if rx allows notify and it's a notify
            if not (rx in ("only", "also") and is_notify):
                continue

        if not value:
            continue

        hexstr = str(value).replace(":", "").replace(" ", "").strip()
        try:
            payload = bytes.fromhex(hexstr)
        except Exception:
            continue

        # timestamp (string) or epoch — just pass through if present
        ts = _get(frame, "frame.time") or _get(frame, "frame.time_epoch")
        # try to normalize epoch (seconds) to ISO if it looks numeric
        if ts and isinstance(ts, str) and ts.replace(".", "", 1).isdigit():
            try:
                ts_f = float(ts)
                ts_iso = datetime.utcfromtimestamp(ts_f).isoformat(timespec="microseconds") + "Z"
                ts = ts_iso
            except Exception:
                pass

        att_op = (
            "Write Request" if is_write else
            ("Handle Value Notification" if is_notify else str(method))
        )

        yield {
            "ts": ts or "",
            "att_op": att_op,
            "att_handle": _norm_handle(handle_val),
            "len": len(payload),
            "bytes_hex": payload.hex(),
        }


def write_jsonl(rows: Iterable[Dict[str, Any]], out: TextIO, pretty: bool = False) -> None:
    if pretty:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False, indent=2) + "\n")
    else:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
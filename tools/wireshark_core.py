# custom_components/chihiros/chihiros_doser_control/wireshark_core.py
from __future__ import annotations
import json, base64
from typing import Any, Dict, Iterable, Iterator, TextIO, Optional

def _iter_records(stream: TextIO) -> Iterator[Dict[str, Any]]:
    """Yield Wireshark export records from either a JSON array or NDJSON stream."""
    first = stream.read(1)
    if not first:
        return
    if first == "[":
        buf = "[" + stream.read()
        arr = json.loads(buf)
        for x in arr:
            yield x
    else:
        line = first + stream.readline()
        if line.strip():
            yield json.loads(line)
        for line in stream:
            if line.strip():
                yield json.loads(line)

def _layers_of(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return flattened layers dict (Wireshark fields are often 1-item lists)."""
    src = rec.get("_source", rec)
    layers = src.get("layers", src.get("_source.layers")) or {}
    if not isinstance(layers, dict):
        return {}
    out = {}
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
    # sometimes Wireshark dumps plain hex without 0x
    try:
        return f"0x{int(s, 16):x}"
    except Exception:
        return s

def _btatt_value_to_bytes(v: str) -> bytes:
    hexstr = v.replace(":", "").replace(" ", "").strip()
    return bytes.fromhex(hexstr)

def parse_wireshark_stream(
    stream: TextIO,
    *,
    handle: str = "0x0010",
    op: str = "write",          # write|notify|any
    rx: str = "no",             # no|also|only
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

    for rec in _iter_records(stream):
        layers = _layers_of(rec)
        frame = layers.get("frame", {})
        btatt = layers.get("btatt", {})
        if not isinstance(btatt, dict):
            continue

        method = _get(btatt, "btatt.opcode.method") or _get(btatt, "btatt.opcode")
        hval   = _get(btatt, "btatt.handle")
        # value can live directly or inside value_tree
        value  = btatt.get("btatt.value")
        if not value and isinstance(btatt.get("btatt.value_tree"), dict):
            value = btatt["btatt.value_tree"].get("btatt.value")

        # Normalize possible list wrappers
        if isinstance(method, list): method = method[0]
        if isinstance(hval, list):   hval   = hval[0]
        if isinstance(value, list):  value  = value[0]

        is_write  = (method == "0x12")
        is_notify = (method == "0x1b")

        if op == "write" and not is_write:
            continue
        if op == "notify" and not is_notify:
            continue
        if op == "any" and not (is_write or is_notify):
            continue

        # RX selection logic
        nh = _norm_handle(hval)
        if only_rx:
            if not is_notify:        # only notifications
                continue
        else:
            # for write or any modes, enforce handle unless we're allowing RX notify mismatch
            if nh and nh != target:
                if not (allow_rx and is_notify):
                    continue

        if not value:
            continue

        b = _btatt_value_to_bytes(value)
        yield {
            "ts": _get(frame, "frame.time") or _get(frame, "frame.time_epoch"),
            "att_op": "Write Request" if is_write else ("Handle Value Notification" if is_notify else method),
            "att_handle": hval,
            "bytes_hex": b.hex(),
            "bytes_b64": base64.b64encode(b).decode("ascii"),
            "len": len(b),
        }

def write_jsonl(
    rows: Iterable[Dict[str, Any]],
    out: TextIO,
    *,
    pretty: bool = False
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

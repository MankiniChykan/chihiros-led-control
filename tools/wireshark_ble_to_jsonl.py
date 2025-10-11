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
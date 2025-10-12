# tools/btsnoop_to_jsonl.py
from __future__ import annotations

import struct
import base64
from pathlib import Path
from typing import Iterable, BinaryIO, Iterator, Dict, Any, TextIO
from datetime import datetime, timezone

# BTSnoop constants
_MAGIC = b"btsnoop\0"
_HDR_LEN = 16
_REC_HDR_LEN = 24
# BTSnoop timestamps are "microseconds since midnight, January 1, year 0".
# Convert to UNIX epoch (1970-01-01T00:00:00Z):
_BTSNOOP_UNIX_DELTA_US = 62135596800 * 1_000_000  # 62,135,596,800 seconds

def _open_binary(path: Path) -> BinaryIO:
    return path.open("rb")

def iter_btsnoop_records(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Yield dict rows from a btsnoop_hci.log file with ISO-8601 UTC timestamps.
    Each row contains: ts, dir (in|out), flags, orig_len, incl_len, bytes_hex, bytes_b64.
    """
    with _open_binary(path) as f:
        hdr = f.read(_HDR_LEN)
        if len(hdr) != _HDR_LEN or hdr[:8] != _MAGIC:
            raise ValueError("Not a BTSnoop file.")

        while True:
            h = f.read(_REC_HDR_LEN)
            if not h:
                break
            if len(h) != _REC_HDR_LEN:
                raise ValueError("Truncated BTSnoop record header.")
            orig_len, incl_len, flags, drops, ts_us = struct.unpack(">IIIIQ", h)
            payload = f.read(incl_len)
            if len(payload) != incl_len:
                raise ValueError("Truncated BTSnoop record payload.")

            # Convert BTSnoop ts → UNIX epoch, tz-aware
            unix_us = ts_us - _BTSNOOP_UNIX_DELTA_US
            dt = datetime.fromtimestamp(unix_us / 1_000_000, tz=timezone.utc)
            ts_iso = dt.isoformat(timespec="microseconds").replace("+00:00", "Z")

            # Direction bit: bit0: 0=in (received), 1=out (sent)
            direction = "out" if (flags & 0x01) else "in"

            yield {
                "ts": ts_iso,
                "dir": direction,
                "flags": int(flags),
                "orig_len": int(orig_len),
                "incl_len": int(incl_len),
                "bytes_hex": payload.hex(),
                "bytes_b64": base64.b64encode(payload).decode("ascii"),
            }

def write_jsonl(rows: Iterable[Dict[str, Any]], out: TextIO, pretty: bool = False) -> None:
    import json
    if pretty:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False, indent=2) + "\n")
    else:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
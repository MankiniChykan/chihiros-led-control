# custom_components/chihiros/chihiros_doser_control/protocol.py
"""
Chihiros Doser — low-level protocol helpers.

This module provides *wire-compatible* utilities for the A5/165 (doser) and
5B/LED (report/totals) command families observed on the Chihiros dosing pump
line, matching captured device behavior.

Key features (all aligned to the logs/spec you provided):

• Nordic UART UUIDs:
    - We write on UART_RX and listen on UART_TX.

• Frame building:
    - A5/165 encoder for dosing-side traffic:
        [cmd, 0x01, len(params)+5, msg_hi, msg_lo, mode, *params, checksum]
      The XOR checksum is computed over the body (starting at version byte).
      Message IDs avoid 0x5A in either byte; payload bytes are sanitized so
      accidental 0x5A values are emitted as 0x59 for firmware compatibility.
    - 5B/LED encoder for totals/report frames:
        [0x5B, 0x01, len(params)+2, msg_hi, msg_lo, mode, *params, checksum]

• Manual dose (confirmed):
    - Mode 27 (0x1B) with params [ch, 0, 0, hi, lo]
      where hi/lo encode milliliters using 25.6-bucket + 0.1 remainder
      (see _split_ml_25_6 and ml_from_25_6).

• Weekly schedule (confirmed):
    - Mode 27 (0x1B) with params
      [channel, weekday_mask (0..127), enabled(0/1), HH(0..23), MM(0..59), dose×10 (0..255)]
      plus tolerant parsing for the alternate hi/lo amount flavor found in logs.

• Interval / repeats (from logs):
    - Mode 22 (0x16) with params [channel, interval_minutes, times_per_day].

• Activation / flags:
    - Mode 32 (0x20) with params [channel, catch_up(0/1), active(0/1)].

• Time set:
    - 90/09 (YY, MM, DOW, HH, MM, SS).

• Totals (LED/report side):
    - 0x5B with modes 0x34 (preferred) and 0x22 (alt fw). Decoding assumes ml×10
      pairs by default and auto-falls back to ml×100 if needed, returning 4
      channel values padded with zeros if fewer are present.

• Log parsing → records → state:
    - Robust extractors from PrettyTable-style "Encode Message" blocks AND
      JSON/JSONL lines.
    - Frame decoder yields normalized events (time_set/activate/dose_entry/
      manual_dose/timer/unknown).
    - DeviceState builder collapses events into a consumable per-channel view.
    - CTL exporter emits simple key=value lines that are easy to diff.

• Reliable writes:
    - send_and_await_ack() writes a frame and awaits the device’s ACK on
      the same msg_hi/msg_lo, with heartbeat “dwell” handling and one retry.
    - program_channel_weekly_with_interval() emits the proven cadence
      (mode32 → mode27 → mode22) so schedules “latch” as in the phone app.

Anything outside this documented behavior was intentionally left out.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional, Iterable, Union, Callable
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR
import json
import re
import time

__all__ = [
    "UART_SERVICE", "UART_RX", "UART_TX",
    "CMD_MANUAL_DOSE", "MODE_MANUAL_DOSE", "CMD_LED_QUERY",
    "_split_ml_25_6",
    "dose_ml",
    "build_weekly_entry", "build_interval_entry",
    "build_totals_query_5b", "build_totals_query", "build_totals_probes",
    "parse_totals_frame",
    "decode_weekdays", "ml_from_25_6",
    "parse_frame", "parse_log_blob", "decode_records",
    "DeviceState", "ChannelState", "TimerState", "build_device_state", "to_ctl_lines",
    "interpret_param_burst",
    "send_and_await_ack",
    "program_channel_weekly_with_interval",
]

# ────────────────────────────────────────────────────────────────
# Nordic UART UUIDs (write to RX, notify on TX)
# ────────────────────────────────────────────────────────────────
UART_SERVICE = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX      = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # write
UART_TX      = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # notify

# ────────────────────────────────────────────────────────────────
# Command families
# ────────────────────────────────────────────────────────────────
CMD_MANUAL_DOSE  = 0xA5  # 165, "doser"
MODE_MANUAL_DOSE = 0x1B  # 27 (manual dose AND weekly schedule shape)
CMD_LED_QUERY    = 0x5B  # 91, LED/report side, includes totals

# ────────────────────────────────────────────────────────────────
# Message-ID generator (avoid 0x5A in either byte)
# ────────────────────────────────────────────────────────────────
_last_msg_id: Tuple[int, int] = (0, 0)

def _next_msg_id() -> Tuple[int, int]:
    hi, lo = _last_msg_id
    lo = (lo + 1) & 0xFF
    if lo == 0x5A:
        lo = (lo + 1) & 0xFF
    if lo == 0:
        hi = (hi + 1) & 0xFF
        if hi == 0x5A:
            hi = (hi + 1) & 0xFF
    globals()['_last_msg_id'] = (hi, lo)
    return hi, lo

# ────────────────────────────────────────────────────────────────
# Checksums / param sanitization
# ────────────────────────────────────────────────────────────────
def _xor_checksum(buf: bytes) -> int:
    if len(buf) < 2:
        return 0
    c = buf[1]
    for b in buf[2:]:
        c ^= b
    return c & 0xFF

def _sanitize_params(params: List[int]) -> List[int]:
    """Avoid 0x5A in payload bytes (some firmwares disallow it)."""
    out: List[int] = []
    for p in params:
        b = int(p) & 0xFF
        out.append(0x59 if b == 0x5A else b)
    return out

# ────────────────────────────────────────────────────────────────
# Encoders: A5 (doser) & 5B (LED-style)
# ────────────────────────────────────────────────────────────────
def _encode(cmd: int, mode: int, params: List[int]) -> bytes:
    """
    A5-style frame:
      [cmd, 0x01, len(params)+5, msg_hi, msg_lo, mode, *params, checksum]
    If checksum == 0x5A, rotate msg-id (do not mutate params).
    """
    ps = _sanitize_params(params)
    body = b""
    chk = 0
    for _ in range(8):
        hi, lo = _next_msg_id()
        body = bytes([cmd, 0x01, len(ps) + 5, hi, lo, mode, *ps])
        chk = _xor_checksum(body)
        if chk != 0x5A:
            break
    return body + bytes([chk])

def encode_5b(mode: int, params: List[int]) -> bytes:
    """
    0x5B-style frame:
      [0x5B, 0x01, len(params)+2, msg_hi, msg_lo, mode, *params, checksum]
    """
    ps = _sanitize_params(params)
    body = b""
    chk = 0
    for _ in range(8):
        hi, lo = _next_msg_id()
        body = bytes([CMD_LED_QUERY, 0x01, len(ps) + 2, hi, lo, mode, *ps])
        chk = _xor_checksum(body)
        if chk != 0x5A:
            break
    return body + bytes([chk])

# ────────────────────────────────────────────────────────────────
# mL encoding (25.6 bucket + 0.1 remainder) — used by manual dose
# ────────────────────────────────────────────────────────────────
def _split_ml_25_6(total_ml: Union[float, int, str]) -> tuple[int, int]:
    """
    Encode ml as (hi, lo) with 25.6-mL buckets (+0.1-mL remainder).
      hi = floor(ml / 25.6)
      lo = round((ml - hi*25.6) * 10)  # 0..255 (0.1 mL)
    Normalize exact multiples so 25.6 -> (1,0). Clamp 0.2..999.9.
    """
    if isinstance(total_ml, str):
        s = total_ml.replace(",", ".")
    else:
        s = str(total_ml)

    q = Decimal(s).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if q < Decimal("0.2") or q > Decimal("999.9"):
        raise ValueError("ml must be within 0.2..999.9")

    hi = int((q / Decimal("25.6")).to_integral_value(rounding=ROUND_FLOOR))
    rem = (q - Decimal(hi) * Decimal("25.6")).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    lo  = int((rem * 10).to_integral_value(rounding=ROUND_HALF_UP))
    if lo == 256:
        hi += 1
        lo  = 0
    return hi & 0xFF, lo & 0xFF

def ml_from_25_6(hi_buckets: int, tenths_remainder: int) -> float:
    return round(25.6 * hi_buckets + 0.1 * tenths_remainder, 1)

# ────────────────────────────────────────────────────────────────
# Public write: immediate one-shot dose (manual)
# ────────────────────────────────────────────────────────────────
async def dose_ml(client, channel_1based: int, ml: Union[float, int, str]) -> None:
    """
    Immediate, one-shot dose on the selected channel.

    Protocol (confirmed):
      MODE=27, PARAMS=[ch0..3, 0x00, 0x00, ml_hi, ml_lo]
      ml_hi = floor(ml / 25.6), ml_lo = round(remainder*10)
    """
    ch = max(1, min(int(channel_1based), 4)) - 1  # 0-based on wire
    ml_hi, ml_lo = _split_ml_25_6(ml)
    pkt = _encode(CMD_MANUAL_DOSE, MODE_MANUAL_DOSE, [ch, 0x00, 0x00, ml_hi, ml_lo])
    await client.write_gatt_char(UART_RX, pkt, response=True)

# ────────────────────────────────────────────────────────────────
# NEW: builders for weekly entry & interval repeats
# ────────────────────────────────────────────────────────────────
def build_weekly_entry(channel: int, weekday_mask: int, enabled: bool, hh: int, mm: int, dose_tenths: int) -> bytes:
    """
    Weekly schedule row (confirmed Mode 27 shape):
      [channel, weekday_mask (0..127), enabled(0/1), HH(0..23), MM(0..59), dose×10 (0..255)]
    Returns an A5/27 frame.
    """
    ch = int(channel) & 0xFF
    mask = int(weekday_mask) & 0x7F
    en = 1 if enabled else 0
    HH = max(0, min(int(hh), 23))
    MM = max(0, min(int(mm), 59))
    d10 = max(0, min(int(dose_tenths), 255))
    return _encode(CMD_MANUAL_DOSE, 0x1B, [ch, mask, en, HH, MM, d10])

def build_interval_entry(channel: int, interval_min: int, times_per_day: int) -> bytes:
    """
    Interval/repeat config (Mode 22 shape seen in logs):
      [channel, interval_minutes, times_per_day]
    Returns an A5/22 frame.
    """
    ch = int(channel) & 0xFF
    iv = max(0, min(int(interval_min), 255))
    times = max(0, min(int(times_per_day), 255))
    return _encode(CMD_MANUAL_DOSE, 0x16, [ch, iv, times])

# ────────────────────────────────────────────────────────────────
# Totals helpers (LED-style 0x5B only)
# ────────────────────────────────────────────────────────────────
def build_totals_query_5b(*, mode_5b: int = 0x34, seq: Optional[int] = None) -> bytes:
    """
    Prefer 0x5B daily totals query; default mode 0x34 (some fw use 0x22).

    NOTE: `seq` is accepted for API compatibility with higher layers that
    may supply an application-level sequence. The internal message-id is
    always generated here; `seq` is intentionally ignored.
    """
    _ = seq  # ignored; message-id comes from _next_msg_id()
    return encode_5b(mode_5b, [])

def build_totals_query() -> bytes:
    """Back-compat single-frame helper; still LED-first (0x34)."""
    return build_totals_query_5b(mode_5b=0x34)

def build_totals_probes() -> list[bytes]:
    """
    Return a small set of viable totals queries across firmwares.
    STRICTLY LED (0x5B): try 0x34, 0x22, and 0x1E (seen in your logs).
    """
    frames: list[bytes] = []
    for m in (0x34, 0x22, 0x1E):
        frames.append(encode_5b(m, []))
        frames.append(encode_5b(m, []))  # send each twice to wake some firmwares
    # de-dup preserving order
    seen, uniq = set(), []
    for f in frames:
        b = bytes(f)
        if b not in seen:
            seen.add(b); uniq.append(f)
    return uniq

# ────────────────────────────────────────────────────────────────
# Daily totals parsing (LED 0x5B / 91) — default ml×10; auto-detect ×100
# ────────────────────────────────────────────────────────────────
def _decode_u16_be(hi: int, lo: int) -> int:
    return ((hi & 0xFF) << 8) | (lo & 0xFF)

def parse_totals_frame(payload: bytes | bytearray) -> Optional[List[float]]:
    """
    Parse 0x5B daily totals (Mode 0x34; some fw 0x22).
    Confirmed in device logs: pairs encode **ml×10**.
    """
    if not isinstance(payload, (bytes, bytearray)) or len(payload) < 16:
        return None
    if payload[0] != CMD_LED_QUERY:
        return None

    params = list(payload[6:-1])
    if len(params) < 2:
        return None

    # Optional 2-byte header [hdr0, ch_count]
    start_idx = 0
    ch_count: Optional[int] = None
    if len(params) >= 10:
        maybe_count = params[1]
        if 1 <= maybe_count <= 4 and (len(params) - 2) >= maybe_count * 2:
            ch_count = maybe_count
            start_idx = 2
    if ch_count is None:
        ch_count = 4 if len(params) - start_idx >= 8 else (len(params) - start_idx) // 2
        if ch_count <= 0:
            return None

    data = params[start_idx:start_idx + ch_count * 2]
    raw_pairs = [_decode_u16_be(data[i], data[i + 1] if i + 1 < len(data) else 0) for i in range(0, len(data), 2)]

    def scale_to_ml(v: int) -> float:
        ml10 = v / 10.0
        if ml10 <= 2000.0:
            return round(ml10, 2)
        return round(v / 100.0, 2)

    values = [scale_to_ml(v) for v in raw_pairs]
    while len(values) < 4:
        values.append(0.0)
    return values[:4]

# ────────────────────────────────────────────────────────────────
# Weekday helpers
# ────────────────────────────────────────────────────────────────
WEEKDAY_BITS = {
    64: "monday",
    32: "tuesday",
    16: "wednesday",
    8:  "thursday",
    4:  "friday",
    2:  "saturday",
    1:  "sunday",
}

def decode_weekdays(mask: int) -> List[str]:
    if mask == 127:
        return ["everyday"]
    return [name for bit, name in WEEKDAY_BITS.items() if mask & bit]

# ────────────────────────────────────────────────────────────────
# Frame decoders (tolerant) for A5/165 traffic
# ────────────────────────────────────────────────────────────────
def _dose27_is_manual(params: List[int]) -> bool:
    n = len(params)
    if n == 5:
        ch, a, b, hi, lo = params
        return a == 0 and b == 0 and 0 <= hi <= 10 and 0 <= lo <= 255
    if n == 6:
        ch, a, b, x, hi, lo = params
        return a == 0 and b == 0 and 0 <= hi <= 10 and 0 <= lo <= 255
    return False

def _dose27_is_weekly(params: List[int]) -> bool:
    if len(params) != 6:
        return False
    _, mask, en, HH, MM, dose10 = params
    return (0 <= mask <= 127) and (en in (0, 1)) and (0 <= HH <= 23) and (0 <= MM <= 59) and (0 <= dose10 <= 255)

def decode_time_90_9(params: List[int]) -> Dict[str, Any]:
    yy, mm, idx, HH, MM, SS = params
    return {
        "type": "time_set",
        "year": 2000 + yy,
        "month": mm,
        "day_or_week_index": idx,
        "hour": HH,
        "minute": MM,
        "second": SS,
    }

def decode_activate_165_32(params: List[int]) -> Dict[str, Any]:
    if len(params) == 3:
        ch, en, _confirm = params
        return {"type": "activate", "channel": ch, "enabled": bool(en)}
    if len(params) >= 3:
        ch, _z, en = params[:3]
        return {"type": "activate", "channel": ch, "enabled": bool(en)}
    return {"type": "activate", "channel": params[0] if params else 0, "enabled": None}

def decode_dose_165_27(params: List[int]) -> Dict[str, Any]:
    if _dose27_is_manual(params):
        if len(params) == 5:
            ch, _a, _b, hi, tenths = params
        else:
            ch, _a, _b, _x, hi, tenths = params
        ml = ml_from_25_6(hi, tenths)
        return {"type": "manual_dose", "channel": ch, "amount_ml": ml, "raw": params}

    if _dose27_is_weekly(params):
        ch, mask, enable, HH, MM, dose10 = params
        return {
            "type": "dose_entry",
            "channel": ch,
            "weekdays_mask": mask,
            "weekdays": decode_weekdays(mask),
            "enabled": bool(enable),
            "time_hour": HH,
            "time_minute": MM,
            "amount_ml": round(dose10 / 10.0, 1),
            "raw_aux": [],
        }

    if len(params) == 6:
        ch, mask, c1, c2, hi, tenths = params
        obj = {
            "type": "dose_entry",
            "channel": ch,
            "weekdays_mask": mask,
            "weekdays": decode_weekdays(mask),
            "amount_ml": ml_from_25_6(hi, tenths),
            "raw_aux": [c1, c2],
        }
        return obj

    return {"type": "unknown_27", "params": params}

def decode_timer_165_21(params: List[int]) -> Dict[str, Any]:
    ch, t_or_en, hh, mm, r1, r2 = params
    entry = {
        "type": "timer",
        "channel": ch,
        "timer_type": t_or_en,
        "start_hour": hh,
        "start_minute": mm,
    }
    return entry

def parse_frame(cmd_id: int, mode: int, params: List[int]) -> Dict[str, Any]:
    if cmd_id == 90 and mode == 9 and len(params) == 6:
        return decode_time_90_9(params)
    if cmd_id == 165 and mode == 32 and len(params) >= 3:
        return decode_activate_165_32(params)
    if cmd_id == 165 and mode == 27 and (len(params) in (5, 6)):
        return decode_dose_165_27(params)
    if cmd_id == 165 and mode == 21 and len(params) == 6:
        return decode_timer_165_21(params)
    if (cmd_id, mode) in {(90, 4), (165, 4)}:
        return {"type": "control", "cmd": cmd_id, "mode": mode, "params": params}
    return {"type": "unknown", "cmd": cmd_id, "mode": mode, "params": params}

# ────────────────────────────────────────────────────────────────
# Log parsing → records
# ────────────────────────────────────────────────────────────────
_ENCODE_BLOCK_RE = re.compile(
    r"Encode Message.*?Command ID\s*:\s*(\d+).*?Mode\s*:\s*(\d+).*?Parameters\s*:\s*(\[[^\]]*\])",
    re.IGNORECASE | re.DOTALL,
)

def _safe_int_list_from_str(lst_str: str) -> List[int]:
    try:
        raw = json.loads(lst_str)
        if isinstance(raw, list):
            return [int(x) & 0xFF for x in raw]
    except Exception:
        pass
    vals = re.findall(r"-?\d+", lst_str)
    return [int(v) & 0xFF for v in vals]

def parse_log_blob(text: str) -> List[Tuple[int, int, List[int]]]:
    out: List[Tuple[int,int,List[int]]] = []

    for m in _ENCODE_BLOCK_RE.finditer(text):
        try:
            cmd = int(m.group(1))
            mode = int(m.group(2))
            params = _safe_int_list_from_str(m.group(3))
            out.append((cmd, mode, params))
        except Exception:
            continue

    for line in text.splitlines():
        line = line.strip()
        if not line or not (line.startswith("{") or line.startswith("[")):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        def add(cmd: int, mode: int, params: List[int]) -> None:
            out.append((int(cmd), int(mode), [int(x) & 0xFF for x in params]))
        if isinstance(obj, dict):
            if "cmd" in obj and "mode" in obj and "params" in obj:
                add(obj["cmd"], obj["mode"], list(obj["params"]))
        elif isinstance(obj, list):
            if len(obj) == 3 and isinstance(obj[2], list):
                add(obj[0], obj[1], obj[2])

    return out

def decode_records(records: Iterable[Tuple[int,int,List[int]]]) -> List[Dict[str,Any]]:
    return [parse_frame(c, m, p) for (c, m, p) in records]

# ────────────────────────────────────────────────────────────────
# State builder
# ────────────────────────────────────────────────────────────────
from dataclasses import dataclass  # (re-import kept to match original layout)
@dataclass
class TimerState:
    timer_type: Optional[int] = None
    start_hour: Optional[int] = None
    start_minute: Optional[int] = None

@dataclass
class ChannelState:
    channel: int
    enabled: Optional[bool] = None
    amount_ml: Optional[float] = None
    weekdays_mask: Optional[int] = None
    weekdays: List[str] = field(default_factory=list)
    time_hour: Optional[None] = None
    time_minute: Optional[None] = None
    timer: "TimerState" = field(default_factory=TimerState)

    def merge(self, event: Dict[str, Any]) -> None:
        t = event.get("type")
        if t == "activate":
            self.enabled = bool(event.get("enabled"))
        elif t == "dose_entry":
            self.amount_ml = float(event.get("amount_ml"))
            if "weekdays_mask" in event:
                self.weekdays_mask = int(event.get("weekdays_mask"))
                self.weekdays = list(event.get("weekdays", []))
            if "time_hour" in event:
                self.time_hour = int(event.get("time_hour"))
            if "time_minute" in event:
                self.time_minute = int(event.get("time_minute"))
            if "enabled" in event and self.enabled is None:
                self.enabled = bool(event.get("enabled"))
        elif t == "manual_dose":
            self.amount_ml = float(event.get("amount_ml"))
        elif t == "timer":
            self.timer.timer_type = int(event.get("timer_type"))
            self.timer.start_hour = int(event.get("start_hour"))
            self.timer.start_minute = int(event.get("start_minute"))

@dataclass
class DeviceState:
    device_time: Optional[Dict[str, Any]] = None  # from 90/9
    channels: Dict[int, ChannelState] = field(default_factory=dict)
    other_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_time": self.device_time,
            "channels": {ch: asdict(state) for ch, state in self.channels.items()},
            "other_events": self.other_events,
        }

def build_device_state(parsed_rows: List[Dict[str, Any]]) -> DeviceState:
    ds = DeviceState()
    for ev in parsed_rows:
        et = ev.get("type")
        if et == "time_set":
            ds.device_time = ev
            continue
        if et in {"activate", "dose_entry", "manual_dose", "timer"}:
            ch = int(ev["channel"])
            st = ds.channels.get(ch) or ChannelState(channel=ch)
            st.merge(ev)
            ds.channels[ch] = st
            continue
        ds.other_events.append(ev)
    return ds

# ────────────────────────────────────────────────────────────────
# CTL export (flat key=value lines)
# ────────────────────────────────────────────────────────────────
def _weekday_str(mask: Optional[int]) -> str:
    if mask is None:
        return "unknown"
    names = decode_weekdays(int(mask))
    if names == ["everyday"]:
        return "Every day"
    order = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    names_sorted = [n for n in order if n in names]
    short = {"monday":"Mon","tuesday":"Tue","wednesday":"Wed","thursday":"Thu",
             "friday":"Fri","saturday":"Sat","sunday":"Sun"}
    return ",".join(short[n] for n in names_sorted) if names_sorted else "None"

def to_ctl_lines(state: DeviceState) -> List[str]:
    lines: List[str] = []
    if state.device_time:
        dt = state.device_time
        lines.append(
            f"device_time={dt.get('year'):04d}-{dt.get('month'):02d} "
            f"{dt.get('hour'):02d}:{dt.get('minute'):02d}:{dt.get('second'):02d}"
        )
    for ch_idx in sorted(state.channels.keys()):
        st = state.channels[ch_idx]
        chn = ch_idx  # wire 0..3
        if st.enabled is not None:
            lines.append(f"ch{chn}.enabled={'1' if st.enabled else '0'}")
        if st.amount_ml is not None:
            lines.append(f"ch{chn}.amount_ml={st.amount_ml:.1f}")
        if st.weekdays_mask is not None:
            lines.append(f"ch{chn}.weekday_mask={int(st.weekdays_mask)}")
            lines.append(f"ch{chn}.weekdays={_weekday_str(st.weekdays_mask)}")
        hh = st.time_hour if st.time_hour is not None else st.timer.start_hour
        mm = st.time_minute if st.time_minute is not None else st.timer.start_minute
        if hh is not None and mm is not None:
            lines.append(f"ch{chn}.time={int(hh):02d}:{int(mm):02d}")
        if st.timer.timer_type is not None:
            lines.append(f"ch{chn}.timer_type={int(st.timer.timer_type)}")
    return lines

# ────────────────────────────────────────────────────────────────
# Optional: interpret bare params bursts from notify logs
# ────────────────────────────────────────────────────────────────
def interpret_param_burst(params: List[int]) -> Dict[str, Any]:
    n = len(params)
    out: Dict[str, Any] = {"guess": "unknown", "details": {"len": n, "params": params}}

    if n == 6 and 0 <= params[2] <= 23 and 0 <= params[3] <= 59:
        out["guess"] = "maybe_timer_165_21"
        out["details"].update({"channel": params[0], "hour": params[2], "minute": params[3]})
        return out

    if n == 6 and 0 <= params[1] <= 127 and params[2] in (0, 1) and 0 <= params[3] <= 23 and 0 <= params[4] <= 59:
        out["guess"] = "maybe_weekly_165_27"
        out["details"].update({
            "channel": params[0], "weekdays_mask": params[1], "enabled": bool(params[2]),
            "hour": params[3], "minute": params[4], "dose_tenths": params[5],
        })
        return out

    if (n == 5 and params[1] == 0 and params[2] == 0) or (n == 6 and params[1] == 0 and params[2] == 0):
        hi, lo = (params[-2], params[-1])
        if 0 <= hi <= 10 and 0 <= lo <= 255:
            out["guess"] = "maybe_manual_dose_165_27"
            out["details"].update({"channel": params[0], "amount_ml": ml_from_25_6(hi, lo)})
            return out

    if n >= 8 and n % 2 == 0:
        pairs = []
        for i in range(0, min(n, 8), 2):
            v = ((params[i] << 8) | params[i+1])
            pairs.append(round((v / 10.0 if v / 10.0 <= 2000 else v / 100.0), 2))
        if len(pairs) >= 4:
            out["guess"] = "maybe_led_totals"
            out["details"]["totals_ml"] = pairs[:4]
            return out

    return out

# ────────────────────────────────────────────────────────────────
# Minimal incoming frame parser (for ACK/heartbeat matching)
# ────────────────────────────────────────────────────────────────
def _parse_incoming(buf: bytes) -> Optional[Dict[str, Any]]:
    if not buf or len(buf) < 7:
        return None
    cmd = buf[0]
    ver = buf[1]
    length = buf[2]
    msg_hi, msg_lo, mode = buf[3], buf[4], buf[5]
    params = list(buf[6:-1]) if len(buf) > 7 else []
    ck = buf[-1]
    checksum_ok = (_xor_checksum(buf[:-1]) == ck)
    return {
        "cmd": cmd, "ver": ver, "length": length,
        "msg_hi": msg_hi, "msg_lo": msg_lo, "mode": mode,
        "params": params, "checksum_ok": checksum_ok
    }

# ────────────────────────────────────────────────────────────────
# Send/await-ACK with heartbeat dwell & one retry (reselect on miss)
# ────────────────────────────────────────────────────────────────
async def send_and_await_ack(
    client,
    frame: bytes,
    *,
    expect_cmd: int = CMD_MANUAL_DOSE,
    expect_ack_mode: int = 0x07,
    ack_timeout_ms: int = 300,
    heartbeat_min_ms: int = 250,
    heartbeat_max_ms: int = 800,
    retries: int = 1,
    reselect_cb: Optional[Callable[[], "asyncio.Future[Any] | None"]] = None,
) -> Dict[str, Any]:
    if len(frame) < 7:
        raise ValueError("Frame too short")
    msg_hi, msg_lo = frame[3], frame[4]

    loop = asyncio.get_running_loop()
    ack_fut: asyncio.Future = loop.create_future()
    beat_fut: asyncio.Future = loop.create_future()

    def _notify_cb(_hdl, data: bytearray):
        parsed = _parse_incoming(bytes(data))
        if not parsed:
            return
        if (not ack_fut.done()
            and parsed["cmd"] == expect_cmd
            and parsed["mode"] == expect_ack_mode
            and parsed["msg_hi"] == msg_hi
            and parsed["msg_lo"] == msg_lo):
            ack_fut.set_result(parsed)
            return
        if not beat_fut.done() and parsed["cmd"] == 0x04:
            beat_fut.set_result(parsed)

    await client.start_notify(UART_TX, _notify_cb)

    attempt = 0
    last_err: Optional[BaseException] = None

    try:
        while attempt <= retries:
            attempt += 1
            await client.write_gatt_char(UART_RX, frame, response=True)
            try:
                ack = await asyncio.wait_for(ack_fut, timeout=ack_timeout_ms / 1000.0)
            except asyncio.TimeoutError as e:
                last_err = e
                if attempt <= retries:
                    if reselect_cb is not None:
                        try:
                            maybe = reselect_cb()
                            if asyncio.isfuture(maybe) or asyncio.iscoroutine(maybe):
                                await maybe
                        except Exception:
                            pass
                    continue
                raise TimeoutError("ACK timeout (exhausted retries)") from e

            t0 = time.perf_counter()
            try:
                await asyncio.wait_for(beat_fut, timeout=(heartbeat_max_ms / 1000.0))
            except asyncio.TimeoutError:
                dt = (time.perf_counter() - t0) * 1000.0
                target = max(heartbeat_min_ms, 400)
                if dt < target:
                    await asyncio.sleep((target - dt) / 1000.0)

            return ack

        raise TimeoutError("ACK timeout (no ACK after retries)") from last_err
    finally:
        try:
            await client.stop_notify(UART_TX)
        except Exception:
            pass

# ────────────────────────────────────────────────────────────────
# Orchestration: mode 32 → 27 → 22 with cadence so it latches
# ────────────────────────────────────────────────────────────────
async def program_channel_weekly_with_interval(
    client,
    *,
    channel: int,            # wire index 0..3
    weekdays_mask: int,      # 0..127
    enabled: bool,
    hh: int,
    mm: int,
    dose_tenths: int,        # e.g. 75 => 7.5 mL
    interval_min: int,       # e.g. 60
    times_per_day: int,      # e.g. 3
) -> dict:
    ch = int(channel) & 0xFF

    # 32: activate/flags [ch, catch_up(0/1), active(0/1)]
    f32 = _encode(CMD_MANUAL_DOSE, 0x20, [ch, 0, 1 if enabled else 0])

    # 27: weekly row [ch, mask, en, HH, MM, dose×10]
    f27 = build_weekly_entry(ch, weekdays_mask, enabled, hh, mm, dose_tenths)

    # 22: interval/repeats [ch, interval_min, times_per_day]
    f22 = build_interval_entry(ch, interval_min, times_per_day)

    async def _reselect_cb():
        await send_and_await_ack(
            client,
            f32,
            expect_cmd=CMD_MANUAL_DOSE,
            expect_ack_mode=0x07,
            reselect_cb=None,
        )

    ack32 = await send_and_await_ack(
        client, f32, expect_cmd=CMD_MANUAL_DOSE, expect_ack_mode=0x07, reselect_cb=None
    )
    ack27 = await send_and_await_ack(
        client, f27, expect_cmd=CMD_MANUAL_DOSE, expect_ack_mode=0x07, reselect_cb=_reselect_cb
    )
    ack22 = await send_and_await_ack(
        client, f22, expect_cmd=CMD_MANUAL_DOSE, expect_ack_mode=0x07, reselect_cb=_reselect_cb
    )

    return {"ack32": ack32, "ack27": ack27, "ack22": ack22}

# ────────────────────────────────────────────────────────────────
# Quick self-test / example
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_frames = [
        (90, 4,  [1]),
        (90, 9,  [25, 10, 2, 11, 28, 47]),
        (CMD_MANUAL_DOSE, 32, [2, 1, 1]),
        (CMD_MANUAL_DOSE, 27, [2, 127, 1, 0, 27, 75]),
        (CMD_MANUAL_DOSE, 21, [2, 1, 0, 27, 0, 0]),
        (CMD_MANUAL_DOSE, 27, [1, 0, 0, 1, 140]),
    ]
    parsed = decode_records(example_frames)
    state = build_device_state(parsed)

    from pprint import pprint
    print("STATE:")
    pprint(state.to_dict())
    print("\nCTL:")
    for ln in to_ctl_lines(state):
        print(ln)

    print("\nBuilders:")
    print("Weekly:", build_weekly_entry(2, 64, True, 1, 0, 75))
    print("Repeat:", build_interval_entry(2, 60, 3))

    demo = bytes([0x5B, 0x01, 0x0A, 0x00, 0x01, 0x34, 0x00, 0x04, 0x00, 0x03, 0x00, 0x0C, 0x00, 0x87, 0x00, 0xA3, 0x00])
    print("\nTotals parse demo:", parse_totals_frame(demo))

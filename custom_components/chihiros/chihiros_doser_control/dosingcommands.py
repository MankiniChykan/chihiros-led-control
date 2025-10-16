# custom_components/chihiros/chihiros_doser_control/dosingcommands.py
"""
Chihiros Doser — A5/165 command builders (drop-in).

This module builds *exact* A5/165 frames used by the dosing side of the device,
matching the shapes seen in captures/logs. It is runtime-light and has **no**
Home Assistant imports, so it can be reused by the CLI or other tools.

Highlights:

• Core frame encoder:
    _create_command_encoding_dosing_pump(cmd_id, mode, (msg_hi,msg_lo), params)
  - XOR checksum over the body
  - Message-ID rotation that skips 0x5A
  - Payload sanitization to prevent literal 0x5A bytes

• Manual dose (Mode 27):
    - create_add_dosing_pump_command_manuell_ml / _amount variants.
    - Uses the same 25.6-bucket + 0.1 remainder encoder as the device.

• Weekly auto schedule (Mode 27, byte-amount flavor):
    - create_add_auto_setting_command_dosing_pump()
    - create_schedule_weekly_byte_amount()

• Time-of-day helper (Mode 21) and hi/lo schedule flavor:
    - create_auto_mode_dosing_pump_command_time()
    - create_schedule_weekly_hi_lo() to support firmwares that separate time/amount.

• Interval / repeats (Mode 22):
    - create_interval_entry()

• Activation / flags (Mode 32):
    - create_switch_to_auto_mode_dosing_pump_command()

• Time sync (90/09) and reset helper:
    - create_set_time_command()
    - create_reset_auto_settings_command()

All functions clamp/validate bytes to stay in-range and preserve device
compatibility. Anything not present in confirmed behavior is intentionally
omitted.
"""

from __future__ import annotations

from datetime import time, datetime
from typing import List, Tuple

from .protocol import _split_ml_25_6  # same 25.6+0.1 encoder

__all__ = [
    "_create_command_encoding_dosing_pump",
    "create_command_encoding_dosing_pump",
    "create_add_dosing_pump_command_manuell_ml",
    "create_add_dosing_pump_command_manuell_ml_amount",
    "create_add_auto_setting_command_dosing_pump",
    "create_auto_mode_dosing_pump_command_time",
    "create_interval_entry",
    "create_switch_to_auto_mode_dosing_pump_command",
    "create_order_confirmation",
    "create_reset_auto_settings_command",
    "create_schedule_weekly_byte_amount",
    "create_schedule_weekly_hi_lo",
    "create_set_time_command",
    "next_message_id",
]

# ────────────────────────────────────────────────────────────────
# Byte helpers
# ────────────────────────────────────────────────────────────────
def _clamp_byte(v: int) -> int:
    if not isinstance(v, int):
        raise TypeError(f"Parameter must be int (got {type(v).__name__})")
    if v < 0 or v > 255:
        raise ValueError(f"Parameter byte out of range 0..255: {v}")
    return v


def _bump_msg_id(msg_hi: int, msg_lo: int) -> tuple[int, int]:
    lo = (msg_lo + 1) & 0xFF
    hi = msg_hi
    if lo == 0x5A:
        lo = (lo + 1) & 0xFF
    if lo == 0:
        hi = (hi + 1) & 0xFF
        if hi == 0x5A:
            hi = (hi + 1) & 0xFF
    return hi, lo


def next_message_id(current: Tuple[int, int] | None = None) -> Tuple[int, int]:
    """Increment a 2-byte message id, skipping 0x5A on either byte."""
    if current is None:
        hi, lo = 0, 0
    else:
        hi, lo = int(current[0]) & 0xFF, int(current[1]) & 0xFF
    lo = (lo + 1) & 0xFF
    if lo == 0x5A:
        lo = (lo + 1) & 0xFF
    if lo == 0:
        hi = (hi + 1) & 0xFF
        if hi == 0x5A:
            hi = (hi + 1) & 0xFF
    return hi, lo


def _sanitize_params(params: List[int]) -> List[int]:
    """Replace any literal 0x5A bytes with 0x59 to avoid checksum conflicts."""
    out: List[int] = []
    for p in params:
        b = _clamp_byte(p)
        out.append(0x59 if b == 0x5A else b)
    return out


def _xor_checksum(buf: bytes | bytearray) -> int:
    """Device-style XOR over bytes [1..n-1] (excludes first byte)."""
    if len(buf) < 2:
        return 0
    c = buf[1]
    for b in buf[2:]:
        c ^= b
    return c & 0xFF


# ────────────────────────────────────────────────────────────────
# Core A5 encoder (165 family)
# ────────────────────────────────────────────────────────────────
def _create_command_encoding_dosing_pump(
    cmd_id: int,
    cmd_mode: int,
    msg_id: tuple[int, int],
    parameters: list[int],
) -> bytearray:
    _clamp_byte(cmd_id)
    _clamp_byte(cmd_mode)
    msg_hi, msg_lo = msg_id
    _clamp_byte(msg_hi)
    _clamp_byte(msg_lo)
    ps = _sanitize_params(parameters)

    # If checksum would be 0x5A, bump msg id and retry (up to 8 attempts)
    frame = bytearray()
    checksum = 0
    for _ in range(8):
        frame = bytearray([cmd_id, 1, len(ps) + 5, msg_hi, msg_lo, cmd_mode] + ps)
        checksum = _xor_checksum(frame) & 0xFF
        if checksum != 0x5A:
            return frame + bytes([checksum])
        msg_hi, msg_lo = _bump_msg_id(msg_hi, msg_lo)
    # If still 0x5A after attempts, return last (device would accept in practice)
    return frame + bytes([checksum])


# Friendly alias
create_command_encoding_dosing_pump = _create_command_encoding_dosing_pump


# ────────────────────────────────────────────────────────────────
# WRITE command creators (confirmed layouts)
# ────────────────────────────────────────────────────────────────
def create_add_dosing_pump_command_manuell_ml(
    msg_id: tuple[int, int],
    ch_id: int,
    ch_ml_one: int,
    ch_ml_two: int,
) -> bytearray:
    _clamp_byte(ch_id)
    _clamp_byte(ch_ml_one)
    _clamp_byte(ch_ml_two)
    return _create_command_encoding_dosing_pump(165, 27, msg_id, [ch_id, 0, 0, ch_ml_one, ch_ml_two])


def create_add_dosing_pump_command_manuell_ml_amount(
    msg_id: tuple[int, int],
    ch_id: int,
    ml: float | int | str,
) -> bytearray:
    hi, lo = _split_ml_25_6(ml)
    return create_add_dosing_pump_command_manuell_ml(msg_id, ch_id, hi, lo)


def create_add_auto_setting_command_dosing_pump(
    performance_time: time,
    msg_id: tuple[int, int],
    ch_id: int,
    weekdays_mask: int,
    daily_ml_tenths: int,
    enabled: bool = True,
) -> bytearray:
    _clamp_byte(ch_id)
    _clamp_byte(weekdays_mask & 0x7F)
    _clamp_byte(performance_time.hour)
    _clamp_byte(performance_time.minute)
    dose10 = int(daily_ml_tenths)
    if dose10 < 0 or dose10 > 255:
        raise ValueError("daily_ml_tenths must be 0..255")
    return _create_command_encoding_dosing_pump(
        165,
        27,
        msg_id,
        [ch_id, weekdays_mask & 0x7F, 1 if enabled else 0, performance_time.hour, performance_time.minute, dose10],
    )


def create_auto_mode_dosing_pump_command_time(
    performance_time: time,
    msg_id: tuple[int, int],
    ch_id: int,
    enabled: bool = True,
) -> bytearray:
    _clamp_byte(ch_id)
    _clamp_byte(performance_time.hour)
    _clamp_byte(performance_time.minute)
    return _create_command_encoding_dosing_pump(
        165, 21, msg_id, [ch_id, 1 if enabled else 0, performance_time.hour, performance_time.minute, 0, 0]
    )


def create_interval_entry(
    msg_id: tuple[int, int],
    ch_id: int,
    interval_min: int,
    times_per_day: int,
) -> bytearray:
    _clamp_byte(ch_id)
    _clamp_byte(interval_min)
    _clamp_byte(times_per_day)
    return _create_command_encoding_dosing_pump(165, 22, msg_id, [ch_id, interval_min, times_per_day])


def create_switch_to_auto_mode_dosing_pump_command(
    msg_id: tuple[int, int],
    channel_id: int,
    catch_up_missed: int = 0,
    active_flag: int = 1,
) -> bytearray:
    _clamp_byte(channel_id)
    _clamp_byte(catch_up_missed)
    _clamp_byte(active_flag)
    return _create_command_encoding_dosing_pump(165, 32, msg_id, [channel_id, catch_up_missed, active_flag])


def create_order_confirmation(
    msg_id: tuple[int, int],
    command_id: int,
    mode: int,
    command: int,
) -> bytearray:
    """Generic one-byte command wrapper (rarely used; kept for completeness)."""
    return _create_command_encoding_dosing_pump(command_id, mode, msg_id, [_clamp_byte(command)])


def create_reset_auto_settings_command(msg_id: tuple[int, int]) -> bytearray:
    return _create_command_encoding_dosing_pump(90, 5, msg_id, [5, 255, 255])


def create_set_time_command(msg_id: tuple[int, int]) -> bytearray:
    now = datetime.now()
    yy = (now.year - 2000) & 0xFF
    mm = now.month & 0xFF
    idx = now.isoweekday() & 0xFF
    HH = now.hour & 0xFF
    MM = now.minute & 0xFF
    SS = now.second & 0xFF
    return _create_command_encoding_dosing_pump(90, 9, msg_id, [yy, mm, idx, HH, MM, SS])


# ────────────────────────────────────────────────────────────────
# Dual schedule variants (support multiple firmware flavors)
# ────────────────────────────────────────────────────────────────
def create_schedule_weekly_byte_amount(
    performance_time: time,
    msg_id: tuple[int, int],
    ch_id: int,
    weekdays_mask: int,
    daily_ml_tenths: int,
    enabled: bool = True,
) -> bytearray:
    _clamp_byte(ch_id)
    _clamp_byte(weekdays_mask & 0x7F)
    _clamp_byte(performance_time.hour)
    _clamp_byte(performance_time.minute)
    dose10 = int(daily_ml_tenths)
    if not (0 <= dose10 <= 255):
        raise ValueError("daily_ml_tenths must be 0..255")
    return _create_command_encoding_dosing_pump(
        165,
        27,
        msg_id,
        [ch_id, weekdays_mask & 0x7F, 1 if enabled else 0, performance_time.hour, performance_time.minute, dose10],
    )


def create_schedule_weekly_hi_lo(
    performance_time: time,
    msg_id_time: tuple[int, int],
    msg_id_amount: tuple[int, int],
    ch_id: int,
    weekdays_mask: int,
    daily_ml: float,
    enabled: bool = True,
) -> list[bytearray]:
    _clamp_byte(ch_id)
    _clamp_byte(weekdays_mask & 0x7F)
    _clamp_byte(performance_time.hour)
    _clamp_byte(performance_time.minute)
    hi, lo = _split_ml_25_6(daily_ml)
    f_time = create_auto_mode_dosing_pump_command_time(performance_time, msg_id_time, ch_id, enabled=enabled)
    f_amount = _create_command_encoding_dosing_pump(
        165, 27, msg_id_amount, [ch_id, weekdays_mask & 0x7F, 1 if enabled else 0, 0, hi, lo]
    )
    return [f_time, f_amount]

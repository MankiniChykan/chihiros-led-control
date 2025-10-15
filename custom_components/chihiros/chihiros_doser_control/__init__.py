# custom_components/chihiros/chihiros_doser_control/__init__.py
"""
Chihiros Doser — Home Assistant service surface (runtime-only).

OVERVIEW
--------
This module is the *runtime* glue between Home Assistant and the doser protocol
helpers that live in `custom_components/chihiros/chihiros_doser_control/`.
It registers a handful of services (listed below) and implements the BLE I/O
loops needed to talk to the device using HA’s Bluetooth stack plus
`bleak_retry_connector` (which plays nicely with HA’s BLE queue/slots).

Why this file exists separately from the CLI:
- The CLI (`chihirosctl`) needs to import parts of the doser package without
  importing Home Assistant. To keep that import safe, all HA imports happen
  *inside* the `try:` block; if HA isn’t present we export a tiny stub for
  `register_services` which simply raises at call time (but import still works).
- Inside HA, `async_setup_entry` (in the integration `__init__.py`) calls this
  module’s `register_services(hass)` exactly once per HA run. The function
  is idempotent and will no-op on reloads.

SERVICES (must match services.yaml)
-----------------------------------
Provided (and enabled):
• `dose_ml`               — Perform a one-shot manual dose on a channel.
• `enable_auto_mode`     — Put a channel into “auto” mode and sync device time.
• `read_auto_settings`   — Passive listener for auto schedule frames (pretty printed).
• `read_container_status`— Passive listener for container/tank notifications (raw hex).
• `raw_doser_command`    — Advanced: send arbitrary A5/0x5B frames.

Kept for reference but disabled behind flags (see FEATURE FLAGS):
• `read_daily_totals`    — Active read of 0x5B “daily totals” (sensors already do this).
• `set_24h_dose`         — Program 24-hour dosing; left in code for later.

HOW THE “PUSH PATH” WORKS
-------------------------
After `dose_ml` completes, we immediately try a couple of LED-side “totals probe”
frames (0x1E and 0x22 in 0x5B and A5 shapes). If the device replies with a
0x5B “daily totals” frame, we decode it and *dispatch* it via HA’s dispatcher:
    async_dispatcher_send(hass, f"{DOMAIN}_push_totals_{addr_l}", {"ml": [...], "raw": bytes})
The daily-dose sensors subscribe to that signal (per-address) and update without
needing to poll immediately. They also do periodic passive refreshes on their own.

NOTES / PITFALLS
----------------
- BLE address is looked up via device_id (device registry) or can be provided
  directly via the service call. Addresses are normalized to uppercase for HA.
- We use HA’s `establish_connection` wrapper to benefit from retry and slot
  handling. We also guard notification + write with try/except to tolerate
  firmwares that expect writes on TX rather than RX.
- If you add new services, make sure to:
    1) define a voluptuous schema here,
    2) register it in `register_services`,
    3) add the matching entry in `services.yaml`.

This file is designed to be copied as-is into your integration. No project-wide
imports are performed at module import time other than cheap constants/types.
"""

from __future__ import annotations

# Explicit public API so this module is import-safe for CLI/tests regardless of HA.
__all__ = ["register_services"]

# ────────────────────────────────────────────────────────────────
# Import guard so this module can be imported without Home Assistant
# (e.g. when running the CLI). We only define the real implementation
# if HA is available; otherwise we export a friendly stub.
# ────────────────────────────────────────────────────────────────
try:
    import asyncio
    import logging
    import voluptuous as vol

    from homeassistant.core import HomeAssistant, ServiceCall
    from homeassistant.helpers import device_registry as dr
    from homeassistant.exceptions import HomeAssistantError
    from homeassistant.helpers.dispatcher import async_dispatcher_send

    # use HA’s bluetooth helper + bleak-retry-connector (slot-aware, proxy-friendly)
    from homeassistant.components import bluetooth
    from bleak_retry_connector import (
        BleakClientWithServiceCache,
        BLEAK_RETRY_EXCEPTIONS as BLEAK_EXC,
        establish_connection,
    )

    from ..const import DOMAIN  # integration domain
    from . import protocol as dp  # protocol helpers (dose_ml, parse_totals_frame, etc.)
    from .protocol import UART_TX  # notify UUID for totals frames

    # human-friendly weekday handling using your existing encoding helper
    from ..chihiros_led_control.weekday_encoding import (
        WeekdaySelect,
        encode_selected_weekdays,
    )

    HA_AVAILABLE = True
except ModuleNotFoundError:
    HA_AVAILABLE = False

# ────────────────────────────────────────────────────────────────
# Public API when HA is *not* installed (CLI import-safe)
# ────────────────────────────────────────────────────────────────
if not HA_AVAILABLE:
    import logging

    _LOGGER = logging.getLogger(__name__)

    async def register_services(*_args, **_kwargs) -> None:
        raise RuntimeError(
            "Chihiros_doser_control: services require Home Assistant runtime. "
            "This module can be imported safely by the CLI, "
            "but service registration only works inside Home Assistant."
        )

# ────────────────────────────────────────────────────────────────
# Full implementation when HA *is* available
# ────────────────────────────────────────────────────────────────
else:
    _LOGGER = logging.getLogger(__name__)

    # ────────────────────────────────────────────────────────────
    # FEATURE FLAGS — you can re-enable later if needed
    # ────────────────────────────────────────────────────────────
    DISABLE_READ_DAILY_TOTALS: bool = True
    DISABLE_SET_24H_DOSE: bool = True

    # ────────────────────────────────────────────────────────────
    # Schemas
    # ────────────────────────────────────────────────────────────
    DOSE_SCHEMA = vol.Schema({
        vol.Exclusive("device_id", "target"): str,
        vol.Exclusive("address", "target"): str,
        vol.Required("channel"): vol.All(vol.Coerce(int), vol.Range(min=1, max=4)),
        vol.Required("ml"):      vol.All(vol.Coerce(float), vol.Range(min=0.2, max=999.9)),
    })

    READ_TOTALS_SCHEMA = vol.Schema({
        vol.Exclusive("device_id", "target"): str,
        vol.Exclusive("address", "target"): str,
    })

    # 24h config (kept, but registration gated by feature flag)
    SET_24H_SCHEMA = vol.Schema({
        vol.Exclusive("device_id", "target"): str,
        vol.Exclusive("address", "target"): str,

        vol.Required("channel"):   vol.All(vol.Coerce(int), vol.Range(min=1, max=4)),
        vol.Required("daily_ml"):  vol.All(vol.Coerce(float), vol.Range(min=0.2, max=999.9)),

        # Time-of-day (24h) — either provide "time": "HH:MM" OR hour+minute
        vol.Optional("time"): str,
        vol.Optional("hour", default=None): vol.Any(None, vol.All(vol.Coerce(int), vol.Range(min=0, max=23))),
        vol.Optional("minutes", default=None): vol.Any(None, vol.All(vol.Coerce(int), vol.Range(min=0, max=59))),

        # Days — either a numeric mask or human strings ("Mon,Wed", ["Mon","Thu"], "everyday")
        vol.Optional("weekday_mask", default=None): vol.Any(None, vol.All(vol.Coerce(int), vol.Range(min=0, max=0x7F))),
        vol.Optional("weekdays", default=None): vol.Any(str, [str]),

        vol.Optional("catch_up", default=False): vol.Boolean(),
    })

    # Extra service schemas (matching services.yaml)
    ENABLE_AUTO_SCHEMA = vol.Schema({
        vol.Exclusive("device_id", "target"): str,
        vol.Exclusive("address", "target"): str,
        vol.Required("channel"): vol.All(vol.Coerce(int), vol.Range(min=1, max=4)),
    })

    READ_PASSIVE_SCHEMA = vol.Schema({
        vol.Exclusive("device_id", "target"): str,
        vol.Exclusive("address", "target"): str,
        vol.Optional("channel"): vol.All(vol.Coerce(int), vol.Range(min=1, max=4)),
        vol.Optional("timeout_s", default=2.0): vol.All(vol.Coerce(float), vol.Range(min=0.2, max=30.0)),
    })

    RAW_CMD_SCHEMA = vol.Schema({
        vol.Exclusive("device_id", "target"): str,
        vol.Exclusive("address", "target"): str,
        vol.Required("cmd_id"): vol.All(vol.Coerce(int), vol.Range(min=0, max=255)),
        vol.Required("mode"):   vol.All(vol.Coerce(int), vol.Range(min=0, max=255)),
        vol.Required("params"): list,  # list of ints
        vol.Optional("repeats", default=1): vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
    })

    # ────────────────────────────────────────────────────────────
    # Weekday helpers (English ⇄ mask) using your encoding
    # ────────────────────────────────────────────────────────────
    _WEEKDAY_ALIAS = {
        "mon": WeekdaySelect.monday, "monday": WeekdaySelect.monday,
        "tue": WeekdaySelect.tuesday, "tues": WeekdaySelect.tuesday, "tuesday": WeekdaySelect.tuesday,
        "wed": WeekdaySelect.wednesday, "wednesday": WeekdaySelect.wednesday,
        "thu": WeekdaySelect.thursday, "thur": WeekdaySelect.thursday, "thurs": WeekdaySelect.thursday, "thursday": WeekdaySelect.thursday,
        "fri": WeekdaySelect.friday, "friday": WeekdaySelect.friday,
        "sat": WeekdaySelect.saturday, "saturday": WeekdaySelect.saturday,
        "sun": WeekdaySelect.sunday, "sunday": WeekdaySelect.sunday,
        "everyday": "ALL", "every day": "ALL", "any": "ALL", "all": "ALL",
    }

    def _parse_weekdays_to_mask(value, fallback_mask: int = 0x7F) -> int:
        """Accept int mask, string 'Mon,Wed,Fri', or list of day strings; return 0..127 mask."""
        if value is None:
            return fallback_mask & 0x7F
        if isinstance(value, int):
            return value & 0x7F
        if isinstance(value, str):
            tokens = [t.strip() for t in value.replace("/", ",").split(",") if t.strip()]
        else:
            try:
                tokens = [str(t).strip() for t in value]
            except Exception:
                return fallback_mask & 0x7F

        if any(_WEEKDAY_ALIAS.get(t.lower()) == "ALL" for t in tokens):
            return 127

        sels: list[WeekdaySelect] = []
        for t in tokens:
            sel = _WEEKDAY_ALIAS.get(t.lower())
            if isinstance(sel, WeekdaySelect):
                sels.append(sel)
        return encode_selected_weekdays(sels) if sels else (fallback_mask & 0x7F)

    def _weekdays_mask_to_english(mask: int) -> str:
        """Render mask (Mon=64 … Sun=1) as 'Mon, Tue, …' or 'Every day'."""
        try:
            m = int(mask) & 0x7F
        except Exception:
            return "Unknown"
        parts = []
        if m & 64: parts.append("Mon")
        if m & 32: parts.append("Tue")
        if m & 16: parts.append("Wed")
        if m & 8:  parts.append("Thu")
        if m & 4:  parts.append("Fri")
        if m & 2:  parts.append("Sat")
        if m & 1:  parts.append("Sun")
        return "Every day" if len(parts) == 7 else (", ".join(parts) if parts else "None")

    # ────────────────────────────────────────────────────────────
    # Address / entry helpers
    # ────────────────────────────────────────────────────────────
    async def _resolve_address_from_device_id(hass: HomeAssistant, did: str) -> str | None:
        reg = dr.async_get(hass)
        dev = reg.async_get(did)
        if not dev:
            return None

        # 1) Native BT connection stored by HA
        for (conn_type, conn_val) in dev.connections:
            if conn_type == dr.CONNECTION_BLUETOOTH:
                return conn_val

        # 2) Identifiers used by this integration
        for domain, ident in dev.identifiers:
            if domain != DOMAIN:
                continue
            # a) explicit "ble:AA:BB:..."
            if isinstance(ident, str) and ident.startswith("ble:"):
                return ident.split(":", 1)[1]
            # b) (DOMAIN, <config_entry_id>) → hass.data[DOMAIN][entry_id].coordinator.address
            data_by_entry = hass.data.get(DOMAIN, {})
            data = data_by_entry.get(ident)
            if data and hasattr(data.coordinator, "address"):
                return data.coordinator.address

        # 3) Fallback: any linked config entries
        for entry_id in getattr(dev, "config_entries", set()):
            data_by_entry = hass.data.get(DOMAIN, {})
            data = data_by_entry.get(entry_id)
            if data and hasattr(data.coordinator, "address"):
                return data.coordinator.address

        return None

    def _find_entry_id_for_address(hass: HomeAssistant, addr: str) -> str | None:
        """Find the config_entry_id for a BLE address (case-insensitive)."""
        data_by_entry = hass.data.get(DOMAIN, {})
        addr_l = (addr or "").lower()
        for entry_id, data in data_by_entry.items():
            coord = getattr(data, "coordinator", None)
            c_addr = getattr(coord, "address", None)
            if isinstance(c_addr, str) and c_addr.lower() == addr_l:
                return entry_id
        return None

    # ────────────────────────────────────────────────────────────
    # Probe / prelude helpers
    # ────────────────────────────────────────────────────────────
    def _build_totals_probes() -> list[bytes]:
        """Try 0x1E then 0x22, 0x5B-style first then A5-style as fallback."""
        frames: list[bytes] = []
        try:
            if hasattr(dp, "encode_5b"):
                frames.append(dp.encode_5b(0x1E, []))  # some firmwares use 0x1E
                frames.append(dp.encode_5b(0x22, []))  # others use 0x22
        except Exception:
            pass
        try:
            frames.append(dp._encode(dp.CMD_MANUAL_DOSE, 0x1E, []))
        except Exception:
            pass
        try:
            frames.append(dp._encode(dp.CMD_MANUAL_DOSE, 0x22, []))
        except Exception:
            pass
        return frames

    def _build_prelude_frames() -> list[bytes]:
        """Small ‘wake up / confirm’ sequence seen in captures."""
        return [
            dp._encode(90, 4, [1]),   # 90/4 [1]
            dp._encode(165, 4, [4]),  # 165/4 [4]
            dp._encode(165, 4, [5]),  # 165/4 [5]
        ]

    # ────────────────────────────────────────────────────────────
    # Service registration
    # ────────────────────────────────────────────────────────────
    async def register_services(hass: HomeAssistant) -> None:
        # Avoid duplicate registration on reloads
        flag_key = f"{DOMAIN}_doser_services_registered"
        if hass.data.get(flag_key):
            return
        hass.data[flag_key] = True

        # -------------------------
        # Manual one-shot dose
        # -------------------------
        async def _svc_dose(call: ServiceCall):
            data = DOSE_SCHEMA(dict(call.data))

            # Resolve address
            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError("Provide address or a device_id linked to a BLE address")

            addr_u = addr.upper()
            addr_l = addr_u.lower()

            channel = int(data["channel"])
            ml = round(float(data["ml"]), 1)  # protocol is 0.1-mL resolution

            ble_dev = bluetooth.async_ble_device_from_address(hass, addr_u, True)
            if not ble_dev:
                raise HomeAssistantError(f"Could not find BLE device for address {addr_u}")

            got: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

            def _on_notify(_char, payload: bytearray) -> None:
                try:
                    if isinstance(payload, (bytes, bytearray)) and len(payload) >= 6 and payload[0] in (0x5B, 91):
                        _LOGGER.debug(
                            "dose: notify 0x5B mode=0x%02X raw=%s",
                            payload[5], bytes(payload).hex(" ").upper()
                        )
                    vals = dp.parse_totals_frame(payload)
                    if vals and not got.done():
                        got.set_result(bytes(payload))
                except Exception:
                    pass

            client = None
            try:
                client = await establish_connection(BleakClientWithServiceCache, ble_dev, f"{DOMAIN}-dose")
                await client.start_notify(UART_TX, _on_notify)

                await dp.dose_ml(client, channel, ml)

                # Totals probes (0x1E & 0x22; 0x5B first; then A5)
                for frame in _build_totals_probes():
                    try:
                        try:
                            await client.write_gatt_char(dp.UART_RX, frame, response=True)
                        except Exception:
                            await client.write_gatt_char(dp.UART_TX, frame, response=True)
                    except Exception:
                        pass
                    await asyncio.sleep(0.08)

                try:
                    payload = await asyncio.wait_for(got, timeout=8.0)
                    vals = dp.parse_totals_frame(payload) or []
                    async_dispatcher_send(hass, f"{DOMAIN}_push_totals_{addr_l}", {"ml": vals, "raw": payload})
                except asyncio.TimeoutError:
                    _LOGGER.debug("dose: no totals frame received within timeout")

                try:
                    await client.stop_notify(UART_TX)
                except Exception:
                    pass

                if (entry_id := _find_entry_id_for_address(hass, addr_u)):
                    async_dispatcher_send(hass, f"{DOMAIN}_{entry_id}_refresh_totals")

            except BLEAK_EXC as e:
                raise HomeAssistantError(f"BLE temporarily unavailable: {e}") from e
            except Exception as e:
                raise HomeAssistantError(f"Dose failed: {e}") from e
            finally:
                if client:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

        hass.services.async_register(DOMAIN, "dose_ml", _svc_dose, schema=DOSE_SCHEMA)

        # -------------------------
        # Enable auto mode (and sync time)
        # -------------------------
        async def _svc_enable_auto(call: ServiceCall):
            data = ENABLE_AUTO_SCHEMA(dict(call.data))

            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError("Provide address or a device_id linked to a BLE address")

            channel = int(data["channel"])
            ble_dev = bluetooth.async_ble_device_from_address(hass, addr.upper(), True)
            if not ble_dev:
                raise HomeAssistantError(f"Could not find BLE device for address {addr}")

            from .dosingcommands import (
                create_set_time_command,
                create_switch_to_auto_mode_dosing_pump_command,
            )

            client = None
            try:
                client = await establish_connection(BleakClientWithServiceCache, ble_dev, f"{DOMAIN}-enable-auto")
                # 90/09 set time
                msg_hi, msg_lo = 0, 0
                frame_time = create_set_time_command((msg_hi, msg_lo))
                await client.write_gatt_char(dp.UART_RX, frame_time, response=True)

                # 165/32 switch to auto for channel-1 (wire 0-based)
                msg_hi, msg_lo = 0, 1
                wire_ch = max(1, min(channel, 4)) - 1
                frame_auto = create_switch_to_auto_mode_dosing_pump_command((msg_hi, msg_lo), wire_ch, 0, 1)
                await dp.send_and_await_ack(
                    client,
                    bytes(frame_auto),
                    expect_cmd=dp.CMD_MANUAL_DOSE,
                    expect_ack_mode=0x07,
                    ack_timeout_ms=400,
                    heartbeat_min_ms=300,
                    heartbeat_max_ms=900,
                    retries=1,
                )
            except BLEAK_EXC as e:
                raise HomeAssistantError(f"BLE temporarily unavailable: {e}") from e
            except Exception as e:
                raise HomeAssistantError(f"Enable auto failed: {e}") from e
            finally:
                if client:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

        hass.services.async_register(DOMAIN, "enable_auto_mode", _svc_enable_auto, schema=ENABLE_AUTO_SCHEMA)

        # -------------------------
        # Read auto settings (passive)
        # -------------------------
        async def _svc_read_auto(call: ServiceCall):
            data = READ_PASSIVE_SCHEMA(dict(call.data))
            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError("Provide address or a device_id linked to a BLE address")

            ch_id = data.get("channel")
            timeout_s = float(data.get("timeout_s", 2.0))
            ble_dev = bluetooth.async_ble_device_from_address(hass, addr.upper(), True)
            if not ble_dev:
                raise HomeAssistantError(f"Could not find BLE device for address {addr}")

            buf: list[str] = []
            client = None
            try:
                client = await establish_connection(BleakClientWithServiceCache, ble_dev, f"{DOMAIN}-read-auto")

                def _cb(_char, payload: bytearray) -> None:
                    try:
                        b = bytes(payload)
                        buf.append(f'["{b[0] if b else 0}", "{b[5] if len(b)>5 else 0}", {list(b[6:-1])}]')
                    except Exception:
                        pass

                await client.start_notify(UART_TX, _cb)
                await asyncio.sleep(max(0.3, timeout_s))
            except BLEAK_EXC as e:
                raise HomeAssistantError(f"BLE temporarily unavailable: {e}") from e
            finally:
                if client:
                    try:
                        await client.stop_notify(UART_TX)
                    except Exception:
                        pass
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

            # Decode and present results
            try:
                recs = dp.parse_log_blob("\n".join(buf))
                dec = dp.decode_records(recs)
                state = dp.build_device_state(dec)
                lines = dp.to_ctl_lines(state)
                if ch_id is not None:
                    prefix = f"ch{int(ch_id)}."
                    lines = [ln for ln in lines if ln.startswith(prefix)]
            except Exception as e:
                raise HomeAssistantError(f"Decode failed: {e}") from e

            message = "\n".join(lines) if lines else "No auto schedule information observed."
            await hass.services.async_call(
                "persistent_notification", "create",
                {"title": "Chihiros Doser — Auto schedule", "message": message},
                blocking=False,
            )

        hass.services.async_register(DOMAIN, "read_auto_settings", _svc_read_auto, schema=READ_PASSIVE_SCHEMA)

        # -------------------------
        # Read container/tank status (passive, raw hex)
        # -------------------------
        async def _svc_read_container(call: ServiceCall):
            data = READ_PASSIVE_SCHEMA(dict(call.data))
            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError("Provide address or a device_id linked to a BLE address")

            timeout_s = float(data.get("timeout_s", 2.0))
            ble_dev = bluetooth.async_ble_device_from_address(hass, addr.upper(), True)
            if not ble_dev:
                raise HomeAssistantError(f"Could not find BLE device for address {addr}")

            lines: list[str] = []
            client = None
            try:
                client = await establish_connection(BleakClientWithServiceCache, ble_dev, f"{DOMAIN}-read-container")

                def _cb(_char, payload: bytearray) -> None:
                    lines.append(bytes(payload).hex(" ").upper())

                await client.start_notify(UART_TX, _cb)
                await asyncio.sleep(max(0.3, timeout_s))
            except BLEAK_EXC as e:
                raise HomeAssistantError(f"BLE temporarily unavailable: {e}") from e
            finally:
                if client:
                    try:
                        await client.stop_notify(UART_TX)
                    except Exception:
                        pass
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

            msg = (
                "Container/Tank notifications:\n  " + "\n  ".join(lines)
                if lines
                else "No container/tank notifications observed."
            )
            await hass.services.async_call(
                "persistent_notification", "create",
                {"title": "Chihiros Doser — Container status", "message": msg},
                blocking=False,
            )

        hass.services.async_register(DOMAIN, "read_container_status", _svc_read_container, schema=READ_PASSIVE_SCHEMA)

        # -------------------------
        # Raw A5/5B command sender (advanced)
        # -------------------------
        async def _svc_raw_cmd(call: ServiceCall):
            data = RAW_CMD_SCHEMA(dict(call.data))
            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError("Provide address or a device_id linked to a BLE address")

            cmd_id = int(data["cmd_id"])
            mode   = int(data["mode"])
            params = [int(p) & 0xFF for p in (data.get("params") or [])]
            repeats = int(data.get("repeats", 1))

            ble_dev = bluetooth.async_ble_device_from_address(hass, addr.upper(), True)
            if not ble_dev:
                raise HomeAssistantError(f"Could not find BLE device for address {addr}")

            from .dosingcommands import create_command_encoding_dosing_pump

            client = None
            try:
                client = await establish_connection(BleakClientWithServiceCache, ble_dev, f"{DOMAIN}-raw-cmd")
                msg_hi, msg_lo = 0, 0
                for _ in range(repeats):
                    frame = create_command_encoding_dosing_pump(cmd_id, mode, (msg_hi, msg_lo), params)
                    await client.write_gatt_char(dp.UART_RX, frame, response=True)
                    msg_lo = (msg_lo + 1) & 0xFF
            except BLEAK_EXC as e:
                raise HomeAssistantError(f"BLE temporarily unavailable: {e}") from e
            except Exception as e:
                raise HomeAssistantError(f"Raw command failed: {e}") from e
            finally:
                if client:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

        hass.services.async_register(DOMAIN, "raw_doser_command", _svc_raw_cmd, schema=RAW_CMD_SCHEMA)

        # -------------------------
        # Read & print daily totals — DISABLED
        # -------------------------
        if DISABLE_READ_DAILY_TOTALS:
            async def _svc_disabled_read(_call: ServiceCall):
                raise HomeAssistantError("Disabled: 'read_daily_totals' is not available in this build.")
            hass.services.async_register(DOMAIN, "read_daily_totals", _svc_disabled_read)
        else:
            # hass.services.async_register(DOMAIN, "read_daily_totals", _svc_read_totals, schema=READ_TOTALS_SCHEMA)
            pass

        # -------------------------
        # Configure 24-hour dosing — DISABLED
        # -------------------------
        if DISABLE_SET_24H_DOSE:
            async def _svc_disabled_set24h(_call: ServiceCall):
                raise HomeAssistantError("Disabled: 'set_24h_dose' is not available in this build.")
            hass.services.async_register(DOMAIN, "set_24h_dose", _svc_disabled_set24h)
        else:
            # hass.services.async_register(DOMAIN, "set_24h_dose", _svc_set_24h, schema=SET_24H_SCHEMA)
            pass

    # ────────────────────────────────────────────────────────────
    # ORIGINAL (kept for reference) — not registered while disabled
    # ────────────────────────────────────────────────────────────
    async def _svc_read_totals(call: ServiceCall):
        """Original handler retained but **not registered** while disabled."""
        data = READ_TOTALS_SCHEMA(dict(call.data))

        addr = data.get("address")
        if not addr and (did := data.get("device_id")):
            addr = await _resolve_address_from_device_id(hass=call.hass, did=did)
        if not addr:
            raise HomeAssistantError("Provide address or a device_id linked to a BLE address")

        addr_u = addr.upper()
        addr_l = addr_u.lower()

        ble_dev = bluetooth.async_ble_device_from_address(call.hass, addr_u, True)
        if not ble_dev:
            raise HomeAssistantError(f"Could not find BLE device for address {addr_u}")

        got: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

        def _on_notify(_char, payload: bytearray) -> None:
            try:
                if isinstance(payload, (bytes, bytearray)) and len(payload) >= 6 and payload[0] in (0x5B, 91):
                    _LOGGER.debug(
                        "read: notify 0x5B mode=0x%02X raw=%s",
                        payload[5], bytes(payload).hex(" ").upper()
                    )
                vals = dp.parse_totals_frame(payload)
                if vals and not got.done():
                    got.set_result(bytes(payload))
            except Exception:
                pass

        client = None
        try:
            client = await establish_connection(BleakClientWithServiceCache, ble_dev, f"{DOMAIN}-read-totals")
            await client.start_notify(UART_TX, _on_notify)

            for frame in _build_totals_probes():
                try:
                    try:
                        await client.write_gatt_char(dp.UART_RX, frame, response=True)
                    except Exception:
                        await client.write_gatt_char(dp.UART_TX, frame, response=True)
                except Exception:
                    pass
                await asyncio.sleep(0.08)

            try:
                payload = await asyncio.wait_for(got, timeout=8.0)
                vals = dp.parse_totals_frame(payload) or [None, None, None, None]
                msg = (
                    f"Daily totals for {addr_u}:\n"
                    f"  Ch1: {vals[0]} mL\n"
                    f"  Ch2: {vals[1]} mL\n"
                    f"  Ch3: {vals[2]} mL\n"
                    f"  Ch4: {vals[3]} mL"
                )
                await call.hass.services.async_call(
                    "persistent_notification", "create",
                    {"title": "Chihiros Doser — Daily totals", "message": msg},
                    blocking=False,
                )

                async_dispatcher_send(call.hass, f"{DOMAIN}_push_totals_{addr_l}", {"ml": vals, "raw": payload})

                if (entry_id := _find_entry_id_for_address(call.hass, addr_u)):
                    async_dispatcher_send(call.hass, f"{DOMAIN}_{entry_id}_refresh_totals")

            except asyncio.TimeoutError:
                raise HomeAssistantError("No totals frame received (timeout). Try again.")
            finally:
                try:
                    await client.stop_notify(UART_TX)
                except Exception:
                    pass

        except BLEAK_EXC as e:
            raise HomeAssistantError(f"BLE temporarily unavailable: {e}") from e
        except Exception as e:
            raise HomeAssistantError(f"Totals read failed: {e}") from e
        finally:
            if client:
                try:
                    await client.disconnect()
                except Exception:
                    pass

    async def _svc_set_24h(call: ServiceCall):
        """Original handler retained but **not registered** while disabled."""
        data = SET_24H_SCHEMA(dict(call.data))

        # Resolve address
        addr = data.get("address")
        if not addr and (did := data.get("device_id")):
            addr = await _resolve_address_from_device_id(call.hass, did)
        if not addr:
            raise HomeAssistantError("Provide address or a device_id linked to a BLE address")

        addr_u = addr.upper()
        addr_l = addr_u.lower()

        channel = int(data["channel"])
        daily_ml = float(data["daily_ml"])
        catch_up = bool(data["catch_up"])

        # Parse time-of-day: prefer "time": "HH:MM", else hour+minutes
        hour: int | None = None
        minute: int | None = None
        if data.get("time"):
            try:
                parts = str(data["time"]).strip().split(":")
                hour = int(parts[0]); minute = int(parts[1])
            except Exception as e:
                raise HomeAssistantError(f"Invalid time format (expected HH:MM): {data['time']}") from e
        else:
            hour = data.get("hour")
            minute = data.get("minutes")

        if hour is None or minute is None:
            raise HomeAssistantError("Provide either 'time': 'HH:MM' or both 'hour' and 'minutes'.")

        if not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
            raise HomeAssistantError("Time out of range: hour 0..23, minutes 0..59.")

        hour = int(hour)
        minute = int(minute)

        # Build weekday bitmask from human string/list or numeric mask
        weekday_mask = _parse_weekdays_to_mask(
            data.get("weekdays"),
            fallback_mask=(data.get("weekday_mask", 0x7F) if data.get("weekday_mask") is not None else 0x7F),
        )
        days_str = _weekdays_mask_to_english(weekday_mask)

        # Wire channel is 0-based in these frames
        wire_ch = max(1, min(channel, 4)) - 1

        # Split daily mL into (hi, lo) using the same 25.6 + 0.1 method
        hi = int(daily_ml // 25.6)
        lo = int(round((daily_ml - hi * 25.6) * 10))
        if lo == 256:
            hi += 1
            lo = 0
        hi &= 0xFF
        lo &= 0xFF

        # Frames (per your captures):
        #   0x15 → [ch, 1 /* 24h */, hour, minutes, 0, 0]
        #   0x1B → [ch, weekday_mask, 1 /*?*/, 0 /*completed today?*/, hi, lo]
        #   0x20 → [ch, 0 /*?*/, 1 if catch_up else 0]
        f_prelude  = _build_prelude_frames()
        f_schedule = dp._encode(dp.CMD_MANUAL_DOSE, 0x15, [wire_ch, 1, hour, minute, 0, 0])
        f_daily_ml = dp._encode(dp.CMD_MANUAL_DOSE, 0x1B, [wire_ch, weekday_mask, 1, 0, hi, lo])
        f_catchup  = dp._encode(dp.CMD_MANUAL_DOSE, 0x20, [wire_ch, 0, 1 if catch_up else 0])

        # Connect and apply
        ble_dev = bluetooth.async_ble_device_from_address(call.hass, addr_u, True)
        if not ble_dev:
            raise HomeAssistantError(f"Could not find BLE device for address {addr_u}")

        client = None
        try:
            client = await establish_connection(BleakClientWithServiceCache, ble_dev, f"{DOMAIN}-set-24h")

            # Send small prelude (stabilizes programming on some firmwares)
            for frame in f_prelude:
                try:
                    await client.write_gatt_char(dp.UART_RX, frame, response=True)
                except Exception:
                    pass
                await asyncio.sleep(0.08)

            # Then the actual 24h configuration
            for frame in (f_schedule, f_daily_ml, f_catchup):
                try:
                    await client.write_gatt_char(dp.UART_RX, frame, response=True)
                except Exception:
                    try:
                        await client.write_gatt_char(dp.UART_TX, frame, response=True)
                    except Exception:
                        pass
                await asyncio.sleep(0.10)

            # Notify user via persistent_notification
            msg = (
                f"Configured 24-hour dosing for {addr_u}:\n"
                f"  Channel: {channel}\n"
                f"  Daily total: {daily_ml:.1f} mL\n"
                f"  Time: {hour:02d}:{minute:02d}\n"
                f"  Days: {days_str} (mask=0x{weekday_mask:02X})\n"
                f"  Catch-up: {'on' if catch_up else 'off'}"
            )
            await call.hass.services.async_call(
                "persistent_notification", "create",
                {"title": "Chihiros Doser — 24h program set", "message": msg},
                blocking=False,
            )

            # Ask sensors to refresh
            if (entry_id := _find_entry_id_for_address(call.hass, addr_u)):
                async_dispatcher_send(call.hass, f"{DOMAIN}_{entry_id}_refresh_totals")
            async_dispatcher_send(call.hass, f"{DOMAIN}_refresh_totals_{addr_l}")

        except BLEAK_EXC as e:
            raise HomeAssistantError(f"BLE temporarily unavailable: {e}") from e
        except Exception as e:
            raise HomeAssistantError(f"Failed to set 24-hour dosing: {e}") from e
        finally:
            if client:
                try:
                    await client.disconnect()
                except Exception:
                    pass

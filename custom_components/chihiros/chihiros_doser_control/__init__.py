# custom_components/chihiros/chihiros_doser_control/__init__.py
"""
Chihiros Doser — Home Assistant service surface (runtime-only).

This module exposes HA services for Chihiros doser devices and now routes
*all* BLE traffic through the central coordinator created in __init__.py.
The coordinator owns the long-lived BLE session, keeps the link alive, and
fans out notifications to subscribers via the HA dispatcher. Older code paths
that opened ad-hoc connections by resolving MAC addresses have been removed in
favor of delegating every operation to the coordinator-managed session.

Services registered (subject to build flags):
- chihiros.dose_ml
- chihiros.enable_auto_mode
- chihiros.read_auto_settings
- chihiros.read_container_status
- chihiros.raw_doser_command
(“read_daily_totals” and “set_24h_dose” are disabled by default in this build)
"""

from __future__ import annotations

__all__ = ["register_services"]

try:
    import asyncio
    import logging
    import time
    import voluptuous as vol

    from homeassistant.core import HomeAssistant, ServiceCall
    from homeassistant.helpers import device_registry as dr
    from homeassistant.exceptions import HomeAssistantError
    from homeassistant.helpers.dispatcher import (
        async_dispatcher_send,
        async_dispatcher_connect,
    )

    # use HA’s bluetooth helper + bleak-retry-connector (slot-aware, proxy-friendly)
    from bleak_retry_connector import BLEAK_RETRY_EXCEPTIONS as BLEAK_EXC

    from ..const import DOMAIN  # integration domain
    from . import protocol as dp  # protocol helpers (dose_ml, parse_totals_frame, etc.)
    from .protocol import UART_TX  # notify UUID for totals frames

    from ..chihiros_led_control.weekday_encoding import (
        WeekdaySelect,
        encode_selected_weekdays,
    )

    HA_AVAILABLE = True
except ModuleNotFoundError:
    HA_AVAILABLE = False

if not HA_AVAILABLE:
    import logging

    _LOGGER = logging.getLogger(__name__)

    async def register_services(*_args, **_kwargs) -> None:
        raise RuntimeError(
            "Chihiros_doser_control: services require Home Assistant runtime. "
            "This module can be imported safely by the CLI, "
            "but service registration only works inside Home Assistant."
        )

else:
    _LOGGER = logging.getLogger(__name__)

    # Feature toggles (keep disabled unless you explicitly test them)
    DISABLE_READ_DAILY_TOTALS: bool = True
    DISABLE_SET_24H_DOSE: bool = True

    # ────────────────────────────────────────────────────────────
    # Schemas
    # ────────────────────────────────────────────────────────────
    DOSE_SCHEMA = vol.Schema(
        {
            vol.Exclusive("device_id", "target"): str,
            vol.Exclusive("address", "target"): str,
            vol.Required("channel"): vol.All(vol.Coerce(int), vol.Range(min=1, max=4)),
            vol.Required("ml"): vol.All(
                vol.Coerce(float), vol.Range(min=0.2, max=999.9)
            ),
        }
    )

    READ_TOTALS_SCHEMA = vol.Schema(
        {
            vol.Exclusive("device_id", "target"): str,
            vol.Exclusive("address", "target"): str,
        }
    )

    SET_24H_SCHEMA = vol.Schema(
        {
            vol.Exclusive("device_id", "target"): str,
            vol.Exclusive("address", "target"): str,
            vol.Required("channel"): vol.All(vol.Coerce(int), vol.Range(min=1, max=4)),
            vol.Required("daily_ml"): vol.All(
                vol.Coerce(float), vol.Range(min=0.2, max=999.9)
            ),
            vol.Optional("time"): str,
            vol.Optional("hour", default=None): vol.Any(
                None, vol.All(vol.Coerce(int), vol.Range(min=0, max=23))
            ),
            vol.Optional("minutes", default=None): vol.Any(
                None, vol.All(vol.Coerce(int), vol.Range(min=0, max=59))
            ),
            vol.Optional("weekday_mask", default=None): vol.Any(
                None, vol.All(vol.Coerce(int), vol.Range(min=0, max=0x7F))
            ),
            vol.Optional("weekdays", default=None): vol.Any(str, [str]),
            vol.Optional("catch_up", default=False): vol.Boolean(),
        }
    )

    ENABLE_AUTO_SCHEMA = vol.Schema(
        {
            vol.Exclusive("device_id", "target"): str,
            vol.Exclusive("address", "target"): str,
            vol.Required("channel"): vol.All(vol.Coerce(int), vol.Range(min=1, max=4)),
        }
    )

    READ_PASSIVE_SCHEMA = vol.Schema(
        {
            vol.Exclusive("device_id", "target"): str,
            vol.Exclusive("address", "target"): str,
            vol.Optional("channel"): vol.All(vol.Coerce(int), vol.Range(min=1, max=4)),
            vol.Optional("timeout_s", default=2.0): vol.All(
                vol.Coerce(float), vol.Range(min=0.2, max=30.0)
            ),
        }
    )

    RAW_CMD_SCHEMA = vol.Schema(
        {
            vol.Exclusive("device_id", "target"): str,
            vol.Exclusive("address", "target"): str,
            vol.Required("cmd_id"): vol.All(vol.Coerce(int), vol.Range(min=0, max=255)),
            vol.Required("mode"): vol.All(vol.Coerce(int), vol.Range(min=0, max=255)),
            vol.Required("params"): list,
            vol.Optional("repeats", default=1): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=20)
            ),
        }
    )

    _WEEKDAY_ALIAS = {
        "mon": WeekdaySelect.monday,
        "monday": WeekdaySelect.monday,
        "tue": WeekdaySelect.tuesday,
        "tues": WeekdaySelect.tuesday,
        "tuesday": WeekdaySelect.tuesday,
        "wed": WeekdaySelect.wednesday,
        "wednesday": WeekdaySelect.wednesday,
        "thu": WeekdaySelect.thursday,
        "thur": WeekdaySelect.thursday,
        "thurs": WeekdaySelect.thursday,
        "thursday": WeekdaySelect.thursday,
        "fri": WeekdaySelect.friday,
        "friday": WeekdaySelect.friday,
        "sat": WeekdaySelect.saturday,
        "saturday": WeekdaySelect.saturday,
        "sun": WeekdaySelect.sunday,
        "sunday": WeekdaySelect.sunday,
        "everyday": "ALL",
        "every day": "ALL",
        "any": "ALL",
        "all": "ALL",
    }

    def _parse_weekdays_to_mask(value, fallback_mask: int = 0x7F) -> int:
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
        try:
            m = int(mask) & 0x7F
        except Exception:
            return "Unknown"
        parts = []
        if m & 64:
            parts.append("Mon")
        if m & 32:
            parts.append("Tue")
        if m & 16:
            parts.append("Wed")
        if m & 8:
            parts.append("Thu")
        if m & 4:
            parts.append("Fri")
        if m & 2:
            parts.append("Sat")
        if m & 1:
            parts.append("Sun")
        return "Every day" if len(parts) == 7 else (", ".join(parts) if parts else "None")

    # ────────────────────────────────────────────────────────────
    # Address / entry / session helpers
    # ────────────────────────────────────────────────────────────
    async def _resolve_address_from_device_id(hass: HomeAssistant, did: str) -> str | None:
        reg = dr.async_get(hass)
        dev = reg.async_get(did)
        if not dev:
            return None

        for (conn_type, conn_val) in dev.connections:
            if conn_type == dr.CONNECTION_BLUETOOTH:
                return conn_val

        for domain, ident in dev.identifiers:
            if domain != DOMAIN:
                continue
            if isinstance(ident, str) and ident.startswith("ble:"):
                return ident.split(":", 1)[1]
            data_by_entry = hass.data.get(DOMAIN, {})
            data = data_by_entry.get(ident)
            if data and hasattr(data.coordinator, "address"):
                return data.coordinator.address

        for entry_id in getattr(dev, "config_entries", set()):
            data_by_entry = hass.data.get(DOMAIN, {})
            data = data_by_entry.get(entry_id)
            if data and hasattr(data.coordinator, "address"):
                return data.coordinator.address

        return None

    def _find_entry_id_for_address(hass: HomeAssistant, addr: str) -> str | None:
        data_by_entry = hass.data.get(DOMAIN, {})
        addr_l = (addr or "").lower()
        for entry_id, data in data_by_entry.items():
            coord = getattr(data, "coordinator", None)
            c_addr = getattr(coord, "address", None)
            if isinstance(c_addr, str) and c_addr.lower() == addr_l:
                return entry_id
        return None

    def _get_coordinator_for_address(hass: HomeAssistant, addr: str):
        entry_id = _find_entry_id_for_address(hass, addr)
        if not entry_id:
            return None
        data_by_entry = hass.data.get(DOMAIN, {})
        data = data_by_entry.get(entry_id)
        if not data:
            return None
        return getattr(data, "coordinator", None)

    async def _run_via_coordinator(coordinator, fn, *, use_gate: bool = False):
        """Ensure the BLE client is available via the central coordinator."""

        if coordinator is None:
            raise HomeAssistantError("Coordinator unavailable for device")

        gate = getattr(coordinator, "write_gate", None)
        if use_gate and isinstance(gate, asyncio.Lock):
            async with gate:
                return await coordinator.async_with_client(fn)
        return await coordinator.async_with_client(fn)

    # ────────────────────────────────────────────────────────────
    # Probe / prelude helpers
    # ────────────────────────────────────────────────────────────
    def _build_totals_probes() -> list[bytes]:
        frames: list[bytes] = []
        try:
            if hasattr(dp, "encode_5b"):
                # prefer 0x34; also try 0x22 (some fw)
                frames.append(dp.encode_5b(0x34, []))
                frames.append(dp.encode_5b(0x22, []))
        except Exception:
            pass
        # conservative extras (A5-encoded) if needed
        try:
            frames.append(dp._encode(dp.CMD_MANUAL_DOSE, 0x1E, []))
        except Exception:
            pass
        try:
            frames.append(dp._encode(dp.CMD_MANUAL_DOSE, 0x22, []))
        except Exception:
            pass
        # de-dup preserving order
        seen, out = set(), []
        for f in frames:
            b = bytes(f)
            if b not in seen:
                seen.add(b)
                out.append(f)
        return out

    def _build_prelude_frames() -> list[bytes]:
        return [
            dp._encode(90, 4, [1]),
            dp._encode(165, 4, [4]),
            dp._encode(165, 4, [5]),
        ]

    async def _notify(hass: HomeAssistant, title: str, message: str) -> None:
        await hass.services.async_call(
            "persistent_notification",
            "create",
            {"title": title, "message": message},
            blocking=False,
        )

    # ────────────────────────────────────────────────────────────
    # Service registration
    # ────────────────────────────────────────────────────────────
    async def register_services(hass: HomeAssistant) -> None:
        flag_key = f"{DOMAIN}_doser_services_registered"
        if hass.data.get(flag_key):
            return
        hass.data[flag_key] = True

        # -------------------------
        # Manual one-shot dose
        # -------------------------
        async def _svc_dose(call: ServiceCall):
            started = time.perf_counter()
            data = DOSE_SCHEMA(dict(call.data))

            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError(
                    "Provide address or a device_id linked to a BLE address"
                )

            addr_u = addr.upper()
            addr_l = addr_u.lower()
            channel = int(data["channel"])
            ml = round(float(data["ml"]), 1)

            coordinator = _get_coordinator_for_address(hass, addr_u)
            if not coordinator:
                raise HomeAssistantError(
                    "Chihiros device not loaded — ensure the integration is set up"
                )

            totals_text = "no totals within timeout"
            loop = asyncio.get_running_loop()
            got: asyncio.Future[dict] = loop.create_future()

            def _on_push(payload: dict):
                if not got.done():
                    got.set_result(payload)

            unsub = async_dispatcher_connect(
                hass, f"{DOMAIN}_push_totals_{addr_l}", _on_push
            )

            async def _dose_with_client(client):
                await dp.dose_ml(client, channel_1based=channel, ml=ml)
                for frame in _build_totals_probes():
                    try:
                        try:
                            await client.write_gatt_char(
                                dp.UART_RX, frame, response=True
                            )
                        except Exception:
                            await client.write_gatt_char(
                                dp.UART_TX, frame, response=True
                            )
                    except Exception:
                        _LOGGER.debug(
                            "dose_ml(coordinator): probe write failed", exc_info=True
                        )
                    await asyncio.sleep(0.08)

            try:
                try:
                    await _run_via_coordinator(
                        coordinator, _dose_with_client, use_gate=True
                    )
                except BLEAK_EXC as e:
                    _LOGGER.warning("dose_ml: BLE unavailable: %s", e)
                    raise HomeAssistantError(
                        f"BLE temporarily unavailable: {e}"
                    ) from e
                except Exception as e:
                    _LOGGER.exception("dose_ml: failed")
                    raise HomeAssistantError(f"Dose failed: {e}") from e

                try:
                    res = await asyncio.wait_for(got, timeout=8.0)
                    vals = res.get("ml") or []
                    totals_text = f"totals={vals}"
                    _LOGGER.info("dose_ml(coordinator): received totals %s", vals)
                except asyncio.TimeoutError:
                    _LOGGER.info(
                        "dose_ml(coordinator): no totals notify received within 8s"
                    )
            finally:
                try:
                    unsub()
                except Exception:
                    pass

            if (entry_id := _find_entry_id_for_address(hass, addr_u)):
                async_dispatcher_send(
                    hass, f"{DOMAIN}_{entry_id}_refresh_totals"
                )

            elapsed = (time.perf_counter() - started) * 1000
            await _notify(
                hass,
                "Chihiros Doser — Dose Now",
                f"Address: {addr_u}\nChannel: {channel}\nDose: {ml:.1f} mL\nResult: {totals_text}\nTook: {elapsed:.0f} ms",
            )

        hass.services.async_register(
            DOMAIN, "dose_ml", _svc_dose, schema=DOSE_SCHEMA
        )

        # -------------------------
        # Enable auto mode (and sync time)
        # -------------------------
        async def _svc_enable_auto(call: ServiceCall):
            started = time.perf_counter()
            data = ENABLE_AUTO_SCHEMA(dict(call.data))

            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError(
                    "Provide address or a device_id linked to a BLE address"
                )

            channel = int(data["channel"])
            _LOGGER.info("enable_auto_mode: addr=%s ch=%s", addr, channel)

            coordinator = _get_coordinator_for_address(hass, addr.upper())
            if not coordinator:
                raise HomeAssistantError(
                    "Chihiros device not loaded — ensure the integration is set up"
                )

            from .dosingcommands import (
                create_set_time_command,
                create_switch_to_auto_mode_dosing_pump_command,
            )

            msg_hi, msg_lo = 0, 0
            frame_time = create_set_time_command((msg_hi, msg_lo))
            msg_hi, msg_lo = 0, 1
            wire_ch = max(1, min(channel, 4)) - 1
            frame_auto = create_switch_to_auto_mode_dosing_pump_command(
                (msg_hi, msg_lo), wire_ch, 0, 1
            )

            def _cb(_char, payload: bytearray):
                raw = bytes(payload)
                head = f"{raw[0]:02X}" if raw else "??"
                mode = f"{raw[5]:02X}" if len(raw) > 6 else "??"
                _LOGGER.debug(
                    "enable_auto_mode: notify head=%s mode=%s raw=%s",
                    head,
                    mode,
                    raw.hex(" ").upper(),
                )

            coordinator.add_notify_callback(_cb)

            async def _send_frames(client):
                try:
                    await client.write_gatt_char(
                        dp.UART_RX, frame_time, response=True
                    )
                    _LOGGER.debug(
                        "enable_auto_mode: sent time sync %s",
                        frame_time.hex(" ").upper(),
                    )
                except Exception:
                    _LOGGER.debug(
                        "enable_auto_mode: failed to write time sync to RX", exc_info=True
                    )
                    raise

                try:
                    await client.write_gatt_char(
                        dp.UART_RX, frame_auto, response=True
                    )
                    _LOGGER.debug("enable_auto_mode: wrote auto-mode frame to RX")
                except Exception:
                    await client.write_gatt_char(
                        dp.UART_TX, frame_auto, response=True
                    )
                    _LOGGER.debug(
                        "enable_auto_mode: RX failed, wrote to TX instead"
                    )

            try:
                try:
                    await _run_via_coordinator(
                        coordinator, _send_frames, use_gate=True
                    )
                except BLEAK_EXC as e:
                    _LOGGER.warning("enable_auto_mode: BLE unavailable: %s", e)
                    raise HomeAssistantError(
                        f"BLE temporarily unavailable: {e}"
                    ) from e
                except Exception as e:
                    _LOGGER.exception("enable_auto_mode: failed")
                    raise HomeAssistantError(f"Enable auto failed: {e}") from e

                _LOGGER.info("enable_auto_mode: listening 3s for ACK/notify…")
                await asyncio.sleep(3.0)
            finally:
                coordinator.remove_notify_callback(_cb)

            elapsed = (time.perf_counter() - started) * 1000
            await _notify(
                hass,
                "Chihiros Doser — Enable Auto",
                f"Address: {addr}\nChannel: {channel}\nResult: ACK window complete\nTook: {elapsed:.0f} ms",
            )

        hass.services.async_register(
            DOMAIN, "enable_auto_mode", _svc_enable_auto, schema=ENABLE_AUTO_SCHEMA
        )

        # -------------------------
        # Read auto settings (passive)
        # -------------------------
        async def _svc_read_auto(call: ServiceCall):
            data = READ_PASSIVE_SCHEMA(dict(call.data))
            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError(
                    "Provide address or a device_id linked to a BLE address"
                )

            ch_id = data.get("channel")
            timeout_s = float(data.get("timeout_s", 2.0))
            _LOGGER.info(
                "read_auto_settings: addr=%s timeout=%.1fs ch=%s",
                addr,
                timeout_s,
                ch_id,
            )

            buf: list[str] = []
            coordinator = _get_coordinator_for_address(hass, addr.upper())
            if not coordinator:
                raise HomeAssistantError(
                    "Chihiros device not loaded — ensure the integration is set up"
                )

            def _cb(_char, payload: bytearray) -> None:
                try:
                    b = bytes(payload)
                    buf.append(
                        f'["{b[0] if b else 0}", "{b[5] if len(b)>5 else 0}", {list(b[6:-1])}]'
                    )
                    if len(buf) <= 3:
                        _LOGGER.debug(
                            "read_auto_settings: sample notify %s",
                            b.hex(" ").upper(),
                        )
                except Exception:
                    pass

            coordinator.add_notify_callback(_cb)

            async def _listen(_client):
                await asyncio.sleep(max(0.3, timeout_s))

            try:
                try:
                    await coordinator.async_with_client(_listen)
                except BLEAK_EXC as e:
                    _LOGGER.warning("read_auto_settings: BLE unavailable: %s", e)
                    raise HomeAssistantError(
                        f"BLE temporarily unavailable: {e}"
                    ) from e
            finally:
                coordinator.remove_notify_callback(_cb)

            try:
                recs = dp.parse_log_blob("\n".join(buf))
                dec = dp.decode_records(recs)
                state = dp.build_device_state(dec)
                lines = dp.to_ctl_lines(state)
                if ch_id is not None:
                    prefix = f"ch{int(ch_id)}."
                    lines = [ln for ln in lines if ln.startswith(prefix)]
            except Exception as e:
                _LOGGER.exception("read_auto_settings: decode failed")
                raise HomeAssistantError(f"Decode failed: {e}") from e

            message = f"Observed {len(buf)} frame(s)\n\n" + (
                "\n".join(lines) if lines else "No auto schedule information observed."
            )
            await _notify(hass, "Chihiros Doser — Auto schedule", message)

        hass.services.async_register(
            DOMAIN, "read_auto_settings", _svc_read_auto, schema=READ_PASSIVE_SCHEMA
        )

        # -------------------------
        # Read container/tank status (passive, raw hex)
        # -------------------------
        async def _svc_read_container(call: ServiceCall):
            data = READ_PASSIVE_SCHEMA(dict(call.data))
            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError(
                    "Provide address or a device_id linked to a BLE address"
                )

            timeout_s = float(data.get("timeout_s", 2.0))
            _LOGGER.info(
                "read_container_status: addr=%s timeout=%.1fs", addr, timeout_s
            )

            lines: list[str] = []
            coordinator = _get_coordinator_for_address(hass, addr.upper())
            if not coordinator:
                raise HomeAssistantError(
                    "Chihiros device not loaded — ensure the integration is set up"
                )

            def _cb(_char, payload: bytearray) -> None:
                raw = bytes(payload).hex(" ").upper()
                lines.append(raw)
                if len(lines) <= 3:
                    _LOGGER.debug(
                        "read_container_status: sample notify %s", raw
                    )

            coordinator.add_notify_callback(_cb)

            async def _listen(_client):
                await asyncio.sleep(max(0.3, timeout_s))

            try:
                try:
                    await coordinator.async_with_client(_listen)
                except BLEAK_EXC as e:
                    _LOGGER.warning("read_container_status: BLE unavailable: %s", e)
                    raise HomeAssistantError(
                        f"BLE temporarily unavailable: {e}"
                    ) from e
            finally:
                coordinator.remove_notify_callback(_cb)

            preview = "\n  ".join(lines[:8])
            msg = (
                f"Observed {len(lines)} frame(s).\n"
                + ("Sample:\n  " + preview if lines else "No container/tank notifications observed.")
            )
            await _notify(hass, "Chihiros Doser — Container status", msg)

        hass.services.async_register(
            DOMAIN,
            "read_container_status",
            _svc_read_container,
            schema=READ_PASSIVE_SCHEMA,
        )

        # -------------------------
        # Raw A5/5B command sender (advanced)
        # -------------------------
        async def _svc_raw_cmd(call: ServiceCall):
            data = RAW_CMD_SCHEMA(dict(call.data))
            addr = data.get("address")
            if not addr and (did := data.get("device_id")):
                addr = await _resolve_address_from_device_id(hass, did)
            if not addr:
                raise HomeAssistantError(
                    "Provide address or a device_id linked to a BLE address"
                )

            cmd_id = int(data["cmd_id"])
            mode = int(data["mode"])
            params = [int(p) & 0xFF for p in (data.get("params") or [])]
            repeats = int(data.get("repeats", 1))

            _LOGGER.info(
                "raw_doser_command: addr=%s cmd=%d mode=%d params=%s repeats=%d",
                addr,
                cmd_id,
                mode,
                params,
                repeats,
            )

            from .dosingcommands import create_command_encoding_dosing_pump

            coordinator = _get_coordinator_for_address(hass, addr.upper())
            if not coordinator:
                raise HomeAssistantError(
                    "Chihiros device not loaded — ensure the integration is set up"
                )

            async def _send_frames(client):
                sent_local = 0
                msg_hi, msg_lo = 0, 0
                for _ in range(repeats):
                    frame = create_command_encoding_dosing_pump(
                        cmd_id, mode, (msg_hi, msg_lo), params
                    )
                    await client.write_gatt_char(
                        dp.UART_RX, frame, response=True
                    )
                    _LOGGER.debug(
                        "raw_doser_command: wrote %s", frame.hex(" ").upper()
                    )
                    sent_local += 1
                    msg_lo = (msg_lo + 1) & 0xFF
                return sent_local

            try:
                sent = await _run_via_coordinator(
                    coordinator, _send_frames, use_gate=True
                )
            except BLEAK_EXC as e:
                _LOGGER.warning("raw_doser_command: BLE unavailable: %s", e)
                raise HomeAssistantError(
                    f"BLE temporarily unavailable: {e}"
                ) from e
            except Exception as e:
                _LOGGER.exception("raw_doser_command: failed")
                raise HomeAssistantError(f"Raw command failed: {e}") from e

            await _notify(
                hass,
                "Chihiros Doser — Raw command",
                f"Address: {addr}\nCMD={cmd_id} MODE={mode}\nParams={params}\nSent: {sent} frame(s)",
            )

        hass.services.async_register(
            DOMAIN, "raw_doser_command", _svc_raw_cmd, schema=RAW_CMD_SCHEMA
        )

        # -------------------------
        # Read & print daily totals — DISABLED
        # -------------------------
        if DISABLE_READ_DAILY_TOTALS:
            async def _svc_disabled_read(_call: ServiceCall):
                raise HomeAssistantError(
                    "Disabled: 'read_daily_totals' is not available in this build."
                )

            hass.services.async_register(
                DOMAIN, "read_daily_totals", _svc_disabled_read
            )
        else:
            # (retain original implementation here when enabling)
            pass

    # Placeholders for disabled originals (kept for clarity)
    async def _svc_read_totals(call: ServiceCall):
        # … unchanged / disabled …
        pass

    async def _svc_set_24h(call: ServiceCall):
        # … unchanged / disabled …
        pass

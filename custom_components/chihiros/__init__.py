# custom_components/chihiros/__init__.py
"""Chihiros HA integration root module."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from .const import DOMAIN
from homeassistant.helpers.dispatcher import async_dispatcher_send  # safe top-level import

if TYPE_CHECKING:  # pragma: no cover
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigEntry

_LOGGER = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# Small helper
# ────────────────────────────────────────────────────────────────────────────────
def _guess_channel_count(name: str | None) -> int:
    """Best-effort channel count from BLE name."""
    s = (name or "").lower()
    if "d1" in s or "1ch" in s or "1-channel" in s:
        return 1
    if "d2" in s or "2ch" in s or "2-channel" in s:
        return 2
    if "d3" in s or "3ch" in s or "3-channel" in s:
        return 3
    if "d4" in s or "4ch" in s or "4-channel" in s:
        return 4
    return 4  # sensible default


# ────────────────────────────────────────────────────────────────────────────────
# HA entry setup / unload
# ────────────────────────────────────────────────────────────────────────────────
async def async_setup_entry(hass: "HomeAssistant", entry: "ConfigEntry") -> bool:
    """Set up chihiros from a config entry."""
    # Lazy HA imports to keep module import light
    from homeassistant.components import bluetooth
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.const import Platform, CONF_NAME
    from homeassistant.exceptions import ConfigEntryNotReady

    # Device classes
    from .chihiros_led_control.device import BaseDevice, get_model_class_from_name
    from .chihiros_led_control.device.commander1 import Commander1
    from .chihiros_led_control.device.commander4 import Commander4
    from .chihiros_led_control.device.fallback import Fallback
    from .chihiros_led_control.device.generic_rgb import GenericRGB
    from .chihiros_led_control.device.generic_white import GenericWhite
    from .chihiros_led_control.device.generic_wrgb import GenericWRGB

    # Our BLE session coordinator (persistent connection & keepalive)
    from .coordinator import ChihirosDataUpdateCoordinator

    # Model wrapper for hass.data
    from .models import ChihirosData

    # Doser helpers (services + totals decode)
    from .chihiros_doser_control import register_services as register_doser_services
    from .chihiros_doser_control import protocol as dp  # parse_totals_frame, etc.

    if entry.unique_id is None:
        raise ConfigEntryNotReady(f"Entry doesn't have any unique_id {entry.title}")

    address: str = entry.unique_id
    ble_device = bluetooth.async_ble_device_from_address(hass, address.upper(), True)
    if not ble_device:
        raise ConfigEntryNotReady(f"Could not find Chihiros BLE device with address {address}")
    if not ble_device.name:
        raise ConfigEntryNotReady(
            f"Found Chihiros BLE device with address {address} but can not find its name"
        )

    model_class = get_model_class_from_name(ble_device.name)
    chihiros_device: BaseDevice = model_class(ble_device)  # type: ignore[call-arg]

    # If fallback or commander device, allow user-provided name and device_type (white/rgb/wrgb)
    if isinstance(chihiros_device, (Fallback, Commander1, Commander4)):
        # Apply name override if provided
        entry_name = entry.data.get(CONF_NAME)
        if entry_name:
            try:
                ble_device.name = entry_name
            except Exception:
                pass
        # Configure colors based on device_type
        device_type = entry.data.get("device_type")
        if device_type == "rgb":
            chihiros_device = GenericRGB(ble_device)
        elif device_type == "wrgb":
            chihiros_device = GenericWRGB(ble_device)
        else:
            chihiros_device = GenericWhite(ble_device)

    # Create the persistent BLE coordinator
    coordinator = ChihirosDataUpdateCoordinator(
        hass=hass,
        client=chihiros_device,
        ble_device=ble_device,
    )

    # Classification + handy attributes
    is_doser = any(k in (ble_device.name or "").lower() for k in ("doser", "dose", "dydose"))
    coordinator.device_type = "doser" if is_doser else "led"  # type: ignore[attr-defined]
    coordinator.address = address  # type: ignore[attr-defined]

    # Options → explicit enabled channels (subset of 1..4). Fallback to “all 4” for dosers.
    opt_enabled = entry.options.get("enabled_channels")
    if opt_enabled:
        try:
            enabled = sorted({int(x) for x in opt_enabled if 1 <= int(x) <= 4})
        except Exception:
            enabled = [1, 2, 3, 4]
        if not enabled:
            enabled = [1]
        coordinator.enabled_channels = enabled  # type: ignore[attr-defined]
        coordinator.channel_count = len(enabled)  # type: ignore[attr-defined]
    else:
        if is_doser:
            coordinator.enabled_channels = [1, 2, 3, 4]  # type: ignore[attr-defined]
            coordinator.channel_count = 4  # type: ignore[attr-defined]
        else:
            guessed = _guess_channel_count(ble_device.name)
            coordinator.enabled_channels = list(range(1, guessed + 1))  # type: ignore[attr-defined]
            coordinator.channel_count = guessed  # type: ignore[attr-defined]

    # Choose platforms per device type
    platforms_to_load: list[Platform] = (
        [Platform.BUTTON, Platform.NUMBER, Platform.SENSOR]  # include sensors for doser
        if is_doser
        else [Platform.LIGHT, Platform.SWITCH, Platform.SENSOR]
    )

    # Register integration data
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = ChihirosData(entry.title, chihiros_device, coordinator)

    # Start the persistent BLE session now (non-blocking)
    await coordinator.async_start()

    # If this is a doser, attach a notify parser that *pushes* totals updates.
    if is_doser:
        # Optional: a write gate for services to serialize multi-frame operations, if you want it.
        coordinator.write_gate = asyncio.Lock()  # type: ignore[attr-defined]

        def _totals_notify_cb(_char, payload: bytearray) -> None:
            try:
                vals = dp.parse_totals_frame(bytes(payload))
                if vals:
                    async_dispatcher_send(
                        hass,
                        f"{DOMAIN}_push_totals_{address.lower()}",
                        {"ml": vals[:4], "raw": bytes(payload)},
                    )
            except Exception:  # pragma: no cover
                _LOGGER.debug("totals_notify_cb error", exc_info=True)

        coordinator.add_notify_callback(_totals_notify_cb)

        # Register doser services (idempotent inside submodule)
        await register_doser_services(hass)

    # Reload entry when options change
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    # Only load matching platforms
    await hass.config_entries.async_forward_entry_setups(entry, platforms_to_load)
    _LOGGER.debug(
        "Loaded platforms for %s (%s): %s", entry.title, coordinator.device_type, platforms_to_load
    )
    return True


async def async_unload_entry(hass: "HomeAssistant", entry: "ConfigEntry") -> bool:
    """Unload a config entry."""
    from homeassistant.const import Platform
    from .models import ChihirosData

    data: Optional[ChihirosData] = hass.data.get(DOMAIN, {}).get(entry.entry_id)  # type: ignore[assignment]
    if data and getattr(data.coordinator, "device_type", "led") == "doser":
        platforms_to_unload = [Platform.BUTTON, Platform.NUMBER, Platform.SENSOR]
    else:
        platforms_to_unload = [Platform.LIGHT, Platform.SWITCH, Platform.SENSOR]

    # Stop persistent session
    if data:
        try:
            await data.coordinator.async_stop()
        except Exception:  # pragma: no cover
            _LOGGER.debug("coordinator.async_stop() failed", exc_info=True)

    unload_ok = await hass.config_entries.async_unload_platforms(entry, platforms_to_unload)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok


async def _async_update_listener(hass: "HomeAssistant", entry: "ConfigEntry") -> None:
    """Handle options updates by reloading the entry."""
    await hass.config_entries.async_reload(entry.entry_id)


# Expose Options Flow at the component level so HA shows “Configure”
async def async_get_options_flow(config_entry: "ConfigEntry"):
    from .config_flow import ChihirosOptionsFlow
    return ChihirosOptionsFlow(config_entry)

# custom_components/chihiros/config_flow.py
"""Config flow for chihiros integration."""

from __future__ import annotations

import logging
from typing import Any, Dict

import voluptuous as vol
from homeassistant.components.bluetooth import (
    BluetoothServiceInfoBleak,
    async_discovered_service_info,
)
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_ADDRESS, CONF_NAME
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


def _is_doser_name(name: str | None) -> bool:
    if not name:
        return False
    n = name.lower()
    # Seen names: DYDOSE..., DYDOSEDF..., "Chihiros Doser", etc.
    return n.startswith("dydose") or "dose" in n


class ChihirosConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for chihiros."""
    VERSION = 1

    def __init__(self) -> None:
        self._discovery_info: BluetoothServiceInfoBleak | None = None
        self._discovered_device: Any | None = None
        self._discovered_devices: Dict[str, BluetoothServiceInfoBleak] = {}

    # ---------- Bluetooth auto-discovery ----------
    async def async_step_bluetooth(
        self, discovery_info: BluetoothServiceInfoBleak
    ) -> ConfigFlowResult:
        addr = (discovery_info.address or "").upper()
        # Use raise_on_progress=False to avoid abort races when multiple flows run
        await self.async_set_unique_id(addr, raise_on_progress=False)
        self._abort_if_unique_id_configured()

        # Route by advert name first (keep doser path import-light)
        if _is_doser_name(discovery_info.name):
            _LOGGER.debug("BLE discovered Chihiros DOSER: %s", discovery_info.name)
            self._discovery_info = discovery_info
            return await self.async_step_doser_confirm()

        # LED stack only when not a doser
        from .chihiros_led_control.device import get_model_class_from_name
        from .chihiros_led_control.device.commander1 import Commander1
        from .chihiros_led_control.device.commander4 import Commander4
        from .chihiros_led_control.device.fallback import Fallback

        model_class = get_model_class_from_name(discovery_info.name)
        device = model_class(discovery_info.device)
        self._discovery_info = discovery_info
        self._discovered_device = device

        if model_class in (Fallback, Commander1, Commander4):
            return await self.async_step_fallback_config()

        return await self.async_step_bluetooth_confirm()

    async def async_step_bluetooth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        assert self._discovery_info is not None
        addr = (self._discovery_info.address or "").upper()
        title = getattr(self._discovered_device, "name", None) or self._discovery_info.name or "Chihiros LED"

        if user_input is not None:
            return self.async_create_entry(title=title, data={CONF_ADDRESS: addr})

        self._set_confirm_only()
        return self.async_show_form(step_id="bluetooth_confirm", description_placeholders={"name": title})

    async def async_step_fallback_config(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        assert self._discovery_info is not None
        addr = (self._discovery_info.address or "").upper()
        default_name = getattr(self._discovered_device, "name", None) or self._discovery_info.name or "Chihiros LED"

        if user_input is not None:
            title = user_input.get(CONF_NAME, default_name)
            data = {
                CONF_ADDRESS: addr,
                CONF_NAME: title,
                # white|rgb|wrgb (used in __init__.py to pick Generic* model)
                "device_type": user_input["device_type"],
            }
            return self.async_create_entry(title=title, data=data)

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME, default=default_name): str,
                vol.Required("device_type", default="white"): vol.In(["white", "rgb", "wrgb"]),
            }
        )
        return self.async_show_form(step_id="fallback_config", data_schema=schema, errors={})

    # ---------- Doser path ----------
    async def async_step_doser_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        assert self._discovery_info is not None
        addr = (self._discovery_info.address or "").upper()
        title = self._discovery_info.name or "Chihiros Doser"

        if user_input is not None:
            # Store only the BLE address; __init__.py classifies and starts the pinned session.
            return self.async_create_entry(title=title, data={CONF_ADDRESS: addr})

        self._set_confirm_only()
        return self.async_show_form(step_id="doser_confirm", description_placeholders={"name": title})

    # ---------- Manual user-initiated path ----------
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        # Build list of discoverables (minus already-configured)
        if self._discovery_info:
            self._discovered_devices[self._discovery_info.address] = self._discovery_info
        else:
            current_ids = self._async_current_ids()
            for d in async_discovered_service_info(self.hass):
                if not d:
                    continue
                if d.address in current_ids or d.address in self._discovered_devices:
                    continue
                self._discovered_devices[d.address] = d

        if user_input is not None:
            sel_addr = user_input[CONF_ADDRESS]
            si = self._discovered_devices.get(sel_addr)
            if not si:
                return self.async_abort(reason="no_devices_found")

            addr_up = (si.address or "").upper()
            await self.async_set_unique_id(addr_up, raise_on_progress=False)
            self._abort_if_unique_id_configured()

            if _is_doser_name(si.name):
                self._discovery_info = si
                return await self.async_step_doser_confirm()

            from .chihiros_led_control.device import get_model_class_from_name
            from .chihiros_led_control.device.commander1 import Commander1
            from .chihiros_led_control.device.commander4 import Commander4
            from .chihiros_led_control.device.fallback import Fallback

            model_class = get_model_class_from_name(si.name)
            self._discovery_info = si
            self._discovered_device = model_class(si.device)

            if model_class in (Fallback, Commander1, Commander4):
                return await self.async_step_fallback_config()

            title = getattr(self._discovered_device, "name", None) || si.name | | "Chihiros LED"
            return self.async_create_entry(title=title, data={CONF_ADDRESS: addr_up})

        if not self._discovered_devices:
            return self.async_abort(reason="no_devices_found")

        schema = vol.Schema(
            {
                vol.Required(CONF_ADDRESS): vol.In(
                    {si.address: f"{si.name} ({si.address})" for si in self._discovered_devices.values()}
                )
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors={})


class ChihirosOptionsFlow(OptionsFlow):
    """Options flow to select which doser channels are enabled."""
    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry

    async def async_step_init(self, user_input: dict | None = None):
        if user_input is not None:
            raw = user_input.get("enabled_channels", [])
            enabled = sorted({int(x) for x in raw if str(x) in {"1", "2", "3", "4"}}) or [1]
            return self.async_create_entry(
                title="",
                data={"enabled_channels": enabled, "channel_count": len(enabled)},
            )

        current = self._entry.options.get("enabled_channels") or [1, 2, 3, 4]
        default = [str(x) for x in current]
        schema = vol.Schema(
            {
                vol.Required("enabled_channels", default=default): cv.multi_select(
                    {"1": "Channel 1", "2": "Channel 2", "3": "Channel 3", "4": "Channel 4"}
                )
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)

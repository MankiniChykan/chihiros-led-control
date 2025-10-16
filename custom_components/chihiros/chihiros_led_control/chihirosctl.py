# custom_components/chihiros/chihiros_led_control/chihirosctl.py
"""Chihiros LED control CLI entrypoint.

This CLI focuses on LED devices but can also mount the optional Doser and
Template CLIs (when those optional deps are available). All imports are done
defensively so Home Assistant environments are not impacted.
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from typing import Any, List, Optional

import typer
from typing_extensions import Annotated

from bleak import BleakScanner
from rich import print
from rich.table import Table

from .device import get_device_from_address, get_model_class_from_name
from .weekday_encoding import WeekdaySelect

# ────────────────────────────────────────────────────────────────
# Optional sub-apps (mounted if importable)
# ────────────────────────────────────────────────────────────────
# Template CLI
try:
    from ..chihiros_template_control import storage_containers as sc
    from ..chihiros_template_control.chihirostemplatectl import app as template_app  # type: ignore
except Exception:
    sc = None  # type: ignore
    template_app = typer.Typer(help="Template commands unavailable")

    @template_app.callback()
    def _template_unavailable():
        typer.secho(
            "Template CLI is unavailable in this environment.\n"
            "• Ensure optional deps are installed or use the dedicated entry point.",
            fg=typer.colors.YELLOW,
        )

# Doser CLI
try:
    from ..chihiros_doser_control.chihirosdoserctl import app as doser_app  # type: ignore
except Exception:
    doser_app = typer.Typer(help="Doser commands unavailable")

    @doser_app.callback()
    def _doser_unavailable():
        typer.secho(
            "Doser CLI is unavailable in this environment.\n"
            "• Ensure optional deps (e.g. bleak) are installed "
            "or use the dedicated entry point if configured.",
            fg=typer.colors.YELLOW,
        )

# Wireshark helpers (tools are outside HA tree; optional)
try:
    from tools.wireshark.wiresharkctl import app as wireshark_app  # type: ignore
except Exception:
    wireshark_app = typer.Typer(help="Wireshark helpers unavailable")

    @wireshark_app.callback()
    def _wireshark_unavailable():
        typer.secho(
            "Wireshark helpers are unavailable in this environment.\n"
            "Ensure tools/wireshark is on PYTHONPATH and its deps are installed.",
            fg=typer.colors.YELLOW,
        )

# Root app and mounts
app = typer.Typer(help="Chihiros LED control")
app.add_typer(doser_app, name="doser", help="Chihiros doser control")
app.add_typer(template_app, name="template", help="Chihiros template control")
app.add_typer(wireshark_app, name="wireshark", help="Wireshark helpers (parse/peek/encode/tx)")

# ────────────────────────────────────────────────────────────────
# Shared runner for device-bound methods
# ────────────────────────────────────────────────────────────────
def _run_device_func(device_address: str, method_override: Optional[str] = None, **kwargs: Any):
    """
    Invoke a coroutine method on the device.
    - If method_override is None, it uses the caller's function name.
    - Returns the awaited result (so getters can surface values).
    """
    method_name = method_override or inspect.stack()[1][3]

    async def _async_func():
        dev = await get_device_from_address(device_address)
        if not hasattr(dev, method_name):
            print(f"[red]{dev.__class__.__name__} doesn't support {method_name}[/red]")
            raise typer.Abort()
        meth = getattr(dev, method_name)
        return await meth(**kwargs)

    return asyncio.run(_async_func())

# ────────────────────────────────────────────────────────────────
# LED device commands
# ────────────────────────────────────────────────────────────────
@app.command(name="list-devices")
def list_devices(timeout: Annotated[int, typer.Option()] = 5) -> None:
    """List all bluetooth devices."""
    print("Scanning for Bluetooth devices…")
    table = Table("Name", "Address", "Model")
    discovered_devices = asyncio.run(BleakScanner.discover(timeout=timeout))
    chd: list[list[str]] = []
    for idx, device in enumerate(discovered_devices):
        name = device.name or ""
        model_name = "???"
        if name:
            model_class = get_model_class_from_name(name)
            model_name = getattr(model_class, "model_name", "???") or "???"
            if model_name not in ("???", "fallback"):
                chd.insert(idx, [device.address, str(model_name), str(name)])
        table.add_row(name or "(unknown)", device.address, model_name)
    print("Discovered devices:")
    print(table)
    if sc is not None:
        try:
            sc.set_template_device_trusted(chd)  # type: ignore[attr-defined]
        except Exception:
            pass

# turn-on → BaseDevice.get_turn_on
@app.command(name="turn-on")
def ctl_set_turn_on(device_address: str) -> None:
    """Turn on a light."""
    print(f"Connect to device {device_address} and turn on")
    _run_device_func(device_address, method_override="get_turn_on")

# turn-off → BaseDevice.get_turn_off
@app.command(name="turn-off")
def ctl_set_turn_off(device_address: str) -> None:
    """Turn off a light."""
    print(f"Connect to device {device_address} and turn off")
    _run_device_func(device_address, method_override="get_turn_off")

@app.command(name="set-color-brightness")
def set_color_brightness(
    device_address: str,
    color: int,
    brightness: Annotated[int, typer.Argument(min=0, max=140)],
) -> None:
    """Set color brightness of a light."""
    _run_device_func(
        device_address,
        method_override="async_set_color_brightness",
        color=color,
        brightness=brightness,
    )

@app.command(name="set-brightness")
def set_brightness(
    device_address: str,
    brightness: Annotated[int, typer.Argument(min=0, max=140)],
) -> None:
    """Set overall brightness of a light."""
    print("Connect to device …")
    _run_device_func(
        device_address,
        method_override="async_set_brightness",
        brightness=brightness,
    )

# set-rgb-brightness → BaseDevice.get_rgb_brightness
@app.command(name="set-rgb-brightness")
def ctl_set_rgb_brightness(
    device_address: str,
    brightness: Annotated[
        List[int],
        typer.Argument(min=0, max=140, help="One value or 3/4 values: R G B [W], each 0..140"),
    ],
) -> None:
    """Set per-channel RGB/RGBW brightness."""
    print(f"Connect to device {device_address} and set RGB{'W' if len(brightness)==4 else ''} to {brightness} %")
    _run_device_func(
        device_address,
        method_override="get_rgb_brightness",
        brightness=brightness,
    )

# add-setting → BaseDevice.get_add_setting
@app.command(name="add-setting")
def ctl_set_add_setting(
    device_address: str,
    sunrise: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    sunset: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    max_brightness: Annotated[int, typer.Option(max=100, min=0)] = 100,
    ramp_up_in_minutes: Annotated[int, typer.Option(min=0, max=150)] = 0,
    weekdays: Annotated[list[WeekdaySelect], typer.Option()] = [WeekdaySelect.everyday],
) -> None:
    """Add setting to a light."""
    print("Connect to device …")
    _run_device_func(
        device_address,
        method_override="get_add_setting",
        sunrise=sunrise,
        sunset=sunset,
        max_brightness=max_brightness,
        ramp_up_in_minutes=ramp_up_in_minutes,
        weekdays=weekdays,
    )

# add-rgb-setting → BaseDevice.get_add_rgb_setting
@app.command(name="add-rgb-setting")
def ctl_set_add_rgb_setting(
    device_address: str,
    sunrise: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    sunset: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    max_brightness: Annotated[tuple[int, int, int], typer.Option()] = (100, 100, 100),
    ramp_up_in_minutes: Annotated[int, typer.Option(min=0, max=150)] = 0,
    weekdays: Annotated[list[WeekdaySelect], typer.Option()] = [WeekdaySelect.everyday],
) -> None:
    """Add setting to a RGB light."""
    print("Connect to device …")
    _run_device_func(
        device_address,
        method_override="get_add_rgb_setting",
        sunrise=sunrise,
        sunset=sunset,
        max_brightness=max_brightness,
        ramp_up_in_minutes=ramp_up_in_minutes,
        weekdays=weekdays,
    )

# delete-setting → BaseDevice.get_remove_setting
@app.command(name="delete-setting")
def ctl_set_remove_setting(
    device_address: str,
    sunrise: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    sunset: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    ramp_up_in_minutes: Annotated[int, typer.Option(min=0, max=150)] = 0,
    weekdays: Annotated[list[WeekdaySelect], typer.Option()] = [WeekdaySelect.everyday],
) -> None:
    """Remove setting from a light."""
    print("Connect to device …")
    _run_device_func(
        device_address,
        method_override="get_remove_setting",
        sunrise=sunrise,
        sunset=sunset,
        ramp_up_in_minutes=ramp_up_in_minutes,
        weekdays=weekdays,
    )

# reset-settings → BaseDevice.get_reset_settings
@app.command(name="reset-settings")
def ctl_set_reset_settings(device_address: str) -> None:
    """Reset settings from a light."""
    print("Connect to device …")
    _run_device_func(device_address, method_override="get_reset_settings")

# enable-auto-mode → BaseDevice.get_enable_auto_mode
@app.command(name="enable-auto-mode")
def ctl_set_enable_auto_modes(device_address: str) -> None:
    """Enable auto mode in a light."""
    _run_device_func(device_address, method_override="get_enable_auto_mode")

if __name__ == "__main__":
    try:
        app()
    except asyncio.CancelledError:
        pass

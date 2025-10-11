# custom_components/chihiros/chihiros_led_control/chihirosctl.py
"""Chihiros LED control CLI entrypoint."""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime
from typing import Any, List

import typer
from typing_extensions import Annotated

from bleak import BleakScanner
from rich import print
from rich.table import Table

from .device import get_device_from_address, get_model_class_from_name
from .weekday_encoding import WeekdaySelect

# Mount the Template Typer app under "template"
# (robust so LED CLI still works when doser deps/HA are missing)
try:
    from ..chihiros_template_control import storage_containers as sc
    from ..chihiros_template_control.chihirostemplatectl import app as template_app # type: ignore
except Exception:
    # Provide a small stub so `chihirosctl template --help` is informative, not a crash.
    template_app = typer.Typer(help="Template commands unavailable")

    @template_app.callback()
    def _template_unavailable():
        typer.secho(
            "Template CLI is unavailable in this environment.\n"
            "• To use doser commands without Home Assistant, ensure optional deps (e.g. bleak) are installed\n"
            "  or use the dedicated entry point if configured.",
            fg=typer.colors.YELLOW,
        )


# Mount the doser Typer app under "doser"
# (robust so LED CLI still works when doser deps/HA are missing)
try:
    from ..chihiros_doser_control.chihirosdoserctl import app as doser_app  # type: ignore
except Exception:
    # Provide a small stub so `chihirosctl doser --help` is informative, not a crash.
    doser_app = typer.Typer(help="Doser commands unavailable")

    @doser_app.callback()
    def _doser_unavailable():
        typer.secho(
            "Doser CLI is unavailable in this environment.\n"
            "• To use doser commands without Home Assistant, ensure optional deps (e.g. bleak) are installed\n"
            "  or use the dedicated entry point if configured.",
            fg=typer.colors.YELLOW,
        )

# Mount the Wireshark Typer app under "wireshark"
# NOTE: the actual heavy lifting (parsers/decoders) lives outside HA under /tools.
# This package only exposes a Typer app that calls into those tools.
try:
    from ..wireshark import app as wireshark_app  # re-export from custom_components/chihiros/wireshark/__init__.py
except Exception:
    wireshark_app = typer.Typer(help="Wireshark helpers unavailable")

    @wireshark_app.callback()
    def _wireshark_unavailable():
        typer.secho(
            "Wireshark helpers are unavailable in this environment.\n"
            "Make sure the external tools (tools/wireshark_core.py, tools/btsnoop_to_jsonl.py) "
            "are present and importable by custom_components.chihiros.wireshark.wiresharkctl.",
            fg=typer.colors.YELLOW,
        )

app = typer.Typer()
app.add_typer(doser_app, name="doser", help="Chihiros doser control")
app.add_typer(template_app, name="template", help="Chihiros template control")
app.add_typer(wireshark_app, name="wireshark", help="Wireshark helpers (parse/peek/encode/decode/tx)")

# ────────────────────────────────────────────────────────────────
# Shared runner for device-bound methods
# ────────────────────────────────────────────────────────────────

def _run_device_func(device_address: str, **kwargs: Any) -> None:
    command_name = inspect.stack()[1][3]

    async def _async_func() -> None:
        dev = await get_device_from_address(device_address)
        if hasattr(dev, command_name):
            await getattr(dev, command_name)(**kwargs)
        else:
            print(f"{dev.__class__.__name__} doesn't support {command_name}")
            raise typer.Abort()

    asyncio.run(_async_func())

# ────────────────────────────────────────────────────────────────
# LED device commands
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# chihirosctl list-devices
# ────────────────────────────────────────────────────────────────

@app.command(name="list-devices")
def list_devices(timeout: Annotated[int, typer.Option()] = 5) -> None:
    """List all bluetooth devices.

    TODO: add an option to show only Chihiros devices
    """
    print("the search for Bluetooth devices is running")
    table = Table("Name", "Address", "Model")
    discovered_devices = asyncio.run(BleakScanner.discover(timeout=timeout))
    chd = []
    for idx, device in enumerate(discovered_devices):
        name = device.name or ""
        model_name = "???"
        if name:
            model_class = get_model_class_from_name(name)
            # Use safe getattr so we don't assume any class attributes exist
            model_name = getattr(model_class, "model_name", "???") or "???"
            if model_name not in ("???", "fallback"):
                name_chd = [device.address, str(model_name), str(name)]
                chd.insert(idx, name_chd)
        table.add_row(name or "(unknown)", device.address, model_name)
    print("Discovered the following devices:")
    print(table)
    sc.set_template_device_trusted(chd)

# ────────────────────────────────────────────────────────────────
# chihirosctl turn-on <device-address>
# ────────────────────────────────────────────────────────────────

@app.command(name="turn-on")
def turn_on(device_address: str) -> None:
    """Turn on a light."""
    print(f"Connect to device {device_address} and turn on")
    _run_device_func(device_address)

# ────────────────────────────────────────────────────────────────
# chihirosctl turn-off <device-address>
# ────────────────────────────────────────────────────────────────

@app.command(name="turn-off")
def turn_off(device_address: str) -> None:
    """Turn off a light."""
    print(f"Connect to device {device_address} and turn off")
    _run_device_func(device_address)



@app.command()
def set_color_brightness(
    device_address: str,
    color: int,
    brightness: Annotated[int, typer.Argument(min=0, max=140)],
) -> None:
    """Set color brightness of a light."""
    _run_device_func(device_address, color=color, brightness=brightness)

# ────────────────────────────────────────────────────────────────
# chihirosctl set-brightness <device-address> 100
# ────────────────────────────────────────────────────────────────

@app.command(name="set-brightness")
def set_brightness(
    device_address: str, brightness: Annotated[int, typer.Argument(min=0, max=140)]
) -> None:
    print(f"Connect to device ....")
    """Set overall brightness of a light."""
    set_color_brightness(device_address, color=0, brightness=brightness)

# ────────────────────────────────────────────────────────────────
# chihirosctl set-rgb-brightness <device-address> 60 80 100 or 60 80 100 10
#
# Accepts 1, 3, or 4 integers (0..140); total caps enforced in BaseDevice.
#  ────────────────────────────────────────────────────────────────

@app.command(name="set-rgb-brightness")
def set_rgb_brightness(
    device_address: str,
    brightness: Annotated[
        List[int],
        typer.Argument(min=0, max=140, help="One value or 3/4 values: R G B [W], each 0..140"),
    ],
) -> None:
    """Set per-channel RGB/RGBW brightness."""
    print(f"Connect to device {device_address} and set RGB{'W' if len(brightness)==4 else ''} to {brightness} %")
    _run_device_func(device_address, brightness=brightness)


# ────────────────────────────────────────────────────────────────
# chihirosctl add-setting <device-address> 8:00 18:00
# chihirosctl add-setting <device-address> 9:00 18:00 --weekdays monday --weekdays tuesday --ramp-up-in-minutes 30 --max-brightness 75
# ────────────────────────────────────────────────────────────────

@app.command(name="add-setting")
def add_setting(
    device_address: str,
    sunrise: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    sunset: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    max_brightness: Annotated[int, typer.Option(max=100, min=0)] = 100,
    ramp_up_in_minutes: Annotated[int, typer.Option(min=0, max=150)] = 0,
    weekdays: Annotated[list[WeekdaySelect], typer.Option()] = [WeekdaySelect.everyday],
) -> None:
    """Add setting to a light."""
    print(f"Connect to device ....")
    _run_device_func(
        device_address,
        sunrise=sunrise,
        sunset=sunset,
        max_brightness=max_brightness,
        ramp_up_in_minutes=ramp_up_in_minutes,
        weekdays=weekdays,
    )

# ────────────────────────────────────────────────────────────────
# chihirosctl add-rgb-setting <device-address> 8:00 18:00
#
#chihirosctl add-rgb-setting <device-address> 9:00 18:00 --weekdays monday --weekdays tuesday --ramp-up-in-minutes 30 --max-brightness 35 55 75
# ────────────────────────────────────────────────────────────────

@app.command(name="add-rgb-setting")
def add_rgb_setting(
    device_address: str,
    sunrise: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    sunset: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    max_brightness: Annotated[tuple[int, int, int], typer.Option()] = (100, 100, 100),
    ramp_up_in_minutes: Annotated[int, typer.Option(min=0, max=150)] = 0,
    weekdays: Annotated[list[WeekdaySelect], typer.Option()] = [WeekdaySelect.everyday],
) -> None:
    """Add setting to a RGB light."""
    print(f"Connect to device ....")
    _run_device_func(
        device_address,
        sunrise=sunrise,
        sunset=sunset,
        max_brightness=max_brightness,
        ramp_up_in_minutes=ramp_up_in_minutes,
        weekdays=weekdays,
    )
# ────────────────────────────────────────────────────────────────
# chihirosctl delete-setting <device-address> 8:00 18:00
# ────────────────────────────────────────────────────────────────

@app.command(name="delete-setting")
def remove_setting(
    device_address: str,
    sunrise: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    sunset: Annotated[datetime, typer.Argument(formats=["%H:%M"])],
    ramp_up_in_minutes: Annotated[int, typer.Option(min=0, max=150)] = 0,
    weekdays: Annotated[list[WeekdaySelect], typer.Option()] = [WeekdaySelect.everyday],
) -> None:
    """Remove setting from a light."""
    print(f"Connect to device ....")
    _run_device_func(
        device_address,
        sunrise=sunrise,
        sunset=sunset,
        ramp_up_in_minutes=ramp_up_in_minutes,
        weekdays=weekdays,
    )

# ────────────────────────────────────────────────────────────────
# chihirosctl reset-settings <device-address>
# ────────────────────────────────────────────────────────────────

@app.command(name="reset-settings")
def reset_settings(device_address: str) -> None:
    """Reset settings from a light."""
    print(f"Connect to device ....")
    _run_device_func(device_address)

# ────────────────────────────────────────────────────────────────
# chihirosctl enable-auto-mode <device-address>
# ────────────────────────────────────────────────────────────────

@app.command(name="enable-auto-mode")
def enable_auto_mode(device_address: str) -> None:
    """Enable auto mode in a light."""
    _run_device_func(device_address)

if __name__ == "__main__":
    try:
        app()
    except asyncio.CancelledError:
        pass
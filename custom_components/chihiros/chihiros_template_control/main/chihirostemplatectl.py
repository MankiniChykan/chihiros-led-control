from __future__ import annotations

import inspect
import asyncio
from typing import List

import typer
from typing_extensions import Annotated
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakDeviceNotFoundError, BleakError

from .storage_containers import app as app
from datetime import datetime, time, timedelta
from ...chihiros_led_control.main import ctl_command as ctl_cmd
from ...chihiros_led_control.main import msg_command as msg_cmd
from ....helper.weekday_encoding import (WeekdaySelect,encode_selected_weekdays)
from . import storage_containers as sc
from ...chihiros_led_control.main import chihirosctl as ctl
from ...chihiros_doser_control.main.doser_device import DoserDevice, _resolve_ble_or_fail # used by the extra helpers below


# ────────────────────────────────────────────────────────────────
# chihirosctl template load-template-standart <device-address> --template-name ALL 
# or BUKELAND or FRESHWATER or NATURE or RADIANT RGB
# Pfad /custom_components/chihiros/chihiros_template_control/data/chihiros
# ────────────────────────────────────────────────────────────────

### Admin ####
@app.command(name="load-template-standart")
def load_template_standart(
    device_address: str,
    template_name: Annotated[str, typer.Option()]):
    brightness = sc.load_standart_template(device_address, template_name)
    ctl.set_rgb_brightness(device_address, brightness=brightness)


# ────────────────────────────────────────────────────────────────
# chihirosctl template set-template-standart <device-address> --template-name ALL 
# or BUKELAND or FRESHWATER or NATURE or RADIANT RGB
# Pfad /custom_components/chihiros/chihiros_template_control/data/chihiros/
# write protection for users
# ────────────────────────────────────────────────────────────────

@app.command(name="set-template-standart")
def set_template_standart(
    template_name: Annotated[str, typer.Option("--template-name")],
    brightness: Annotated[List[int], typer.Argument(min=0, max=140, help="Parameter list, e.g. 0 0 0 or 0")],
    ) -> None:
       sc.set_template_standart(template_name, brightness)


# ────────────────────────────────────────────────────────────────
# costum templates
# chihirosctl template set-template <device-address> --template-name example
# Pfad /
# number maximum costum templates ???
# ────────────────────────────────────────────────────────────────

@app.command(name="set-template")
def set_template(
    device_address: str,
    template_name: Annotated[str, typer.Option("--template-name")],
    brightness: Annotated[List[int], typer.Argument(min=0, max=140, help="Parameter list, e.g. 0 0 0 or 0")],
    ) -> None:
       sc.set_template(device_address,template_name, brightness)

# ────────────────────────────────────────────────────────────────
# costum templates
# chihirosctl template load-template <device-address> --template-name example
# Pfad /
# number maximum costum templates ???
# ────────────────────────────────────────────────────────────────

@app.command(name="load-template")
def load_template(
    device_address: str,
    template_name: Annotated[str, typer.Option()]):
    brightness = sc.load_template(device_address, template_name)
    ctl.set_rgb_brightness(device_address, brightness=brightness)
    

if __name__ == "__main__":
    try:
        app()
    except asyncio.CancelledError:
        pass 


# custom_components/chihiros/chihiros_template_control/storage_containers.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import asyncio
from datetime import datetime, time, timedelta
from typing import List, Optional

import typer
from typing_extensions import Annotated
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakDeviceNotFoundError, BleakError

# 👈 go up to the common LED package (shared BaseDevice, time sync, weekday utils)
from ..chihiros_led_control.device.base_device import BaseDevice

from ..chihiros_led_control.weekday_encoding import (
    WeekdaySelect,
    encode_selected_weekdays,
)

app = typer.Typer(help="Chihiros template control")

#  count of parameters.
def async_totSize(*args):
    return sum(map(len, args))


def async_max_rgb_check(brightness, size_params):
    if size_params == 3:
       sum_rgb = brightness[0] + brightness[1] + brightness[2]
    if size_params == 4:
       sum_rgb = brightness[0] + brightness[1] + brightness[2] + brightness[3]   
    if sum_rgb > 400 and size_params == 4:
        raise ValueError("The values of RGB (red + green + blue + white) must not exceed 400% please correct")
    if sum_rgb > 300 and size_params == 3:
        raise ValueError("The values of RGB  must not exceed 300% please correct")



def async_load(
        ct_path: str, standart: bool,
    ) -> Dict:
        if standart is False:
           _STORE = Path(__file__).parent.parent.parent.parent.parent.parent / ".chihiros" / ct_path
        if standart is True:
           _STORE = Path(__file__).parent / "data/chihiros" / ct_path
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        if _STORE.exists():
            try:
                return json.loads(_STORE.read_text())
            except Exception:
                return {}
        return {}


def async_save(data: Dict, ct_path: str, standart: bool) -> None:
    if standart is False:
        _STORE = Path(__file__).parent.parent.parent.parent.parent.parent/ ".chihiros" / ct_path
        _STORE.write_text(json.dumps(data, indent=3, sort_keys=True))
    if standart is True:
        _STORE = Path(__file__).parent.parent / "data/chihiros" / ct_path
        _STORE.write_text(json.dumps(data, indent=3, sort_keys=True))


def async_key(param: str) -> str:
        return param.upper()

# ────────────────────────────────────────────────────────────────
# {"trusted": { "<device-address>": { "model_name": "DYU1000D5416857777E", "model_short": "Universal WRGB" } }
# get get_template_device_trusted from file 
# hold at costum template set data from the trusted file
# Avoid spam queries
# ────────────────────────────────────────────────────────────────


def get_template_device_trusted(
    device_addres: str
) -> None:
    data = async_load("trusted.json", False)
    if device_addres in data['trusted']:
        # Print the success message and the value of the key
        return data['trusted'][device_addres]['model_name']
    else:
        return


# ────────────────────────────────────────────────────────────────
# {"trusted": { "<device-address>": { "model_name": "DYU1000D5416857777E", "model_short": "Universal WRGB" } }
# is set with chihirosctl list-devices
# hold at costum template set data from the trusted file
# Avoid spam queries
# ────────────────────────────────────────────────────────────────

def get_set_template_device_trusted(
    list: str,
) -> None:
    data = async_load("trusted.json", False)
    chd  = None
    for idx, device in enumerate(list):
        if (data.get("trusted") is None):
            dev = data.setdefault(str("trusted"), {})
            chd = dev.setdefault(async_key(device[0]), {})
            chd["model_name"]  = str(device[2])
            chd["model_short"] = str(device[1])
        elif not data["trusted"].get(device[0]): 
            dev = data.setdefault(str("trusted"), {})
            chd = dev.setdefault(async_key(device[0]), {})
            chd["model_name"]  = str(device[2])
            chd["model_short"] = str(device[1])
    if chd is not None:
        _save(data, "trusted.json", False)


# ────────────────────────────────────────────────────────────────
# chihirosctl template load-template-standart <device-address> --template-name ALL 
# or BUKELAND or FRESHWATER or NATURE or RADIANT RGB
# Pfad /custom_components/chihiros/chihiros_template_control/data/chihiros
# ────────────────────────────────────────────────────────────────

def get_load_standart_template(
    device_address: str,
    template_name: str
) -> None:
    data = async_load("standart.json", True)
    check_size = len(data.get(async_key("Standart"), {}).get(async_key(template_name)).get(str("color")))
    if check_size == 4:
        red   = data.get(async_key("Standart"), {}).get(async_key(template_name), {}).get(str("color")).get("red")[1]
        green = data.get(async_key("Standart"), {}).get(async_key(template_name), {}).get(str("color")).get("green")[1]
        blue  = data.get(async_key("Standart"), {}).get(async_key(template_name), {}).get(str("color")).get("blue")[1]
        white = data.get(async_key("Standart"), {}).get(async_key(template_name), {}).get(str("color")).get("white")[1]
        typer.echo(f"Standart Template : Mac id={device_address} Template name={template_name} Red={red} Green={green} Blue={blue} White={white}")
        return [red, green, blue, white]
    if check_size == 3:
        red   = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("red")[1]
        green = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("green")[1]
        blue  = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("blue")[1]
        typer.echo(f"Standart Template : Mac id={device_address} Template name={template_name} Red={red} Green={green} Blue={blue}")
        return [red, green, blue]

# ────────────────────────────────────────────────────────────────
# chihirosctl template set-template-standart <device-address> --template-name ALL 
# or BUKELAND or FRESHWATER or NATURE or RADIANT RGB
# Pfad /custom_components/chihiros/chihiros_template_control/data/chihiros/
# write protection for users
# ────────────────────────────────────────────────────────────────


def get_set_template_standart(
    template_name: str,
    brightness: Annotated[List[int], typer.Argument(min=0, max=140, help="Parameter list, e.g. 0 0 0 or 0")],
) -> None:
    data = async_load("standart.json", True)
    check_size = async_totSize(brightness)
    async_max_rgb_check(brightness, check_size)
    
    
    if not async_key(template_name) in data:
        start = data.setdefault(str(async_key("Standart")), {})
        dev = start.setdefault(str(async_key(template_name)), {})
        
        dev["name_org"] = str(template_name)
        chd = dev.setdefault(str("color"), {})
        if check_size == 3:
           chd["red"]         = 0, max(0.100, int(brightness[0])),
           chd["green"]       = 1, max(0.100, int(brightness[1])),
           chd["blue"]        = 2, max(0.100, int(brightness[2])),
        if check_size == 4:
           chd["red"]         = 0, max(0.140, int(brightness[0])),
           chd["green"]       = 1, max(0.140, int(brightness[1])),
           chd["blue"]        = 2, max(0.140, int(brightness[2])),
           chd["white"]       = 3, max(0.140, int(brightness[3])),
        async_save(data, "standart.json", True)
    else:    
        start = data.setdefault(str(async_key("Standart")), {})
        dev = start.setdefault(str(async_key(template_name)), {})
        dev["name_org"] = str(template_name)
        chd = dev.setdefault(str("color"), {})
        if check_size == 3:
           chd["red"]         = 0, max(0.100, int(brightness[0])),
           chd["green"]       = 1, max(0.100, int(brightness[1])),
           chd["blue"]        = 2, max(0.100, int(brightness[2])),
        if check_size == 4:
           chd["red"]         = 0, max(0.140, int(brightness[0])),
           chd["green"]       = 1, max(0.140, int(brightness[1])),
           chd["blue"]        = 2, max(0.140, int(brightness[2])),
           chd["white"]       = 3, max(0.140, int(brightness[3])),
        async_save(data, "standart.jsonn", True)
    if check_size == 4:
       typer.echo(f"Standart template: Template name={template_name} Red={brightness[0]} Green={brightness[1]} Blue={brightness[2]} White={brightness[3]}")
    if check_size == 3:
       typer.echo(f"Standart template: Template name={template_name} Red={brightness[0]} Green={brightness[1]} Blue={brightness[2]}") 


# ────────────────────────────────────────────────────────────────
# costum templates
# chihirosctl template set-template <device-address> --template-name example
# Pfad /
# number maximum costum templates ???
# ────────────────────────────────────────────────────────────────

def get_set_template(
    device_address: str,
    name: str,
    brightness: Annotated[List[int], typer.Argument(min=0, max=140, help="Parameter list, e.g. 0 0 0 or 0")],
) -> None:
    data = async_load("costumer.json", False)
    check_size = async_totSize(brightness)
    async_max_rgb_check(brightness, check_size)
    
    if not async_key(name) in data:
        device_name = get_template_device_trusted(device_address)
        
        start = data.setdefault(str(async_key(device_address)), {})
        start["device_name"] = str(device_name)
        
        dev = start.setdefault(str(async_key(name)), {})
        dev["name_org"] = str(name)
        
        chd = dev.setdefault(str("color"), {})
        if check_size == 3:
           chd["red"]         = 0, max(0.100, int(brightness[0])),
           chd["green"]       = 1, max(0.100, int(brightness[1])),
           chd["blue"]        = 2, max(0.100, int(brightness[2])),
        if check_size == 4:
           chd["red"]         = 0, max(0.140, int(brightness[0])),
           chd["green"]       = 1, max(0.140, int(brightness[1])),
           chd["blue"]        = 2, max(0.140, int(brightness[2])),
           chd["white"]       = 3, max(0.140, int(brightness[3])),
        async_save(data, "costumer.json", False)
    else:    
        device_name = get_template_device_trusted(device_address)
        start = data.setdefault(str(async_key(device_address)), {})
        start["device_name"] = str(device_name)
        
        chd = start.setdefault(str(async_key(name)), {})
        dev["name_org"] = str(name)
        
        chd = dev.setdefault(str("color"), {})
        if check_size == 3:
           chd["red"]         = 0, max(0.100, int(brightness[0])),
           chd["green"]       = 1, max(0.100, int(brightness[1])),
           chd["blue"]        = 2, max(0.100, int(brightness[2])),
        if check_size == 4:
           chd["red"]         = 0, max(0.140, int(brightness[0])),
           chd["green"]       = 1, max(0.140, int(brightness[1])),
           chd["blue"]        = 2, max(0.140, int(brightness[2])),
           chd["white"]       = 3, max(0.140, int(brightness[3])),
        async_save(data, "costumer.json", False)
    if check_size == 4:
       typer.secho(f"Costume template: Mac id={device_address} Template name={name} Red={brightness[0]} Green={brightness[1]} Blue={brightness[2]} White={brightness[3]}",fg=typer.colors.YELLOW)
    if check_size == 3:
       typer.secho(f"Costume template: Mac id={device_address} Template name={name} Red={brightness[0]} Green={brightness[1]} Blue={brightness[2]}",fg=typer.colors.YELLOW,) 

# ────────────────────────────────────────────────────────────────
# costum templates
# chihirosctl template load-template <device-address> --template-name example
# Pfad /
# number maximum costum templates ???
# ────────────────────────────────────────────────────────────────

def get_load_template(
    device_address: str,
    template_name: str
) -> None:
    data = async_load("costumer.json", False)
    check_size = len(data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")))
    if check_size == 4:
        red   = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("red")[1]
        green = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("green")[1]
        blue  = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("blue")[1]
        white = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("white")[1]
        typer.secho(f"Costume Template : Mac id={device_address} Template name={template_name} Red={red} Green={green} Blue={blue} White={white}",fg=typer.colors.YELLOW,)
        return [red, green, blue, white]
    if check_size == 3:
        red   = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("red")[1]
        green = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("green")[1]
        blue  = data.get(async_key(device_address), {}).get(async_key(template_name), {}).get(str("color")).get("blue")[1]
        typer.secho(f"Costume Template : Mac id={device_address} Template name={template_name} Red={red} Green={green} Blue={blue}",fg=typer.colors.YELLOW,)
        return [red, green, blue]

# ────────────────────────────────────────────────────────────────
# costum templates
# chihirosctl template delete-template <device-address> --template-name example
# ────────────────────────────────────────────────────────────────

def get_delete_template(
    device_address: str,
    template_name: str
) -> None:
    data = async_load("costumer.json", False)
    if async_key(template_name) in data[device_address]:
        input_delete=input(f"Do you want from {device_address} this Template {template_name} delete (yes/no)")
        if input_delete.lower() == 'yes': 
            typer.secho(f"ok I delete the template {template_name} from device {device_address}",fg=typer.colors.RED,)
            del data[device_address][async_key(template_name)]
            async_save(data, "costumer.json", False)
        else: 
            typer.secho(f"ok I do not delete the template {template_name} from device {device_address}",fg=typer.colors.YELLOW,)
    else:
        typer.secho(f"The template with the name {template_name} of device {device_address} does not exist",fg=typer.colors.RED,)


def get_show_template(device_address: str) -> None:
    """
    Pretty-print all custom templates stored for a given device address.

    Skips non-template keys (e.g., "device_name") and only prints entries
    that contain a 'color' dict. Handles both RGB and RGBW.
    """
    data = async_load("costumer.json", False)
    key = async_key(device_address)

    section = data.get(key)
    if not isinstance(section, dict) or not section:
        typer.secho(f"No templates found for device '{device_address}'.", fg=typer.colors.YELLOW)
        return

    # Optional pretty table
    try:
        from rich.console import Console
        from rich.table import Table
        use_rich = True
        console = Console()
    except Exception:
        use_rich = False

    rows = []
    for name, entry in section.items():
        # Skip non-template entries (like "device_name": "<name>")
        
        if not isinstance(entry, dict):
            continue
        color = entry.get("color")
        if not isinstance(color, dict):
            continue

        # Extract channel values safely; your schema stores [index, value]
        def _get(ch: str, default: int = 0) -> int:
            v = color.get(ch)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    return int(v[1])
                except Exception:
                    return default
            return default

        r = _get("red", 0)
        g = _get("green", 0)
        b = _get("blue", 0)
        has_w = "white" in color
        w = _get("white", 0) if has_w else None

        rows.append((name, r, g, b, w))

    if not rows:
        typer.secho(f"No color templates stored for '{device_address}'.", fg=typer.colors.YELLOW)
        return
    use_richs = None
    if use_richs:
        hdr = ("Template", "Device" ,"Red", "Green", "Blue", "White")
        table = Table(*hdr)
        for name, r, g, b, w in rows:
            table.add_row(name, str(entry), str(r), str(g), str(b), "" if w is None else str(w))
        console.print(table)
    else:
        typer.echo("Template  |  Red  Green  Blue  White")
        typer.echo("------------------------------------")
        for name, r, g, b, w in rows:
            typer.echo(f"{name:<9} | {entry} | {r:>3} | {g:>3} | {b:>3} | {'' if w is None else f'{w:>3}'}")
              
                  
if __name__ == "__main__":
    app()

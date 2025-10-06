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
from ...chihiros_led_control.main.base_device import BaseDevice
from ...chihiros_led_control.main import msg_command as msg_cmd
from ...chihiros_led_control.main import ctl_command as ctl_cmd
from ....helper.weekday_encoding import (
    WeekdaySelect,
    encode_selected_weekdays,
)

app = typer.Typer(help="Chihiros template control")


def _load(
        ct_path: str, standart: bool,
    ) -> Dict:
        if standart is False:
           _STORE = Path(__file__).parent.parent.parent.parent.parent.parent / ".chihiros" / ct_path
        if standart is True:
           _STORE = Path(__file__).parent.parent / "data/chihiros" / ct_path
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        if _STORE.exists():
            try:
                return json.loads(_STORE.read_text())
            except Exception:
                return {}
        return {}


def _save(data: Dict, ct_path: str, standart: bool) -> None:
    if standart is False:
        _STORE = Path(__file__).parent.parent.parent.parent.parent.parent/ ".chihiros" / ct_path
        _STORE.write_text(json.dumps(data, indent=3, sort_keys=True))
    if standart is True:
        _STORE = Path(__file__).parent.parent / "data/chihiros" / ct_path
        _STORE.write_text(json.dumps(data, indent=3, sort_keys=True))


def _key(param: str) -> str:
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
    data = _load("trusted.json", False)
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

def set_template_device_trusted(
    list: str,
) -> None:
    data = _load("trusted.json", False)
    chd  = None
    for idx, device in enumerate(list):
        if (data.get("trusted") is None):
            dev = data.setdefault(str("trusted"), {})
            chd = dev.setdefault(_key(device[0]), {})
            chd["model_name"]  = str(device[2])
            chd["model_short"] = str(device[1])
        elif not data["trusted"].get(device[0]): 
            dev = data.setdefault(str("trusted"), {})
            chd = dev.setdefault(_key(device[0]), {})
            chd["model_name"]  = str(device[2])
            chd["model_short"] = str(device[1])
    if chd is not None:
        _save(data, "trusted.json", False)


# ────────────────────────────────────────────────────────────────
# chihirosctl template load-template-standart <device-address> --template-name ALL 
# or BUKELAND or FRESHWATER or NATURE or RADIANT RGB
# Pfad /custom_components/chihiros/chihiros_template_control/data/chihiros
# ────────────────────────────────────────────────────────────────

def load_standart_template(
    device_address: str,
    template_name: str
) -> None:
    data = _load("standart.json", True)
    
    check_size = len(data.get(_key("Standart"), {}).get(_key(template_name)).get(str("color")))
    if check_size == 4:
        red   = data.get(_key("Standart"), {}).get(_key(template_name), {}).get(str("color")).get("red")[1]
        green = data.get(_key("Standart"), {}).get(_key(template_name), {}).get(str("color")).get("green")[1]
        blue  = data.get(_key("Standart"), {}).get(_key(template_name), {}).get(str("color")).get("blue")[1]
        white = data.get(_key("Standart"), {}).get(_key(template_name), {}).get(str("color")).get("white")[1]
        typer.echo(f"Standart Template : Mac id={device_address} Template name={template_name} Red={red} Green={green} Blue={blue} White={white}")
        return [red, green, blue, white]
    if check_size == 3:
        red   = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("red")[1]
        green = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("green")[1]
        blue  = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("blue")[1]
        typer.echo(f"Standart Template : Mac id={device_address} Template name={template_name} Red={red} Green={green} Blue={blue}")
        return [red, green, blue]

# ────────────────────────────────────────────────────────────────
# chihirosctl template set-template-standart <device-address> --template-name ALL 
# or BUKELAND or FRESHWATER or NATURE or RADIANT RGB
# Pfad /custom_components/chihiros/chihiros_template_control/data/chihiros/
# write protection for users
# ────────────────────────────────────────────────────────────────


def set_template_standart(
    template_name: str,
    brightness: Annotated[List[int], typer.Argument(min=0, max=140, help="Parameter list, e.g. 0 0 0 or 0")],
) -> None:
    data = _load("standart.json", True)
    check_size = ctl_cmd._totSize(brightness)
    ctl_cmd._max_rgb_check(brightness, check_size)
    
    
    if not _key(template_name) in data:
        start = data.setdefault(str(_key("Standart")), {})
        dev = start.setdefault(str(_key(template_name)), {})
        
        dev["name_org"] = str(template_name),
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
        _save(data, "standart.json", True)
    else:    
        start = data.setdefault(str(_key("Standart")), {})
        dev = start.setdefault(str(_key(template_name)), {})
        dev["name_org"] = str(template_name),
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
        _save(data, "standart.jsonn", True)
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

def set_template(
    device_address: str,
    name: str,
    brightness: Annotated[List[int], typer.Argument(min=0, max=140, help="Parameter list, e.g. 0 0 0 or 0")],
) -> None:
    data = _load("costumer.json", False)
    check_size = ctl_cmd._totSize(brightness)
    ctl_cmd._max_rgb_check(brightness, check_size)
    
    if not _key(name) in data:
        device_name = get_template_device_trusted(device_address)
        
        start = data.setdefault(str(_key(device_address)), {})
        start["device_name"] = str(device_name),
        
        dev = start.setdefault(str(_key(name)), {})
        dev["name_org"] = str(name),
        
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
        _save(data, "costumer.json", False)
    else:    
        device_name = get_template_device_trusted(device_address)
        start = data.setdefault(str(_key(device_address)), {})
        start["device_name"] = str(device_name),
        
        chd = start.setdefault(str(_key(name)), {})
        dev["name_org"] = str(name),
        
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
        _save(data, "costumer.json", False)
    if check_size == 4:
       typer.echo(f"Costume template: Mac id={device_address} Template name={name} Red={brightness[0]} Green={brightness[1]} Blue={brightness[2]} White={brightness[3]}")
    if check_size == 3:
       typer.echo(f"Costume template: Mac id={device_address} Template name={name} Red={brightness[0]} Green={brightness[1]} Blue={brightness[2]}") 

# ────────────────────────────────────────────────────────────────
# costum templates
# chihirosctl template load-template <device-address> --template-name example
# Pfad /
# number maximum costum templates ???
# ────────────────────────────────────────────────────────────────

def load_template(
    device_address: str,
    template_name: str
) -> None:
    data = _load("costumer.json", False)
    check_size = len(data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")))
    if check_size == 4:
        red   = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("red")[1]
        green = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("green")[1]
        blue  = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("blue")[1]
        white = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("white")[1]
        typer.echo(f"Costume Template : Mac id={device_address} Template name={template_name} Red={red} Green={green} Blue={blue} White={white}")
        return [red, green, blue, white]
    if check_size == 3:
        red   = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("red")[1]
        green = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("green")[1]
        blue  = data.get(_key(device_address), {}).get(_key(template_name), {}).get(str("color")).get("blue")[1]
        typer.echo(f"Costume Template : Mac id={device_address} Template name={template_name} Red={red} Green={green} Blue={blue}")
        return [red, green, blue]


if __name__ == "__main__":
    app()
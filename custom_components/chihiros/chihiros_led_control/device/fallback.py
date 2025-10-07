"""Module defining fallback device."""

from ..main.base_device import BaseDevice


class Fallback(BaseDevice):
    """Fallback device used when a device is not completely supported yet."""

    _model_name = "fallback"
    _model_codes = []
    _colors: dict[str, int] = {
        "white": 0,
        "red": 0,
        "green": 1,
        "blue": 2,
    }

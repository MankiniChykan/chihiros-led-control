"""Z Light TINY device Model."""

from ..main.base_device import BaseDevice


class ZLightTiny(BaseDevice):
    """Z Light TINY device Class."""

    _model_name = "Z Light TINY"
    _model_codes = ["DYSSD", "DYZSD"]
    _colors: dict[str, int] = {
        "white": 0,
        "warm": 1,
    }

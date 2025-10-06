"""Module defining commands generation functions."""

import datetime
from ...chihiros_led_control.main import msg_command as msg_cmd


def create_manual_setting_command(
    msg_id: tuple[int, int], color: int, brightness_level: int
) -> bytearray:
    """Set brightness.

    param: color: 0-2 (0 is red, 1 is green, 2 is blue; on non-RGB models, 0 is white)
    param: brightness_level: 0 - 100
    """
    return msg_cmd._create_command_encoding(90, 7, msg_id, [color, brightness_level])


def create_add_auto_setting_command(
    msg_id: tuple[int, int],
    sunrise: datetime.time,
    sunset: datetime.time,
    brightness: tuple[int, int, int],
    ramp_up_minutes: int,
    weekdays: int,
) -> bytearray:
    """Add auto setting.

    brightness: tuple of 3 ints for red, green, and blue brightness, respectively
                on non-RGB models, set to (white brightness, 255, 255)
    weekdays: int resulting of selection bit mask
              (Monday Tuesday Wednesday Thursday Friday Saturday Sunday) in decimal
    """
    parameters = [
        sunrise.hour,
        sunrise.minute,
        sunset.hour,
        sunset.minute,
        ramp_up_minutes,
        weekdays,
        *brightness,
        255,
        255,
        255,
        255,
        255,
    ]

    return msg_cmd._create_command_encoding(165, 25, msg_id, parameters)


def create_delete_auto_setting_command(
    msg_id: tuple[int, int],
    sunrise: datetime.time,
    sunset: datetime.time,
    ramp_up_minutes: int,
    weekdays: int,
) -> bytearray:
    """Create delete auto setting command."""
    return create_add_auto_setting_command(
        msg_id, sunrise, sunset, (255, 255, 255), ramp_up_minutes, weekdays
    )


def create_reset_auto_settings_command(msg_id: tuple[int, int]) -> bytearray:
    """Create reset auto setting command."""
    return msg_cmd._create_command_encoding(90, 5, msg_id, [5, 255, 255])


def create_switch_to_auto_mode_command(msg_id: tuple[int, int]) -> bytearray:
    """Create switch auto setting command."""
    return msg_cmd_create_command_encoding(90, 5, msg_id, [18, 255, 255])


 #  count of parameters.
def _totSize(*args):
    return sum(map(len, args))


def _max_rgb_check(brightness, size_params):
    if size_params == 3:
       sum_rgb = brightness[0] + brightness[1] + brightness[2]
    if size_params == 4:
       sum_rgb = brightness[0] + brightness[1] + brightness[2] + brightness[3]   
    if sum_rgb > 400 and size_params == 4:
        raise ValueError("The values of RGB (red + green + blue + white) must not exceed 400% please correct")
    if sum_rgb > 300 and size_params == 3:
        raise ValueError("The values of RGB  must not exceed 300% please correct")

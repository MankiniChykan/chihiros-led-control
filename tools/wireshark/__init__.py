# custom_components/chihiros/wireshark/__init__.py
from __future__ import annotations
from .wiresharkctl import app  # re-export Typer app for chihirosctl
__all__ = ["app"]
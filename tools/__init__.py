# tools/__init__.py
"""Utilities/tools package (optional re-exports)."""
try:
    from .wireshark import app as wireshark_app  # re-export as wireshark_app
except Exception:
    wireshark_app = None

__all__ = ["wireshark_app"]

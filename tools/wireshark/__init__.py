# tools/wireshark/__init__.py
from __future__ import annotations

try:
    from .wiresharkctl import app  # Typer app
except Exception:
    # Fallback so importing tools.wireshark never raises
    import typer
    app = typer.Typer(help="Wireshark helpers unavailable (missing deps or runtime).")

__all__ = ["app"]
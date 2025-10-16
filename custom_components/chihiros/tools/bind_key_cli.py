"""English / Deutsch usage instructions for the bind key CLI.

English:
    This helper logs in to Xiaomi's Mi Cloud and prints the MiBeacon bind key.
    1. Install the integration in the same environment as this script.
    2. Run ``python -m custom_components.chihiros.tools.bind_key_cli fetch``.
    3. Provide your Mi Account username, password, device DID and region when
       prompted or via command-line options.
    4. Copy the reported bind key for use in other tools or Home Assistant.

    ## Retrieving the Xiaomi MiBeacon bind key
    Xiaomi encrypts the MiBeacon payloads broadcast by Chihiros hardware. To decrypt them you
    need the per-device bind key that is provisioned when the light or doser is paired with a Mi
    Account in the Mi Home app. This repository ships a helper CLI that logs into Xiaomi's cloud
    and requests the key for you.

    1. Pair your device with Mi Home and note the ``did`` shown in the device info page.
    2. Install the project in a virtual environment as shown above (``pip install -e .``).
    3. Run the helper and provide your Mi Account credentials when prompted:

       ``chihiros-bind-key fetch --username you@example.com --did <mi-device-id> --region de``

       * Use the two-letter region that matches your Mi Home account (``cn``, ``de``, ``us``,
         ``sg``, ...). The default is ``de`` (Europe).
       * Add ``--json`` to print the decrypted Xiaomi response or ``--all`` to list every key
         returned by the API.

    The command prints a table with the bind key, MAC address, and product identifier. Copy the
    bind key into your Home Assistant configuration or CLI environment to enable MiBeacon
    decryption.

Deutsch:
    Dieses Hilfsprogramm meldet sich bei Xiaomis Mi Cloud an und zeigt den
    MiBeacon-Bindeschlüssel an.
    1. Installieren Sie die Integration in derselben Umgebung wie dieses Skript.
    2. Führen Sie ``python -m custom_components.chihiros.tools.bind_key_cli fetch`` aus.
    3. Geben Sie bei der Abfrage oder über Befehlszeilenoptionen Ihre
       Mi-Account-E-Mail, Ihr Passwort, die Geräte-DID und die Region ein.
    4. Kopieren Sie den ausgegebenen Bindeschlüssel für andere Werkzeuge oder
       Home Assistant.

    ## Abrufen des Xiaomi-MiBeacon-Bindeschlüssels
    Xiaomi verschlüsselt die von Chihiros-Geräten ausgesendeten MiBeacon-Nutzdaten. Zur Entschlüsselung
    benötigen Sie den gerätespezifischen Bindeschlüssel, der beim Koppeln der Lampe oder Dosierpumpe mit
    einem Mi-Account in der Mi-Home-App bereitgestellt wird. Dieses Repository enthält ein CLI-Hilfsprogramm,
    das sich bei Xiaomis Cloud anmeldet und den Schlüssel für Sie abruft.

    1. Koppeln Sie Ihr Gerät mit Mi Home und notieren Sie die ``did`` auf der Geräteseite.
    2. Installieren Sie das Projekt wie oben beschrieben in einer virtuellen Umgebung (``pip install -e .``).
    3. Führen Sie das Hilfsprogramm aus und geben Sie bei Aufforderung Ihre Mi-Account-Zugangsdaten ein:

       ``chihiros-bind-key fetch --username you@example.com --did <mi-device-id> --region de``

       * Verwenden Sie den zweibuchstabigen Regionscode, der Ihrem Mi-Home-Konto entspricht (``cn``, ``de``,
         ``us``, ``sg`` ...). Standard ist ``de`` (Europa).
       * Ergänzen Sie ``--json``, um die entschlüsselte Xiaomi-Antwort auszugeben, oder ``--all``, um alle vom API
         gelieferten Schlüssel aufzulisten.

    Der Befehl gibt eine Tabelle mit Bindeschlüssel, MAC-Adresse und Produktkennung aus. Kopieren Sie den
    Bindeschlüssel in Ihre Home-Assistant-Konfiguration oder Ihre CLI-Umgebung, um die MiBeacon-Entschlüsselung
    zu aktivieren.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

import typer
from micloud.micloud import MiCloud
from micloud.micloudexception import MiCloudAccessDenied, MiCloudException
from requests import RequestException
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(help="Authenticate with Xiaomi's Mi Cloud and fetch the BLE bind key for a device.")


def _iter_result_nodes(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield key entries regardless of how the cloud response is shaped."""

    if payload is None:
        return

    if isinstance(payload, dict):
        # The cloud responses have been seen as {"result": [...]}, {"keys": [...]}, or nested under "list".
        if "result" in payload:
            yield from _iter_result_nodes(payload["result"])
            return
        if "keys" in payload:
            yield from _iter_result_nodes(payload["keys"])
            return
        if "list" in payload:
            yield from _iter_result_nodes(payload["list"])
            return
        # Some responses wrap the key entry in a dict already at the desired level.
        if "bindKey" in payload or "bind_key" in payload:
            yield payload  # pragma: no cover - depends on cloud variant
            return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
            else:
                yield from _iter_result_nodes(item)


@app.callback()
def _configure_logging(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging for Xiaomi cloud calls."),
) -> None:
    """Configure logging when the CLI boots."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    ctx.obj = {"verbose": verbose}


@app.command("fetch")
def fetch_bind_key(
    username: str = typer.Option(..., "--username", "-u", prompt=True, help="Mi Account e-mail or phone number."),
    password: str = typer.Option(
        ..., "--password", "-p", prompt=True, hide_input=True, confirmation_prompt=False, help="Mi Account password."
    ),
    did: str = typer.Option(..., "--did", help="Xiaomi device identifier returned by Mi Home."),
    region: str = typer.Option(
        "de",
        "--region",
        "-r",
        help="Two-letter region of the Xiaomi cloud shard (e.g. cn, de, us, sg).",
        show_default=True,
    ),
    show_all: bool = typer.Option(
        False,
        "--all",
        help="Display every bind key record returned by the cloud instead of just the requested DID.",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Print the decrypted Xiaomi cloud response as JSON for troubleshooting.",
    ),
) -> None:
    """Log in to Mi Cloud and print the BLE bind key for the requested device."""

    cloud = MiCloud(username=username, password=password)
    cloud.default_server = region.lower()

    console.log("Authenticating with Xiaomi cloud…")
    try:
        if not cloud.login():
            console.print("[bold red]Login failed. Check the provided credentials and try again.[/bold red]")
            raise typer.Exit(code=1)
    except MiCloudAccessDenied as err:
        console.print(f"[bold red]Access denied by Xiaomi cloud:[/bold red] {err}")
        raise typer.Exit(code=1)
    except MiCloudException as err:
        console.print(f"[bold red]Failed to log in to Xiaomi cloud:[/bold red] {err}")
        raise typer.Exit(code=1)

    payload = {"dids": [did]}
    params = {"data": json.dumps(payload, separators=(",", ":"))}

    console.log(f"Requesting bind key from the {region.upper()} shard…")
    try:
        response = cloud.request_country("/bluetooth/keys", region.lower(), params)
    except MiCloudAccessDenied as err:
        console.print(f"[bold red]The Xiaomi API rejected the request:[/bold red] {err}")
        raise typer.Exit(code=1)
    except MiCloudException as err:
        console.print(f"[bold red]Failed to fetch bind key:[/bold red] {err}")
        raise typer.Exit(code=1)
    except RequestException as err:
        console.print(f"[bold red]Network error while calling Xiaomi cloud:[/bold red] {err}")
        raise typer.Exit(code=1)

    try:
        response_json = json.loads(response)
    except json.JSONDecodeError as err:
        console.print(f"[bold red]Unexpected response from Xiaomi cloud:[/bold red] {err}\n{response}")
        raise typer.Exit(code=1)

    if output_json:
        console.print_json(data=response_json)

    entries = list(_iter_result_nodes(response_json))
    if not entries:
        console.print("[bold red]Xiaomi cloud did not return any bind key records.[/bold red]")
        raise typer.Exit(code=1)

    target_entries = entries if show_all else [entry for entry in entries if entry.get("did") == did]
    if not target_entries:
        console.print(
            "[bold red]No bind key record matched the requested DID.[/bold red] "
            "Use --all to inspect the decrypted response."
        )
        raise typer.Exit(code=1)

    table = Table(title="Mi Beacon Bind Keys", show_header=True, header_style="bold cyan")
    table.add_column("DID")
    table.add_column("MAC", justify="center")
    table.add_column("Product")
    table.add_column("Bind Key")

    for entry in target_entries:
        bind_key = entry.get("bindKey") or entry.get("bind_key")
        if not bind_key:
            continue
        table.add_row(
            entry.get("did", "?"),
            (entry.get("mac") or entry.get("deviceMac") or "").upper(),
            str(entry.get("productId") or entry.get("model") or ""),
            bind_key,
        )

    if not table.rows:
        console.print("[bold red]The Xiaomi response did not include a bindKey field.[/bold red]")
        raise typer.Exit(code=1)

    console.print(Panel.fit(table, title="Success", border_style="green"))
    console.print(
        "[green]Copy the bind key above into your Home Assistant or CLI configuration to enable MiBeacon decryption.[/green]"
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation convenience
    app()

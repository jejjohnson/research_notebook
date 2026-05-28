"""Centralised credential loading for geospatial services.

Every accessor in this module follows the same shape:

1. Try environment variables first — this covers both `.env` (loaded
   below via `python-dotenv`) and CI secrets injected directly.
2. Fall back to the service-native credentials file (`~/.netrc`,
   `~/.config/earthengine/credentials`, …) so contributors who already
   ran the service's own auth command don't need to duplicate.
3. Raise `CredentialsMissingError` with sign-up + setup instructions
   pointing at `.env.example` if nothing is found.

`load_dotenv` runs once at import time via `find_dotenv(usecwd=True)`,
which walks up from the working directory to find the repo-root `.env`
regardless of where Python was launched from (notebook vs. shell vs.
pixi task).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


# Idempotent. Won't override env vars that are already set (e.g., CI).
load_dotenv(find_dotenv(usecwd=True))


class CredentialsMissingError(RuntimeError):
    """Raised when a required credential cannot be located."""


# ---------------------------------------------------------------------------
# NASA Earthdata (earthaccess)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarthdataCreds:
    """NASA Earthdata Login username + password."""

    username: str
    password: str


_EARTHDATA_NETRC_MACHINE = "urs.earthdata.nasa.gov"


def earthdata() -> EarthdataCreds:
    """Return Earthdata credentials from env vars or `~/.netrc`."""
    user = os.environ.get("EARTHDATA_USERNAME")
    pw = os.environ.get("EARTHDATA_PASSWORD")
    if user and pw:
        return EarthdataCreds(user, pw)

    netrc_creds = _earthdata_from_netrc()
    if netrc_creds is not None:
        return netrc_creds

    raise CredentialsMissingError(
        "Earthdata credentials not found. Either:\n"
        "  1. Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD in .env\n"
        "     (see .env.example at repo root).\n"
        "  2. Run once in Python:\n"
        "         import earthaccess; earthaccess.login(persist=True)\n"
        "     to write ~/.netrc — the module reads that as a fallback.\n"
        "Sign up: https://urs.earthdata.nasa.gov/users/new"
    )


def _earthdata_from_netrc(path: Path | None = None) -> EarthdataCreds | None:
    """Best-effort `~/.netrc` lookup; returns None on any failure."""
    import netrc

    netrc_path = path if path is not None else Path.home() / ".netrc"
    if not netrc_path.is_file():
        return None
    try:
        parsed = netrc.netrc(str(netrc_path))
    except (netrc.NetrcParseError, OSError):
        return None
    auth = parsed.authenticators(_EARTHDATA_NETRC_MACHINE)
    if not auth:
        return None
    login, _, password = auth
    if not login or not password:
        return None
    return EarthdataCreds(login, password)


# ---------------------------------------------------------------------------
# Google Earth Engine
# ---------------------------------------------------------------------------


_GEE_INTERACTIVE_PATH = Path.home() / ".config" / "earthengine" / "credentials"

# Shared setup-guidance suffix appended to every GEE-related
# CredentialsMissingError so users always see the same next steps,
# whether nothing is configured at all or an env var points at a
# missing file.
_GEE_SETUP_GUIDANCE = (
    "Either:\n"
    "  1. Set GEE_SERVICE_ACCOUNT_JSON (or GOOGLE_APPLICATION_CREDENTIALS)\n"
    "     in .env to a service-account JSON path (recommended for\n"
    "     headless / CI).\n"
    "  2. Run `earthengine authenticate` once locally for interactive\n"
    "     use — the module reads ~/.config/earthengine/credentials.\n"
    "Service account guide:\n"
    "  https://developers.google.com/earth-engine/guides/service_account"
)


def gee_credentials_path() -> Path:
    """Return the path to GEE credentials.

    Headless mode: `GEE_SERVICE_ACCOUNT_JSON` (or `GOOGLE_APPLICATION_CREDENTIALS`)
    pointing at a GCP service-account JSON.

    Interactive mode: the file written by `earthengine authenticate`
    at `~/.config/earthengine/credentials`.
    """
    for var in ("GEE_SERVICE_ACCOUNT_JSON", "GOOGLE_APPLICATION_CREDENTIALS"):
        raw = os.environ.get(var)
        if not raw:
            continue
        path = Path(raw)
        if not path.is_file():
            raise CredentialsMissingError(
                f"{var}={raw} but no file exists at that path.\n\n"
                + _GEE_SETUP_GUIDANCE
            )
        return path

    if _GEE_INTERACTIVE_PATH.is_file():
        return _GEE_INTERACTIVE_PATH

    raise CredentialsMissingError(
        "Google Earth Engine credentials not found. " + _GEE_SETUP_GUIDANCE
    )


# ---------------------------------------------------------------------------
# Microsoft Planetary Computer (optional)
# ---------------------------------------------------------------------------


def planetary_computer_key() -> str | None:
    """Return the optional MPC subscription key, or None for anonymous use.

    Anonymous access works for every public collection in the
    `satellite_viewer.SENSORS` registry. A subscription key only buys
    higher rate limits and access to private collections — most users
    don't need it.
    """
    return os.environ.get("PC_SDK_SUBSCRIPTION_KEY")


__all__ = [
    "CredentialsMissingError",
    "EarthdataCreds",
    "earthdata",
    "gee_credentials_path",
    "planetary_computer_key",
]

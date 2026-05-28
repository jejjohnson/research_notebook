"""Offline tests for satellite_viewer.credentials.

We never touch real services here — every test isolates env vars and
filesystem state via fixtures so credentials never leak in from the
host running the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from satellite_viewer import credentials as cred


# ---------------------------------------------------------------------------
# Shared isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Hide every credential-relevant env var and redirect HOME to a tmpdir.

    Applied to every test in this module so no host-machine `.netrc` or
    `~/.config/earthengine/credentials` leaks in and causes a false-positive
    "credentials found".
    """
    for var in (
        "EARTHDATA_USERNAME",
        "EARTHDATA_PASSWORD",
        "GEE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "PC_SDK_SUBSCRIPTION_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    # `Path.home()` honours HOME on POSIX; on Windows it uses USERPROFILE.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    # The module caches its interactive-GEE path at import time using
    # Path.home(); rebind it for the duration of the test.
    monkeypatch.setattr(
        cred,
        "_GEE_INTERACTIVE_PATH",
        tmp_path / ".config" / "earthengine" / "credentials",
    )


# ---------------------------------------------------------------------------
# Earthdata
# ---------------------------------------------------------------------------


def test_earthdata_from_env(monkeypatch):
    monkeypatch.setenv("EARTHDATA_USERNAME", "alice")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "secret")
    creds = cred.earthdata()
    assert creds.username == "alice"
    assert creds.password == "secret"


def test_earthdata_from_netrc(tmp_path: Path):
    netrc = tmp_path / ".netrc"
    netrc.write_text("machine urs.earthdata.nasa.gov login bob password hunter2\n")
    netrc.chmod(0o600)  # netrc parser warns on permissive modes
    creds = cred._earthdata_from_netrc(netrc)
    assert creds is not None
    assert creds.username == "bob"
    assert creds.password == "hunter2"


def test_earthdata_netrc_without_matching_machine_returns_none(tmp_path: Path):
    netrc = tmp_path / ".netrc"
    netrc.write_text("machine example.com login bob password hunter2\n")
    netrc.chmod(0o600)
    assert cred._earthdata_from_netrc(netrc) is None


def test_earthdata_raises_when_nothing_set():
    with pytest.raises(cred.CredentialsMissingError) as excinfo:
        cred.earthdata()
    msg = str(excinfo.value)
    # The error should point users at both fix paths and the sign-up URL.
    # We deliberately don't substring-match the signup FQDN — CodeQL's
    # py/incomplete-url-substring-sanitization rule fires on that
    # pattern even in tests, so check for unambiguous brand + section
    # words instead.
    assert "EARTHDATA_USERNAME" in msg
    assert ".netrc" in msg
    assert "Sign up" in msg
    assert "earthdata" in msg.lower()


# ---------------------------------------------------------------------------
# Google Earth Engine
# ---------------------------------------------------------------------------


def test_gee_from_env_service_account(tmp_path: Path, monkeypatch):
    sa = tmp_path / "sa.json"
    sa.write_text('{"type":"service_account"}')
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT_JSON", str(sa))
    assert cred.gee_credentials_path() == sa


def test_gee_from_google_application_credentials(tmp_path: Path, monkeypatch):
    sa = tmp_path / "sa.json"
    sa.write_text('{"type":"service_account"}')
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))
    assert cred.gee_credentials_path() == sa


def test_gee_env_pointing_at_missing_file_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT_JSON", str(tmp_path / "missing.json"))
    with pytest.raises(cred.CredentialsMissingError) as excinfo:
        cred.gee_credentials_path()
    msg = str(excinfo.value)
    assert "missing.json" in msg
    # README promises every CredentialsMissingError carries the same
    # setup guidance — including this branch.
    assert "earthengine authenticate" in msg
    assert "service_account" in msg


def test_gee_from_interactive_credentials_file(tmp_path: Path):
    interactive = tmp_path / ".config" / "earthengine" / "credentials"
    interactive.parent.mkdir(parents=True)
    interactive.write_text('{"refresh_token":"…"}')
    assert cred.gee_credentials_path() == interactive


def test_gee_raises_when_nothing_set():
    with pytest.raises(cred.CredentialsMissingError) as excinfo:
        cred.gee_credentials_path()
    msg = str(excinfo.value)
    assert "GEE_SERVICE_ACCOUNT_JSON" in msg
    assert "earthengine authenticate" in msg


# ---------------------------------------------------------------------------
# Planetary Computer (optional — None is a valid return)
# ---------------------------------------------------------------------------


def test_planetary_computer_key_unset_returns_none():
    assert cred.planetary_computer_key() is None


def test_planetary_computer_key_from_env(monkeypatch):
    monkeypatch.setenv("PC_SDK_SUBSCRIPTION_KEY", "abc123")
    assert cred.planetary_computer_key() == "abc123"

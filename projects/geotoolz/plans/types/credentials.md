---
title: Credential Protocol
subject: Core types
subtitle: Typed credential surface for cloud auth
short_title: Credentials
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, types, credentials, auth
---

> **Parent:** [README.md](README.md) — Core types.
> **Status:** design proposal. Motivated by the credential-plumbing snippets in [Tutorial Ch. 3 §9](../../georeader_tutorial/03_rasterio_reader.md) and the duplication of `mars_data_ops/utils/filesystem.py` across downstream projects.
> **Scope:** a typed `Credential` Protocol that wraps the env-var-setting patterns currently re-implemented in every project, plus per-cloud helpers and a config-file entry point — all in `georeader.credentials`.

---

## Summary

Today, credentials for `RasterioReader` flow through process environment variables. The pattern works (GDAL picks them up automatically) but every project that uses `georeader` re-implements the env-var setup logic — `mars_data_ops/utils/filesystem.py` is ~800 lines of "read from a config file, set the right env vars, optionally fetch a managed-identity token first." The duplication is real and the surface is awkward.

This design proposes a typed `Credential` Protocol with concrete subclasses per cloud / auth mode, plus a `from_config(...)` helper for the config-file pattern. Readers (`RasterioReader`, future `AsyncGeoTIFFReader`) accept a `credential=` kwarg that applies the credential's env vars (or, in the fsspec / opener paths, threads the credential through the relevant constructor) without forcing global state.

The today-pattern (set env vars once, construct readers anywhere) keeps working — the proposed `Credential` is opt-in. Users who don't construct one get GDAL's existing env-var behaviour for free.

---

## Motivation

Three pressures make a typed credential surface worth doing:

1. **Every project re-implements the env-var setup.** [`mars_data_ops/utils/filesystem.py`](../../georeader_tutorial/03_rasterio_reader.md) is the canonical example: ~800 lines of `os.environ[...] = ...` plus auth-priority logic plus a config-file entry point. The same code appears (slightly differently) in every downstream pipeline that touches Azure storage. Promoting it into `georeader` eliminates the duplication and gives the package a stable credential API.

2. **Global env-var state is ergonomically awkward.** Two `RasterioReader` instances reading from two Azure accounts in one process have to clobber `AZURE_STORAGE_ACCOUNT` between calls or fall through to the [HTTPS-embedded-SAS workaround](../../georeader_tutorial/03_rasterio_reader.md#mode-3--https-with-embedded-sas-fallback). Tests that want to swap credentials require `monkeypatch.setenv` instead of constructor kwargs. Per-reader credential isolation would solve both.

3. **Managed-identity tokens expire.** The current pattern fetches a token once at startup via `DefaultAzureCredential` and writes it to `AZURE_STORAGE_ACCESS_TOKEN`. If the process runs longer than the token TTL (~1 hour by default), reads start failing with 401 and the user has no in-package recourse. A typed credential that knows how to refresh would handle this transparently.

The status quo absorbs each of these by re-implementing the same fix every time. A typed `Credential` lets the package own the answer once.

---

## Primer for newcomers

> **ELI5.** A static credential is like a **key that always opens the door**. A dynamic credential (managed identity) is like a **guest pass the building issues you for one hour** — secure, but you have to ask for a new one when it expires. The `Credential` Protocol says "I don't care which kind you have; just hand me something I can use to unlock the door," and quietly handles the renewal if you have a guest pass.

### Cloud auth — static vs dynamic credentials

**What it is.** Two broad classes of cloud credential. **Static**: a fixed string (or pair of strings) that's valid until rotated — e.g., AWS access key + secret, an Azure SAS token, a GCS service-account JSON file. **Dynamic**: a credential that's minted on demand and expires after some TTL — e.g., AWS STS session tokens, Azure managed-identity bearer tokens, OIDC JWTs.

**How it works.** Static credentials sit in a config file or environment variable; the cloud SDK signs requests with them directly. Dynamic credentials require an upstream call to mint: `azure.identity.DefaultAzureCredential().get_token(...)` reaches the IMDS endpoint (a host-local HTTP service that talks to the cloud control plane), gets a short-lived bearer token, and you sign requests with that. The TTL is typically 1 hour — after that, you need a refresh.

**What this means for us.** The `Credential` Protocol's two subclass families reflect this split. Static creds (`AzureSASCredential`, `AWSStaticCredential`, `GCSServiceAccountCredential`) have a no-op `refresh()`. Dynamic creds (`AzureManagedIdentityCredential`, `AWSProfileCredential`-with-SSO) have a real `refresh()` that re-fetches the token. Same `apply()` surface; different lifecycle underneath.

### Process environment variables (the GDAL pattern)

**What it is.** GDAL — the C library that reads cloud rasters — discovers credentials from `os.environ`. Set `AWS_ACCESS_KEY_ID` (or `AZURE_STORAGE_SAS_TOKEN`, etc.) before opening a file, and GDAL just works. There's no API call to "configure GDAL with credentials."

**How it works.** Inside `RasterioReader`, the code does `with rasterio.Env(**rio_env_options): rasterio.open(path)`. `rasterio.Env(...)` is a context manager that snapshots `os.environ`, overlays the provided keys, opens GDAL with that environment, then restores on exit. GDAL's libcurl reads `AWS_*`, `AZURE_STORAGE_*`, `GOOGLE_APPLICATION_CREDENTIALS` from the env and signs HTTP requests accordingly.

**What this means for us.** The today-pattern (set env vars at app startup) works because `rasterio.Env(...)` inherits the process env. The proposed pattern (`Credential.apply(env)`) merges credential keys into a per-call dict that's passed to `rasterio.Env(**dict)` — same C-side mechanism, different Python-side scoping. Per-reader isolation flows from this: two readers with two different `apply()`-ed dicts don't see each other's credentials.

### Managed identity (IMDS)

**What it is.** Azure-specific: when code runs inside Azure compute (VM, AKS pod, Function), the platform exposes a metadata endpoint (`http://169.254.169.254/metadata/identity/...`) that mints bearer tokens. The code asks the endpoint for a token; the platform authenticates the request based on the compute's assigned identity. AWS has the equivalent (`http://169.254.169.254/latest/meta-data/iam/...`), GCP too.

**How it works.** `azure.identity.DefaultAzureCredential()` walks a chain (env vars → managed identity → developer CLI auth → ...) and uses whichever auth mode succeeds. In production-on-Azure, that's almost always managed identity — no static credentials in the deployed code. The token comes back as `JWT_string` + `expires_on` (a Unix timestamp).

**What this means for us.** `AzureManagedIdentityCredential.apply()` calls `get_token(...)`, caches the result, and returns env vars. The cache is keyed on the scope (`https://storage.azure.com/.default`); if the cached token is within 60 seconds of expiry, `apply()` triggers a refresh first. Long-running processes don't silently fail at the 1-hour mark.

```{mermaid}
sequenceDiagram
    participant App
    participant Cred as AzureManagedIdentityCredential
    participant IMDS as IMDS endpoint
    participant Cloud as Azure Blob

    App->>Cred: apply(env)
    alt token cached and not near expiry
        Cred-->>App: env with bearer token
    else token expired or missing
        Cred->>IMDS: GET /metadata/identity/oauth2/token
        IMDS-->>Cred: {token, expires_on}
        Cred->>Cred: cache
        Cred-->>App: env with bearer token
    end
    App->>Cloud: GET blob<br/>(Authorization: Bearer ...)
    Cloud-->>App: 206 Partial Content
```

### Bearer tokens and TTL

**What it is.** A *bearer token* is a credential where possession of the token (without further proof) authorises the holder. JWT strings, Azure access tokens, OAuth access tokens are all bearer tokens. TTL (time-to-live) is the deliberately short validity window — typically 1 hour — that limits exposure if the token leaks.

**How it works.** The token is sent with each request as `Authorization: Bearer <token>`. The server validates the signature against the issuer's public key (no shared secret needed). When TTL expires, requests start failing with 401 Unauthorized; the client must request a new token from the issuer (e.g., refresh via IMDS).

**What this means for us.** The "401-retry-with-refresh" pattern in [`reader_rasterio.md`](../georeader/reader_rasterio.md) Proposal 2 exists because bearer tokens fail mid-pipeline. Catching one 401, calling `credential.refresh()`, retrying once — that's the standard fix. The `Credential` Protocol carries a `refresh()` method specifically for this; static creds implement it as a no-op.

```{mermaid}
stateDiagram-v2
    [*] --> Empty: __init__
    Empty --> Fetching: apply() called
    Fetching --> Valid: token retrieved
    Valid --> Valid: apply() within TTL
    Valid --> NearExpiry: less than 60s left
    NearExpiry --> Fetching: apply() triggers refresh
    Valid --> Expired: time passed
    Expired --> Fetching: apply() triggers refresh
    Valid --> Fetching: 401 from server
```

---

## Goals

- **A `Credential` Protocol** that wraps the "apply this credential to a process / reader before opening files" operation. Concrete subclasses per cloud / auth mode.
- **Per-cloud helpers** for the common construction patterns: `AzureSASCredential`, `AzureManagedIdentityCredential`, `AWSStaticCredential`, `AWSProfileCredential`, `GCSServiceAccountCredential`.
- **A `from_config(...)` factory** that reads a config object (configparser, dict, dataclass) and returns the right credential subclass.
- **Refresh-aware tokens.** Credentials backed by short-lived tokens know how to refresh on demand. Reader integration is wired through [`reader_rasterio.md`](../georeader/reader_rasterio.md).
- **Backward compatibility.** The today-pattern (set env vars, construct reader, GDAL picks them up) keeps working unchanged. The new `Credential` types are opt-in.

---

## Non-goals

- **Replacing `azure-identity` / `boto3` / `google-auth`.** The credential subclasses *wrap* SDK credential objects; they don't reimplement them.
- **Authoring a config file format.** `from_config(...)` accepts whatever shape the user has — configparser, dict, pydantic model — and reads from named keys. No new schema.
- **Changing how GDAL reads credentials.** GDAL still reads from process env vars in the GDAL-VSI path. The `Credential` Protocol is the *Python* side of the boundary; it sets the env vars and refreshes tokens, but the actual cred consumption is GDAL's job.
- **Credential storage / vault integration.** This design is about the in-process API. Where the credential strings come from (config file, environment, vault, AWS Secrets Manager, …) is the user's call.

---

## Constraints

- **Env vars are GDAL's contract.** GDAL reads cloud credentials from environment variables — there's no per-call credential argument in libcurl's GDAL integration. The package crosses that boundary in two ways: `Credential.apply_to_os_environ()` mutates `os.environ` (the today-pattern, set globals at startup) and `Credential.apply(env)` returns a fresh dict the reader merges into `rasterio.Env(**env)` (per-call, per-reader isolation). Both ultimately feed GDAL's libcurl through the env-var mechanism; they differ only in whether the env scope is process-global or per-call.
- **`apply()` is pure; `apply_to_os_environ()` mutates.** The Protocol design uses two methods so callers can choose. Per-reader isolation in `RasterioReader(credential=...)` uses `apply()` and merges into the local `rasterio.Env(...)` — no `os.environ` mutation, no global-state collisions between readers in one process.
- **Per-reader isolation works on every bytes path.** GDAL-VSI gets a per-call env dict via `rasterio.Env(**apply(env))`. fsspec gets credentials inside its filesystem constructor. Custom openers close over their own credential. None of these touch `os.environ`.
- **Refresh logic lives at the reader layer**, not in the `Credential` itself. The `Credential` knows *how* to refresh (`refresh()` method, which may also be invoked internally by `apply()` if the credential's TTL is near-expiry); the reader decides *when* to retry on a 401 (one refresh + one retry, then propagate). This split keeps the credential testable in isolation.
- **The today-pattern can't break.** Existing `os.environ['AZURE_STORAGE_*'] = ...` code keeps working — `RasterioReader()` without a `credential=` kwarg inherits from `os.environ` exactly as today.

---

## The `Credential` Protocol

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Credential(Protocol):
    """A cloud credential — applies itself to a process environment.

    Implementations wrap one specific auth mode (static SAS token,
    managed identity, AWS profile, GCS service account, ...) and know
    how to express that mode as the env vars GDAL expects.
    """

    def apply(self, env: dict[str, str] | None = None) -> dict[str, str]:
        """Return a dict of env-var keys/values to set for this credential.

        If `env` is given, mutate and return it; otherwise return a fresh dict.
        Pure function — does not touch os.environ. Reader code applies the
        result inside its rasterio.Env(...) wrap.
        """
        ...

    def apply_to_os_environ(self) -> None:
        """Set the credential's env vars on os.environ.

        Convenience for the today-pattern (set once at app startup).
        Equivalent to: os.environ.update(self.apply()).
        """
        ...

    def refresh(self) -> None:
        """Refresh the credential if it's backed by an expiring token.

        Static credentials are no-ops. Managed-identity / OIDC / etc.
        re-fetch the token. Called by readers on 401-retry.
        """
        ...
```

The Protocol is `runtime_checkable` so user code can `isinstance(x, Credential)` for the duck-typed case. Concrete implementations don't need to inherit — any class with the three methods satisfies the Protocol structurally.

---

## Concrete credential types

### Azure

```python
@dataclass
class AzureSASCredential:
    account: str
    sas_token: str

    def apply(self, env=None):
        env = dict(env or {})
        env['AZURE_STORAGE_ACCOUNT'] = self.account
        env['AZURE_STORAGE_SAS_TOKEN'] = self.sas_token
        return env

    def apply_to_os_environ(self): os.environ.update(self.apply())
    def refresh(self): pass    # static


@dataclass
class AzureConnectionStringCredential:
    connection_string: str

    def apply(self, env=None):
        env = dict(env or {})
        env['AZURE_STORAGE_CONNECTION_STRING'] = self.connection_string
        return env

    def apply_to_os_environ(self): os.environ.update(self.apply())
    def refresh(self): pass


class AzureManagedIdentityCredential:
    """Refreshable bearer token via azure.identity.DefaultAzureCredential."""

    def __init__(self, account: str, *, client_id: str | None = None,
                 scope: str = 'https://storage.azure.com/.default'):
        from azure.identity import DefaultAzureCredential
        self.account = account
        self.scope = scope
        self._cred = (
            DefaultAzureCredential(managed_identity_client_id=client_id)
            if client_id else DefaultAzureCredential()
        )
        self._token: str | None = None
        self._expires_on: float | None = None

    def apply(self, env=None):
        if self._token is None or self._expired():
            self.refresh()
        env = dict(env or {})
        env['AZURE_STORAGE_ACCOUNT'] = self.account
        env['AZURE_STORAGE_ACCESS_TOKEN'] = self._token
        return env

    def apply_to_os_environ(self): os.environ.update(self.apply())

    def refresh(self) -> None:
        token = self._cred.get_token(self.scope)
        self._token = token.token
        self._expires_on = token.expires_on

    def _expired(self) -> bool:
        # 60s safety margin
        import time
        return self._expires_on is None or self._expires_on - time.time() < 60
```

### AWS

```python
@dataclass
class AWSStaticCredential:
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None
    region: str | None = None
    requester_pays: bool = False

    def apply(self, env=None):
        env = dict(env or {})
        env['AWS_ACCESS_KEY_ID'] = self.access_key_id
        env['AWS_SECRET_ACCESS_KEY'] = self.secret_access_key
        if self.session_token: env['AWS_SESSION_TOKEN'] = self.session_token
        if self.region: env['AWS_REGION'] = self.region
        if self.requester_pays: env['AWS_REQUEST_PAYER'] = 'requester'
        return env

    def apply_to_os_environ(self): os.environ.update(self.apply())
    def refresh(self): pass


class AWSProfileCredential:
    """Reads from ~/.aws/credentials; refreshable via boto3 if SSO."""

    def __init__(self, profile: str, region: str | None = None):
        import boto3
        self.profile = profile
        self.region = region
        self._session = boto3.Session(profile_name=profile)
        self._cached: dict[str, str] | None = None

    def apply(self, env=None):
        if self._cached is None: self.refresh()
        env = dict(env or {})
        env.update(self._cached)
        return env

    def apply_to_os_environ(self): os.environ.update(self.apply())

    def refresh(self) -> None:
        c = self._session.get_credentials().get_frozen_credentials()
        out = {
            'AWS_ACCESS_KEY_ID': c.access_key,
            'AWS_SECRET_ACCESS_KEY': c.secret_key,
        }
        if c.token: out['AWS_SESSION_TOKEN'] = c.token
        if self.region: out['AWS_REGION'] = self.region
        self._cached = out
```

### GCS

```python
@dataclass
class GCSServiceAccountCredential:
    credentials_path: str
    project: str | None = None

    def apply(self, env=None):
        env = dict(env or {})
        env['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
        if self.project: env['GOOGLE_CLOUD_PROJECT'] = self.project
        return env

    def apply_to_os_environ(self): os.environ.update(self.apply())
    def refresh(self): pass
```

---

## `from_config(...)` factory

The config-file entry point that today's `mars_data_ops.fs_access_from_config(...)` re-implements:

```python
def from_config(
    config: configparser.ConfigParser | dict | Any,
    *,
    section: str = 'azure.storage',
    use_managed_identity: bool = False,
) -> Credential:
    """Build a Credential from a config object.

    Reads named keys, walks an explicit auth-priority order, returns the
    first matching credential type. For Azure (the most varied case):

        managed_identity > connection_string > sas_token

    For AWS:

        profile > static_access_key

    For GCS:

        credentials_path

    Section name selects the cloud (`azure.storage`, `aws`, `gcs`).
    """
    ...
```

Implementation detail: this dispatches on `section` to the right per-cloud helper, which then reads the right named keys. Same auth-priority logic as `mars_data_ops/utils/filesystem.py:617-703`, just promoted to the package.

---

## How readers use it

A `RasterioReader` constructed with a `credential=` kwarg threads it through the `rasterio.Env(...)` wrap:

```python
class RasterioReader(GeoData):
    def __init__(self, paths, *, credential: Credential | None = None,
                 rio_env_options: dict | None = None, **kwargs):
        self._credential = credential
        self.rio_env_options = rio_env_options or RIO_ENV_OPTIONS_DEFAULT
        ...

    def _get_rio_options_path(self, path: str) -> dict:
        opts = dict(self.rio_env_options)
        if self._credential is not None:
            opts = self._credential.apply(opts)
        return get_rio_options_path(opts, path)
```

Now per-reader isolation works for the GDAL-VSI path: at the moment `rasterio.Env(...)` is constructed, the credential's env vars are merged into the `rio_env_options` dict (not into `os.environ` globally), so two readers with two different credentials don't collide.

The fsspec path (`fs=fsspec_fs`) and the opener path (`opener=callable`) take credentials through their respective objects' constructors and ignore the `credential=` kwarg if both are given. See [`reader_rasterio.md`](../georeader/reader_rasterio.md) for the full integration spec including refresh-on-401 retry and SAS-fallback rewriting.

---

## Connections to other designs

| Design | How it touches `Credential` |
|---|---|
| [Tutorial Ch. 3 §9](../../georeader_tutorial/03_rasterio_reader.md) | The today-pattern that this design replaces. Reading Ch. 3 first is the right way to understand what env-var-soup looks like in practice. |
| [`reader_protocol.md`](../georeader/reader_protocol.md) §"Credential handling" | Articulates where credentials live in each of the three bytes paths. This Protocol is the typed surface for the GDAL-VSI path; the other two paths use their own native credential objects. |
| [`reader_rasterio.md`](../georeader/reader_rasterio.md) | Wires `credential=` into the `RasterioReader` refactor; specifies refresh-on-401 retry, SAS-fallback path-rewriting, and multi-account isolation. |
| [Reader reconciliation](../georeader/README.md) | Both readers (`RasterioReader`, future `AsyncGeoTIFFReader`) accept a `credential=` kwarg. Different paths consume it differently; same Protocol surface. |
| [`bytestore.md`](bytestore.md) | Cloud byte transport for `AsyncGeoTIFFReader` is delegated to upstream [`obspec`](https://github.com/developmentseed/obspec); we don't ship a `ByteStore` Protocol. Credentials flow into the underlying `obstore.S3Store` / `GCSStore` / `AzureStore` (which all satisfy `obspec.AsyncStore`) via `Credential.to_obstore_*_store(...)` helpers. See [`bytestore.md`](bytestore.md). |

---

## Open questions

### 1. Where does `Credential` live — in `georeader` core or as a `[creds]` extra?

The Azure / AWS / GCS subclasses each pull a real SDK (`azure-identity`, `boto3`, `google-auth`). Hard deps would balloon the install footprint. Three options:

- **Hard deps** — every install gets all three SDKs. Simplest, biggest install.
- **`[creds]` extra** — `pip install georeader-spaceml[creds]` enables the typed credentials. Default users keep using env vars manually.
- **Per-cloud extras** — `[azure-creds]`, `[aws-creds]`, `[gcs-creds]`. Most surgical; most complicated to document.

**Tentative pick: `[creds]` extra** (single optional dep that pulls all three SDKs). Per-cloud extras if install-size complaints arrive.

### 2. Should `apply_to_os_environ` be the default `apply` method?

The two-method shape (`apply` returns dict, `apply_to_os_environ` mutates global) is for the per-reader isolation case (which uses `apply`) and the today-pattern case (which uses `apply_to_os_environ`). Could collapse to one method that takes a destination dict:

```python
def apply(self, env: dict[str, str] | None = os.environ) -> dict[str, str]: ...
```

Default to `os.environ`, override with `{}` for isolation. Cleaner API, but `os.environ` is mutable global state and using it as a default is a bit of a smell.

**Tentative pick: keep two methods.** The two callsites have genuinely different intentions; one method with a magical default obscures that.

### 3. Refresh policy — automatic or explicit?

Reader-level refresh on 401 is the obvious answer. Open questions:

- **How many retries?** One refresh + one retry is the standard pattern.
- **Backoff?** Probably no — refresh is fast and retry should be immediate.
- **Should `apply()` auto-refresh expired tokens, or wait for a 401?** Auto-refresh in `apply()` is cheaper (avoids the 401 round-trip) but requires the credential to know its TTL. Wait-for-401 is simpler but slower on the failure path. The `AzureManagedIdentityCredential` snippet above auto-refreshes; could be relaxed.

### 4. STS session token / OIDC for AWS

AWS via SSO / STS / OIDC is increasingly common in modern enterprises. The `AWSProfileCredential` above handles this via `boto3.Session`, but the actual refresh is delegated to boto3's own logic. Whether to expose a more explicit `AWSSSOCredential` / `AWSOIDCCredential` is open — depends on how often users want to construct these directly vs going through profile-based config.

### 5. Should a `from_env_vars()` constructor exist?

For symmetry with the today-pattern: read whatever's currently in `os.environ`, return the appropriate credential object. Useful for "I've already set env vars; now wrap that as a typed object so my downstream code is consistent." Probably worth adding. Tentative shape: `AzureSASCredential.from_env()`, `AWSStaticCredential.from_env()`, etc. — class methods that read the canonical env vars.

---

## Alternatives considered

- **Just promote `mars_data_ops/filesystem.py` verbatim.** Rejected: that module is Azure-only, written before the package needed cross-cloud credential support, and tightly coupled to a specific config-file format. A typed Protocol is more flexible.
- **Use `azure-identity` / `boto3` / `google-auth` credential objects directly.** Rejected: each SDK has its own credential type with its own API. The point of the `Credential` Protocol is to give a single surface; user code shouldn't have to branch on which SDK is in play.
- **Use `fsspec.AbstractFileSystem` as the credential carrier.** Rejected: that's the fsspec-path credential locus, but it doesn't help GDAL-VSI users. fsspec is one of three paths, not the universal answer.
- **Don't bother — keep the env-var pattern.** Rejected: the duplication is real and the multi-account / refresh-token cases are unsolved. A typed Protocol is the smallest API surface that fixes both.

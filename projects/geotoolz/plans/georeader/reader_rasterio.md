---
title: RasterioReader improvements
subject: georeader design
subtitle: Credential ergonomics and GDAL-knob proposals
short_title: RasterioReader
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, georeader, rasterio, credentials
---

> **Parent:** [README.md](README.md) — Reader reconciliation.
> **Sibling:** [`reader_protocol.md`](reader_protocol.md) (the broader Protocol-conformance refactor).
> **Status:** design proposal. Motivated by the credential-plumbing patterns documented in [Tutorial Ch. 3 §9](../../georeader_tutorial/03_rasterio_reader.md) and the typed credential surface in [`types/credentials.md`](../types/credentials.md).
> **Scope:** `RasterioReader`-specific ergonomics improvements that don't fit the broader Protocol refactor — credential wiring, retry logic, multi-account isolation, ergonomic GDAL knobs.

---

## Why this issue exists

[`reader_protocol.md`](reader_protocol.md) optionally widens `RasterioReader`'s constructor with `opener=` / `fs=` / `rio_open_kwargs=` knobs (additive — `RasterioReader` already conforms to `GeoData`).
That widening is about *bytes-path triage*; it doesn't try to fix the awkward parts of the today-API.

This document is about the awkward parts.
It builds on:

- The credential-plumbing snippets in [Tutorial Ch. 3 §9](../../georeader_tutorial/03_rasterio_reader.md) — env-var-first auth, manual managed-identity token fetching, HTTPS-embedded-SAS fallback.
- The typed `Credential` Protocol proposed in [`types/credentials.md`](../types/credentials.md) — wraps the env-var setup and adds refresh.
- Recurring pain points users hit when they need finer control over GDAL config than `rio_env_options=` gives them.

Four concrete proposals, each scoped to a single PR. None are blocking on the others; each can land independently after `reader_protocol.md` ships.

---

## Primer for newcomers

> **ELI5.** Today, cloud credentials are like a **sticky note on the office fridge** — anyone in the room can read them, and there's only one note.
> The proposed change is each person carrying their own credential card, so two people working on different cloud accounts don't keep replacing each other's sticky notes.

### Env-var-based auth (the GDAL pattern)

**What it is.** GDAL — the C library underneath `rasterio` — reads cloud credentials from process environment variables: `AWS_ACCESS_KEY_ID`, `GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_STORAGE_*`.
There's no per-call credential argument; GDAL just looks at `os.environ` when it opens a file.

**How it works.** Your app sets the env vars once at startup (`os.environ['AZURE_STORAGE_SAS_TOKEN'] = '...'`).
Every subsequent `RasterioReader(...)` constructor implicitly inherits those values via `rasterio.Env(**rio_env_options)`, which copies the GDAL env into a context-managed scope.
The credentials flow through the C layer; Python never touches them per-call.

**What this means for us.** It's *easy* — set vars once, ignore credentials in pipeline code.
But it's *global* — two readers in one process can't use different credentials without clobbering each other.
The proposed `credential=` kwarg on `RasterioReader` (Proposal 1) merges credentials into `rio_env_options` *per call*, giving per-reader isolation without abandoning the GDAL contract.

### Managed identity and refreshable tokens

**What it is.** When code runs inside cloud compute (Azure VM, AKS pod, AWS EC2 with an IAM role), the platform mints a short-lived bearer token via a metadata-service endpoint instead of using static credentials.
Common in production; basically never in laptops.

**How it works.** Code calls `azure.identity.DefaultAzureCredential().get_token(...)` (or `boto3.Session().get_credentials()` for AWS) — under the hood that hits the cloud's IMDS / IAM endpoint and returns a JWT bearer token.
The token typically expires in 1 hour.
GDAL accepts the token as `AZURE_STORAGE_ACCESS_TOKEN` (or for AWS, the credential gets converted to standard `AWS_*` env vars).

**What this means for us.** Long-running pipelines (overnight batches, async services) silently start failing at the 1-hour mark with 401 Unauthorized.
Today's pattern fetches the token once at startup and never refreshes.
Proposal 2 (refresh-aware tokens) makes the typed `Credential` know how to refresh and the reader retry once on 401 — turning a silent failure into transparent recovery.

```{mermaid}
sequenceDiagram
    participant App
    participant Reader as RasterioReader
    participant Cred as Credential
    participant GDAL
    participant Blob as Azure Blob

    App->>Reader: read_window(w)
    Reader->>Cred: apply(env)
    Cred-->>Reader: env with token
    Reader->>GDAL: rasterio.Env(env).read
    GDAL->>Blob: GET (Bearer token)
    Blob-->>GDAL: 401 expired
    GDAL-->>Reader: RasterioIOError
    Reader->>Cred: refresh()
    Cred-->>Reader: new token cached
    Reader->>GDAL: retry read
    GDAL->>Blob: GET (new token)
    Blob-->>GDAL: 206 partial content
    GDAL-->>Reader: bytes
    Reader-->>App: GeoTensor
```

### Multi-account isolation

**What it is.** A single process needs to read from two different cloud accounts at the same time — e.g., source data in one S3 bucket, destination in another with different credentials.

**How it works (today, awkward).** Set env vars for account A → construct reader A → swap env vars for account B → construct reader B. Then if you ever call `reader_a.read_window()` after the swap, you read with account B's credentials.
The HTTPS-embedded-SAS workaround is one fix (the credential travels in the URL, not the env), but it only works for SAS tokens.

**What this means for us.** Proposal 1 fixes this directly: `credential=` is per-reader, applied inside the `rasterio.Env(...)` scope of just that reader.
Two readers with two `Credential` instances see two different credential sets.
No global env-var clobbering.
(Caveat: GDAL's underlying HTTP client uses libcurl, which has process-global SSL state; the isolation is at the credential layer, not the connection-pool layer.)

```{mermaid}
flowchart TD
    Start[RasterioReader credential=]
    Start --> Q{Which path?}
    Q -->|GDAL VSI default| Env[merge into rio_env_options<br/>per call]
    Q -->|fs=fsspec_fs| FsCred[ignored — fs carries cred]
    Q -->|opener=callable| OpCred[ignored — opener closes over cred]
    Env --> GDAL[rasterio.Env reads from local env dict]
    FsCred --> Fsspec[fsspec uses constructor cred]
    OpCred --> Custom[callable uses closure cred]
```

---

## Proposal 1 — Explicit `credential=` kwarg on `RasterioReader`

### Today

Credentials are global process state.
Two `RasterioReader` instances reading from two Azure accounts in one process collide at the env-var layer:

```python
# Account A
os.environ['AZURE_STORAGE_ACCOUNT'] = 'account_a'
os.environ['AZURE_STORAGE_SAS_TOKEN'] = 'sv=...&sig=A...'
reader_a = RasterioReader('https://account_a.blob.core.windows.net/.../scene_a.tif')

# Account B — clobbers Account A's env vars
os.environ['AZURE_STORAGE_ACCOUNT'] = 'account_b'
os.environ['AZURE_STORAGE_SAS_TOKEN'] = 'sv=...&sig=B...'
reader_b = RasterioReader('https://account_b.blob.core.windows.net/.../scene_b.tif')

# Now if reader_a.read_window(...) is called, it uses Account B's credentials (wrong)
```

The HTTPS-embedded-SAS workaround is the standard answer — but it requires the user to know to invoke it, and only handles SAS-token credentials.

### Proposed

`RasterioReader` accepts a `credential=` kwarg (typed against the [`Credential`](../types/credentials.md) Protocol).
The credential's env vars are merged into `rio_env_options` *per call*, not into `os.environ` globally:

```python
from georeader.credentials import AzureSASCredential

reader_a = RasterioReader(
    'https://account_a.blob.core.windows.net/.../scene_a.tif',
    credential=AzureSASCredential(account='account_a', sas_token='sv=...&sig=A...'),
)
reader_b = RasterioReader(
    'https://account_b.blob.core.windows.net/.../scene_b.tif',
    credential=AzureSASCredential(account='account_b', sas_token='sv=...&sig=B...'),
)

# Both work — credentials are isolated per reader
gt_a = reader_a.read_window(window)
gt_b = reader_b.read_window(window)
```

Implementation:

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

The credential's `apply(env)` returns the env dict with credential keys merged.
`rasterio.Env(**opts)` then sees the per-reader credential without touching global `os.environ`.

### What this preserves

- **The today-pattern still works.** Construct `RasterioReader` without `credential=`, and the existing env-var inheritance from `os.environ` is unchanged.
- **The fsspec / opener paths are unaffected.** Those paths carry their own credential locus (the `fs` object's construction or the opener's closure).
  When both `credential=` and `fs=` are given, the `credential=` is ignored with a warning.

### Acceptance criteria

- `RasterioReader` accepts `credential: Credential | None = None`.
- When given, credential env vars merge into `rio_env_options` per call (verified by reading two readers with two creds in one process and confirming both succeed).
- Backward-compatible: existing constructor calls without `credential=` keep working.

---

## Proposal 2 — Refresh-aware tokens for managed identity

### Today

The managed-identity snippet calls `credential.get_token(...).token` once at startup and writes the result to `AZURE_STORAGE_ACCESS_TOKEN`.
Tokens typically expire in 1 hour.
Long-running pipelines silently start failing with 401 once the token expires:

```python
# At startup
credential = DefaultAzureCredential()
token = credential.get_token('https://storage.azure.com/.default').token
os.environ['AZURE_STORAGE_ACCESS_TOKEN'] = token

# 90 minutes later — token has expired, this raises
gt = RasterioReader('az://...').read_window(window)
# rasterio.RasterioIOError: HTTP 401 Unauthorized
```

The user's only recourse today: catch the error, refresh the token, set the env var, retry.
Every project does it differently.

### Proposed

`RasterioReader.read_window(...)` (and friends) catches 401 errors, calls `self._credential.refresh()`, and retries once:

```python
def read_window(self, window):
    try:
        return self._do_read_window(window)
    except rasterio.RasterioIOError as exc:
        if self._credential is not None and _is_unauthorized(exc):
            self._credential.refresh()
            return self._do_read_window(window)  # one retry
        raise
```

The `Credential` Protocol's `refresh()` method is implemented as a no-op for static credentials and as a real token-fetch for managed-identity types (see [`types/credentials.md`](../types/credentials.md)).

The refresh policy:

- **One refresh + one retry on 401.** No backoff (refresh is fast).
  Second 401 propagates.
- **Auto-refresh on `apply()` if token expired.** Avoids the 401 round-trip when the credential knows its TTL. The `AzureManagedIdentityCredential` snippet in `types/credentials.md` does this with a 60-second safety margin.
- **No refresh for non-401 errors.** A 403 (permission denied) or 404 (not found) is not a credential issue and should propagate immediately.

### What this preserves

- Static credentials are unaffected — `refresh()` is a no-op.
- Failures unrelated to auth still propagate immediately.
- Users who want refresh-on-failure but don't use the typed `Credential` can implement their own retry logic; this proposal is about making it easy when the typed surface is in use.

### Acceptance criteria

- `RasterioReader` retries once on 401 when `credential=` is set and `credential.refresh()` is callable.
- Verified by mocking a credential whose `refresh()` flips the token, and reading a 401-then-200 sequence.
- `read_window`, `read_bounds`, `read_geoslice`, `load` all wrap the retry logic.

---

## Proposal 3 — Auto-rewrite paths for the SAS-fallback case

### Today

GDAL sometimes ignores `AZURE_STORAGE_SAS_TOKEN` for paths that don't go through the canonical `az://` form — typically `https://account.blob.core.windows.net/container/blob` paths fail with 401 even though the SAS token is set.
The workaround in `mars_data_ops/utils/filesystem.py:336-358`:

```python
def pathasroothttps(self, path: str) -> str:
    path_https = path.replace(self.root, self.root_https())
    if self.sas_token is not None:
        sep = '&' if '?' in path_https else '?'
        path_https += f"{sep}{self.sas_token.lstrip('?')}"
    return path_https
```

User has to know to call `pathasroothttps(path)` before constructing the reader.
Easy to forget; debugging the resulting 401 is annoying.

### Proposed

When `RasterioReader` is constructed with an Azure SAS credential and the path is an HTTPS Azure URL, automatically rewrite the path to embed the SAS token as a query string:

```python
def _maybe_rewrite_path(self, path: str) -> str:
    if not isinstance(self._credential, AzureSASCredential):
        return path
    if not _is_azure_https(path):
        return path
    # Embed SAS in query string
    sep = '&' if '?' in path else '?'
    return f"{path}{sep}{self._credential.sas_token.lstrip('?')}"
```

User code becomes:

```python
# Today
reader = RasterioReader(pathasroothttps('https://account.blob.core.windows.net/.../blob.tif'))

# Proposed
reader = RasterioReader(
    'https://account.blob.core.windows.net/.../blob.tif',
    credential=AzureSASCredential(account='account', sas_token='sv=...&sig=...'),
)
```

The detection heuristic (`_is_azure_https`) should be conservative — only rewrite paths matching `https://*.blob.core.windows.net/*` or `https://*.dfs.core.windows.net/*`. Other HTTPS paths are passed through untouched.

### What this preserves

- Users who already use `pathasroothttps`-style preprocessing keep working — the rewrite is idempotent (a path with the SAS already embedded won't get it embedded twice).
- Paths the heuristic doesn't recognise as Azure are untouched.
- The detection is opt-in via `credential=AzureSASCredential(...)` — without it, the reader does no path rewriting.

### Acceptance criteria

- `https://account.blob.core.windows.net/...` + `AzureSASCredential` → rewritten path with `?sv=...&sig=...` appended.
- `s3://...` + any credential → unchanged.
- Path that already contains `?sv=...` → not double-embedded.

---

## Proposal 4 — Surface common GDAL knobs as kwargs

### Today

Users who need to tune GDAL HTTP behaviour have to override `rio_env_options` and know the right env-var names:

```python
RasterioReader(
    'https://...',
    rio_env_options={
        **RIO_ENV_OPTIONS_DEFAULT,
        'GDAL_HTTP_TIMEOUT': '60',
        'GDAL_HTTP_MAX_RETRY': '5',
        'GDAL_HTTP_RETRY_DELAY': '2',
        'CPL_VSIL_CURL_USE_HEAD': 'NO',
    },
)
```

Discoverability is poor.
The env-var names aren't memorable.
Casing matters.
String-typed integer values are fragile.

### Proposed

A handful of common GDAL knobs get first-class kwargs that translate to `rio_env_options` internally:

```python
RasterioReader(
    'https://...',
    http_timeout=60,                # → GDAL_HTTP_TIMEOUT
    http_max_retry=5,               # → GDAL_HTTP_MAX_RETRY
    http_retry_delay=2,             # → GDAL_HTTP_RETRY_DELAY
    cache_max_bytes=4 * 10**9,      # → GDAL_CACHEMAX
    disable_head_requests=True,     # → CPL_VSIL_CURL_USE_HEAD=NO
)
```

The knobs are typed (int / bool), the names are Python-idiomatic, and `rio_env_options=` still works as the escape hatch for less-common settings.

Initial set of kwargs (open for discussion):

| Kwarg | Maps to | Notes |
|---|---|---|
| `http_timeout: int` | `GDAL_HTTP_TIMEOUT` | seconds |
| `http_max_retry: int` | `GDAL_HTTP_MAX_RETRY` | retries on transient errors |
| `http_retry_delay: int` | `GDAL_HTTP_RETRY_DELAY` | seconds |
| `cache_max_bytes: int` | `GDAL_CACHEMAX` | overrides 2 GB default |
| `disable_head_requests: bool` | `CPL_VSIL_CURL_USE_HEAD=NO` | for buckets that don't allow HEAD |
| `requester_pays: bool` | `AWS_REQUEST_PAYER=requester` | AWS only |

### What this preserves

- `rio_env_options=` still works for arbitrary GDAL options.
- When both `rio_env_options=` and the typed kwargs are given, the typed kwargs override (last write wins for the keys they touch).
- Defaults for the typed kwargs are `None` (don't set the env var at all), so the today-defaults in `RIO_ENV_OPTIONS_DEFAULT` keep applying.

### Acceptance criteria

- All six kwargs accepted by `RasterioReader.__init__`.
- Verified each kwarg sets the right env var in the rasterio.Env context.
- Combining `cache_max_bytes=` with `rio_env_options={'GDAL_CACHEMAX': ...}` resolves predictably (typed kwarg wins).

---

## What this issue does NOT do

Out of scope for this design (each is its own potential follow-up):

- **Async retry / async credentials.** Async support lives in [`reader_async_geotiff.md`](reader_async_geotiff.md).
  Whether `AsyncGeoData`-conformant readers accept the same `credential=` Protocol is a question for that design.
- **Per-reader cache isolation.** Today's `GDAL_CACHEMAX` is process-global. Per-reader caches would require a much deeper change (custom block manager).
- **Connection pool configuration.** GDAL's HTTP client doesn't expose pool tuning to PROJ-aware code; if needed, the answer is to switch to `AsyncGeoTIFFReader`, which uses obstore directly.
- **Vault / secrets-manager integration.** That's the user's responsibility upstream of constructing the `Credential`.

---

## Sequencing and dependencies

| Step | Depends on |
|---|---|
| Proposal 1 (`credential=` kwarg) | [`reader_protocol.md`](reader_protocol.md) (the refactor itself) + [`types/credentials.md`](../types/credentials.md) (the Protocol). |
| Proposal 2 (refresh-on-401) | Proposal 1. |
| Proposal 3 (SAS path rewrite) | Proposal 1. |
| Proposal 4 (typed GDAL kwargs) | [`reader_protocol.md`](reader_protocol.md). Independent of Proposals 1–3. |

Proposal 4 can ship first since it's purely about ergonomics.
Proposals 1–3 are coupled and should land together (or at least 1 first, then 2 and 3 in parallel).

---

## Open questions

### 1. Should `credential=` be on `RasterioReader` only, or on the abstract `GeoData` / `AsyncGeoData` Protocols?

If we want `AsyncGeoTIFFReader` (or any future reader) to accept the same kwarg, it has to be in the Protocol.
But those readers have their own credential locus (the `obspec.AsyncStore` they're constructed with), so adding `credential=` to the Protocol creates a question: which path wins when both `credential=` and a credential-bearing store are given?

**Tentative pick:** keep `credential=` on `RasterioReader` only for now.
Other readers route credentials through `Credential.to_obstore_*_store(...)` and pass the resulting store via `store=`.
Promote to the Protocol later if the duplication becomes painful.

### 2. Should the SAS path-rewrite be opt-in (kwarg) or always-on?

Always-on with `AzureSASCredential` is the proposal here.
Opt-out via `auto_rewrite_paths=False` if false-positive rewrites turn out to be a problem in practice.
**Tentative pick: always-on.**

### 3. Should typed kwargs (Proposal 4) include sensor-specific knobs?

E.g., `read_subdatasets=True` for HDF5 / NetCDF. These are GDAL-driver-specific and the kwargs proliferate quickly.
**Tentative pick: keep typed kwargs to the http- and cache-tier knobs.** Driver-specific options stay in `rio_open_kwargs=` per the existing pattern.

### 4. Multi-credential single-reader

A reader pointed at two paths in two clouds (`stack=True` with mixed S3 and Azure) doesn't have a clean answer in this design — `credential=` is a single object.
Options: reject mixed-cloud stacks at construction time, or accept `credential=Sequence[Credential]` aligned to `paths`.
**Tentative pick: reject at construction time** (with a clear error).
Mixed-cloud is rare enough that the user can build two readers and stack themselves.

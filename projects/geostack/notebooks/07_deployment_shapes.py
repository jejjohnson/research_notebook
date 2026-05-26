# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "07 — Deployment shapes (13 patterns)"
# ---
#
# # Deployment shapes — a recipe gallery
#
# A gallery of patterns showing where the `geotoolz` operator algebra fits —
# across notebook exploration, ETL, ML, web APIs, and operational tooling.
#
# The unifying claim: the *same* operator graph runs across all of these.
# What changes between cases is **who drives** — a notebook user, an HTTP
# request, a scheduler, a sampler, a tile renderer, or another graph.
#
# > **About the dummy code.** Each section preserves the *shape* of the
# > deployment pattern (which class is loaded where, who calls `pipe(gt)`,
# > how the result flows out) but uses stand-in `Operator` subclasses for
# > domain ops we haven't shipped yet, and skips actually starting servers /
# > orchestrators. Treat each section as a *runnable structural sketch*.
# >
# > When real domain operators (`gz.cloud.MaskClouds`, `gz.indices.NDVI`,
# > ...) land in v0.2+, swap the dummy stand-ins for them and the rest of
# > the code stands as-is.
#
# The thirteen sections below correspond to the thirteen recipes; your
# rendered docs will pick them up as a sidebar TOC automatically.
#

# %% [markdown]
# ## Shared setup — stand-in domain operators & fake catalog
#
# Reused across most cases. Each fake op is a thin `Operator` subclass that takes the shape of the equivalent design-doc op (`MaskClouds`, `NDVI`, ...) but does trivial work over numbers.

# %%
import hashlib

import numpy as np
from geotoolz import (
    Graph,
    Identity,
    Input,
    Lambda,
    ModelOp,
    Operator,
    Sequential,
    Tap,
)


# Stand-in domain operators ------------------------------------------------
class _FakeMaskClouds(Operator):
    """Stand-in for ``gz.cloud.MaskClouds``."""

    def __init__(self, *, qa_band: int = -1, bits=(10, 11)) -> None:
        self.qa_band = qa_band
        self.bits = tuple(bits)

    def _apply(self, x):
        # Dummy: zero-out 'cloudy' pixels (negative values)
        arr = np.asarray(x).copy()
        arr[arr < 0] = 0
        return arr

    def get_config(self) -> dict:
        return {"qa_band": self.qa_band, "bits": list(self.bits)}


class _FakeNDVI(Operator):
    def __init__(self, *, red_idx: int = 2, nir_idx: int = 3) -> None:
        self.red_idx, self.nir_idx = red_idx, nir_idx

    def _apply(self, x):
        # Dummy: arr → arr / 10
        return np.asarray(x) / 10.0

    def get_config(self) -> dict:
        return {"red_idx": self.red_idx, "nir_idx": self.nir_idx}


class _FakeToBOA(Operator):
    def __init__(self, *, sun_zenith_band: int = -2) -> None:
        self.sun_zenith_band = sun_zenith_band

    def _apply(self, x):
        return np.asarray(x) * 1.02

    def get_config(self) -> dict:
        return {"sun_zenith_band": self.sun_zenith_band}


class _FakeDarkObject(Operator):
    def __init__(self, *, percentile: int = 1) -> None:
        self.percentile = percentile

    def _apply(self, x):
        arr = np.asarray(x)
        return arr - np.percentile(arr, self.percentile)

    def get_config(self) -> dict:
        return {"percentile": self.percentile}


# Stand-in catalog / reader -------------------------------------------------
class FakeRow:
    def __init__(self, scene_id: str, payload):
        self.id = scene_id
        self.payload = payload


class FakeCatalog:
    def __init__(self, rows):
        self._rows = list(rows)

    def iter_rows(self):
        yield from self._rows

    def __len__(self):
        return len(self._rows)


def make_demo_catalog(n: int = 5) -> FakeCatalog:
    rng = np.random.default_rng(0)
    rows = [
        FakeRow(f"scene_{i:03d}", rng.integers(-2, 100, size=(4,))) for i in range(n)
    ]
    return FakeCatalog(rows)


def fake_save_cog(arr, dest: str) -> str:
    """No-op stand-in for georeader.save_cog. Returns the path it pretended to write."""
    return dest


print("Shared fixtures ready — fake ops, FakeCatalog, fake_save_cog.")

# %% [markdown]
# ## 1. Notebook exploration
#
# **The setting.** The first place a new operator earns its keep. A
# researcher debugging *why does my output look like that?* needs to see
# intermediate state without exploding the pipeline into ten named
# variables. `Tap` is the under-rated trick — an identity operator with a
# side effect lets users live inside the chain instead of breaking it
# apart for inspection.
#
# **Driver:** human at a notebook keyboard.
#

# %%
# A typical exploration pipeline with inline observation
seen = []

viz = Sequential(
    [
        _FakeMaskClouds(qa_band=-1, bits=[10, 11]),
        Tap(lambda gt: seen.append(("after_mask", float(np.nanmean(gt))))),
        _FakeDarkObject(percentile=2),
        Tap(lambda gt: seen.append(("after_correction", float(np.nanmean(gt))))),
        _FakeNDVI(red_idx=2, nir_idx=3),
    ]
)

scene = np.array([10, -5, 50, 100])
result = viz(scene)

print("final result:", result)
print()
print("intermediate stats logged inline:")
for stage, mean in seen:
    print(f"  {stage:18s}  mean = {mean:.3f}")

# %% [markdown]
# **Tradeoff.** The `fn` in `Tap` is a closure, so `Tap` is *not*
# config-round-trippable — its `get_config()` is a debug repr. That's the
# correct choice for a debug primitive. Production pipelines should not
# contain `Tap`. A `forbid_in_yaml=True` flag on the class prevents
# accidental escape from the notebook.

# %% [markdown]
# ## 2. ETL pipelines
#
# **The setting.** A one-off notebook script becomes a scheduled job that
# runs nightly across thousands of scenes. The two failure modes the
# library is meant to remove: **rewriting the pipeline** to fit the
# orchestrator, and **divergence** between the research notebook version
# and the production version. ETL here means: catalog drives the
# iteration; same operator graph applies to every row; results land in
# storage.
#
# **Driver:** scheduler (cron / Airflow / etc.) → `CatalogPipeline` →
# `reader → op → writer` per row.
#

# %%
# Same operator graph used in the notebook above
etl = Sequential(
    [
        _FakeMaskClouds(qa_band=-1, bits=[10, 11]),
        _FakeDarkObject(percentile=1),
        _FakeToBOA(sun_zenith_band=-2),
        _FakeNDVI(red_idx=2, nir_idx=3),
    ]
)


class CatalogPipeline:
    """Stand-in for the v0.2+ ``gz.catalog_ops.CatalogPipeline``."""

    def __init__(self, catalog, op, *, on_row_error: str = "raise") -> None:
        self.catalog, self.op = catalog, op
        if on_row_error not in ("raise", "log_and_continue"):
            raise ValueError("on_row_error must be 'raise' or 'log_and_continue'")
        self.on_row_error = on_row_error
        self.errors: list = []

    def run(self) -> list:
        results = []
        for row in self.catalog.iter_rows():
            try:
                # In real code: gt = reader.read_geoslice(row.to_geoslice())
                gt = row.payload
                out = self.op(gt)
                results.append((row.id, out))
            except Exception as e:
                if self.on_row_error == "raise":
                    raise
                self.errors.append((row.id, str(e)))
                print(f"  [skip] {row.id}: {e}")
        return results


catalog = make_demo_catalog(n=5)
runner = CatalogPipeline(catalog, etl, on_row_error="log_and_continue")
results = runner.run()

print(f"\nprocessed {len(results)} rows from {len(catalog)} candidates")
for row_id, out in results:
    print(f"  {row_id}  →  {out}")
print(f"errors: {len(runner.errors)}")

# %% [markdown]
# **Tradeoff.** Process-pool parallelism crosses pickle boundaries — every
# operator and every reader must be pickleable. Closures, lambdas, and
# non-default-pickled cloud credentials break here, often silently. The
# `on_row_error="log_and_continue"` default is a hedge against the "one
# bad scene crashes the 10k-scene job" anti-pattern; the cost is that
# quiet failures need monitoring (count of skipped rows per run, failure
# rate alerts).

# %% [markdown]
# ## 3. ML training and inference
#
# **The setting.** Train/serve skew is a perennial source of silent ML
# bugs — training preprocessing diverges from inference preprocessing
# over time, the model performs well in evaluation and badly in
# production, and nobody notices until a stakeholder spots the regression
# months later. The library's claim: *the same `preprocess` object* lives
# in both paths, by construction.
#
# **Driver:** training loop (DataLoader) at train time; tile sampler +
# `ApplyToChips` at inference time. **Both** use the same `preprocess`.
#

# %%
# === Shared preprocessing — defined ONCE ===
preprocess = Sequential(
    [
        _FakeMaskClouds(),
        _FakeDarkObject(percentile=2),
        _FakeToBOA(),
    ]
)


# === Training augmentation — train-only ===
class _RandomFlip(Operator):
    def __init__(self, *, p: float = 0.5, rng=None) -> None:
        self.p = p
        self.rng = rng or np.random.default_rng()

    def _apply(self, x):
        if self.rng.random() < self.p:
            return np.asarray(x)[..., ::-1]
        return x

    def get_config(self) -> dict:
        return {"p": self.p}


augment = Sequential([_RandomFlip(p=0.5, rng=np.random.default_rng(42))])


# === Training-time dataloader (in real code this would be torch.IterableDataset) ===
def training_batches(catalog, *, n_batches: int = 3):
    rng = np.random.default_rng(0)
    for _ in range(n_batches):
        rows = list(catalog.iter_rows())
        row = rows[rng.integers(0, len(rows))]
        gt = row.payload
        x = augment(preprocess(gt))  # both run at train time
        yield x


# === Inference-time pipeline (same preprocess, no augment) ===
class _FakeUNet:
    """Fake model: returns input shifted."""

    def __call__(self, arr):
        return arr + 1000


infer = Sequential(
    [
        preprocess,
        ModelOp(_FakeUNet()),
    ]
)


catalog = make_demo_catalog(n=10)

print("Training batches (with augmentation):")
for i, batch in enumerate(training_batches(catalog, n_batches=3)):
    print(f"  batch {i}: {batch}")

print("\nInference (same preprocess + model, no augmentation):")
for row in list(catalog.iter_rows())[:3]:
    print(f"  {row.id}: {infer(row.payload)}")

# %% [markdown]
# **Tradeoff.** Closures inside augment Operators (e.g. a captured RNG)
# break DataLoader workers — each worker gets the same seed and produces
# correlated batches. The fix is per-Operator `seed=` arguments and a
# `worker_init_fn` that re-seeds. Chip-stitching overlap (`stride <
# chip_size`) costs ~30% more reads for smoother boundaries; for fast
# inference set `stride == chip_size` and accept seam artefacts.

# %% [markdown]
# ## 4. LitServe inference API
#
# **The setting.** The model ships as an HTTP service, a long-lived
# process holding the GPU and serving request/response over JSON. LitServe
# is a good fit because its decode/predict/encode split lines up exactly
# with read → operator-graph → serialise.
#
# **Driver:** HTTP request → LitAPI lifecycle (`decode_request` →
# `predict` → `encode_response`). The pipeline is loaded once in `setup`.
#
# > The cell below shows the *structure* of a LitAPI handler without
# > actually starting LitServe (no GPU in this environment). The
# > `decode_request` / `predict` / `encode_response` methods are
# > exercised on a stand-in request.
#

# %%
import json as _json


class MethanePipelineAPI:
    """Sketch of a litserve.LitAPI subclass — not actually launched."""

    def setup(self, device):
        """Load the pipeline once at process startup."""
        self.pipeline = Sequential(
            [
                _FakeMaskClouds(qa_band=-1, bits=[10, 11]),
                _FakeDarkObject(percentile=1),
                _FakeNDVI(red_idx=2, nir_idx=3),
            ]
        )
        cfg_canon = _json.dumps(self.pipeline.get_config(), sort_keys=True)
        self.config_hash = hashlib.sha256(cfg_canon.encode()).hexdigest()[:16]

    def decode_request(self, req: dict):
        # In real code: gt = georeader.read_from_url(req["url"], bounds=req["bbox"])
        return np.asarray(req["payload"])

    def predict(self, gt):
        return self.pipeline(gt)

    def encode_response(self, gt_out) -> dict:
        return {
            "result": gt_out.tolist(),
            "stats": {
                "mean": float(np.nanmean(gt_out)),
                "min": float(np.nanmin(gt_out)),
                "max": float(np.nanmax(gt_out)),
            },
            "config_hash": self.config_hash,
        }


api = MethanePipelineAPI()
api.setup(device="cpu")
fake_request = {
    "url": "s3://bucket/scene_001.tif",
    "bbox": [-1, -1, 1, 1],
    "payload": [10, -5, 50, 100],
}
print("Request payload:", fake_request["payload"])
print()
decoded = api.decode_request(fake_request)
predicted = api.predict(decoded)
response = api.encode_response(predicted)

print("Response:")
print(_json.dumps(response, indent=2, default=float))


# %% [markdown]
# **Tradeoff.** Including `config_hash` in every response is cheap
# insurance — clients can detect "the model changed under me" without
# out-of-band coordination. Cold start is dominated by `setup()`: keep
# heavy state (model weights, calibration tables, target spectra) in
# long-lived attributes, never construct them per `predict()`. Batch
# dynamically (LitServe's `max_batch_size`) only if the underlying model
# genuinely benefits — geospatial operators often don't, since the
# chip-size dimension already saturates the GPU.

# %% [markdown]
# ## 5. Backend API (FastAPI, multi-pipeline)
#
# **The setting.** A platform team running many models wants **one**
# service that hosts them all. Pipeline registry pattern: load every
# pipeline at startup, route based on URL path. Trades GPU pinning for
# flexibility — an operations team can roll out a new pipeline by adding
# a YAML entry and restarting.
#
# **Driver:** HTTP request → router → registered pipeline. The
# `/pipelines/{name}` endpoint matters more than people think:
# `get_config()` lets the API describe its own behaviour to clients
# without out-of-band docs.
#
# > Stand-in: a `PipelineService` class that mirrors a FastAPI router
# > without actually starting Uvicorn. The HTTP plumbing is one
# > decorator-call layer above what's shown here.
#


# %%
class PipelineService:
    """Routes requests to named pipelines. Returns config + result."""

    def __init__(self, pipelines: dict) -> None:
        self.pipelines = pipelines
        self.hashes = {
            name: hashlib.sha256(
                _json.dumps(p.get_config(), sort_keys=True).encode()
            ).hexdigest()[:12]
            for name, p in pipelines.items()
        }

    def process(self, name: str, payload) -> dict:
        """@app.post('/process/{name}') equivalent."""
        if name not in self.pipelines:
            raise KeyError(f"unknown pipeline: {name}")
        gt = np.asarray(payload)
        out = self.pipelines[name](gt)
        out_uri = fake_save_cog(out, f"s3://outputs/{name}_demo.tif")
        return {"out_uri": out_uri, "config_hash": self.hashes[name]}

    def describe(self, name: str) -> dict:
        """@app.get('/pipelines/{name}') equivalent."""
        return self.pipelines[name].get_config()

    def list_pipelines(self) -> dict:
        """@app.get('/pipelines') equivalent."""
        return dict(self.hashes)


# Three named pipelines registered at startup
service = PipelineService(
    {
        "methane_v3": Sequential([_FakeMaskClouds(), _FakeDarkObject(percentile=1)]),
        "methane_v4": Sequential([_FakeMaskClouds(), _FakeDarkObject(percentile=2)]),
        "ndvi_etl": Sequential([_FakeMaskClouds(), _FakeNDVI()]),
    }
)

print("Pipeline registry:")
for name, h in service.list_pipelines().items():
    print(f"  {name:12s}  hash={h}")

print("\n/process/methane_v3 →", service.process("methane_v3", [10, 20, 30]))

print("\n/pipelines/ndvi_etl describe:")
print(_json.dumps(service.describe("ndvi_etl"), indent=2))

# %% [markdown]
# **Tradeoff.** Loading every pipeline at startup grows memory linearly
# in the number of registered pipelines — fine for a few dozen, painful
# at hundreds. Above that scale, lazy-load on first request and cache.
# Hot reload (swap a YAML without restart) sounds attractive but
# interacts badly with in-flight requests holding references to the old
# graph; safer to require a process restart and let the load balancer
# drain.

# %% [markdown]
# ## 6. User-uploaded pipelines for viz
#
# **The setting.** The democratisation case. A scientist or analyst — not
# necessarily a Python engineer — wants to compose a pipeline from a UI,
# see the result, and share the recipe. Letting a user submit YAML *as a
# request* is a natural mapping, but it bites the moment you call
# `hydra.utils.instantiate()` on untrusted input: a malicious payload
# like `_target_: os.system, args: ["rm -rf /"]` is arbitrary code
# execution.
#
# **The fix** is a sandboxed loader that walks the config tree and
# rejects any `_target_` not in a pre-vetted public registry.
#
# **Driver:** Streamlit / web UI textbox → POST → server-side sandboxed
# loader → pipeline → render.
#

# %%
# A tiny registry — the trustworthy public surface
ALLOWED = {
    "Sequential": Sequential,
    "_FakeMaskClouds": _FakeMaskClouds,
    "_FakeNDVI": _FakeNDVI,
    "_FakeDarkObject": _FakeDarkObject,
    "Identity": Identity,
}


class UntrustedTargetError(Exception):
    def __init__(self, target):
        self.target = target
        super().__init__(f"Operator '{target}' is not in the public registry")


def load_dict_sandboxed(cfg: dict):
    """Sandboxed builder. Refuses any class not in ALLOWED."""
    if not isinstance(cfg, dict):
        return cfg
    target = cfg.get("_target_")
    if target is None:
        return cfg
    if target not in ALLOWED:
        raise UntrustedTargetError(target)
    cls = ALLOWED[target]
    # Recursively build child configs
    kwargs = {k: load_dict_sandboxed(v) for k, v in cfg.items() if k != "_target_"}
    # 'operators' for Sequential is a list of child configs
    if "operators" in kwargs:
        kwargs["operators"] = [load_dict_sandboxed(c) for c in kwargs["operators"]]
    return cls(**kwargs)


# A safe submission
safe_yaml = {
    "_target_": "Sequential",
    "operators": [
        {"_target_": "_FakeMaskClouds", "qa_band": -1, "bits": [10, 11]},
        {"_target_": "_FakeNDVI", "red_idx": 2, "nir_idx": 3},
    ],
}

pipe = load_dict_sandboxed(safe_yaml)
print("safe submission ran:", pipe(np.array([10, -5, 50])))


# A malicious submission
evil_yaml = {
    "_target_": "Sequential",
    "operators": [
        {"_target_": "os.system", "args": ["rm -rf /"]},
    ],
}

try:
    load_dict_sandboxed(evil_yaml)
except UntrustedTargetError as e:
    print(f"\nrejected: {e}")


# %% [markdown]
# **Tradeoff.** The sandbox doesn't solve resource exhaustion — a
# malicious user can still submit a config that allocates huge
# intermediates or schedules a million `Sequential` steps. Time and
# memory limits per request are mandatory. The other side of the same
# coin: this pattern also gates *internal* contributors — a colleague's
# experimental Operator can't run in the demo until it lands in the
# registry, which is a feature for governance and a friction point for
# researchers. The right balance is a fast registry-promotion process,
# not a permissive sandbox.

# %% [markdown]
# ## 7. Tile server (z/x/y → PNG)
#
# **The setting.** Slippy-map UX — a user pans a Leaflet/Mapbox view, the
# browser issues XYZ tile requests, the server renders each tile on
# demand. Distinct from a generic backend API: every request is `(z, x,
# y, layer)`, the response is a 256×256 PNG, p99 must be sub-second, a
# single map view fans out to dozens of concurrent requests.
#
# **Driver:** browser map → XYZ tile request → layer registry → operator
# graph runs once per tile (often cached upstream by a CDN).
#
# Operators in this hot loop run ~10⁴ times per minute, so per-call
# construction work is the enemy. Load LUTs / state **once** in
# `__init__`, never per call.
#


# %%
class _FakeColormap(Operator):
    """Stand-in for ``gz.viz.Colormap``. Builds its LUT once at construction."""

    def __init__(
        self, *, name: str = "viridis", vmin: float = 0, vmax: float = 1
    ) -> None:
        self.name, self.vmin, self.vmax = name, vmin, vmax
        # one-time work — reused across every call (the whole point)
        rng = np.random.default_rng({"viridis": 0, "RdYlGn": 1}.get(name, 2))
        self.lut = rng.integers(0, 255, size=(256, 4), dtype=np.uint8)

    def _apply(self, arr):
        normed = np.clip((arr - self.vmin) / (self.vmax - self.vmin), 0, 1)
        return self.lut[(normed * 255).astype(np.uint8)]

    def get_config(self) -> dict:
        return {"name": self.name, "vmin": self.vmin, "vmax": self.vmax}


# Layer registry
LAYERS = {
    "ndvi": Sequential(
        [
            _FakeMaskClouds(),
            _FakeNDVI(),
            _FakeColormap(name="RdYlGn", vmin=-1, vmax=1),
        ]
    ),
    "true_color": Sequential(
        [
            _FakeMaskClouds(),
            Lambda(lambda x: np.clip(x / 100.0, 0, 1), name="scale_for_rgb"),
        ]
    ),
}


def tile_handler(layer: str, z: int, x: int, y: int):
    """@app.get('/tiles/{layer}/{z}/{x}/{y}.png') equivalent."""
    # In real code:
    #   gt = catalog.read_tile(mercantile.xy_bounds(x, y, z), size=(256, 256))
    fake_tile = np.linspace(-1, 1, 16).reshape(4, 4) * 10
    rgb = LAYERS[layer](fake_tile)
    # In real code: return Response(rgb.to_png_bytes(), media_type="image/png")
    return f"<png {rgb.shape} for layer={layer} z={z} x={x} y={y}>"


# Simulate a few tile requests
for z, x, y in [(10, 512, 256), (10, 513, 256), (10, 512, 257)]:
    print(tile_handler("ndvi", z, x, y))


# %% [markdown]
# **Tradeoff.** A tile cache (Redis, CloudFront, on-disk) usually matters
# more than Python-level performance — the same tile gets requested by
# every user panning the same area. With caching, the operator graph runs
# once per distinct tile and the API mostly serves bytes. Without
# caching, every operator's hot-path performance is exposed. Stateful
# operators (e.g. learned percentile clip) need careful handling: the
# state must be loaded once at startup, not per request.

# %% [markdown]
# ## 8. Data validation / QC as operators
#
# **The setting.** Data goes wrong in predictable ways — wrong CRS, wrong
# resolution, missing bands, all-NaN scenes from a failed download.
# Catching these at the right pipeline stage is the difference between
# "model trained on garbage for a week" and "pipeline halted on the first
# bad scene." The library contribution is making **assertions look like
# operators**: pass-through operators that raise (or warn) on contract
# violations, droppable anywhere in the chain.
#
# **Driver:** the same pipeline that does the work runs the assertions —
# no separate validation framework, no out-of-band tests.
#


# %%
class QCError(Exception):
    pass


class AssertValueRange(Operator):
    def __init__(
        self, *, min_val: float, max_val: float, on_fail: str = "raise"
    ) -> None:
        if on_fail not in ("raise", "warn"):
            raise ValueError("on_fail must be 'raise' or 'warn'")
        self.min, self.max, self.on_fail = min_val, max_val, on_fail

    def _apply(self, x):
        arr = np.asarray(x)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if lo < self.min or hi > self.max:
            msg = f"range [{lo:.3g}, {hi:.3g}] outside [{self.min}, {self.max}]"
            if self.on_fail == "raise":
                raise QCError(msg)
            print(f"  WARNING: {msg}")
        return x

    def get_config(self) -> dict:
        return {"min_val": self.min, "max_val": self.max, "on_fail": self.on_fail}


class AssertValidFraction(Operator):
    def __init__(self, *, min_valid: float, on_fail: str = "raise") -> None:
        self.min_valid, self.on_fail = min_valid, on_fail

    def _apply(self, x):
        arr = np.asarray(x, dtype=float)
        valid = float(np.mean(~np.isnan(arr)))
        if valid < self.min_valid:
            msg = f"valid fraction {valid:.1%} below {self.min_valid:.0%}"
            if self.on_fail == "raise":
                raise QCError(msg)
            print(f"  WARNING: {msg}")
        return x

    def get_config(self) -> dict:
        return {"min_valid": self.min_valid, "on_fail": self.on_fail}


# Stack of assertions reused at multiple checkpoints
qc = Sequential(
    [
        AssertValueRange(min_val=0, max_val=10_000, on_fail="warn"),
        AssertValidFraction(min_valid=0.5, on_fail="warn"),
    ]
)

# Drop into ETL like a tracer round
etl_with_qc = Sequential(
    [
        _FakeMaskClouds(),
        qc,  # first checkpoint
        _FakeToBOA(),
        qc,  # again, post-correction
        _FakeNDVI(),
    ]
)

# Good input — silent
print("good input (silent):")
out = etl_with_qc(np.array([10.0, 50.0, 200.0, 1000.0]))
print("  result:", out)

# Bad input — warnings fire
print("\nbad input (warnings fire):")
out = etl_with_qc(np.array([10.0, 50.0, 200.0, 99_999.0]))
print("  result:", out)

# Strict mode
print("\nstrict mode raises:")
strict_qc = AssertValueRange(min_val=0, max_val=100)
try:
    strict_qc(np.array([5, 25, 9999]))
except QCError as e:
    print(f"  caught: {e}")


# %% [markdown]
# **Tradeoff.** Where a QC step fires (raise vs warn) is a per-deployment
# policy. A research notebook wants `on_fail="warn"` so exploration isn't
# interrupted; a production ETL wants `on_fail="raise"` plus alerting. The
# same YAML can ship with `on_fail="${qc_policy}"` and let the loader
# inject the right value. The performance cost is real — full-array
# reductions (`nanmin`, `nanmax`) on large carriers aren't free. For
# tile-server hot paths, sample a small fraction of pixels rather than
# the full array.

# %% [markdown]
# ## 9. Pinned / hashed regulatory artifact
#
# **The setting.** A scientific paper, a regulatory submission, or an
# attribution claim from operational monitoring needs to be **byte-exact
# reproducible** years later. "Re-run the pipeline" requires three things
# to be pinned together: the operator graph, the inputs (with content
# hashes, not just URIs), and the dependency stack. Treat the operator
# graph as **the deliverable**, not just a way to compute one.
#
# **Driver:** publication time → `Artifact.freeze(pipeline, inputs,
# deps_lock)` → tarball goes into archive. Years later → `Artifact.load(path).rerun()`.
#


# %%
class Artifact:
    """Reproducibility envelope around an Operator graph.

    In real life this is a tarball of (pipeline_yaml, hashes, deps_lock,
    metadata.json). Here we keep it in-memory.
    """

    def __init__(
        self, *, pipeline_cfg, pipeline_hash, inputs, deps_lock, metadata
    ) -> None:
        self.pipeline_cfg = pipeline_cfg
        self.pipeline_hash = pipeline_hash
        self.inputs = inputs
        self.deps_lock = deps_lock
        self.metadata = metadata

    @classmethod
    def freeze(
        cls, pipeline: Operator, *, inputs: dict, deps_lock: str, metadata: dict
    ) -> "Artifact":
        cfg = pipeline.get_config()
        canon = _json.dumps(cfg, sort_keys=True)
        return cls(
            pipeline_cfg=cfg,
            pipeline_hash=hashlib.sha256(canon.encode()).hexdigest(),
            inputs=inputs,
            deps_lock=deps_lock,
            metadata=metadata,
        )


# === At publication time ===
pipeline = Sequential(
    [
        _FakeMaskClouds(qa_band=-1, bits=[10, 11]),
        _FakeDarkObject(percentile=1),
        _FakeNDVI(red_idx=2, nir_idx=3),
    ]
)

scene_payload = np.array([10.0, -5.0, 50.0, 100.0])

artifact = Artifact.freeze(
    pipeline,
    inputs={
        "scene_uri": "s3://imeo/emit/2024Q3/scene_001.nc",
        "scene_sha256": hashlib.sha256(scene_payload.tobytes()).hexdigest(),
    },
    deps_lock="geotoolz==0.0.1\nnumpy==1.26.0\n# ... full lockfile here ...",
    metadata={"author": "ej", "date": "2024-09-15", "doi": "10.xxxx/mars-2024Q3"},
)

print("pipeline hash:", artifact.pipeline_hash[:16] + "...")
print("input hash   :", artifact.inputs["scene_sha256"][:16] + "...")
print("metadata     :", artifact.metadata)

# === Years later: re-instantiate and re-run ===
# In real code, gz.repro.load() rebuilds the pipeline from artifact.pipeline_cfg
# via the same sandboxed loader as case 6.
print()
print("Re-running yields identical output:")
print("  run 1:", pipeline(scene_payload))
print("  run 2:", pipeline(scene_payload))


# %% [markdown]
# **Tradeoff.** A `poetry.lock` alone is **not enough** for true
# byte-exactness — BLAS implementations, GPU drivers, libc versions, even
# the order in which floating-point reductions are evaluated all affect
# numerical output. Honest framing: the artifact *guarantees* the operator
# graph and inputs; it *aspires to* numerical reproducibility, with the
# deps_lock as a best-effort. For genuine bit-exactness, ship a Docker
# image hash alongside the artifact. External URIs decay over years; for
# true long-term archive, the artifact should optionally include the
# *bytes* of small inputs, not just their hashes.

# %% [markdown]
# ## 10. Active learning / uncertainty-driven sampling
#
# **The setting.** Labels are expensive (an expert annotator costs more
# than a GPU-hour). The classic active-learning loop: train a model, score
# the unlabelled pool by uncertainty, send the highest-uncertainty
# examples for labelling, retrain. The library contribution: a sampler
# that **consumes operator output** to decide what to sample next.
#
# **Driver:** training-loop orchestrator → `ActiveSampler` →
# `base_sampler ∘ score_op → top-K`. Same composition primitives, but the
# iterator is now informed by inference.
#


# %%
class _FakeUQModel:
    """Stand-in for a model that returns (prediction, uncertainty)."""

    def predict_with_uncertainty(self, arr):
        # Dummy: prediction = mean, sigma = std
        return {"pred": float(np.mean(arr)), "sigma": float(np.std(arr))}


class _ExtractSigma(Operator):
    """GeoTensor-like → scalar (just the uncertainty field)."""

    def _apply(self, output_dict):
        return output_dict["sigma"]

    def get_config(self) -> dict:
        return {}


# A scoring pipeline that ends in a scalar (the uncertainty estimate)
score_pipeline = Sequential(
    [
        _FakeMaskClouds(),
        ModelOp(_FakeUQModel(), method="predict_with_uncertainty"),
        _ExtractSigma(),
    ]
)


class ActiveSampler:
    """Stand-in for ``gz.sampling.ActiveSampler``."""

    def __init__(
        self, *, base: FakeCatalog, score_op: Operator, k: int = 5, mode: str = "top"
    ) -> None:
        self.base, self.score_op, self.k, self.mode = base, score_op, k, mode

    def __call__(self) -> list:
        scored = []
        for row in self.base.iter_rows():
            score = float(self.score_op(row.payload))
            scored.append((row, score))
        scored.sort(key=lambda s: s[1], reverse=(self.mode == "top"))
        return scored[: self.k]


catalog = make_demo_catalog(n=12)
sampler = ActiveSampler(base=catalog, score_op=score_pipeline, k=3, mode="top")
top_uncertain = sampler()

print("Top 3 most uncertain scenes (highest sigma):")
for row, sigma in top_uncertain:
    print(f"  {row.id}  sigma={sigma:.3f}  payload={row.payload.tolist()}")


# %% [markdown]
# **Tradeoff.** The dirty secret of active learning: **scoring is itself
# expensive** — you're running model inference over the entire unlabelled
# pool just to pick the next batch. For large catalogs this dominates the
# labelling budget you were trying to save. Mitigations: random
# sub-sampling before scoring (score 10% of the pool, label the top 1%),
# or tiered scoring (cheap heuristic filters first, expensive model
# second). Pure top-k by uncertainty also under-explores — pair with a
# `mode="diverse"` option that uses a clustering step on chip embeddings
# before picking, to avoid a top-100 that's all one land-cover type.

# %% [markdown]
# ## 11. Cross-sensor fusion (the `Graph` case)
#
# **The setting.** Combining modalities — radar with optical, optical
# with DEM, optical with weather reanalysis. Each input has different
# spatial extent, resolution, CRS, and acquisition time, and the fusion
# rule isn't a single linear chain. This is where `Sequential` runs out
# and `Graph` earns its keep.
#
# **Driver:** the applied scientist describes the DAG declaratively
# (`opt → mask → align ← sar`) instead of as a tangle of intermediate
# variables. The same `Graph` can serialise to YAML and run in any
# deployment shape from the preceding cases.
#


# %%
class _FakeLeeSpeckle(Operator):
    def __init__(self, *, window: int = 7) -> None:
        self.window = window

    def _apply(self, x):
        # Dummy: 'smooth' by mean
        arr = np.asarray(x, dtype=float)
        return arr - arr.mean() + 5.0  # standin

    def get_config(self) -> dict:
        return {"window": self.window}


class _FakeAlignAndStack(Operator):
    """Stand-in for the v0.2+ fusion op. Stacks aligned inputs along a new axis."""

    def __init__(self, *, target_grid: str = "optical") -> None:
        self.target_grid = target_grid

    def _apply(self, *inputs):
        # Dummy: just stack as 1-D arrays
        return np.stack([np.asarray(x).flatten()[:4] for x in inputs])

    def get_config(self) -> dict:
        return {"target_grid": self.target_grid}


class _FakeCropClassifier:
    def __call__(self, stack):
        return {
            "class_id": int(np.argmax(stack.sum(axis=0))),
            "scores": stack.sum(axis=0).tolist(),
        }


# Build the graph
opt = Input("optical")
sar = Input("sar")
dem = Input("dem")

opt_clean = _FakeMaskClouds(qa_band=-1, bits=[10, 11])(opt)
sar_smooth = _FakeLeeSpeckle(window=7)(sar)
fused = _FakeAlignAndStack(target_grid="optical")(opt_clean, sar_smooth, dem)
classify = ModelOp(_FakeCropClassifier())(fused)

g = Graph(
    inputs={"optical": opt, "sar": sar, "dem": dem},
    outputs={
        "crop_map": classify,
        "fused_stack": fused,  # multiple outputs are free
    },
)

print("graph:", g)
print()

# Run with dummy inputs
result = g(
    optical=np.array([10, 20, -5, 100]),
    sar=np.array([-3, -7, -2, -8]),
    dem=np.array([100, 110, 120, 105]),
)
print("crop_map   :", result["crop_map"])
print("fused_stack:\n", result["fused_stack"])


# %% [markdown]
# **Tradeoff.** `AlignAndStack` is the load-bearing operator and the
# place fusion pipelines silently fail — picking which input defines the
# target grid (`target_grid="optical"`) discards information from the
# other inputs (SAR gets resampled to the optical grid, losing the SAR's
# native resolution). For some workflows that's exactly right; for others
# (super-resolution from SAR to optical, for example) the choice is wrong.
# Make the choice explicit and test it. Multi-input arity also
# complicates Operator validation — a `Graph` with a missing input fails
# at runtime, not construction time, unless you add explicit
# input-presence checks.

# %% [markdown]
# ## 12. Workflow orchestrator task units
#
# **The setting.** Production scheduling — Airflow, Prefect, Dagster.
# The orchestrator owns retry, parallelism, monitoring, alerting, and DAG
# topology; the operator graph owns the science. The interface between
# them is small: **pickleable task units** that the orchestrator can submit
# to workers.
#
# **Driver:** Prefect / Airflow / Dagster flow definition.
# Each task wraps `(op_yaml_path, slice, source_uri)` and reconstructs the
# operator graph on the worker side.
#
# > The example below shows a Prefect-style task wrapper as a plain
# > function (no decorators) so it runs in this environment. In real code
# > you'd add `@task(retries=3, retry_delay_seconds=60)`.
#


# %%
# In real Prefect code:
#   @task(retries=3, retry_delay_seconds=60)
def run_op(op_factory, slice_dict: dict, source_uri: str) -> dict:
    """Task: reconstruct the op on the worker, run it on one slice."""
    # In real code: op = gz.serialization.load_yaml(op_yaml_path)
    op = op_factory()
    # In real code: gt = georeader.RasterioReader.open(source_uri).read_geoslice(...)
    gt = np.asarray(slice_dict["payload"])
    out = op(gt)
    out_uri = fake_save_cog(out, f"s3://daily/{slice_dict['id']}.tif")
    return {"out_uri": out_uri, "stats_mean": float(np.mean(out))}


# In real Prefect code:
#   @flow
def daily_methane_flow(date: str) -> list:
    """Flow: one task per slice in the catalog."""
    catalog = make_demo_catalog(n=5)

    # The op factory is pickleable — closures over module-level
    # classes are fine. Avoid closures over local mutable state.
    def build_op():
        return Sequential(
            [
                _FakeMaskClouds(qa_band=-1, bits=[10, 11]),
                _FakeDarkObject(percentile=1),
                _FakeNDVI(red_idx=2, nir_idx=3),
            ]
        )

    results = []
    for row in catalog.iter_rows():
        # In real Prefect:
        #   futures.append(run_op.submit(build_op, slice_dict, source_uri))
        slice_dict = {"id": row.id, "payload": row.payload.tolist()}
        results.append(run_op(build_op, slice_dict, f"s3://bucket/{row.id}.tif"))

    return results


print("Flow output:")
for r in daily_methane_flow("2024-09-15"):
    print(f"  {r}")

# %% [markdown]
# **Tradeoff.** Pickling crosses worker boundaries, so the same
# constraints as case 2 apply — no closures, no captured RNGs, no
# unbounded factories. Retries need to be **idempotent**: if `run_op` is
# retried, the output COG path is the same and is overwritten, not
# duplicated. State that lives outside the operator graph (database
# writes, message queues) is the orchestrator's problem — don't bake side
# effects into operators. Alerting on failure rate (rather than first
# failure) is appropriate at the orchestrator layer; an operator-level
# `on_fail="raise"` would defeat that.

# %% [markdown]
# ## 13. Pipeline diffing / A-B regression
#
# **The setting.** You're about to roll out a new version of a production
# pipeline — a tweaked atmospheric correction, a new cloud-mask
# threshold, a calibration update. The release-management question: *did
# anything regress that I didn't intend?* Because operator graphs
# serialise, you can both **diff** them (config-level) and **run them
# side-by-side** on the same input (numerical-level).
#
# **Driver:** release engineer or CI job runs `diff(old, new)` (textual
# config diff) and `AB(old, new).run(catalog)` (numerical comparison).
#

# %%
import difflib


def config_diff(op_a: Operator, op_b: Operator) -> str:
    """Human-readable diff of two operator graphs' configs."""
    a_lines = _json.dumps(op_a.get_config(), indent=2, sort_keys=True).splitlines()
    b_lines = _json.dumps(op_b.get_config(), indent=2, sort_keys=True).splitlines()
    diff = difflib.unified_diff(a_lines, b_lines, lineterm="", n=2)
    return "\n".join(diff)


class _MeanAbsDiff:
    def __call__(self, a, b):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


class AB:
    """Run two pipelines on the same inputs; report per-slice metric."""

    def __init__(self, op_a: Operator, op_b: Operator, *, metric) -> None:
        self.a, self.b, self.metric = op_a, op_b, metric

    def run(self, catalog) -> dict:
        rows = []
        for row in catalog.iter_rows():
            gt = row.payload
            ya, yb = self.a(gt), self.b(gt)
            rows.append((row.id, float(self.metric(ya, yb))))
        rows.sort(key=lambda r: -r[1])
        return {
            "n": len(rows),
            "mean_metric": float(np.mean([r[1] for r in rows])),
            "max_metric": rows[0][1],
            "max_slice": rows[0][0],
            "per_slice": rows,
        }


old = Sequential(
    [
        _FakeMaskClouds(),
        _FakeDarkObject(percentile=1),  # old: percentile=1
        _FakeNDVI(),
    ]
)

new = Sequential(
    [
        _FakeMaskClouds(),
        _FakeDarkObject(percentile=2),  # new: percentile=2
        _FakeLeeSpeckle(window=5),  # new: added pre-step
        _FakeNDVI(),
    ]
)


print("=== Config diff (textual) ===")
print(config_diff(old, new))
print()

print("=== Numerical A/B over the catalog ===")
catalog = make_demo_catalog(n=5)
report = AB(old, new, metric=_MeanAbsDiff()).run(catalog)
print(f"n          : {report['n']}")
print(f"mean diff  : {report['mean_metric']:.4g}")
print(f"max diff   : {report['max_metric']:.4g} ({report['max_slice']})")
print()
print("per-slice (sorted by diff):")
for sid, d in report["per_slice"]:
    print(f"  {sid}  diff={d:.4g}")

# %% [markdown]
# **Tradeoff.** Mean RMSE alone can hide bimodal regressions — most
# slices fine, a few catastrophically worse. Always look at per-tile
# distributions and the worst-N tiles, not just aggregate stats. Choosing
# the metric is itself a design decision: RMSE is wrong for probability
# outputs (use cross-entropy or KL), wrong for hard classifications (use
# confusion-matrix shifts). The `config_diff` is human-readable but
# doesn't capture *semantic* difference — a parameter rename that's
# mathematically equivalent shows up as a diff anyway. Pair config-level
# diffing with numerical AB testing, and trust the numerics.

# %% [markdown]
# ## The pattern
#
# The operator graph is useful **wherever you'd otherwise hand-roll glue
# between input → transform → output**. What changes between cases is
# *who drives*: a notebook user, an HTTP request, a scheduler, a sampler,
# a tile renderer, or another graph. The library's job is to keep the
# graph itself a stable artifact across all of them — written once,
# audited once, run anywhere.
#
# ## Where next
#
# - [Concepts](https://github.com/jejjohnson/geotoolz/blob/main/docs/concepts.md) — the model behind the primitives.
# - [Composition core walkthrough](01_composition_core.ipynb) — primitives demonstrated against scalars.
# - [Pipeline idioms](02_pipeline_idioms.ipynb) — recipe gallery of observer / control-flow / composition / QC patterns, including build-your-own implementations for the v0.2+ ops.
# - [Core API reference](https://github.com/jejjohnson/geotoolz/blob/main/docs/api/core.md) — operator-by-operator docs.
#

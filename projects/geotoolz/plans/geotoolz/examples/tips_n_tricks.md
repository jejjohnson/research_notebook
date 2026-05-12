---
title: geotoolz pipeline tricks
subject: geotoolz examples
subtitle: Small composable Operators that punch above their weight
short_title: Pipeline tricks
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geotoolz, operators, examples, remote-sensing
---

> **Status:** companion to [`geotoolz.md`](../geotoolz.md) — codifies the v0.1 idiom library of observers, control flow, assertions, and small building blocks.
> **Audience:** anyone composing `geotoolz` pipelines beyond the canned operator surface; anyone deciding which speculative tricks belong in v0.1 vs later.

A reference of small composable Operators that punch above their weight.
The meta-pattern across all of them: **the `Operator` interface is general enough to express things that aren’t just transforms** — side effects, assertions, profiling, control flow, error handling, caching, and metadata propagation all become first-class composable units.

Drop them into a `Sequential` like any other op.
They follow the same config-round-trip rules, plug into Hydra-zen YAML, and respect the carrier contract (`GeoTensor → GeoTensor`, even when they don’t transform).

## Contents

- [Inspection / introspection (Tap family)](#inspection--introspection-tap-family)
  - [Tap](#tap) · [Snapshot](#snapshot) · [TimeIt / Profile](#timeit--profile) · [Histogram](#histogram) · [ShapeTrace](#shapetrace) · [Spy / Hook](#spy--hook) · [Diff](#diff)
- [Control flow](#control-flow)
  - [Branch](#branch) · [Switch](#switch) · [Try / Fallback](#try--fallback) · [Coalesce](#coalesce) · [Retry](#retry)
- [Composition](#composition)
  - [Fanout](#fanout) · [ApplyToBands](#applytobands) · [Cache / Memoize](#cache--memoize)
- [Stateful / ML](#stateful--ml)
  - [Mode](#mode) · [Provenance / Watermark](#provenance--watermark)
- [Validation / QC (assertion family)](#validation--qc-assertion-family)
  - [AssertX](#assertx-recap) · [Quarantine](#quarantine)
- [Small but load-bearing building blocks](#small-but-load-bearing-building-blocks)
  - [Identity](#identity) · [Const](#const) · [Lambda](#lambda) · [Sink](#sink) · [Subsample](#subsample)
- [Two design rules](#two-design-rules)

-----

## Inspection / introspection (Tap family)

The simplest, most powerful family: identity Operators with side effects.
They let users observe a pipeline without breaking the chain.

### Tap

The seed pattern.
An identity Operator with a side effect — the `fn` runs, the GeoTensor flows through unchanged.
Already in the core surface; included here because the rest of this section builds on it.

```python
gz.Sequential([
    gz.cloud.MaskClouds(...),
    gz.core.Tap(lambda gt: print(f"after mask: {np.isnan(gt).mean():.1%} NaN")),
    gz.indices.NDVI(...),
])
```

### Snapshot

`Tap` that **stores** instead of prints.
After the pipeline runs, every named intermediate is available for plotting, debugging, or comparison.

```python
snap = gz.core.Snapshot()
viz = gz.Sequential([
    gz.cloud.MaskClouds(...),  snap.at("after_mask"),
    gz.correction.TOAToBOA(...), snap.at("after_correction"),
    gz.indices.NDVI(...),       snap.at("ndvi"),
])
viz(gt)

plt.imshow(snap["after_mask"].values)
plt.imshow(snap["ndvi"].values, cmap="RdYlGn")
```

Implementation:

```python
class Snapshot:
    def __init__(self):
        self._store: dict[str, GeoTensor] = {}
    def at(self, key: str) -> Operator:
        return _SnapshotTap(self._store, key)
    def __getitem__(self, key): return self._store[key]
    def keys(self): return self._store.keys()

class _SnapshotTap(Operator):
    def __init__(self, store, key): self.store, self.key = store, key
    def _apply(self, gt): self.store[self.key] = gt; return gt
```

**Tradeoff.** Stores *references*, not copies — if an in-place op downstream mutates the array, your snapshot sees it too.
Add `copy=True` for safety in exploratory work; leave default `False` for hot loops.

### TimeIt / Profile

Per-step latency in a Sequential. Drop in once, get a profile for free.

```python
prof = gz.core.Profile()
ops = [gz.cloud.MaskClouds(...), gz.correction.TOAToBOA(...), gz.indices.NDVI(...)]
pipe = gz.Sequential([prof.wrap(op) for op in ops])

pipe(gt)
prof.report()
# MaskClouds:  12 ms
# TOAToBOA:   340 ms   ← the suspect
# NDVI:         8 ms
# total:      360 ms
```

Implementation:

```python
class Profile:
    def __init__(self): self._timings: dict[str, list[float]] = defaultdict(list)
    def wrap(self, op):
        return _TimedOp(op, self._timings)
    def report(self):
        for name, ts in self._timings.items():
            print(f"{name}: mean={np.mean(ts)*1e3:.1f}ms n={len(ts)}")

class _TimedOp(Operator):
    def __init__(self, inner, store):
        self.inner, self.store = inner, store
    def _apply(self, gt):
        t0 = time.perf_counter()
        out = self.inner(gt)
        self.store[type(self.inner).__name__].append(time.perf_counter() - t0)
        return out
```

**Tradeoff.** `wrap()` adds one Python frame per op — negligible compared to GeoTensor work but visible in microbenchmark territory.
For tile-server hot paths, profile during development and remove before deploying.

### Histogram

`Tap` that captures **distributions**, not point summaries.
Useful for “is this band shifted from last week?” or “did my correction blow out the bright end?”

```python
hist = gz.core.Histogram(bins=50)
pipe = gz.Sequential([
    op1, hist.at("pre_correction"),
    op2, hist.at("post_correction"),
])
pipe(gt)
hist.plot(overlay=True)               # both distributions on one axis
hist.compare("post_correction", reference="/data/ref_dist.npz")  # vs golden
```

**Tradeoff.** Cheap *per call* (one `np.histogram`), expensive in *aggregate* if you bin every chip in a 10k-tile run.
For ETL monitoring, sample 1% of tiles rather than all of them.

### ShapeTrace

Logs shape, dtype, CRS, transform, and bands at every step.
Invaluable when an op silently strips a band or changes a transform.

```python
gz.Sequential([
    gz.core.ShapeTrace(),
    op1,
    gz.core.ShapeTrace(),
    op2,
    gz.core.ShapeTrace(),
])(gt)
# step 0: (4, 256, 256) float32 EPSG:32630 res=10  bands=B02,B03,B04,B08
# step 1: (4, 256, 256) float32 EPSG:32630 res=10  bands=B02,B03,B04,B08
# step 2: (1, 256, 256) float32 EPSG:32630 res=10  bands=ndvi   ← collapsed
```

A diff-mode is the killer feature — only logs *changes* between steps, so a 20-op pipeline doesn’t drown you in 20 identical lines.

```python
gz.core.ShapeTrace(mode="diff_only")
```

### Spy / Hook

Register a callback that fires whenever an op of a specific *type* runs anywhere in the graph.
Same idea as PyTorch forward hooks, but for operator graphs — cross-cutting instrumentation without modifying every Sequential.

**Hooks are scoped, never global.** A naive global registry — hooks registered in one module firing in another — is a footgun the size of a barn (debug output appearing in production, tests leaking hooks into other tests).
The primary API is a `with` block:

```python
# Debug: "why is MatchedFilter being called 47 times?"
def trace_mf(op, input_gt, output_gt):
    print(f"MF call: input shape {input_gt.shape}, "
          f"output stats {output_gt.summary_stats()}")

with gz.core.Spy.scoped() as spy:
    spy.on(gz.hyperspectral.MatchedFilter, trace_mf)
    big_pipeline(scene_gt)  # trace_mf fires per MatchedFilter call
# scope exits — hooks deregister; subsequent calls are silent again
```

Implementation:

```python
import contextvars

# Process-scoped stack of active Spy contexts. Empty by default —
# calls outside `Spy.scoped()` pay zero cost (empty-list iteration).
_active_spies: contextvars.ContextVar[tuple["Spy", ...]] = \
    contextvars.ContextVar("geotoolz_spy_stack", default=())

class Spy:
    def __init__(self):
        self._hooks: dict[type, list[Callable]] = defaultdict(list)

    @classmethod
    @contextmanager
    def scoped(cls):
        spy = cls()
        token = _active_spies.set(_active_spies.get() + (spy,))
        try:
            yield spy
        finally:
            _active_spies.reset(token)

    def on(self, op_type: type, fn: Callable) -> None:
        self._hooks[op_type].append(fn)

    @staticmethod
    def fire(op, input_gt, output_gt) -> None:
        for spy in _active_spies.get():
            for fn in spy._hooks.get(type(op), ()):
                fn(op, input_gt, output_gt)
```

`Operator.__call__` invokes `Spy.fire(self, args, output)` after `_apply`.
When no `Spy.scoped()` block is active, `_active_spies.get()` returns `()` and the firing path is a single empty-tuple iteration.

**Tradeoff.** Scoping is the safe default but it doesn't help when the behaviour you want to debug is *deep inside a worker process*.
For that case, the worker can enter its own `Spy.scoped()` at startup — the contextvar behaves correctly under `concurrent.futures` workers.
Avoid promoting a "global / sticky" `Spy.on` to a public API; if a test or a notebook needs the global feel, wrap the whole session in `with Spy.scoped() as s: ...` and register at the top.

### Diff

Compares output to a stored reference; raises on drift beyond a tolerance.
The **pytest of operator pipelines** — drop inline during refactoring to catch silent regressions.

```python
gz.Sequential([
    op1, op2,
    gz.core.Diff.against("/refs/post_correction.tif", atol=1e-4),
    op3,
])(gt)
# raises DiffError("max abs diff 0.0023 exceeds atol=1e-4 at index (0, 142, 89)")
```

Workflow: bless a reference once you trust the pipeline output, then `Diff` catches any subsequent change.
Particularly useful when refactoring a `Sequential` into a `Graph` or swapping a primitive — the numbers should match, and `Diff` proves it.

```python
# bless mode (run once when you know the pipeline is correct)
gz.core.Diff.bless("/refs/post_correction.tif")(gt)
```

**Tradeoff.** Reference files drift over time as legitimate improvements ship.
Pair with version pinning: `Diff.against("/refs/v3/...", atol=...)` per pipeline version, never overwrite a blessed reference in place.

-----

## Control flow

The Operator interface is general enough to express conditional execution, fallbacks, and retries.
Same composition primitives, more interesting graphs.

### Branch

Conditional execution based on a predicate over the GeoTensor.
Apply correction only if the input warrants it; otherwise pass through.

```python
gz.Sequential([
    gz.core.Branch(
        predicate=lambda gt: cloud_fraction(gt) < 0.3,
        if_true=gz.correction.TOAToBOA(...),
        if_false=gz.core.Identity(),
    ),
    gz.indices.NDVI(...),
])
```

The predicate is a callable rather than an Operator because it’s a *boolean decision* about an input, not a transform of it.
The two arms are Operators because they perform the work.

### Switch

Multi-way Branch — dispatch on a metadata field (sensor, season, region):

```python
gz.core.Switch(
    key=lambda gt: gt.metadata["sensor"],
    cases={
        "S2":      gz.presets.s2.S2_L2A_NDVI(),
        "Landsat": gz.presets.landsat.L8_BOA_NDVI(),
        "MODIS":   gz.presets.modis.MOD13_NDVI(),
    },
    default=gz.core.Raise("unknown sensor in {key}"),
)
```

Use this for cross-sensor pipelines where the *same downstream analysis* needs sensor-specific preprocessing.
The downstream op stays one Sequential; the sensor branching lives in one place.

**Tradeoff.** A `Switch` with five branches whose bodies share 80% of their operators is a code smell — refactor into `Sequential([common_prefix, Switch(...) for the divergent step])`.
Don’t let `Switch` become a copy-paste vehicle.

### Try / Fallback

Robust to upstream flakiness — if the first op fails, try the fallback.
Useful for ML inference where a remote model is occasionally unavailable, or for batch jobs that should survive transient errors.

```python
gz.core.Try(
    primary=gz.inference.ModelOp(model_v2, batch_size=8),
    fallback=gz.inference.ModelOp(model_v1, batch_size=4),  # smaller batch on OOM
    on=(torch.cuda.OutOfMemoryError, ConnectionError),
)
```

Always specify the exception types in `on=`.
Catching bare `Exception` masks real bugs.

**Tradeoff.** Silent fallback can hide deteriorating production conditions — “the v2 model has been OOMing for two weeks but Try kept covering it up.” Pair with metrics: log every fallback as a counter, alert when the rate exceeds a threshold.

### Coalesce

First-non-empty across sources — the “S2 first, fall back to Landsat if S2 is too cloudy, fall back to MODIS if Landsat is also unavailable” pattern.

```python
gz.core.Coalesce([
    s2_pipeline,
    landsat_pipeline,
    modis_pipeline,
], is_empty=lambda gt: np.isnan(gt).mean() > 0.7)
```

Distinct from `Try` — `Coalesce` cascades on a *quality predicate* (the output is bad), `Try` cascades on an *exception* (the operator failed).
They compose naturally:

```python
gz.core.Coalesce([
    gz.core.Try(s2_v2, fallback=s2_v1, on=(ModelError,)),  # S2 with fallback model
    landsat_pipeline,
])
```

### Retry

Wraps an op with retry + backoff.
Most useful for `ModelOp` over a remote API or any op touching the network.

```python
gz.core.Retry(
    op=gz.inference.ModelOp(remote_endpoint),
    attempts=3,
    backoff="exponential",   # 1s, 2s, 4s
    on=(ConnectionError, TimeoutError),
)
```

**Tradeoff.** Retries hide latency from upstream callers — a tile server request that nominally takes 200ms can spike to 7 seconds because of two retries.
For latency-sensitive paths, prefer `Try` with a fast fallback over `Retry` with backoff.

-----

## Composition

Building blocks for graphs that aren’t pure linear chains.

### Fanout

One input, multiple outputs — useful when you want N derived products from one read instead of re-reading the source N times.

```python
products = gz.core.Fanout({
    "ndvi":  gz.indices.NDVI(red_idx=2, nir_idx=3),
    "ndwi":  gz.indices.NDWI(green_idx=1, nir_idx=3),
    "rgb":   gz.presets.s2.S2_L2A_RGB(),
})(gt)
# {"ndvi": GeoTensor, "ndwi": GeoTensor, "rgb": GeoTensor}
```

For computing many indices on one scene, Fanout reads once and computes once per index.
Compared to running three separate Sequentials, it’s both faster (one read) and more honest (the input GeoTensor is genuinely shared).

A `Graph` with a single input and N outputs expresses the same thing more formally.
`Fanout` is the sugar for the common case.

### ApplyToBands

Split-apply-combine over the band axis — applies the inner op to each band independently, recombines.
Mirrors xarray’s `.groupby()` shape but for bare arrays.

```python
# Apply Lee speckle filter independently to each polarisation channel
gz.core.ApplyToBands(gz.sar.LeeSpeckle(window=7), axis=0)(sar_gt)
```

Composable with everything: drop into a Sequential, wrap in Cache, profile with TimeIt.
The inner op becomes a regular Operator that just happens to run N times.

**Tradeoff.** Naive implementation is a Python-level for-loop over bands — fine for ~10 bands, painful for hyperspectral with ~200. For hyperspectral, the inner op should ideally be vectorisable across the band axis directly; `ApplyToBands` is the fallback when it isn’t.

### Cache / Memoize

Hashes input + operator config, caches the result.
Saves hours during iterative analysis where you keep re-running the same expensive prefix.

```python
expensive_prefix = gz.core.Cache(
    gz.Sequential([
        gz.correction.TOAToBOA(sun_zenith_band=-2),
        gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    ]),
    backend="disk",
    path="~/.geotoolz_cache",
)
# First run: 30s. Every subsequent run on the same input: 0.1s.
```

Implementation sketch:

```python
class Cache(Operator):
    def __init__(self, inner, *, backend="memory", path=None):
        self.inner = inner
        self.store = _make_backend(backend, path)

    def _apply(self, gt):
        key = self._key(gt)
        if key in self.store:
            return self.store[key]
        out = self.inner(gt)
        self.store[key] = out
        return out

    def _key(self, gt):
        return hashlib.sha256(
            (gt.content_hash() + self.inner.config_hash()).encode()
        ).hexdigest()
```

The cache key is `(input_hash, inner.config_hash())`.
Both must be stable — the input hash is on `GeoTensor` content (or a checksum of the source bytes), the config hash is what `get_config()` returns canonicalised.

**Tradeoff.** Memory backend is fast but unbounded — wrap with an LRU. Disk backend is restart-friendly but you need to garbage-collect old entries.
Cache must opt-in (`Cache(...)` wraps explicitly), never silent — if `Cache` ever becomes a default, debugging “why does my pipeline produce stale output” becomes a nightmare.

-----

## Stateful / ML

Operators that hold state across calls or interact with the broader runtime.

### Mode

Train / eval switching for pipelines that mix deterministic preprocessing with train-only operators (augmentation, dropout-like ops).
The lesson from PyTorch is that *implicit, instance-level, sticky* `train()` / `eval()` is a footgun — "I forgot to call `.eval()` on the batchnorm" is the well-known bug.
`geotoolz` takes the **scoped, explicit** path: mode is a context manager on a Sequential, not a sticky attribute on the operator graph.

```python
pipe = gz.Sequential([
    preprocess,
    gz.augment.RandomFlip(p=0.5).train_only(),
    gz.augment.RandomCrop(...).train_only(),
])

# Train: enter the train scope explicitly, augmentations apply inside.
with pipe.mode("train"):
    train_x = pipe(scene_gt)

# Eval: explicit again. Outside any `with` block, calling the pipe raises.
with pipe.mode("eval"):
    val_x = pipe(scene_gt)
```

Implementation:

```python
from contextlib import contextmanager

class Sequential(Operator):
    _mode: str | None = None  # unset by default — calls outside `mode(...)` raise

    @contextmanager
    def mode(self, m: str):
        if m not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {m!r}")
        prev, self._mode = self._mode, m
        try:
            yield self
        finally:
            self._mode = prev

class _ModeGated(Operator):
    """Wraps an operator that only fires under a specific mode."""
    def __init__(self, inner, *, mode: str):
        self.inner, self.required_mode = inner, mode

    def _apply(self, gt, *, _seq_mode: str):
        if _seq_mode == self.required_mode:
            return self.inner(gt)
        return gt
```

`Sequential._apply` threads the current `_mode` to `_ModeGated` children explicitly (a private kwarg or a contextvar — either works; the contextvar form nests cleanly for `Sequential`-in-`Sequential`).
Either way, **there is no "current mode" without an active `with` block** — a defensive default that prevents the PyTorch bug at construction time.

**Tradeoff.** Slightly more verbose than `pipe.train()` / `pipe.eval()`.
The verbosity is the feature — every train-mode invocation is grep-able in source, and code that forgets to enter the scope fails loudly instead of silently producing augmented "validation" tensors.

### Provenance / Watermark

Operator graph stamps lineage metadata into the output GeoTensor.
Survives subsequent transformations, lands in the COG header on save.
Now your output COGs *know* what produced them.

```python
pipe = gz.core.Provenance.wrap(my_methane_pipeline)
out = pipe(scene_gt)

out.metadata["provenance"]
# {
#   "pipeline_hash": "abc123...",
#   "operators":     ["MaskFromQABits", "DarkObjectSubtraction", "MatchedFilter"],
#   "inputs":        {"scene_uri": "s3://...", "sha256": "..."},
#   "geotoolz_version": "0.4.2",
#   "timestamp":     "2026-05-10T14:32:00Z",
# }

georeader.save_cog(out, "/out/methane.tif")  # provenance baked into COG tags
```

Pairs naturally with the pinned-artifact pattern from the [use-cases doc](usecases.md#9-pinned--hashed-regulatory-artifact) — the provenance metadata in the COG references the artifact hash, so a consumer years later can chase the COG back to the exact pipeline that produced it.

**Tradeoff.** Provenance metadata grows with every wrap — a deeply-nested Graph produces a large lineage record.
Cap at a reasonable depth, or store the full graph hash and a small operator-name summary rather than every config. Don’t let provenance bloat overshadow the pixel data.

-----

## Validation / QC (assertion family)

The other identity-with-side-effect family — pass-through Operators that *check* invariants rather than observe state.
See [the QC use case](usecases.md#8-data-validation--qc-as-operators) for the full pattern.

### AssertX (recap)

A family of pass-through Operators that raise (or warn) on contract violations.
Drop anywhere in a pipeline.

```python
gz.Sequential([
    gz.qc.AssertCRSEquals("EPSG:32630"),
    gz.qc.AssertResolutionWithin((9.5, 10.5)),
    gz.qc.AssertValidFraction(min_valid=0.5),
    gz.qc.AssertValueRange(min_val=0, max_val=10_000, on_fail="warn"),
])
```

### Quarantine

Non-raising QC: routes bad GeoTensors to a sidecar location for later debugging, and lets the pipeline continue.
The “log_and_continue” of QC.

```python
gz.Sequential([
    op1,
    gz.qc.Quarantine(
        check=gz.qc.AssertValidFraction(min_valid=0.5),
        sink="s3://debug/quarantine/",
        on_quarantine=lambda gt, err: log.warn(f"quarantined: {err}"),
    ),
    op2,
])
```

Behaviour: if the inner check passes, GeoTensor flows through.
If it fails, the GeoTensor is written to the sink with the failure reason as sidecar metadata, the pipeline returns a sentinel (e.g. an all-NaN GeoTensor of the same shape), and downstream ops skip it gracefully.

The orchestrator gets a “look here for bad data” pile that grows over time — a free dataset of failure modes for debugging and improving upstream readers.

**Tradeoff.** Quarantine is a *hedge* — it trades immediate failure for delayed debugging.
Don’t quarantine errors that indicate genuine bugs (wrong CRS, wrong band order); those should `raise`.
Quarantine is for *expected edge cases* in the data (corrupt scenes, partial downloads, sensor glitches).

-----

## Small but load-bearing building blocks

Boring on their own, indispensable in combination.

### Identity

Explicit no-op.
The right thing to put in a `Branch` default, a `Switch` unmatched case, or anywhere a pipeline structurally needs an Operator but you have no work to do.

```python
class Identity(Operator):
    def _apply(self, gt): return gt
    def get_config(self): return {}
```

```python
gz.core.Branch(predicate=is_clean, if_true=gz.core.Identity(), if_false=cleanup)
```

Use `Identity` rather than passing `None`.
It serialises, it composes, it shows up in `repr()` and `ShapeTrace` output.
Being explicit about no-ops makes them visible.

### Const

Returns a fixed GeoTensor regardless of input.
Great for golden test fixtures and as a synthetic source in `Switch` defaults.

```python
class Const(Operator):
    def __init__(self, gt: GeoTensor):
        self.gt = gt
    def _apply(self, _): return self.gt
    def get_config(self): return {"shape": self.gt.shape, "crs": str(self.gt.crs)}
```

```python
test_pipeline = gz.Sequential([
    gz.core.Const(synthetic_gt),
    real_pipeline,                # exercises real_pipeline against known input
])
```

### Lambda

Inline-function escape hatch when writing a full Operator subclass is overkill.

```python
class Lambda(Operator):
    def __init__(self, fn, *, name: str = "lambda"):
        self.fn, self.name = fn, name
    def _apply(self, gt): return self.fn(gt)
    def get_config(self): return {"name": self.name}  # debug repr only
```

```python
gz.Sequential([
    gz.cloud.MaskClouds(...),
    gz.core.Lambda(lambda gt: gt * 0.0001, name="scale_to_reflectance"),
    gz.indices.NDVI(...),
])
```

**Tradeoff.** Closures aren’t config-round-trippable.
`Lambda` should be flagged `forbid_in_yaml=True` — research-only, never ships to prod.
The first time a `Lambda` recurs, refactor it into a real Operator subclass with a name.

### Sink

A terminal write Operator that’s still composable.
The classic write op (e.g. `WriteCOG`) returns nothing, which means it can’t be `Tap`-ped or `Snapshot`-ed and breaks cleanly into other ops.
`Sink` writes *and* returns the input unchanged.

```python
class Sink(Operator):
    def __init__(self, write_fn):
        self.write_fn = write_fn
    def _apply(self, gt):
        self.write_fn(gt)
        return gt   # ← unchanged

# Usage: write to disk *and* keep going
gz.Sequential([
    op1,
    gz.core.Sink(lambda gt: georeader.save_cog(gt, "/intermediate.tif")),
    op2,                  # still gets the GeoTensor
])
```

Useful for debugging (“what did the pipeline look like at step 3?”), for checkpointing long pipelines (write intermediate to disk in case the rest crashes), and for branching analysis (write intermediate, continue with final product).

### Subsample

Random pixel sample inside a `Tap`-style op, for fast viz off a full-resolution GeoTensor without re-reading.

```python
gz.Sequential([
    full_resolution_pipeline,
    gz.core.Tap(gz.core.Subsample(fraction=0.01).then(plot_histogram)),
])
```

Or as a standalone transform that returns a smaller GeoTensor (decimated, with an updated transform):

```python
small_gt = gz.core.Subsample(stride=10)(big_gt)  # every 10th pixel
plt.imshow(small_gt.values)
```

**Tradeoff.** Random subsampling biases summary statistics for non-uniform fields (e.g., concentrated NDVI hotspots in mostly-bare scenes).
For fair visualisation, use stride-based subsampling; for fair statistics, use weighted random sampling.

-----

## Two design rules

These keep the surface tractable as the trick library grows.

### 1. Honest naming

Don’t disguise an assertion as a transform.
Don’t disguise a side effect as a computation.
Users should be able to scan a Sequential and immediately see which steps mutate, which observe, which guard, and which control flow.

```python
# Good — the role of each step is obvious
gz.Sequential([
    gz.cloud.MaskClouds(...),                       # transform
    gz.qc.AssertValidFraction(min_valid=0.5),       # guard
    gz.core.Tap(log_stats),                         # observe
    gz.core.Branch(predicate=..., if_true=..., if_false=Identity()),  # control flow
    gz.indices.NDVI(...),                           # transform
])
```

Naming convention (suggested):

- `qc.AssertX` for assertions
- `core.Tap`, `core.Snapshot`, `core.Histogram`, `core.TimeIt`, `core.ShapeTrace` for observers
- `core.Branch`, `core.Switch`, `core.Try`, `core.Coalesce`, `core.Retry` for control flow
- everything else is a transform

### 2. Round-trip discipline

Operators that hold closures (`Tap`, `Lambda`, `Branch` with a callable predicate, `Spy` hooks) **cannot** round-trip to YAML faithfully.
Their `get_config()` is a debug repr, not a faithful serialisation.

The library should:

- Flag those Operators with `forbid_in_yaml = True`
- Have the YAML loader refuse to instantiate any operator with that flag set
- Have the YAML dumper raise (loudly) if asked to serialise a graph containing one

Production pipelines never contain closures.
This keeps the “operator graph as audit artifact” guarantee from the [regulatory artifact use case](usecases.md#9-pinned--hashed-regulatory-artifact) honest — every operator in a regulatory artifact has a stable config, every config round-trips, every artifact reruns to the same answer.

-----

## The shape

Once these primitives exist, **most user “Operator subclass” needs go away.**

|Instead of writing… |Compose…                                                      |
|--------------------|--------------------------------------------------------------|
|`LoggedNDVI`        |`Tap(log) | NDVI(...)`                                        |
|`OptionalCorrection`|`Branch(predicate, if_true=Correction(), if_false=Identity())`|
|`RobustModelOp`     |`Retry(ModelOp(...), attempts=3) | Try(..., fallback=...)`    |
|`BandwiseSpeckle`   |`ApplyToBands(LeeSpeckle(...), axis=0)`                       |
|`CachedPipeline`    |`Cache(my_pipeline, backend="disk")`                          |
|`TimedPipeline`     |`Profile().wrap(my_pipeline)`                                 |
|`MultiOutputModel`  |`Fanout({"a": op_a, "b": op_b})`                              |
|`WithLogging`       |`Tap(log_stats)`                                              |
|`WithProvenance`    |`Provenance.wrap(my_pipeline)`                                |
|`S2OrLandsatNDVI`   |`Switch(key="sensor", cases={...})`                           |

The library’s primitive set does the work; users compose.
Adding a new trick adds one Operator, not a family of variants.
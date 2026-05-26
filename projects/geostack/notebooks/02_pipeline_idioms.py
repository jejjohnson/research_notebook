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
# title: "02 — Pipeline idioms"
# ---
#
# # Pipeline idioms — a recipe gallery
#
# `geotoolz` re-exports the core composition algebra from `pipekit`, which ships
# observers (`Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram`), control
# flow (`Branch`, `Switch`, `Try`, `Coalesce`, `Retry`), caching (`Cache`,
# `Memoize`), QC (`AssertShape`, `AssertDType`, `AssertHasAttribute`,
# `Quarantine`), parallelism (`Thread`/`Process`/`Async`/`BatchedMap`), and
# the basic building blocks. This notebook is the **tour**: each section
# imports the shipped op, drops it into a `Sequential`, and shows the shape.
#
# Where the shipped primitive does not cover a pattern (e.g. `Spy`, `Diff`,
# `Provenance`, `Subsample`, `ApplyToBands`), the section also shows a
# **build-your-own** subclass that does. Treat those as recipes — the
# composition surface is general enough to express them in a few lines.
#
# Most cells run on plain integers or dicts (no `GeoTensor` setup). The
# shapes translate directly to real RS pipelines — the
# [operators_lake_tahoe](03_operators_lake_tahoe.ipynb) notebook exercises the
# same observability primitives on a real Sentinel-2 scene.
#

# %% [markdown]
# ## Shared setup — stand-in domain operators
#
# A few minimal `Operator` subclasses standing in for the domain ops we haven't shipped yet. They're not pretending to be real — they're just named placeholders so the *shape* of a `Sequential([MaskClouds(...), NDVI(...)])` is recognisable.

# %%
from geotoolz import Operator, Sequential


class Add(Operator):
    """Tiny test operator: x → x + n."""

    def __init__(self, n: int) -> None:
        self.n = n

    def _apply(self, x: int) -> int:
        return x + self.n

    def get_config(self) -> dict:
        return {"n": self.n}


class _NamedStep(Operator):
    """Named identity-with-tag stand-in for any domain op (NDVI, MaskClouds, ...)."""

    def __init__(self, name: str, *, factor: float = 1.0) -> None:
        self.name = name
        self.factor = factor

    def _apply(self, x):
        return x * self.factor

    def get_config(self) -> dict:
        return {"name": self.name, "factor": self.factor}


# A 'pipeline' that stands in for a real preprocessing chain
demo_pipe = Sequential(
    [
        _NamedStep("MaskClouds", factor=1.0),
        _NamedStep("PercentileClip", factor=0.98),
        _NamedStep("NDVI", factor=0.5),
    ]
)
print(demo_pipe)
print("demo_pipe(10) =", demo_pipe(10))

# %% [markdown]
# ## 1. Inspection / introspection (the Tap family)
#
# Identity operators with side effects — they let you observe a pipeline
# mid-flight without breaking the chain. Drop them between steps to log
# stats, capture intermediates, profile, or validate against a reference.
#

# %% [markdown]
# ### Tap — fire a callback (shipped)
#
# The seed pattern. Calls `fn(x)`, passes input through unchanged. Returns
# of `fn` are ignored — `Tap` is for side effects, not transforms.

# %%
from geotoolz import Tap


seen = []
pipe = Sequential(
    [
        Add(1),
        Tap(lambda x: seen.append(("after Add(1)", x))),
        Add(10),
        Tap(lambda x: seen.append(("after Add(10)", x))),
    ]
)
out = pipe(0)
print("result :", out)
print("log    :", seen)

# %% [markdown]
# ### Snapshot — capture intermediates by name (shipped)
#
# A controller (not an `Operator` itself) that produces snapshot ops via
# `snap.at(key)`. After the pipeline runs, intermediates are keyed in
# `snap.captures[key]`. Useful for "what did step 3 produce?" without
# re-running.

# %%
from geotoolz import Snapshot


snap = Snapshot()
pipe = Sequential(
    [
        Add(1),
        snap.at("after_first"),
        Add(10),
        snap.at("after_second"),
        Add(100),
    ]
)
pipe(0)
print("captured:", snap.captures)

# %% [markdown]
# ### `Profile` — per-step timings (shipped)
#
# `Profile.wrap(op)` returns a timing-instrumented wrapper. `prof.report()`
# summarises calls / total / mean per operator class.

# %%
from pipekit import Profile


prof = Profile()
pipe = Sequential([prof.wrap(Add(1)), prof.wrap(Add(10)), prof.wrap(Add(100))])
for _ in range(100):
    pipe(0)
print("Per-step timings:")
for name, stats in prof.report().items():
    print(
        f"  {name:18s}  mean={stats['mean'] * 1e6:7.1f} µs  "
        f"calls={int(stats['calls'])}  total={stats['total'] * 1e3:.2f} ms"
    )

# %% [markdown]
# **Tradeoff.** `wrap()` adds one Python frame per op — negligible compared
# to GeoTensor work but visible in microbenchmark territory.

# %% [markdown]
# ### `Histogram` — capture distributions (shipped)
#
# Controller analogous to `Snapshot`, but each `hist.at(key)` records a
# histogram of the values at that point. Bin counts accumulate across
# calls — useful for "did this band shift from last week?" or "did my
# correction blow out the bright end?"

# %%
import numpy as np
from pipekit import Histogram


hist = Histogram(bins=5, to_array=lambda a: np.asarray(a, dtype=float).ravel().tolist())
pipe = Sequential(
    [
        hist.at("input"),
        _NamedStep("PercentileClip", factor=0.5),
        hist.at("after_clip"),
    ]
)
pipe(np.arange(20))
print("input    counts:", hist.results["input"]["counts"])
print("clipped  counts:", hist.results["after_clip"]["counts"])

# %% [markdown]
# **Tradeoff.** Cheap per call, expensive in aggregate if you bin every
# chip in a 10k-tile run. For ETL monitoring, sample 1% of tiles instead
# of every one.

# %% [markdown]
# ### `ShapeTrace` — log shape/dtype/CRS at each step (shipped)
#
# Drop one between steps of a `Sequential` to log what's happening to the
# carrier. `mode="diff_only"` suppresses lines that don't change.

# %%
from geotoolz import Lambda, ShapeTrace


trace = ShapeTrace(mode="diff_only")
pipe = Sequential(
    [
        trace,
        Lambda(
            lambda x: np.asarray([x, x * 2, x * 3], dtype=np.int16), name="to_array"
        ),
        trace,
        Lambda(lambda a: a.astype(np.float32), name="to_float"),
        trace,
    ]
)
pipe(7)

# %% [markdown]
# ### `Spy` / `Hook` — cross-cutting hooks (build-your-own)
#
# Process-scoped hooks that fire whenever an operator of a specific type
# runs, anywhere in the graph. Same idea as PyTorch forward hooks. Not
# in pipekit's surface today; the pattern uses `contextvars` to keep
# hooks scoped (never global):

# %%
import contextvars
from contextlib import contextmanager


_active_spies: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "geotoolz_spy_stack", default=()
)


class Spy:
    def __init__(self) -> None:
        self._hooks: dict[type, list] = {}

    @classmethod
    @contextmanager
    def scoped(cls):
        spy = cls()
        token = _active_spies.set((*_active_spies.get(), spy))
        try:
            yield spy
        finally:
            _active_spies.reset(token)

    def on(self, op_type: type, fn) -> None:
        self._hooks.setdefault(op_type, []).append(fn)


def _fire(op, x_in, x_out):
    for spy in _active_spies.get():
        for fn in spy._hooks.get(type(op), ()):
            fn(op, x_in, x_out)


# Wrap call sites manually for the demo (in v0.2, Operator.__call__ does this).
class _TrackedAdd(Add):
    def __call__(self, x):
        out = super().__call__(x)
        _fire(self, x, out)
        return out


pipe = Sequential([_TrackedAdd(1), _TrackedAdd(10), _TrackedAdd(100)])

with Spy.scoped() as spy:
    spy.on(_TrackedAdd, lambda op, xi, xo: print(f"  {op}: {xi} → {xo}"))
    pipe(0)

print()
print("outside the scope — silent:")
pipe(0)


# %% [markdown]
# **Tradeoff.** Scoping is the safe default. A naive global registry —
# hooks fire across modules and tests — is a footgun. The `contextvars`
# implementation here costs an empty-tuple iteration when no scope is
# active.

# %% [markdown]
# ### `Diff` — compare against a reference (build-your-own)
#
# The pytest of operator pipelines. Drop inline to catch silent
# regressions during refactors. Not in pipekit's surface today; the
# pattern is short enough to keep inline where you need it:


# %%
class DiffError(Exception):
    pass


class Diff(Operator):
    """Compare input against a stored reference; raise on drift beyond atol."""

    forbid_in_yaml = True

    def __init__(self, *, reference, atol: float = 1e-6) -> None:
        self.reference = np.asarray(reference)
        self.atol = atol

    def _apply(self, x):
        arr = np.asarray(x)
        diff = float(np.max(np.abs(arr - self.reference)))
        if diff > self.atol:
            raise DiffError(f"max abs diff {diff:.4g} exceeds atol={self.atol}")
        return x

    def get_config(self) -> dict:
        return {"atol": self.atol, "ref_shape": self.reference.shape}


# Bless once when you trust the output:
golden = np.array([1, 11, 111])

# Then catches drift on subsequent runs:
arr_in = np.array([0, 10, 110])
pipe = Sequential(
    [
        Lambda(lambda a: a + 1, name="add1"),
        Diff(reference=golden, atol=1e-6),
    ]
)
print("matches golden:", pipe(arr_in))

try:
    pipe(np.array([0, 10, 111]))  # off by one in last element
except DiffError as e:
    print("caught regression:", e)

# %% [markdown]
# **Tradeoff.** Reference files drift over time as legitimate
# improvements ship. Pair with version pinning (`Diff(reference=v3_ref)`)
# per pipeline version — don't overwrite a blessed reference in place.

# %% [markdown]
# ## 2. Control flow
#
# Conditional execution, fallbacks, retries — same composition primitives,
# more interesting graphs.
#

# %% [markdown]
# ### Branch — binary predicate (shipped)
#
# Apply `if_true` when `predicate(x)` is truthy, else `if_false`. Default
# `if_false=Identity()`.

# %%
from geotoolz import Branch, Identity


guarded = Branch(
    predicate=lambda x: x > 0,
    if_true=Add(100),
    if_false=Identity(),  # default — pass-through for non-positive
)
print("guarded(5) =", guarded(5))
print("guarded(-5)=", guarded(-5))

# %% [markdown]
# ### Switch — multi-way dispatch (shipped)
#
# `key(x)` selects from `cases`; falls through to `default` (default
# `Identity()`) on misses.

# %%
from geotoolz import Switch


router = Switch(
    key=lambda x: "even" if x % 2 == 0 else "odd",
    cases={
        "even": Add(1),
        "odd": Sequential([Add(100), Add(100)]),
    },
)
print("router(4) =", router(4))
print("router(3) =", router(3))


# %% [markdown]
# ### `Try` / `Fallback` — exception cascade (shipped)
#
# Robust to upstream flakiness — try primary, fall back to a fallback op
# if it raises a listed exception type. Uncaught exception types
# propagate; there is no implicit `Exception` catch-all.

# %%
from pipekit import Try


class Boom(Operator):
    def _apply(self, x):
        raise ConnectionError(f"network unreachable on input {x}")


robust = Try(primary=Boom(), fallback=Add(0), on=(ConnectionError,))
print("robust(7) =", robust(7))


# %% [markdown]
# **Tradeoff.** Silent fallback can hide deteriorating production
# conditions. Pair with metrics — log every fallback as a counter, alert
# when the rate exceeds a threshold.

# %% [markdown]
# ### `Coalesce` — first acceptable result wins (shipped)
#
# "S2 first, fall back to Landsat if S2 is too cloudy, then MODIS."
# Each source runs on the same input; the first whose output passes
# `is_ok` is returned. Distinct from `Try`, which cascades on an
# exception rather than a quality predicate.

# %%
from pipekit import Coalesce


def is_positive(out):
    return out > 0


coalesced = Coalesce(
    sources=[
        Add(-100),  # produces negative — rejected
        Add(-10),  # still negative — rejected
        Add(50),  # positive — wins
    ],
    is_ok=is_positive,
)
print("coalesced(5) =", coalesced(5))


# %% [markdown]
# ### `Retry` — exponential backoff (shipped)
#
# Wrap an op with retry + exponential backoff. Attempt `i` waits
# `base_delay * (2 ** (i - 1))` before retrying; the last attempt does
# not sleep before re-raising. Useful for `ModelOp` calls to remote
# endpoints or flaky S3 reads.

# %%
from pipekit import Retry


_attempts = {"count": 0}


class FlakeyOp(Operator):
    def _apply(self, x):
        _attempts["count"] += 1
        if _attempts["count"] < 3:
            raise TimeoutError(f"transient failure (attempt {_attempts['count']})")
        return x + 1000


robust_call = Retry(op=FlakeyOp(), attempts=5, base_delay=0.0, on=(TimeoutError,))
print("retrying...")
out = robust_call(7)
print(f"succeeded after {_attempts['count']} attempts, out = {out}")

# %% [markdown]
# **Tradeoff.** Retries hide latency from upstream callers — a tile-server
# request that nominally takes 200ms can spike to 7s after two retries.
# For latency-sensitive paths, prefer `Try` with a fast fallback.

# %% [markdown]
# ## 3. Composition
#
# Building blocks for graphs that aren't pure linear chains.
#

# %% [markdown]
# ### Fanout — one input, many outputs (shipped)
#
# Sugar over a single-input `Graph`. Computes each branch on the same
# input, returns a dict keyed by branch name.

# %%
from geotoolz import Fanout


products = Fanout(
    {
        "doubled": Lambda(lambda x: x * 2, name="double"),
        "squared": Lambda(lambda x: x * x, name="square"),
        "labelled": _NamedStep("classify", factor=10),
    }
)
print(products(7))


# %% [markdown]
# ### `ApplyToBands` — split-apply-combine over an axis (build-your-own)
#
# Apply the inner op independently to each slice along an axis,
# recombine. Not in pipekit today (no shipped opinion on axis
# convention); minimal recipe:


# %%
class ApplyToBands(Operator):
    """Apply ``inner`` to each slice along ``axis``, stack results."""

    def __init__(self, inner: Operator, *, axis: int = 0) -> None:
        self.inner = inner
        self.axis = axis

    def _apply(self, arr):
        arr = np.asarray(arr)
        slices = [
            self.inner(np.take(arr, i, axis=self.axis))
            for i in range(arr.shape[self.axis])
        ]
        return np.stack(slices, axis=self.axis)

    def get_config(self) -> dict:
        return {
            "inner": type(self.inner).__name__,
            "inner_config": self.inner.get_config(),
            "axis": self.axis,
        }


# stand-in: scale each "band" by a factor of 10
op = ApplyToBands(Lambda(lambda x: x * 10, name="scale"), axis=0)
arr = np.arange(12).reshape(3, 4)
print("input  :")
print(arr)
print("per-band scaled:")
print(op(arr))

# %% [markdown]
# **Tradeoff.** Naive Python-level loop — fine for ~10 bands, painful for
# hyperspectral (~200). For hyperspectral, the inner op should ideally
# vectorise across the band axis directly; `ApplyToBands` is the fallback
# when it can't.

# %% [markdown]
# ### `Cache` / `Memoize` — hash input + config (shipped)
#
# Saves hours during iterative analysis where you keep re-running the
# same expensive prefix. The cache key is `(input_hash, op.config_hash)`,
# so any change to the wrapped op's `get_config()` invalidates entries.

# %%
import hashlib
import json

from pipekit import Cache


cached = Cache(Sequential([Add(1), Add(10)]))
for _ in range(5):
    cached(0)
print(f"5 calls → {cached.hits} hits, {cached.misses} miss")
print("result :", cached(0))

# %% [markdown]
# **Tradeoff.** In-memory cache is fast but unbounded — wrap with an LRU
# for long-running services. A disk backend is restart-friendly but you
# need to garbage-collect old entries. **Cache must opt-in** (`Cache(...)`
# wraps explicitly), never silent — if `Cache` ever became a default,
# debugging "why does my pipeline produce stale output?" would be a
# nightmare.

# %% [markdown]
# ## 4. Stateful / ML
#
# Operators that hold state across calls or interact with the broader
# runtime.
#

# %% [markdown]
# ### `Mode` — scoped train / eval switching (build-your-own)
#
# The lesson from PyTorch is that *implicit, instance-level, sticky*
# `train()` / `eval()` is a footgun. `geotoolz` will take the **scoped,
# explicit** path in v0.2 — mode is a context manager on a `Sequential`,
# not a sticky attribute on the graph. Pattern:

# %%
_current_mode: contextvars.ContextVar = contextvars.ContextVar(
    "geotoolz_pipeline_mode", default=None
)


@contextmanager
def pipeline_mode(name: str):
    """Enter a named mode for the duration of the with block."""
    if name not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval', got {name!r}")
    token = _current_mode.set(name)
    try:
        yield name
    finally:
        _current_mode.reset(token)


class ModeGated(Operator):
    """Wraps an op that only fires under a specific mode."""

    forbid_in_yaml = True

    def __init__(self, inner: Operator, *, mode: str) -> None:
        self.inner, self.required_mode = inner, mode

    def _apply(self, x):
        if _current_mode.get() == self.required_mode:
            return self.inner(x)
        return x

    def get_config(self) -> dict:
        return {"mode": self.required_mode, "inner": type(self.inner).__name__}


# Augmentation only fires in train mode; deterministic preprocess always runs.
pipe = Sequential(
    [
        Add(1),  # preprocess (always)
        ModeGated(Add(1000), mode="train"),  # augmentation (train only)
    ]
)

with pipeline_mode("train"):
    print("train mode:", pipe(0))

with pipeline_mode("eval"):
    print("eval mode :", pipe(0))

# Outside any scope: mode is None, so ModeGated is silent
print("no mode   :", pipe(0))


# %% [markdown]
# **Tradeoff.** Slightly more verbose than `pipe.train()` /
# `pipe.eval()` — the verbosity is the feature. Every train-mode invocation
# is grep-able in source. Code that forgets to enter the scope fails
# loudly (or silently does the deterministic path) instead of producing
# augmented "validation" tensors.

# %% [markdown]
# ### `Provenance` / `Watermark` — stamp lineage into outputs (build-your-own)
#
# Stamp pipeline metadata into the output so consumers years later can
# trace it back to the exact recipe. Minimal version:


# %%
class Provenance(Operator):
    """Wraps a pipeline; attaches a metadata dict on its way out.

    Real pipelines write this to the output COG's tags via georeader.
    """

    forbid_in_yaml = True

    def __init__(self, inner: Operator) -> None:
        self.inner = inner

    def _apply(self, x):
        out = self.inner(x)
        cfg = self.inner.get_config()
        cfg_canon = json.dumps(cfg, sort_keys=True)
        provenance = {
            "pipeline_hash": hashlib.sha256(cfg_canon.encode()).hexdigest()[:16],
            "operators": _op_names(self.inner),
            "geotoolz_version": "0.0.1",
        }
        # In real code: out.metadata["provenance"] = provenance
        return {"data": out, "_provenance": provenance}

    def get_config(self) -> dict:
        return {"inner": type(self.inner).__name__}


def _op_names(op: Operator) -> list:
    if isinstance(op, Sequential):
        return [type(o).__name__ for o in op.operators]
    return [type(op).__name__]


tracked = Provenance(Sequential([Add(1), Add(10)]))
result = tracked(0)
print("data        :", result["data"])
print("provenance  :")
for k, v in result["_provenance"].items():
    print(f"  {k}: {v}")


# %% [markdown]
# **Tradeoff.** Provenance metadata grows with every wrap — a
# deeply-nested `Graph` produces a large lineage record. Cap at a
# reasonable depth, or store the full graph hash plus a short
# operator-name summary rather than every config. Don't let provenance
# bloat overshadow the pixel data.

# %% [markdown]
# ## 5. Validation / QC (the assertion family)
#
# The other identity-with-side-effect family — pass-through ops that
# **check invariants** rather than observe state.
#

# %% [markdown]
# ### `AssertShape` / `AssertDType` / `AssertHasAttribute` (shipped)
#
# Pass-through operators that raise `QCError` on contract violation. The
# shipped family covers shape, dtype, and attribute presence/value; for
# *domain-specific* checks (value range, CRS, cloud fraction) write a
# small subclass that follows the same pattern.

# %%
from pipekit import AssertDType, AssertHasAttribute, AssertShape, QCError


class FakeCarrier:
    """Minimal stand-in with the attributes the Assert* family inspects."""

    def __init__(self, shape, dtype, crs="EPSG:4326"):
        self.shape = shape
        self.dtype = dtype
        self.crs = crs


# Three contracts in one Sequential — input must be (4, H, W) float32 with a CRS.
contract = (
    AssertShape((4, None, None)) | AssertDType("float32") | AssertHasAttribute("crs")
)
print("good carrier passes through:", contract(FakeCarrier((4, 256, 256), "float32")))

print("\nbad shape raises:")
try:
    contract(FakeCarrier((3, 256, 256), "float32"))
except QCError as e:
    print(" ", e)


# Domain-specific range check, written in the same pass-through shape:
class AssertValueRange(Operator):
    """Pass-through; raise QCError on values outside [min_val, max_val]."""

    def __init__(self, *, min_val: float, max_val: float) -> None:
        self.min, self.max = min_val, max_val

    def _apply(self, x):
        arr = np.asarray(x)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if lo < self.min or hi > self.max:
            raise QCError(
                f"range [{lo:.3g}, {hi:.3g}] outside [{self.min}, {self.max}]"
            )
        return x

    def get_config(self) -> dict:
        return {"min_val": self.min, "max_val": self.max}


print("\nrange check on raw values:")
range_check = AssertValueRange(min_val=0, max_val=100)
try:
    range_check(np.array([5, 25, 150]))
except QCError as e:
    print(" caught:", e)


# %% [markdown]
# **The bigger pattern.** Every QC check is a research-time "I bet this
# won't break" turned into a permanent runtime guard. CI gets a
# pipeline-correctness suite for free, and pipelines self-document their
# preconditions.

# %% [markdown]
# ### `Quarantine` — non-raising QC (shipped)
#
# If a `check(x)` predicate returns False, route the bad input to a
# `sentinel`, optionally call an `on_quarantine` hook, and keep the
# batch alive. The "log_and_continue" of QC. Distinct from the
# `Assert*` family above, which *raises*.

# %%
from pipekit import Quarantine


log: list = []

pipe = Sequential(
    [
        Quarantine(
            check=lambda arr: float(np.nanmax(arr)) <= 100,
            sentinel=np.array([-1]),  # marker for "quarantined"
            on_quarantine=lambda arr: log.append(
                f"quarantined max={float(np.nanmax(arr))}"
            ),
        ),
    ]
)

print("good:", pipe(np.array([5, 25])))
print("bad: ", pipe(np.array([5, 999])))
print("log :", log)

# %% [markdown]
# **Tradeoff.** Quarantine is a *hedge* — it trades immediate failure for
# delayed debugging. Don't quarantine errors that indicate genuine bugs
# (wrong CRS, wrong band order); those should `raise` via the `Assert*`
# family. Quarantine is for *expected edge cases* in the data (corrupt
# scenes, partial downloads, sensor glitches).

# %% [markdown]
# ## 6. Small but load-bearing building blocks
#
# Boring on their own, indispensable in combination.
#

# %% [markdown]
# ### Identity / Const / Lambda / Sink (all shipped)
#
# Recap: `Identity` for explicit no-op, `Const` for fixed-value, `Lambda`
# for inline callables, `Sink` for composable terminal writes.

# %%
from geotoolz import Const, Sink


# Branch with explicit Identity in if_false — self-documenting no-op
opt_clean = Branch(
    predicate=lambda x: x > 0,
    if_true=Sequential([Add(100), Add(-50)]),
    if_false=Identity(),
)
print("opt_clean(10) =", opt_clean(10))
print("opt_clean(-5) =", opt_clean(-5))


# Sink: write side effect + pass through
written: list = []
checkpoint_pipe = Sequential(
    [
        Add(1),
        Sink(written.append, name="checkpoint_after_first"),
        Add(10),
    ]
)
print("checkpoint_pipe(0) =", checkpoint_pipe(0))
print("checkpoint stored  :", written)


# Const: fixed value, useful for synthetic sources / golden tests
golden_input = Const(np.arange(5))
synth_test = Sequential(
    [
        golden_input,  # ignores actual input
        Lambda(lambda a: a.sum(), name="sum"),
    ]
)
print("synth test sum:", synth_test("real_input_ignored"))


# %% [markdown]
# ### `Subsample` — stride decimation (build-your-own)
#
# Random or stride-based pixel subsampling for fast visualisation off a
# full-resolution carrier. Two shapes — `Tap`-style (sample-and-call)
# and standalone (return decimated GeoTensor):


# %%
class Subsample(Operator):
    """Stride-decimate the last two axes. Standalone transform."""

    def __init__(self, *, stride: int = 10) -> None:
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.stride = stride

    def _apply(self, arr):
        arr = np.asarray(arr)
        return arr[..., :: self.stride, :: self.stride]

    def get_config(self) -> dict:
        return {"stride": self.stride}


big = np.arange(100).reshape(10, 10)
small = Subsample(stride=3)(big)
print("input shape   :", big.shape)
print("decimated shape:", small.shape)
print(small)

# %% [markdown]
# **Tradeoff.** Random subsampling biases summary statistics for
# non-uniform fields (e.g., concentrated NDVI hotspots in mostly-bare
# scenes). For fair visualisation, use stride; for fair statistics, use
# weighted random sampling.

# %% [markdown]
# ## 7. Two design rules
#
# Two small disciplines keep the surface tractable as the trick library
# grows.
#
# ### Honest naming
#
# Don't disguise an assertion as a transform. Don't disguise a side
# effect as a computation. Users should be able to scan a `Sequential`
# and immediately see which steps **mutate**, which **observe**, which
# **guard**, and which **control flow**.
#
# ```python
# Sequential([
#     _NamedStep("MaskClouds"),                          # transform
#     AssertValueRange(min_val=0, max_val=10_000,        # guard
#                      on_fail="warn"),
#     Tap(log_stats),                                    # observe
#     Branch(predicate=..., if_true=..., if_false=Identity()),  # control flow
#     _NamedStep("NDVI"),                                # transform
# ])
# ```
#
# Suggested naming:
#
# | Family | Prefix |
# |---|---|
# | Assertions | `qc.Assert*` |
# | Observers | `core.Tap`, `core.Snapshot`, `core.Histogram`, `core.Profile`, `core.ShapeTrace` |
# | Control flow | `core.Branch`, `core.Switch`, `core.Try`, `core.Coalesce`, `core.Retry` |
# | Transforms | everything else |
#
# ### Round-trip discipline
#
# Operators that hold closures (`Tap`, `Lambda`, `Branch`, `Switch`,
# `Sink`, `ModelOp`, the build-your-own ones in this notebook with
# `forbid_in_yaml = True`) **cannot** round-trip to YAML faithfully. Their
# `get_config()` is a debug repr.
#
# Future YAML loaders should:
#
# - Refuse to instantiate any operator with `forbid_in_yaml = True`
# - Refuse to dump a graph containing one
#
# Production pipelines never contain closures. This keeps "operator graph
# as audit artifact" honest — every operator in a regulatory artifact has
# a stable config, every config round-trips, every artifact reruns to the
# same answer.
#

# %% [markdown]
# ## 8. The shape
#
# Once these primitives exist, **most user "Operator subclass" needs go
# away.**
#
# | Instead of writing… | Compose… |
# |---|---|
# | `LoggedNDVI` | `Tap(log) \| NDVI(...)` |
# | `OptionalCorrection` | `Branch(predicate, if_true=Correction(), if_false=Identity())` |
# | `RobustModelOp` | `Retry(ModelOp(...), attempts=3) \| Try(..., fallback=...)` |
# | `BandwiseSpeckle` | `ApplyToBands(LeeSpeckle(...), axis=0)` |
# | `CachedPipeline` | `Cache(my_pipeline)` |
# | `TimedPipeline` | `Profile().wrap(my_pipeline)` |
# | `MultiOutputModel` | `Fanout({"a": op_a, "b": op_b})` |
# | `WithLogging` | `Tap(log_stats)` |
# | `WithProvenance` | `Provenance(my_pipeline)` |
# | `S2OrLandsatNDVI` | `Switch(key=lambda gt: gt.metadata["sensor"], cases={...})` |
#
# The library's primitive set does the work; users compose. Adding a new
# trick adds one Operator, not a family of variants.
#
# ## Where next
#
# - [Concepts](https://github.com/jejjohnson/geotoolz/blob/main/docs/concepts.md) — the model behind the primitives.
# - [Composition core walkthrough](01_composition_core.ipynb) — primitives demonstrated against scalars.
# - [Deployment shapes](07_deployment_shapes.ipynb) — same primitives in 13 deployment patterns (notebook, ETL, FastAPI, tile server, orchestrator, regulatory artifact, ...).
# - [Core API reference](https://github.com/jejjohnson/geotoolz/blob/main/docs/api/core.md) — operator-by-operator docs.
#

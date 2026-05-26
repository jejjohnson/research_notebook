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
# title: "01 — Composition core walkthrough"
# ---
#
# # Composition core — walkthrough
#
# This notebook exercises every primitive in `geotoolz.core` end-to-end against
# plain Python integers. **No `GeoTensor` setup required** — the composition
# algebra is carrier-agnostic, so the same code that runs here works against
# ndarrays, scalars, or `GeoTensor`s once domain operators land.
#
# Skim it top-to-bottom for a tour, or jump to a section using the table of
# contents in your viewer. Every cell is executable.
#
# For the model behind the primitives, see the [Concepts](https://github.com/jejjohnson/geotoolz/blob/main/docs/concepts.md) page.
#

# %% [markdown]
# ## 1. Defining an Operator
#
# An `Operator` is a class with two methods: `_apply` (the work) and
# `get_config` (a JSON-serialisable dict of the constructor args, used for
# `__repr__` and round-trip).

# %%
from geotoolz import Operator


class Add(Operator):
    """Add a constant."""

    def __init__(self, n: int) -> None:
        self.n = n

    def _apply(self, x: int) -> int:
        return x + self.n

    def get_config(self) -> dict:
        return {"n": self.n}


op = Add(5)
print(repr(op))
print("eager call:", op(10))

# %% [markdown]
# ## 2. `Sequential` — linear composition
#
# A `Sequential` threads the output of each operator into the next. The
# `|` operator (inherited from `Operator`) builds one too, and flattens
# nested `Sequential`s automatically.

# %%
from geotoolz import Sequential


pipe = Sequential([Add(1), Add(10), Add(100)])
print("Sequential:", pipe(0))

# Same thing via the pipe operator
piped = Add(1) | Add(10) | Add(100)
print("piped:    ", piped(0))
print("flattened len:", len(piped))  # Sequential[3], not nested

# %% [markdown]
# ### `get_config()` recurses through a `Sequential`
#
# Every operator's config is serialised. Lists of ops become lists of
# `{"class": ..., "config": ...}` dicts — friendly to hydra-zen, picklable,
# human-readable in tracebacks.

# %%
import json


cfg = Sequential([Add(1), Add(2), Add(3)]).get_config()
print(json.dumps(cfg, indent=2))

# %% [markdown]
# ## 3. Dual-mode `__call__` — eager vs graph
#
# The same operator works two ways:
#
# - `op(value)` → runs `_apply` (eager)
# - `op(input_node)` → returns a `Node` (graph construction)
#
# The dispatch is automatic — `__call__` checks the argument type.

# %%
from geotoolz import Input


# Eager: pass a value
print("eager:", Add(5)(10))

# Graph mode: pass an Input
x = Input("x")
node = Add(5)(x)
print("graph node:", type(node).__name__)
print("node operator:", node.operator)
print("node parents:", node.parents)

# %% [markdown]
# ## 4. `Graph` — symbolic multi-input / multi-output composition
#
# When you need branching outputs or multi-input fusion, `Graph` is the
# shape. Build it by calling operators on `Input` placeholders, then wrap
# the result with `Graph(inputs=..., outputs=...)`.

# %%
from geotoolz import Graph


class Sum2(Operator):
    """Sum two inputs."""

    def _apply(self, a: int, b: int) -> int:
        return a + b

    def get_config(self) -> dict:
        return {}


# Build a small graph:
#   x → a → c
#   x → b → c (c = a + b)
x = Input("x")
a = Add(1)(x)
b = Add(2)(x)
c = Sum2()(a, b)

g = Graph(inputs={"x": x}, outputs={"a": a, "b": b, "c": c})
print(g)
print(g(x=10))  # {"a": 11, "b": 12, "c": 23}

# %% [markdown]
# `Graph` topologically sorts the nodes, evaluates each exactly once, and
# returns a dict keyed by output name. Cycles and unreachable inputs are
# caught at construction time.

# %% [markdown]
# ## 5. `Fanout` — one input → many outputs (sugar over `Graph`)
#
# For the common single-input / multi-output case, `Fanout` is more concise
# than a hand-built `Graph`.

# %%
from geotoolz import Fanout, Lambda


products = Fanout(
    {
        "doubled": Lambda(lambda x: x * 2, name="double"),
        "squared": Lambda(lambda x: x * x, name="square"),
        "negated": Lambda(lambda x: -x, name="negate"),
    }
)
print(products(7))

# %% [markdown]
# ## 6. Observers — `Tap`, `Snapshot`, `ShapeTrace`
#
# Identity operators with side effects. The value flows through unchanged
# while something useful happens on the side.

# %% [markdown]
# ### `Tap` — fire a callback, pass through
#
# The seed pattern. The callback's return value is ignored — `Tap` is for
# side effects, not transforms.

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

result = pipe(0)
print("result:", result)
for label, val in seen:
    print(f"  {label}: {val}")

# %% [markdown]
# ### `Snapshot` — capture intermediates by name
#
# A controller (not an `Operator` itself) that produces snapshot-taking
# operators via `snap.at(key)`. After the pipeline runs, intermediates are
# keyed by name in `snap[key]`.

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
        snap.at("final"),
    ]
)

pipe(0)
print("captured keys:", list(snap.keys()))
print("after_first :", snap["after_first"])
print("after_second:", snap["after_second"])
print("final       :", snap["final"])

# %% [markdown]
# ### `ShapeTrace` — log carrier metadata at each step
#
# Useful for debugging "what happened to my GeoTensor between steps?". Falls
# back to `getattr(..., None)` for objects that don't have `shape` / `dtype`
# / `crs` (like our integers below), so the same op works on any carrier.

# %%
import numpy as np
from geotoolz import ShapeTrace


trace = ShapeTrace()
pipe = Sequential(
    [
        trace,
        Lambda(
            lambda x: np.asarray([x, x * 2, x * 3], dtype=np.int16), name="to_array"
        ),
        trace,
    ]
)
pipe(7)

# %% [markdown]
# ## 7. Control flow — `Branch`, `Switch`
#
# The `Operator` interface is general enough to express conditionals.
# `Branch` is the binary case; `Switch` is multi-way.

# %%
from geotoolz import Branch, Identity


guarded = Branch(
    predicate=lambda x: x > 0,
    if_true=Add(100),
    if_false=Identity(),  # default; pass-through for non-positive
)
print("guarded(5) :", guarded(5))  # positive → +100
print("guarded(-5):", guarded(-5))  # non-positive → unchanged

# %%
from geotoolz import Switch


dispatcher = Switch(
    key=lambda x: "even" if x % 2 == 0 else "odd",
    cases={
        "even": Add(1),
        "odd": Sequential([Add(100), Add(100)]),
    },
)
print("dispatcher(4):", dispatcher(4))
print("dispatcher(3):", dispatcher(3))

# %% [markdown]
# ## 8. Small but load-bearing building blocks
#
# `Identity`, `Const`, `Lambda`, `Sink` are tiny on their own. In combination
# they replace most one-off `Operator` subclasses.

# %%
from geotoolz import Const, Identity, Lambda, Sink


# Identity: explicit no-op
print("Identity:", Identity()(42))

# Const: ignore input, return a fixed value
print("Const   :", Const("HELLO")(123))

# Lambda: inline-callable; use Operator subclass for anything reusable
print("Lambda  :", Lambda(lambda x: x.upper(), name="upper")("hello"))

# Sink: side-effect terminal that returns input — composable
written = []
sink_pipe = Sequential(
    [
        Add(1),
        Sink(written.append, name="checkpoint"),
        Add(10),
    ]
)
print("Sink    :", sink_pipe(0))
print("written :", written)

# %% [markdown]
# ## 9. `ModelOp` — framework-agnostic inference
#
# `ModelOp` wraps any callable. Use it for sklearn (`method="predict"`), torch
# (plain call), JAX (plain call), or any user-supplied function. With
# `batch_size=N`, it chunks the input along axis 0 and concatenates the
# results — handy when the whole input doesn't fit in GPU memory.

# %%
from geotoolz import ModelOp


class FakeSklearn:
    """sklearn-style — only ``predict`` works, not ``__call__``."""

    def predict(self, arr):
        return arr * 10


arr = np.arange(6).reshape(6, 1)
op = ModelOp(FakeSklearn(), method="predict", batch_size=2)
print("input:")
print(arr.ravel())
print("output:")
print(op(arr).ravel())

# %% [markdown]
# ## 10. Pickling — operator graphs as artifacts
#
# The "operator graph as audit artifact" pattern depends on pickling working.
# Every YAML-safe operator in `geotoolz.core` round-trips through `pickle`
# cleanly. Operators flagged `forbid_in_yaml = True` (`Tap`, `Lambda`,
# `Branch`, `Switch`, `Sink`, `ModelOp`) hold closures and cannot reproducibly
# serialise — use `Operator` subclasses with named `get_config()` for those
# paths instead.

# %%
import pickle


pipe = Sequential([Add(1), Add(2), Add(3)])
restored = pickle.loads(pickle.dumps(pipe))
print("pickled config matches:", restored.get_config() == pipe.get_config())
print("restored call:", restored(0))

# %% [markdown]
# ## 11. Putting it together
#
# A non-trivial pipeline using several primitives at once:

# %%
snap = Snapshot()
log = []

pipeline = Sequential(
    [
        Add(1),
        Tap(lambda x: log.append(("entered branch", x))),
        Branch(
            predicate=lambda x: x % 2 == 0,
            if_true=Sequential([Add(100), snap.at("even_path")]),
            if_false=Sequential([Add(1000), snap.at("odd_path")]),
        ),
        snap.at("after_branch"),
    ]
)

# Even path
print("pipeline(1) =", pipeline(1))  # 1 + 1 = 2 → +100 → 102

# Odd path
print("pipeline(0) =", pipeline(0))  # 0 + 1 = 1 → +1000 → 1001

print()
print("snap keys:    ", list(snap.keys()))
print("snap[final]:  ", snap["after_branch"])
print("log:          ", log)

# %% [markdown]
# ## Where next
#
# - The [Core API reference](https://github.com/jejjohnson/geotoolz/blob/main/docs/api/core.md) documents each operator with
#   its constructor signature and config keys.
# - The [Concepts](https://github.com/jejjohnson/geotoolz/blob/main/docs/concepts.md) page explains the model behind these
#   primitives.
# - Domain operators (radiometry, indices, cloud masking, sampling,
#   inference) land in v0.2+.
#

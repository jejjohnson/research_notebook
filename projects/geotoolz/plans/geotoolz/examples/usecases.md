---
title: geotoolz use cases
subject: geotoolz examples
subtitle: A gallery of deployment patterns for the Operator algebra
short_title: Use cases
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - UNEP
      - IMEO
      - MARS
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: design, geotoolz, operators, use-cases, remote-sensing
---

> **Status:** companion to [`geotoolz.md`](../geotoolz.md) — the missing "where does this thing live in a deployment?" gallery, covering notebook, ETL, ML, inference API, tile server, QC, reproducibility, and orchestrator patterns.
> **Audience:** anyone deciding whether the operator-graph abstraction fits a given workflow; reviewers checking the v0.1 surface against real deployment shapes before committing.

A gallery of patterns showing where the `geotoolz` operator algebra fits — across notebook exploration, ETL, ML training and inference, web APIs, and operational tooling.
Each case includes context (who’s driving, what they’re trying to do), a **demo API** (calling code), an **example implementation sketch** where useful, and a short **tradeoffs** note covering when *not* to use the pattern.

The unifying claim of the library: the *same* operator graph runs across all of these.
What changes between cases is **who drives** — a notebook user, an HTTP request, a scheduler, a sampler, or another graph.

> Code blocks below are pseudocode — illustrative, not executable as-is.
> They assume the API surface from [`plans/geotoolz/`](https://jejjohnson.github.io/research_notebook/geotoolz) and the catalog/sampler shapes from [`plans/geodatabase/`](https://jejjohnson.github.io/research_notebook/geocatalog) and [`plans/types/geoslice.md`](https://jejjohnson.github.io/research_notebook/geoslice).

## Contents

1. [Notebook exploration](#1-notebook-exploration)
2. [ETL pipelines](#2-etl-pipelines)
3. [ML training and inference](#3-ml-training-and-inference)
4. [Lightning / LitServe inference API](#4-lightning--litserve-inference-api)
5. [Backend API (FastAPI, multi-pipeline)](#5-backend-api-fastapi-multi-pipeline)
6. [User-uploaded pipelines for viz](#6-user-uploaded-pipelines-for-viz)
7. [Tile server (z/x/y → PNG)](#7-tile-server-zxy--png)
8. [Data validation / QC as operators](#8-data-validation--qc-as-operators)
9. [Pinned / hashed regulatory artifact](#9-pinned--hashed-regulatory-artifact)
10. [Active learning / uncertainty-driven sampling](#10-active-learning--uncertainty-driven-sampling)
11. [Cross-sensor fusion (the `Graph` case)](#11-cross-sensor-fusion-the-graph-case)
12. [Workflow orchestrator task units](#12-workflow-orchestrator-task-units)
13. [Pipeline diffing / A-B regression](#13-pipeline-diffing--a-b-regression)
14. [Cases considered but not expanded](#cases-considered-but-not-expanded)

-----

## 1. Notebook exploration

**The setting.** The first place a new operator earns its keep.
A researcher trying out a spectral index variant, a postdoc tuning a cloud-mask threshold on a stubborn scene, or anyone debugging *why does my output look like that* — the common need is **seeing intermediate state** without exploding the pipeline into ten named variables.
The pitch of the library at this stage isn’t “production ready”; it’s “the same shape you’ll deploy later.”

**Demo API.**

```python
import geotoolz as gz
import numpy as np
import matplotlib.pyplot as plt

viz = gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    gz.core.Tap(lambda gt: print(f"after mask: {np.isnan(gt).mean():.1%} NaN")),
    gz.radiometry.PercentileClip(p_min=2, p_max=98),
    gz.indices.NDVI(red_idx=2, nir_idx=3),
])

ndvi_gt = viz(scene_gt)
plt.imshow(ndvi_gt.values, cmap="RdYlGn")
```

`Tap` is the under-rated trick — an identity Operator with a side effect.
It lets users live inside the chain instead of breaking it apart for inspection.
Per-step stats, intermediate plots, lightweight asserts — all stay inline.

**Implementation sketch.**

```python
class Tap(Operator):
    """Identity operator with a side effect — does not modify the GeoTensor."""

    def __init__(self, fn):
        self.fn = fn

    def _apply(self, gt):
        self.fn(gt)
        return gt

    def get_config(self):
        return {"fn": repr(self.fn)}  # debug repr only — not round-trippable
```

**Tradeoffs.** Because `fn` is a closure, `Tap` is *not* config-round-trippable; its `get_config()` is a debug repr, not faithful YAML. That’s the correct choice for a debug primitive — production pipelines should not contain `Tap`.
A serialiser flag (`forbid_in_yaml=True`) on the class prevents accidental escape from the notebook.

-----

## 2. ETL pipelines

**The setting.** A researcher’s one-off “load → mask → correct → write COG” script becomes a scheduled job that has to run nightly across thousands of scenes.
The two failure modes the library is meant to remove: **rewriting the pipeline** to fit the orchestrator, and **divergence** between the research notebook version and the production version.
ETL here means: catalog drives the iteration; same operator graph applies to every row; results land in cloud storage.

**Demo API.** Loads the pipeline from a YAML config so the same artefact can be audited, version-pinned, and re-run.

```python
import geotoolz as gz
import georeader

catalog = georeader.catalog.open_catalog("s3://bucket/s2_eu.parquet")
etl = gz.serialization.load_yaml("pipelines/ndvi_etl.yaml")

gz.catalog_ops.CatalogPipeline(
    catalog,
    etl,
    n_workers=8,
    on_row_error="log_and_continue",
).run()
```


```yaml
# Sentinel-2 NDVI ETL pipeline.
# Cloud-mask, atmospheric-correct, compute NDVI, write COG.
_target_: geotoolz.core.Sequential
operators:

  # 1. Mask cloud + cirrus from the QA60 band.
  - _target_: geotoolz.cloud.MaskClouds
    qa_band: -1
    bits: [10, 11]

  # 2. Dark-object atmospheric correction.
  - _target_: geotoolz.correction.DarkObjectSubtraction
    percentile: 1

  # 3. TOA -> BOA reflectance using the per-pixel sun-zenith band.
  - _target_: geotoolz.correction.TOAToBOA
    sun_zenith_band: -2

  # 4. Compute NDVI and append it as a new band (preserves originals).
  - _target_: geotoolz.indices.AppendIndex
    name: ndvi
    index:
      _target_: geotoolz.indices.NDVI
      red_idx: 2
      nir_idx: 3

  # 5. Write to S3 with a templated filename.
  - _target_: geotoolz.catalog_ops.WriteCOG
    path_template: "s3://outputs/ndvi/{tile}_{date}.tif"
    overviews: [2, 4, 8, 16]
    compress: deflate
```

**Implementation sketch.** `CatalogPipeline` is itself an Operator — same `_apply` shape, just iterates rows.

```python
class CatalogPipeline(Operator):
    def __init__(self, catalog, op, *, n_workers=1, reader_cls=None,
                 on_row_error="raise"):
        self.catalog, self.op, self.n_workers = catalog, op, n_workers
        self.reader_cls = reader_cls or georeader.RasterioReader
        self.on_row_error = on_row_error  # "raise" | "log_and_continue"

    def run(self):
        if self.n_workers == 1:
            for row in self.catalog.iter_rows():
                self._process(row)
        else:
            with ProcessPoolExecutor(self.n_workers) as ex:
                list(ex.map(self._process, self.catalog.iter_rows()))

    def _process(self, row):
        try:
            with self.reader_cls.open(row.filepath) as reader:
                gt = reader.read_geoslice(row.to_geoslice())
            self.op(gt)  # terminal WriteCOG inside op handles output
        except Exception as e:
            if self.on_row_error == "raise":
                raise
            log.error("row %s failed: %s", row.id, e)
```

**Tradeoffs.** Process-pool parallelism crosses pickle boundaries — every Operator and every reader must be pickleable.
Closures, lambdas, and non-default-pickled cloud credentials break here, often silently.
The `on_row_error="log_and_continue"` default is a hedge against the “one bad scene crashes the 10k-scene job” anti-pattern; the cost is that quiet failures need monitoring (count of skipped rows per run, failure rate alerts).

-----

## 3. ML training and inference

**The setting.** Train/serve skew is a perennial source of silent ML bugs — training preprocessing diverges from inference preprocessing over time, the model performs well in evaluation and badly in production, and nobody notices until a stakeholder spots the regression months later.
The library’s claim here: *the same `preprocess` object* lives in both paths, by construction.
Augmentation differs (train-only), but the deterministic transforms are shared.

**Demo API.**

```python
import geotoolz as gz
import torch
import numpy as np

preprocess = gz.Sequential([
    gz.radiometry.ToFloat32(),
    gz.radiometry.PercentileClip(p_min=2, p_max=98, axis=(-2, -1)),
    gz.cloud.ApplyMask(gz.cloud.MaskFromQABits(qa_band=-1, bits=[10, 11])),
])

augment = gz.Sequential([
    gz.augment.RandomFlip(p=0.5),
    gz.augment.RandomRotate90(),
])

# === Training ===
class GeoDataset(torch.utils.data.IterableDataset):
    def __init__(self, catalog, preprocess, augment, reader_cls):
        self.cat, self.pre, self.aug = catalog, preprocess, augment
        self.reader_cls = reader_cls
        self.sampler = gz.sampling.RandomSampler(chip_size=(256, 256), length=10_000)

    def __iter__(self):
        for sl in self.sampler(self.cat):
            with self.reader_cls.open(sl.source_uri) as r:
                gt = r.read_geoslice(sl)
            x = self.aug(self.pre(gt))      # both train-time
            yield torch.from_numpy(np.asarray(x))

# === Inference (same preprocessor, no augment) ===
infer = gz.inference.ApplyToChips(
    sampler=gz.sampling.GridSampler(chip_size=(256, 256), stride=(224, 224)),
    chip_op=gz.Sequential([
        preprocess,                          # same object, identical config
        gz.inference.ModelOp(model, batch_size=8),
    ]),
    stitcher=gz.sampling.Stitch(method="average"),
)
pred_gt = infer(scene_gt)
```

**Tradeoffs.** Closures inside augment Operators (e.g. a captured RNG) break DataLoader workers — each worker gets the same seed and produces correlated batches.
The fix is per-Operator `seed=` arguments and a `worker_init_fn` that re-seeds.
Stitch overlap (`stride=224 < chip_size=256`) costs ~30% more reads for smoother boundaries; for fast inference set `stride == chip_size` and accept seam artefacts.

-----

## 4. Lightning / LitServe inference API

**The setting.** The model needs to ship as an HTTP service, with a long-lived process that holds the GPU and serves request/response over JSON. The shape is distinct from batch ETL because: requests arrive one-at-a-time (or in micro-batches), cold start matters, and the response payload usually isn’t a COG written to S3 but a small JSON encoding the result.
LitServe is well-suited because its decode/predict/encode split lines up exactly with read → operator-graph → serialise.

**Demo API.** Loads the methane attribution pipeline at startup; `predict` is a single Operator call.

```python
import litserve as ls
import geotoolz as gz
import georeader

class MethanePipelineAPI(ls.LitAPI):
    def setup(self, device):
        self.pipeline = gz.serialization.load_yaml("pipelines/methane_v3.yaml")

    def decode_request(self, req):
        return georeader.read_from_url(req["url"], bounds=tuple(req["bbox"]))

    def predict(self, gt):
        return self.pipeline(gt)  # GeoTensor in, GeoTensor out

    def encode_response(self, gt_out):
        return {
            "cog_b64":     gt_out.to_cog_bytes_b64(),
            "stats":       gt_out.summary_stats(),
            "config_hash": self.pipeline.config_hash(),
        }

ls.LitServer(MethanePipelineAPI(), accelerator="gpu").run(port=8000)
```


```yaml
# Methane attribution pipeline — v3
# Used in MARS operational reporting. Hyperspectral matched-filter retrieval
# of methane plumes from EMIT scenes.
_target_: geotoolz.core.Sequential
operators:

  # 1. Mask invalid pixels using QA60 cloud + cirrus bits.
  - _target_: geotoolz.cloud.ApplyMask
    mask_op:
      _target_: geotoolz.cloud.MaskFromQABits
      qa_band: -1
      bits: [10, 11]

  # 2. Dark-object atmospheric correction (TOA -> BOA approximation).
  - _target_: geotoolz.correction.DarkObjectSubtraction
    percentile: 1
    axis: [-2, -1]

  # 3. Hyperspectral matched filter against CH4 absorption signature.
  - _target_: geotoolz.hyperspectral.MatchedFilter
    target_path: /data/spectra/ch4_signature_v2.npy
    axis: 0

  # 4. Soft QC — warn but don't halt. Out-of-range scores indicate likely
  #    upstream calibration or masking issues.
  - _target_: geotoolz.qc.AssertValueRange
    min_val: -100.0
    max_val: 100.0
    on_fail: warn
```

**Tradeoffs.** Including `config_hash` in every response is cheap insurance — clients can detect “the model changed under me” without out-of-band coordination.
Cold start is dominated by `setup()`: keep heavy state (model weights, calibration tables, target spectra) in long-lived attributes, never construct them per `predict()`.
Batch dynamically (LitServe’s `max_batch_size`) only if the underlying model genuinely benefits — geospatial Operators often don’t, since the chip-size dimension already saturates the GPU.

-----

## 5. Backend API (FastAPI, multi-pipeline)

**The setting.** A platform team running many models (different sensors, different products, different pipeline versions) wants **one** service that hosts them all.
The pipeline registry pattern: load every pipeline at startup, route based on URL path.
Compared to LitServe (one model, one process), this trades GPU pinning for flexibility — an operations team can roll out a new pipeline by adding a YAML entry and restarting.

**Demo API.**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4
import geotoolz as gz
import georeader

app = FastAPI()

PIPELINE_REGISTRY = {
    "methane_v3": "pipelines/methane_v3.yaml",
    "methane_v4": "pipelines/methane_v4_candidate.yaml",
    "ndvi_etl":   "pipelines/ndvi_etl.yaml",
}
PIPELINES = {
    name: gz.serialization.load_yaml(path)
    for name, path in PIPELINE_REGISTRY.items()
}

class ProcessRequest(BaseModel):
    url:  str
    bbox: tuple[float, float, float, float]

@app.post("/process/{name}")
def process(name: str, req: ProcessRequest):
    pipe = PIPELINES[name]
    gt = georeader.read_from_url(req.url, bounds=req.bbox)
    out_gt = pipe(gt)
    cog_url = georeader.save_cog(out_gt, f"s3://outputs/{uuid4()}.tif")
    return {"cog_url": cog_url, "config_hash": pipe.config_hash()}

@app.get("/pipelines/{name}")
def describe(name: str):
    """Return the full pipeline config so the API documents itself."""
    return PIPELINES[name].get_config()

@app.get("/pipelines")
def list_pipelines():
    return {name: pipe.config_hash() for name, pipe in PIPELINES.items()}
```

The `/pipelines/{name}` endpoint matters more than people think — `get_config()` lets the API describe its own behaviour to clients without out-of-band docs.
The `/pipelines` listing with hashes lets monitoring detect drift between deployments.

**Tradeoffs.** Loading every pipeline at startup grows memory linearly in the number of registered pipelines — fine for a few dozen, painful at hundreds.
Above that scale, lazy-load on first request and cache.
Hot reload (swap a YAML without restart) sounds attractive but interacts badly with in-flight requests holding references to the old graph; safer to require a process restart and let the load balancer drain.

-----

## 6. User-uploaded pipelines for viz

**The setting.** The democratisation case.
A scientist or analyst — not necessarily a Python engineer — wants to compose a pipeline from a UI, see the result on a map, and share the recipe with collaborators.
Letting a user submit YAML *as a request* is a natural mapping, but it bites the moment you call `hydra.utils.instantiate()` on untrusted input: a malicious payload like `_target_: os.system, args: ["rm -rf /"]` is arbitrary code execution.

**The fix** is a sandboxed loader that walks the config tree and rejects any `_target_` not in a pre-vetted public registry of Operators.

```python
# In geotoolz.serialization
ALLOWED = gz.registry.public_operators()  # set of fully-qualified names

class UntrustedTargetError(Exception):
    def __init__(self, target):
        self.target = target
        super().__init__(f"Operator '{target}' is not in the public registry")

def load_yaml_sandboxed(text: str) -> Operator:
    cfg = yaml.safe_load(text)
    return _build(cfg, allowed=ALLOWED)
```

**Demo API** (Streamlit):

```python
import streamlit as st
import geotoolz as gz

yaml_text = st.text_area("Pipeline YAML", height=300)

if st.button("Run"):
    try:
        pipe = gz.serialization.load_yaml_sandboxed(yaml_text)
    except gz.serialization.UntrustedTargetError as e:
        st.error(f"Operator not allowed: `{e.target}`. Only registered operators may be used.")
    except Exception as e:
        st.error(f"YAML error: {e}")
    else:
        bounds = st.text_input("Bounds (xmin,ymin,xmax,ymax)")
        scene = read_demo_scene(bounds)
        result = pipe(scene)
        st.image(result.to_rgb_array())
        st.json(pipe.get_config())  # show what the user actually ran
```

**Tradeoffs.** The sandbox doesn’t solve resource exhaustion — a malicious user can still submit a YAML that allocates huge intermediates or schedules a million `Sequential` steps.
Time and memory limits per request are mandatory.
The other side of the same coin: this pattern also gates *internal* contributors — a colleague’s experimental Operator can’t run in the demo until it lands in the registry, which is a feature for governance and a friction point for researchers.
The right balance is a fast registry-promotion process, not a permissive sandbox.

-----

## 7. Tile server (z/x/y → PNG)

**The setting.** Slippy-map UX — a user pans a Leaflet/Mapbox view of their data, the browser issues XYZ tile requests, the server renders each tile on demand.
This is *distinct* from a generic backend API because the request shape and latency budget are different: every request is `(z, x, y, layer)`, the response is a 256×256 PNG, p99 must be sub-second, and a single map view fans out to dozens of concurrent requests.
The operator graph here is **the rendering pipeline** — read pixels → process → colormap → encode.

**Demo API.** Layer registry lives in a YAML so layers can be added without code changes.

```python
from fastapi import FastAPI, Response
import mercantile
import geotoolz as gz

app = FastAPI()
LAYERS = gz.serialization.load_yaml("pipelines/tile_layers.yaml")  # dict of Sequentials

@app.get("/tiles/{layer}/{z}/{x}/{y}.png")
def tile(layer: str, z: int, x: int, y: int):
    bbox = mercantile.xy_bounds(x, y, z)
    gt = catalog.read_tile(bbox, target_size=(256, 256), crs="EPSG:3857")
    rgb_gt = LAYERS[layer](gt)
    return Response(rgb_gt.to_png_bytes(), media_type="image/png")
```


```yaml
# Tile-server layer registry.
# Each top-level key is a layer name; the value is a Sequential pipeline that
# takes a 256x256 GeoTensor read from the catalog and returns a 256x256 RGB(A)
# GeoTensor ready to encode as PNG.

# === Vegetation index, color-mapped red->yellow->green ===
ndvi:
  _target_: geotoolz.core.Sequential
  operators:
    - _target_: geotoolz.cloud.MaskClouds
      qa_band: -1
      bits: [10, 11]
    - _target_: geotoolz.indices.NDVI
      red_idx: 2
      nir_idx: 3
    - _target_: geotoolz.viz.Colormap
      name: RdYlGn
      vmin: -1.0
      vmax: 1.0

# === True-color preset (sensible brightness defaults baked in) ===
true_color:
  _target_: geotoolz.presets.s2.S2_L2A_RGB_v1
  brightness: 2.0

# === False-color (NIR / Red / Green) for vegetation visualisation ===
false_color_nir:
  _target_: geotoolz.core.Sequential
  operators:
    - _target_: geotoolz.cloud.MaskClouds
      qa_band: -1
      bits: [10, 11]
    - _target_: geotoolz.radiometry.SelectBands
      indices: [3, 2, 1]            # NIR, Red, Green
    - _target_: geotoolz.radiometry.PercentileClip
      p_min: 2
      p_max: 98
      axis: [-2, -1]
    - _target_: geotoolz.radiometry.MinMax

# === Categorical scene-classification layer (SCL) ===
scl_class:
  _target_: geotoolz.viz.CategoricalColormap
  legend_path: /data/legends/scl_legend.json
```


**Implementation sketch.** Operators in this hot loop run ~10⁴ times per minute, so per-call construction work is the enemy.
`Colormap` should load its LUT *once* in `__init__`:

```python
class Colormap(Operator):
    def __init__(self, *, name: str, vmin: float, vmax: float):
        # one-time work — reused across every call
        self.lut = matplotlib.colormaps[name](np.linspace(0, 1, 256))
        self.vmin, self.vmax = vmin, vmax

    def _apply(self, gt):
        # hot path — no allocations beyond the output
        normed = np.clip((gt - self.vmin) / (self.vmax - self.vmin), 0, 1)
        rgba = self.lut[(normed * 255).astype(np.uint8)]
        return GeoTensor(values=rgba, transform=gt.transform, crs=gt.crs)
```

**Tradeoffs.** A tile cache (Redis, CloudFront, on-disk) usually matters more than Python-level performance — the same tile gets requested by every user panning the same area.
With caching, the operator graph runs once per distinct tile and the API mostly serves bytes.
Without caching, every operator’s hot-path performance is exposed.
Stateful operators (e.g. learned percentile clip) need careful handling here: the state must be loaded once at startup, not per-request.

-----

## 8. Data validation / QC as operators

**The setting.** Data goes wrong in predictable ways — wrong CRS, wrong resolution, missing bands, all-NaN scenes from a failed download.
Catching these at the right pipeline stage is the difference between “model trained on garbage for a week” and “pipeline halted on the first bad scene.” The library contribution is making **assertions look like operators**: pass-through Operators that raise (or warn) on contract violations, droppable anywhere in the chain.

**Demo API.**

```python
import geotoolz as gz

qc = gz.Sequential([
    gz.qc.AssertCRSEquals("EPSG:32630"),
    gz.qc.AssertResolutionWithin((9.5, 10.5)),
    gz.qc.AssertValidFraction(min_valid=0.5),     # at least 50% non-NaN
    gz.qc.AssertValueRange(min_val=0, max_val=10_000, on_fail="warn"),
    gz.qc.AssertSchema(bands=["B02", "B03", "B04", "B08", "QA60"]),
])

# Drop into ETL like a tracer round
etl = gz.Sequential([
    gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11]),
    qc,                                            # halts on first hard failure
    gz.correction.TOAToBOA(sun_zenith_band=-2),
    qc,                                            # again, post-correction
    gz.indices.NDVI(red_idx=2, nir_idx=3),
])
```

**Implementation sketch.**

```python
class QCError(Exception): pass

class AssertValueRange(Operator):
    def __init__(self, *, min_val: float, max_val: float, on_fail: str = "raise"):
        assert on_fail in ("raise", "warn")
        self.min, self.max, self.on_fail = min_val, max_val, on_fail

    def _apply(self, gt):
        arr = np.asarray(gt)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if lo < self.min or hi > self.max:
            msg = f"range [{lo:.3g}, {hi:.3g}] outside [{self.min}, {self.max}]"
            if self.on_fail == "raise":
                raise QCError(msg)
            warnings.warn(msg, stacklevel=2)
        return gt  # passes through unchanged
```

The bigger pattern: every QC check is a research-time *“I bet this won’t break”* turned into a permanent runtime guard.
CI gets a pipeline-correctness suite for free, and pipelines self-document their preconditions.

**Tradeoffs.** Where a QC step fires (raise vs warn) is a per-deployment policy.
A research notebook wants `on_fail="warn"` so exploration isn’t interrupted; a production ETL wants `on_fail="raise"` plus alerting.
The same YAML can ship with `on_fail="${qc_policy}"` and let the loader inject the right value.
The performance cost is real — full-array reductions (`nanmin`, `nanmax`) on large GeoTensors aren’t free.
For tile-server hot paths, sample a small fraction of pixels rather than the full array.

-----

## 9. Pinned / hashed regulatory artifact

**The setting.** A scientific paper, a regulatory submission, or an attribution claim from MARS-style operational monitoring needs to be **byte-exact reproducible** years later. “Re-run the pipeline” requires three things to be pinned together: the operator graph, the inputs (with content hashes, not just URIs), and the dependency stack (Python and binary).
Treat the operator graph as **the deliverable**, not just a way to compute one.

**Demo API.**

```python
import geotoolz as gz

# === At publication time ===
pipeline = gz.serialization.load_yaml("pipelines/methane_v3.yaml")
artifact = gz.repro.freeze(
    pipeline,
    inputs={
        "scene_uri":    "s3://imeo/emit/2024Q3/EMIT_L2A_RFL_001_20240805T...nc",
        "scene_sha256": "ab12cd34ef56...",
    },
    deps_lock="poetry.lock",
    metadata={
        "author": "ej",
        "date":   "2024-09-15",
        "doi":    "10.xxxx/mars-2024Q3",
    },
)
artifact.save("/regulatory/mars_methane_2024Q3.gtar")

# === Years later ===
art = gz.repro.load("/regulatory/mars_methane_2024Q3.gtar")
assert art.pipeline_hash == "expected-hash-from-paper"
print(art.config_yaml)        # human-readable
result = art.rerun()           # bit-exact — *if* deps reproduce
```

**Implementation sketch.** The artifact is a tarball of `(pipeline_yaml, config_hash, input_hashes, deps_lock, metadata.json)` — small, archivable, diffable across versions.

```python
class Artifact:
    pipeline_yaml: str
    pipeline_hash: str           # sha256 of canonicalised yaml
    inputs:        dict[str, str] # uri → sha256
    deps_lock:     str           # full lockfile contents
    metadata:      dict

    @classmethod
    def freeze(cls, pipeline, inputs, deps_lock, metadata):
        cfg = pipeline.get_config()
        canon = json.dumps(cfg, sort_keys=True)
        return cls(
            pipeline_yaml=yaml.safe_dump(cfg),
            pipeline_hash=hashlib.sha256(canon.encode()).hexdigest(),
            inputs=inputs,
            deps_lock=Path(deps_lock).read_text(),
            metadata=metadata,
        )
```

**Tradeoffs.** A `poetry.lock` alone is not enough for true byte-exactness — BLAS implementations, GPU drivers, libc versions, even the order in which floating-point reductions are evaluated all affect numerical output.
Honest framing: the artifact *guarantees* the operator graph and inputs; it *aspires to* numerical reproducibility, with the deps_lock as a best-effort.
For genuine bit-exactness, ship a Docker image hash alongside the artifact.
External URIs decay over years; for true long-term archive, the artifact should optionally include the *bytes* of small inputs, not just their hashes.

-----

## 10. Active learning / uncertainty-driven sampling

**The setting.** Labels are expensive (an expert annotator costs more than a GPU-hour).
The classic active-learning loop: train a model, score the unlabeled pool by uncertainty, send the highest-uncertainty examples for labelling, retrain.
The library contribution: a sampler that **consumes operator output** to decide what to sample next.
Same composition primitives, but the iterator is now informed by inference.

**Demo API.**

```python
import geotoolz as gz

# Score every chip by predictive uncertainty.
score_pipeline = gz.Sequential([
    preprocess,
    gz.inference.ModelOp(uq_model, method="predict_with_uncertainty"),
    gz.reduce.MeanScalar(field="sigma"),  # GeoTensor → scalar
])

# Manual loop (one-off use)
scored = []
for sl in gz.sampling.GridSampler(chip_size=(256, 256))(catalog):
    gt = reader.read_geoslice(sl)
    scored.append((sl, score_pipeline(gt)))
top_k = sorted(scored, key=lambda s: -s[1])[:100]

# Codified version once you've used the pattern twice
top_k = gz.sampling.ActiveSampler(
    base=gz.sampling.GridSampler(chip_size=(256, 256)),
    score_op=score_pipeline,
    k=100,
    mode="top",
)(catalog)  # → Iterator[GeoSlice]
```

**Implementation sketch.**

```python
class ActiveSampler:
    def __init__(self, *, base, score_op, k, mode="top"):
        self.base, self.score_op, self.k, self.mode = base, score_op, k, mode

    def __call__(self, catalog):
        scored = []
        for sl in self.base(catalog):
            gt = sl.load()  # via injected reader
            scored.append((sl, float(self.score_op(gt))))

        scored.sort(key=lambda s: s[1], reverse=(self.mode == "top"))
        for sl, _ in scored[: self.k]:
            yield sl
```

**Tradeoffs.** The dirty secret of active learning: **scoring is itself expensive** — you’re running model inference over the entire unlabeled pool just to pick the next batch. For large catalogs this dominates the labelling budget you were trying to save.
Mitigations: random sub-sampling before scoring (score 10% of the pool, label the top 1%), or tiered scoring (cheap heuristic filters first, expensive model second).
Pure top-k by uncertainty also under-explores — pair with a `mode="diverse"` option that uses a clustering step on chip embeddings before picking, to avoid a top-100 that’s all one land-cover type.

-----

## 11. Cross-sensor fusion (the `Graph` case)

**The setting.** Combining modalities — radar with optical, optical with DEM, optical with weather reanalysis.
Each input has different spatial extent, resolution, CRS, and acquisition time, and the fusion rule isn’t a single linear chain.
This is where `Sequential` runs out and `Graph` earns its keep.
The user is an applied scientist building a multi-modal classifier; the value is *expressing the DAG declaratively* instead of as a tangle of intermediate variables.

**Demo API.**

```python
import geotoolz as gz

opt = gz.core.Input("optical")
sar = gz.core.Input("sar")
dem = gz.core.Input("dem")

opt_clean = gz.cloud.MaskClouds(qa_band=-1, bits=[10, 11])(opt)
sar_db    = gz.sar.LinearToDB()(gz.sar.LeeSpeckle(window=7)(sar))
fused     = gz.fusion.AlignAndStack(target_grid="optical")(opt_clean, sar_db, dem)
classify  = gz.inference.ModelOp(crop_classifier)(fused)

g = gz.core.Graph(
    inputs={"optical": opt, "sar": sar, "dem": dem},
    outputs={
        "crop_map":    classify,
        "fused_stack": fused,    # multiple outputs are free
    },
)

out = g(optical=s2_gt, sar=s1_gt, dem=dem_gt)
# out["crop_map"] and out["fused_stack"] are both GeoTensors
```

The same `Graph` can serialise to YAML and run in any deployment shape from the preceding cases — backend API endpoint, scheduled job, notebook session.
Multi-output graphs are particularly useful for downstream debugging: ship the intermediate `fused_stack` alongside the final classification so consumers can audit it.

**Tradeoffs.** `AlignAndStack` is the load-bearing operator and the place fusion pipelines silently fail — picking which input defines the target grid (`target_grid="optical"`) discards information from the other inputs (SAR gets resampled to the optical grid, losing the SAR’s native resolution).
For some workflows that’s exactly right; for others (super-resolution from SAR to optical, for example) the choice is wrong.
Make the choice explicit and test it.
Multi-input arity also complicates Operator validation — a `Graph` with a missing input fails at runtime, not construction time, unless you add explicit input-presence checks.

-----

## 12. Workflow orchestrator task units

**The setting.** Production scheduling — Airflow, Prefect, Dagster.
The orchestrator owns retry, parallelism, monitoring, alerting, and DAG topology; the operator graph owns the science.
The interface between them is small: **pickleable task units** that the orchestrator can submit to workers.
An Operator graph already is one — you just need a thin task wrapper that reconstructs it from a YAML path on the worker side.

**Demo API** (Prefect):

```python
from prefect import flow, task
import geotoolz as gz
import georeader

@task(retries=3, retry_delay_seconds=60)
def run_op(op_yaml: str, slice_dict: dict, source_uri: str) -> dict:
    op = gz.serialization.load_yaml(op_yaml)
    sl = GeoSlice.from_dict(slice_dict)
    with georeader.RasterioReader.open(source_uri) as r:
        gt = r.read_geoslice(sl)
    out = op(gt)
    out_uri = georeader.save_cog(out, f"s3://daily/{slice_dict['hash']}.tif")
    return {"out_uri": out_uri, "stats": out.summary_stats()}

@flow
def daily_methane_flow(date: str):
    catalog = georeader.catalog.open_catalog("s3://...")
    op_yaml = "pipelines/methane_v3.yaml"
    futures = [
        run_op.submit(op_yaml, sl.to_dict(), sl.source_uri)
        for sl in catalog.slices_for_date(date)
    ]
    return [f.result() for f in futures]
```

The Operator stays a serialisable YAML artifact; the orchestrator owns retry, parallelism, monitoring.
Clean separation between *what to compute* and *how to schedule it*.

**Tradeoffs.** Pickling crosses worker boundaries, so the same constraints as case 2 apply — no closures, no captured RNGs, no unbounded factories.
Retries need to be **idempotent**: if `run_op` is retried, the output COG path is the same and is overwritten, not duplicated.
State that lives outside the operator graph (database writes, message queues) is the orchestrator’s problem — don’t bake side effects into Operators.
Alerting on failure rate (rather than first failure) is appropriate at the orchestrator layer; an operator-level `on_fail="raise"` would defeat that.

-----

## 13. Pipeline diffing / A-B regression

**The setting.** You’re about to roll out a new version of a production pipeline — a tweaked atmospheric correction, a new cloud-mask threshold, a calibration update.
The release-management question: *did anything regress that I didn’t intend?* Because operator graphs serialise, you can both **diff** them (config-level) and **run them side-by-side** on the same input (numerical-level).

**Demo API.**

```python
import geotoolz as gz

old = gz.serialization.load_yaml("pipelines/methane_v3.yaml")
new = gz.serialization.load_yaml("pipelines/methane_v4_candidate.yaml")

print(gz.diff.config_diff(old, new))
# - DarkObjectSubtraction.percentile: 1
# + DarkObjectSubtraction.percentile: 2
# + LeeSpeckle (added before MatchedFilter): {window: 5}

reg = gz.diff.AB(old, new, metric=gz.metrics.RMSE())
report = reg.run(catalog.sample(100))
report.summary()
# AB summary
#   n_slices:       100
#   mean RMSE:      0.0143
#   max RMSE:       0.0891 (slice s2_T29SND_20240714)
#   per-band drift: [0.011, 0.013, 0.019, 0.012]
```


```yaml
# Methane attribution pipeline — v4 (candidate)
# Differences from v3:
#   - DarkObjectSubtraction percentile bumped from 1 -> 2 (more aggressive haze removal)
#   - LeeSpeckle pre-step added before MatchedFilter (suppresses detector speckle)
_target_: geotoolz.core.Sequential
operators:

  - _target_: geotoolz.cloud.ApplyMask
    mask_op:
      _target_: geotoolz.cloud.MaskFromQABits
      qa_band: -1
      bits: [10, 11]

  # NEW: speckle suppression before retrieval.
  - _target_: geotoolz.sar.LeeSpeckle
    window: 5

  - _target_: geotoolz.correction.DarkObjectSubtraction
    percentile: 2          # was 1
    axis: [-2, -1]

  - _target_: geotoolz.hyperspectral.MatchedFilter
    target_path: /data/spectra/ch4_signature_v2.npy
    axis: 0

  - _target_: geotoolz.qc.AssertValueRange
    min_val: -100.0
    max_val: 100.0
    on_fail: warn
```


**Implementation sketch.**

```python
class AB:
    def __init__(self, op_a, op_b, *, metric):
        self.a, self.b, self.metric = op_a, op_b, metric

    def run(self, catalog):
        rows = []
        for sl in catalog:
            gt = sl.load()
            ya, yb = self.a(gt), self.b(gt)
            rows.append({"slice": sl.id, "metric": float(self.metric(ya, yb))})
        return ABReport(rows)
```

**Tradeoffs.** Mean RMSE alone can hide bimodal regressions — most slices fine, a few catastrophically worse.
Always look at per-tile distributions and the worst-N tiles, not just aggregate stats.
Choosing the metric is itself a design decision: RMSE is wrong for probability outputs (use cross-entropy or KL), wrong for hard classifications (use confusion-matrix shifts).
The `config_diff` is human-readable but doesn’t capture *semantic* difference — a parameter rename that’s mathematically equivalent shows up as a diff anyway.
Pair config-level diffing with numerical AB testing, and trust the numerics.

-----

## Cases considered but not expanded

A few patterns came up but didn’t earn a full section either because they’re covered elsewhere or because they’re explicitly out of scope:

- **Foundation-model embedding stores.** Same shape as [#10](#10-active-learning--uncertainty-driven-sampling), but the operator output is a dense vector you write to a vector DB. See Example O in the [GeoCatalog Phase 1 doc](https://jejjohnson.github.io/research_notebook/geocatalog).
- **Differentiable / JAX path.** The “future work” case — same Operator surface, but `_apply` becomes JAX-traceable.
  Worth mocking up once `coordax` integration matures.
- **Streaming (Flink / Beam).** Explicitly out-of-scope per the [`geotoolz.md`](https://jejjohnson.github.io/research_notebook/geotoolz) honest-scope section.
  Each Operator could be wrapped as a Flink UDF, but composition guarantees would be lost.
- **Edge / mobile inference.** `ModelOp` ships the model, not the surrounding pre/post-processing operators.
  Out of scope for the operator-graph abstraction.

## The pattern

The operator graph is useful **wherever you’d otherwise hand-roll glue between input → transform → output**.
What changes between cases is *who drives*: a notebook user, an HTTP request, a scheduler, a sampler, a tile renderer, or another graph.
The library’s job is to keep the graph itself a stable artefact across all of them — written once, audited once, run anywhere.
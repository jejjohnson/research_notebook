# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Satellite time-series viewer — notebook subapp
#
# Pair of widgets + a leafmap map that lets you draw an AOI, pick a
# sensor, hit "Search", and inspect the intersecting STAC items
# (footprints, timeline, thumbnails) **before downloading anything**.
#
# Same backend as the Panel subapp (`satellite_viewer.search`) — this
# notebook is just a lighter-weight presentation layer that lives next
# to your other experiments.

# %%
from __future__ import annotations

import datetime as dt

import geopandas as gpd
import ipywidgets as W
import leafmap
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from shapely.geometry import box, shape

from satellite_viewer import SENSORS, search


# %% [markdown]
# ## Controls
#
# Edit the bbox inputs or draw a rectangle on the map (the most recent
# draw wins on the next Search click). Sentinel-2 is selected by
# default since it has the cleanest preview thumbnails.

# %%
sensor_picker = W.SelectMultiple(
    options=list(SENSORS),
    value=("sentinel-2-l2a",),
    description="Sensors",
    rows=6,
)
date_start = W.DatePicker(
    description="Start", value=dt.date.today() - dt.timedelta(days=60)
)
date_end = W.DatePicker(description="End", value=dt.date.today())
cloud_max = W.IntSlider(
    description="Cloud %", min=0, max=100, value=40, continuous_update=False
)
max_items = W.BoundedIntText(
    description="Max items", min=1, max=500, value=50
)
# Lake Tahoe default.
minx = W.FloatText(description="W lon", value=-120.20)
miny = W.FloatText(description="S lat", value=38.95)
maxx = W.FloatText(description="E lon", value=-119.90)
maxy = W.FloatText(description="N lat", value=39.25)
go_btn = W.Button(description="Search", button_style="primary")
status = W.HTML()

controls = W.VBox(
    [
        sensor_picker,
        W.HBox([date_start, date_end]),
        cloud_max,
        max_items,
        W.HBox([minx, miny]),
        W.HBox([maxx, maxy]),
        go_btn,
        status,
    ]
)

# %%
m = leafmap.Map(center=(39.10, -120.05), zoom=9, draw_control=True)
display(W.HBox([controls, m]))


# %% [markdown]
# ## Helpers


# %%
def _aoi_from_controls():
    return box(minx.value, miny.value, maxx.value, maxy.value)


def _aoi_from_draw():
    """If the user drew a rectangle on the map, prefer that AOI."""
    last = m.draw_last_feature
    if last is None:
        return None
    return shape(last["geometry"])


def _redraw_map(aoi, hits: gpd.GeoDataFrame) -> None:
    for layer in list(m.layers)[1:]:
        m.remove_layer(layer)
    m.add_geojson(
        gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326").__geo_interface__,
        layer_name="AOI",
        style={"color": "#1565c0", "weight": 3, "fillOpacity": 0.05},
    )
    if not hits.empty:
        m.add_geojson(
            hits[["geometry", "id", "sensor"]].__geo_interface__,
            layer_name="scenes",
            style={"color": "#43a047", "weight": 1, "fillOpacity": 0.08},
        )


def _plot_timeline(hits: gpd.GeoDataFrame) -> None:
    if hits.empty:
        print("no hits")
        return
    fig, ax = plt.subplots(figsize=(10, 2.5))
    df = hits.copy()
    df["y"] = df["sensor"].astype("category").cat.codes
    sc = ax.scatter(
        df["datetime"],
        df["y"],
        c=df["cloud_cover"].fillna(-1),
        cmap="viridis",
        s=40,
    )
    ax.set_yticks(range(df["sensor"].nunique()))
    ax.set_yticklabels(
        df["sensor"].astype("category").cat.categories.tolist()
    )
    ax.set_xlabel("acquisition time")
    ax.set_title("Timeline of intersecting scenes")
    fig.colorbar(sc, ax=ax, label="cloud cover %")
    fig.autofmt_xdate()
    plt.show()


def _show_thumbnails(hits: gpd.GeoDataFrame, n: int = 8) -> None:
    rows = hits[hits["preview_url"].notna()].head(n)
    if rows.empty:
        print("no previewable thumbnails")
        return
    images = []
    for _, row in rows.iterrows():
        img = W.Image(
            value=_fetch(row["preview_url"]),
            format="png",
            width=180,
            height=180,
        )
        cap = W.HTML(
            f"<small><b>{row['sensor']}</b><br>"
            f"{pd.Timestamp(row['datetime']).strftime('%Y-%m-%d %H:%M')}"
            f"</small>"
        )
        images.append(W.VBox([img, cap]))
    display(W.HBox(images, layout=W.Layout(flex_flow="row wrap")))


def _fetch(url: str) -> bytes:
    import requests

    return requests.get(url, timeout=30).content


# %% [markdown]
# ## Wire up the button


# %%
output = W.Output()
display(output)


def _on_click(_btn):
    output.clear_output()
    aoi = _aoi_from_draw() or _aoi_from_controls()
    start = dt.datetime.combine(date_start.value, dt.time())
    end = dt.datetime.combine(date_end.value, dt.time(23, 59))
    status.value = "<i>searching...</i>"

    frames = []
    for key in sensor_picker.value:
        cfg = SENSORS[key]
        cl = cloud_max.value if cfg.cloud_field is not None else None
        try:
            hits = search(
                key,
                aoi,
                start,
                end,
                max_items=max_items.value,
                cloud_lt=cl,
            )
        except Exception as exc:  # noqa: BLE001
            with output:
                print(f"{key} failed: {type(exc).__name__}: {exc}")
            continue
        frames.append(hits)

    if not frames:
        status.value = "<b>no results</b>"
        return
    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(
        combined, geometry="geometry", crs="EPSG:4326"
    )

    _redraw_map(aoi, combined)
    with output:
        _plot_timeline(combined)
        _show_thumbnails(combined)
        display(
            combined.drop(columns="geometry").head(20).style.set_caption(
                f"{len(combined)} hits (showing first 20)"
            )
        )
    status.value = f"<b>{len(combined)}</b> scene(s) found."


go_btn.on_click(_on_click)

"""v4 M4.0 prototype — global 0.1° satellite-coverage heatmap.

Run:

    uv run --no-project --with-editable "projects/satellite_climatology[app]" \
        streamlit run projects/satellite_climatology/apps/global_heatmap_streamlit.py

Available layer comes from public STAC (Planetary Computer). The optional
Acquired layer reads an external PostGIS holdings table configured via a
gitignored .env (see .env.example) — no private code or credentials in the
repo. Gap = max(available - held, 0).
"""

from __future__ import annotations

import datetime as dt

import folium
import matplotlib
import numpy as np
import streamlit as st


matplotlib.use("Agg")
import pyproj
from folium.plugins import Draw
from matplotlib import pyplot as plt
from satellite_climatology import GridSpec, scan_and_grid, select
from satellite_climatology.holdings import (
    HoldingsUnavailable,
    fetch_holdings,
    holdings_stats,
)
from satellite_climatology.sensors import SENSORS
from shapely import wkt as shp_wkt
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from streamlit_folium import st_folium


st.set_page_config(page_title="Coverage heatmap (v4 M4.0)", layout="wide")

BBOX_PRESETS = {
    "Permian Basin (fast)": (-104.5, 30.5, -101.5, 33.0),
    "CONUS": (-125.0, 24.0, -66.0, 50.0),
    "Europe": (-11.0, 35.0, 31.0, 60.0),
    "Global (use MODIS)": (-180.0, -90.0, 180.0, 90.0),
}
METRIC_LABELS = {
    "scenes_count": "Available scenes",
    "cloud_free_scene_count": "Available cloud-free",
    "held_count": "Acquired (held)",
    "held_clear_count": "Acquired cloud-free",
    "gap_unmet": "Gap (available - held)",
}
AGGS = ["rate", "total", "recent", "worst"]
DEF_START, DEF_END = dt.date(2025, 6, 1), dt.date(2025, 6, 30)


def buffer_point_km(pt, km: float):
    """Buffer a lon/lat point by ``km`` (accurate local azimuthal projection)."""
    aeqd = pyproj.CRS(proj="aeqd", lat_0=pt.y, lon_0=pt.x, datum="WGS84", units="m")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", aeqd, always_xy=True).transform
    inv = pyproj.Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True).transform
    return shp_transform(inv, shp_transform(fwd, pt).buffer(km * 1000))


@st.cache_data(show_spinner="Scanning STAC + gridding…")
def _scan(sensor, kind, key, start, end, max_items, res):
    grid = GridSpec(res)
    if kind == "bbox":
        return scan_and_grid(
            sensor, tuple(key), str(start), str(end), grid, max_items=int(max_items)
        )
    return scan_and_grid(
        sensor,
        None,
        str(start),
        str(end),
        grid,
        aoi=shp_wkt.loads(key),
        max_items=int(max_items),
    )


@st.cache_data(show_spinner="Querying holdings DB…")
def _held(kind, key, start, end, res):
    grid = GridSpec(res)
    if kind == "bbox":
        gdf = fetch_holdings(bbox=tuple(key), start=str(start), end=str(end))
    else:
        gdf = fetch_holdings(aoi=shp_wkt.loads(key), start=str(start), end=str(end))
    return holdings_stats(gdf, str(start), str(end), grid)


def build_map(ds, metric, agg, cmap_name, aoi_wkt=None):
    da = select(ds, metric, agg=agg)
    arr = np.asarray(da.values, dtype="float32")
    pos = arr > 0
    vmax = max(float(np.nanpercentile(arr[pos], 98)) if pos.any() else 1.0, 1.0)
    rgba = (matplotlib.colormaps[cmap_name](np.clip(arr / vmax, 0, 1)) * 255).astype(
        "uint8"
    )
    rgba[..., 3] = np.where(pos, 205, 0).astype("uint8")

    r = float(ds.attrs["resolution"])
    lat, lon = da.lat.values, da.lon.values
    south, north = float(lat.min()) - r / 2, float(lat.max()) + r / 2
    west, east = float(lon.min()) - r / 2, float(lon.max()) + r / 2

    m = folium.Map(tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[[south, west], [north, east]],
        opacity=0.75,
        name=METRIC_LABELS[metric],
        mercator_project=True,
    ).add_to(m)
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "polygon": True,
            "rectangle": True,
            "marker": True,
        },
    ).add_to(m)
    if aoi_wkt:
        folium.GeoJson(
            shape(shp_wkt.loads(aoi_wkt)).__geo_interface__,
            name="AOI",
            style_function=lambda _f: {
                "color": "#00e5ff",
                "weight": 2,
                "fillOpacity": 0.0,
            },
        ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    if pos.any():
        ys, xs = np.where(pos)
        m.fit_bounds(
            [
                [float(lat[ys.max()]) - r / 2, float(lon[xs.min()]) - r / 2],
                [float(lat[ys.min()]) + r / 2, float(lon[xs.max()]) + r / 2],
            ]
        )
    else:
        m.fit_bounds([[south, west], [north, east]])
    return m, vmax


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Coverage scan")
sensor = st.sidebar.selectbox(
    "Sensor",
    list(SENSORS),
    help="\n".join(f"{k}: {v.description}" for k, v in SENSORS.items()),
)
area_mode = st.sidebar.radio("Area", ["Preset region", "Draw on map"])
region = "Permian Basin (fast)"
if area_mode == "Preset region":
    region = st.sidebar.selectbox("Preset", list(BBOX_PRESETS))
else:
    st.sidebar.caption("Draw a rectangle / polygon / marker, then Scan.")
bbox = BBOX_PRESETS[region]
buffer_km = st.sidebar.slider(
    "Point buffer (km)",
    0,
    100,
    25,
    help="A drawn marker becomes a circular AOI of this radius.",
)
c0, c1 = st.sidebar.columns(2)
start = c0.date_input("Start", DEF_START)
end = c1.date_input("End", DEF_END)
max_items = st.sidebar.number_input("Max items (cap)", 100, 20000, 2000, step=100)
res = st.sidebar.select_slider("Grid resolution (°)", [0.1, 0.25, 0.5], value=0.1)
acquired_on = st.sidebar.checkbox(
    "Show Acquired (holdings DB)",
    value=False,
    help="Reads an external PostGIS table via .env creds.",
)
scan = st.sidebar.button("Scan & grid", type="primary", width="stretch")
st.sidebar.markdown("---")
agg = st.sidebar.selectbox("Time aggregation", AGGS, help="rate = mean / month")
cmap_name = st.sidebar.selectbox("Colormap", ["magma", "viridis", "inferno", "cividis"])


# ---------------------------------------------------------------------------
# Initial scan + state
# ---------------------------------------------------------------------------
if "ds" not in st.session_state:
    st.session_state.ds = _scan(sensor, "bbox", bbox, DEF_START, DEF_END, 2000, 0.1)
    st.session_state.meta = (sensor, region, str(DEF_START), str(DEF_END))
    st.session_state.aoi_wkt = None
    st.session_state.held_err = None
ds = st.session_state.ds

st.title("Satellite coverage heatmap — 0.1° global grid")
metrics = [m for m in METRIC_LABELS if m in ds.data_vars]
metric = st.radio("Layer", metrics, format_func=METRIC_LABELS.get, horizontal=True)


# ---------------------------------------------------------------------------
# Map (render first so we can read any drawing)
# ---------------------------------------------------------------------------
m, vmax = build_map(ds, metric, agg, cmap_name, aoi_wkt=st.session_state.aoi_wkt)
map_state = st_folium(
    m, height=560, width=None, returned_objects=["all_drawings"], key="cov_map"
)

drawings = (map_state or {}).get("all_drawings") or []
if area_mode == "Draw on map" and drawings:
    geom = shape(drawings[-1]["geometry"])
    if geom.geom_type == "Point" and buffer_km > 0:
        geom = buffer_point_km(geom, buffer_km)
        label = f"drawn point +{buffer_km} km"
    else:
        label = f"drawn {geom.geom_type}"
    next_kind, next_key, next_label, next_wkt = "aoi", geom.wkt, label, geom.wkt
else:
    next_kind, next_key, next_label, next_wkt = "bbox", bbox, region, None

if scan:
    new = _scan(sensor, next_kind, next_key, start, end, max_items, res)
    err = None
    if acquired_on:
        try:
            held = _held(next_kind, next_key, start, end, res)
            new = new.merge(held)
            if "scenes_count" in new and "held_count" in new:
                new["gap_unmet"] = (new["scenes_count"] - new["held_count"]).clip(min=0)
        except HoldingsUnavailable as e:
            err = str(e)
    st.session_state.ds = new
    st.session_state.meta = (sensor, next_label, str(start), str(end))
    st.session_state.aoi_wkt = next_wkt
    st.session_state.held_err = err
    st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
s_sensor, s_area, s0, s1 = st.session_state.meta
left, right = st.columns([3, 1])
with left:
    st.caption(
        f"**{s_sensor}** · {s_area} · {s0} → {s1} · "
        f"**{ds.attrs.get('n_items', '?')}** scenes · {ds.sizes['time']} bin(s) · "
        f"{agg} · scale 0 → {vmax:.0f}"
    )
    if st.session_state.held_err:
        st.warning(f"Acquired layer unavailable — {st.session_state.held_err}")
    if ds.attrs.get("n_items", 0) >= max_items:
        st.warning(f"Hit the {max_items}-item cap — counts are a lower bound.")
    if area_mode == "Draw on map" and not drawings:
        st.info("Draw a shape on the map, then press **Scan & grid**.")
with right:
    fig, ax = plt.subplots(figsize=(3, 0.45))
    fig.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(0, vmax), cmap_name),
        cax=ax,
        orientation="horizontal",
    )
    ax.tick_params(labelsize=6)
    st.pyplot(fig, use_container_width=True)

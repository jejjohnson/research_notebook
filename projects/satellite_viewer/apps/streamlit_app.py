"""Streamlit subapp: AOI -> sensor preview, single-file rerun-on-change script.

Run:

    pixi run -e satellite-viewer streamlit-app

Or directly:

    streamlit run projects/satellite_viewer/apps/streamlit_app.py

Workflow:

1. Pick sensor(s), date range, optional cloud-cover cap in the sidebar.
2. Draw a rectangle on the map (or keep the Lake Tahoe default bbox).
3. Click "Search" — the rest of the page re-renders with footprints,
   a timeline, thumbnails, and the raw results table.

Nothing is downloaded — every preview is a STAC `rendered_preview` URL.
"""

from __future__ import annotations

import datetime as dt

import altair as alt
import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from satellite_viewer import SENSORS, search
from shapely.geometry import box, shape
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Satellite Time-Series Viewer",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

st.sidebar.title("Search")
sensors = st.sidebar.multiselect(
    "Sensors",
    options=list(SENSORS),
    default=["sentinel-2-l2a"],
    help="Polar-orbiting / tasked sensors on Microsoft Planetary Computer.",
)
today = dt.date.today()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(today - dt.timedelta(days=60), today),
    min_value=dt.date(2015, 1, 1),
    max_value=today,
)
cloud_lt = st.sidebar.slider("Max cloud cover %", 0, 100, 40)
max_items = st.sidebar.number_input(
    "Max items / sensor", min_value=1, max_value=500, value=50
)

st.sidebar.markdown("---")
st.sidebar.markdown("**AOI bbox** (WGS84)")
col_w, col_e = st.sidebar.columns(2)
minx = col_w.number_input("W lon", value=-120.20, step=0.05, format="%.4f")
maxx = col_e.number_input("E lon", value=-119.90, step=0.05, format="%.4f")
col_s, col_n = st.sidebar.columns(2)
miny = col_s.number_input("S lat", value=38.95, step=0.05, format="%.4f")
maxy = col_n.number_input("N lat", value=39.25, step=0.05, format="%.4f")
go = st.sidebar.button("Search", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# Map (folium + Draw plugin). The map is interactive on every rerun; if
# the user drew a polygon it overrides the bbox text inputs.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Basemaps. Added to every map as selectable base layers; folium's
# LayerControl renders a radio toggle (top-right corner) so the user can
# switch street / satellite / topo / natural without a Streamlit rerun.
# All are key-free XYZ tile services. The first entry is the default.
# ---------------------------------------------------------------------------

_ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services"
_BASEMAPS = [
    # (display name, tiles, attribution -- None lets folium fill it in)
    ("Street (OSM)", "OpenStreetMap", None),
    (
        "Satellite (Esri)",
        f"{_ESRI}/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
        "Esri World Imagery",
    ),
    (
        "Topographic (Esri)",
        f"{_ESRI}/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}",
        "Esri World Topo",
    ),
    (
        "Natural (NatGeo)",
        f"{_ESRI}/NatGeo_World_Map/MapServer/tile/{{z}}/{{y}}/{{x}}",
        "Esri NatGeo",
    ),
    (
        "Terrain (OpenTopoMap)",
        "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "OpenTopoMap (CC-BY-SA)",
    ),
]


def _add_basemaps(fmap: folium.Map) -> None:
    """Add every basemap as a base layer; the first added is the default."""
    for name, tiles, attr in _BASEMAPS:
        folium.TileLayer(
            tiles=tiles,
            attr=attr,
            name=name,
            overlay=False,
            control=True,
        ).add_to(fmap)


st.title("Satellite Time-Series Viewer")

bbox_aoi = box(minx, miny, maxx, maxy)
m = folium.Map(
    location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=9, tiles=None
)
_add_basemaps(m)
Draw(
    export=False,
    draw_options={
        "polyline": False,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polygon": True,
        "rectangle": True,
    },
).add_to(m)
folium.GeoJson(
    gpd.GeoSeries([bbox_aoi], crs="EPSG:4326").__geo_interface__,
    name="bbox AOI",
    style_function=lambda _f: {
        "color": "#1565c0",
        "weight": 2,
        "fillOpacity": 0.04,
    },
).add_to(m)

folium.LayerControl(collapsed=True).add_to(m)
map_state = st_folium(m, height=500, width=None, key="aoi_map")


def _resolve_aoi():
    """Last-drawn polygon wins over the bbox text inputs."""
    drawings = (map_state or {}).get("all_drawings") or []
    if drawings:
        return shape(drawings[-1]["geometry"])
    return bbox_aoi


# ---------------------------------------------------------------------------
# Search + results
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_search(
    sensor: str,
    aoi_wkt: str,
    start: dt.datetime,
    end: dt.datetime,
    cloud_lt: float | None,
    max_items: int,
) -> pd.DataFrame:
    """Cache key is by-value; we serialise the AOI to WKT to make it hashable."""
    from shapely import wkt

    aoi = wkt.loads(aoi_wkt)
    gdf = search(sensor, aoi, start, end, cloud_lt=cloud_lt, max_items=max_items)
    # Return a plain DataFrame for Streamlit's hashing; reconstitute
    # geometry from WKT on the consumer side.
    out = pd.DataFrame(gdf.drop(columns="geometry"))
    out["geometry_wkt"] = gdf.geometry.to_wkt()
    return out


if go and sensors:
    aoi = _resolve_aoi()
    start_dt = dt.datetime.combine(start_date, dt.time())
    end_dt = dt.datetime.combine(end_date, dt.time(23, 59))

    frames: list[pd.DataFrame] = []
    progress = st.progress(0.0, text="searching…")
    for i, key in enumerate(sensors, 1):
        cfg = SENSORS[key]
        cl = cloud_lt if cfg.cloud_field is not None else None
        try:
            frames.append(
                _cached_search(key, aoi.wkt, start_dt, end_dt, cl, int(max_items))
            )
        except Exception as exc:  # surfaced to the UI, not swallowed
            st.error(f"{key} failed: {type(exc).__name__}: {exc}")
        progress.progress(i / len(sensors), text=f"searched {key}")
    progress.empty()

    if not frames:
        st.warning("No results.")
        st.stop()

    combined = pd.concat(frames, ignore_index=True)
    n = len(combined)
    st.success(f"Found **{n}** scene(s) across {len(sensors)} sensor(s).")

    # --- Footprints on the map (render below the input map) -----------------
    st.subheader("Footprints")
    from shapely import wkt

    footprints = gpd.GeoSeries(
        [wkt.loads(w) for w in combined["geometry_wkt"]], crs="EPSG:4326"
    )
    m2 = folium.Map(
        location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=8, tiles=None
    )
    _add_basemaps(m2)
    folium.GeoJson(
        gpd.GeoSeries([aoi], crs="EPSG:4326").__geo_interface__,
        name="AOI",
        style_function=lambda _f: {
            "color": "#1565c0",
            "weight": 3,
            "fillOpacity": 0.05,
        },
    ).add_to(m2)
    folium.GeoJson(
        gpd.GeoDataFrame(
            {"sensor": combined["sensor"]}, geometry=footprints
        ).__geo_interface__,
        name="scenes",
        style_function=lambda _f: {
            "color": "#43a047",
            "weight": 1,
            "fillOpacity": 0.08,
        },
    ).add_to(m2)
    folium.LayerControl(collapsed=True).add_to(m2)
    st_folium(m2, height=420, width=None, returned_objects=[], key="result_map")

    # --- Timeline -----------------------------------------------------------
    st.subheader("Timeline")
    # A single "acquisition" lane on y; the satellite is identified by colour
    # + legend (and on hover), so the chart reads as "when was anything
    # acquired, and by which sensor".
    timeline = (
        alt.Chart(combined.assign(lane="acquisition"))
        .mark_circle(size=90, opacity=0.75)
        .encode(
            x=alt.X("datetime:T", title="acquisition time"),
            y=alt.Y(
                "lane:N",
                title=None,
                axis=alt.Axis(labels=False, ticks=False, domain=False),
            ),
            color=alt.Color("sensor:N", title="satellite"),
            tooltip=["id", "datetime", "cloud_cover", "sensor"],
        )
        .properties(height=160)
    )
    st.altair_chart(timeline, use_container_width=True)

    # --- Thumbnails ---------------------------------------------------------
    st.subheader("Thumbnails")
    rows = combined[combined["preview_url"].notna()].head(12)
    if rows.empty:
        st.caption("No previewable thumbnails for this result set.")
    else:
        cols = st.columns(4)
        for i, (_, row) in enumerate(rows.iterrows()):
            with cols[i % 4]:
                st.image(row["preview_url"], width="stretch")
                ts = pd.Timestamp(row["datetime"])
                when = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "date n/a"
                caption = f"**{row['sensor']}** — {when}"
                if pd.notna(row.get("cloud_cover")):
                    caption += f" — cloud {row['cloud_cover']:.0f}%"
                st.caption(caption)

    # --- Raw table ----------------------------------------------------------
    st.subheader("Results table")
    st.dataframe(combined.drop(columns="geometry_wkt"), width="stretch", height=320)
else:
    st.info(
        "Pick sensors and a date range in the sidebar, optionally draw a "
        "polygon on the map, then click **Search**."
    )

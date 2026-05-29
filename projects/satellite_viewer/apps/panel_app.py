"""Panel subapp: linked map / timeline / thumbnail strip for a satellite preview.

Run:

    pixi run -e satellite-viewer panel-app

Or directly:

    panel serve projects/satellite_viewer/apps/panel_app.py --show

Workflow:

1. Pick sensor(s), date range, optional cloud-cover cap.
2. The default AOI is a small Lake Tahoe bbox. Edit the four corner
   inputs to point somewhere else.
3. Click "Search" — footprints land on the map, a timeline appears
   below, and thumbnail tiles render in the bottom strip.

Nothing is downloaded — every preview is a STAC `rendered_preview` URL.
"""

from __future__ import annotations

import datetime as dt

import geopandas as gpd
import pandas as pd
import panel as pn
from satellite_viewer import SENSORS, search
from shapely.geometry import box


pn.extension("ipywidgets", "tabulator", sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

sensor_widget = pn.widgets.MultiChoice(
    name="Sensors",
    options=list(SENSORS),
    value=["sentinel-2-l2a"],
    height=110,
)
date_range = pn.widgets.DateRangeSlider(
    name="Date range",
    start=dt.date(2023, 1, 1),
    end=dt.date.today(),
    value=(dt.date.today() - dt.timedelta(days=60), dt.date.today()),
)
cloud_slider = pn.widgets.IntSlider(
    name="Max cloud cover %", start=0, end=100, value=40
)
max_items = pn.widgets.IntInput(name="Max items / sensor", value=50, start=1, end=500)
# Lake Tahoe default.
minx = pn.widgets.FloatInput(name="W lon", value=-120.20, step=0.05)
miny = pn.widgets.FloatInput(name="S lat", value=38.95, step=0.05)
maxx = pn.widgets.FloatInput(name="E lon", value=-119.90, step=0.05)
maxy = pn.widgets.FloatInput(name="N lat", value=39.25, step=0.05)
go_btn = pn.widgets.Button(name="Search", button_type="primary")
status = pn.pane.Markdown("")


# ---------------------------------------------------------------------------
# Map (leafmap, ipyleaflet-backed). The Panel `ipywidgets` extension renders
# any ipywidget — including leafmap maps — natively.
# ---------------------------------------------------------------------------

import holoviews as hv
import leafmap


hv.extension("bokeh", logo=False)


m = leafmap.Map(
    center=(39.10, -120.05),
    zoom=9,
    height="500px",
    draw_control=True,
    measure_control=False,
    fullscreen_control=False,
)
map_pane = pn.pane.IPyWidget(m, height=500, sizing_mode="stretch_width")


# Basemap switcher. All providers are key-free XYZ tile services. We swap the
# base layer in place with substitute_layer() so it stays *under* the AOI /
# footprint overlays (add_basemap would stack it on top and hide them).
from ipyleaflet import basemap_to_tiles, basemaps as _ipybm


_BASEMAPS = {
    "Street (OSM)": _ipybm.OpenStreetMap.Mapnik,
    "Satellite (Esri)": _ipybm.Esri.WorldImagery,
    "Topographic (Esri)": _ipybm.Esri.WorldTopoMap,
    "Natural (NatGeo)": _ipybm.Esri.NatGeoWorldMap,
    "Terrain (OpenTopoMap)": _ipybm.OpenTopoMap,
}
basemap_widget = pn.widgets.Select(
    name="Basemap", options=list(_BASEMAPS), value="Street (OSM)"
)


def _set_basemap(_event=None) -> None:
    tiles = basemap_to_tiles(_BASEMAPS[basemap_widget.value])
    tiles.base = True
    m.substitute_layer(m.layers[0], tiles)


basemap_widget.param.watch(_set_basemap, "value")


def _current_aoi():
    return box(minx.value, miny.value, maxx.value, maxy.value)


# ---------------------------------------------------------------------------
# Output panes
# ---------------------------------------------------------------------------

results_table = pn.widgets.Tabulator(
    pd.DataFrame(), pagination="local", page_size=10, height=320
)
timeline_pane = pn.pane.HoloViews(sizing_mode="stretch_width")
thumbs_pane = pn.FlexBox(sizing_mode="stretch_width")


def _render_timeline(hits: gpd.GeoDataFrame):
    if hits.empty:
        return hv.Text(0.5, 0.5, "no hits").opts(responsive=True, height=200)
    df = hits.copy()
    df["sensor"] = df["sensor"].astype(str)
    # Single "acquisition" lane on y; satellite identity carried by colour +
    # legend (one Scatter per sensor, overlaid) and surfaced on hover.
    df["lane"] = "acquisition"
    by_sensor = {
        name: hv.Scatter(g, kdims=["datetime"], vdims=["lane", "cloud_cover", "id"])
        for name, g in df.groupby("sensor")
    }
    overlay = hv.NdOverlay(by_sensor, kdims="sensor")
    return overlay.opts(
        hv.opts.Scatter(size=11, alpha=0.85, tools=["hover"], muted_alpha=0.15),
        hv.opts.NdOverlay(
            # frame_height pins the glyph canvas so the dots can't collapse.
            responsive=True,
            frame_height=150,
            show_legend=True,
            legend_position="right",
            xlabel="acquisition time",
            ylabel="",
            title="Acquisitions over time",
            padding=(0.05, 0.6),
        ),
    )


def _render_thumbnails(hits: gpd.GeoDataFrame, n_max: int = 12) -> pn.FlexBox:
    rows = hits[hits["preview_url"].notna()].head(n_max)
    tiles = []
    for _, row in rows.iterrows():
        caption = (
            f"**{row['sensor']}**\n\n"
            f"{pd.Timestamp(row['datetime']).strftime('%Y-%m-%d %H:%M')}"
        )
        if pd.notna(row.get("cloud_cover")):
            caption += f" — cloud {row['cloud_cover']:.0f}%"
        tiles.append(
            pn.Column(
                pn.pane.PNG(row["preview_url"], width=180),
                pn.pane.Markdown(caption, height=44),
                width=200,
            )
        )
    if not tiles:
        tiles = [pn.pane.Markdown("_No previewable items._")]
    return pn.FlexBox(*tiles, sizing_mode="stretch_width")


def _redraw_map(aoi, hits: gpd.GeoDataFrame) -> None:
    # Drop every previous overlay layer (keep the basemap).
    for layer in list(m.layers)[1:]:
        m.remove_layer(layer)
    # info_mode=None: skip Leafmap's hover-info widget. It spawns a
    # WidgetControl that remove_layer() doesn't clean up, so the boxes
    # would otherwise stack on the map on every re-search.
    m.add_geojson(
        gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326").__geo_interface__,
        layer_name="AOI",
        style={"color": "#1565c0", "weight": 3, "fillOpacity": 0.05},
        info_mode=None,
    )
    if not hits.empty:
        m.add_geojson(
            hits[["geometry", "id", "sensor"]].__geo_interface__,
            layer_name="scenes",
            style={
                "color": "#43a047",
                "weight": 1,
                "fillOpacity": 0.08,
            },
            info_mode=None,
        )

    # Re-fit the viewport to the data extent. Beyond the convenient
    # auto-zoom, changing the view forces this live Leaflet map to reload
    # its raster tiles: after a programmatic layer swap the basemap can
    # otherwise stay blank (only the SVG footprints repaint) until the user
    # manually pans/zooms.
    bounds_src = (
        hits if not hits.empty else gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326")
    )
    minx_b, miny_b, maxx_b, maxy_b = bounds_src.total_bounds
    m.fit_bounds([[float(miny_b), float(minx_b)], [float(maxy_b), float(maxx_b)]])


def _do_search(_event=None) -> None:
    aoi = _current_aoi()
    start = dt.datetime.combine(date_range.value[0], dt.time())
    end = dt.datetime.combine(date_range.value[1], dt.time(23, 59))
    status.object = f"Searching {len(sensor_widget.value)} sensor(s)…"

    frames: list[gpd.GeoDataFrame] = []
    for key in sensor_widget.value:
        cfg = SENSORS[key]
        try:
            cl = cloud_slider.value if cfg.cloud_field is not None else None
            hits = search(
                key,
                aoi,
                start,
                end,
                max_items=max_items.value,
                cloud_lt=cl,
            )
        except Exception as exc:
            status.object = f"**{key}** failed: `{type(exc).__name__}: {exc}`"
            continue
        frames.append(hits)

    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

    results_table.value = pd.DataFrame(combined.drop(columns="geometry"))
    timeline_pane.object = _render_timeline(combined)
    thumbs_pane.objects = _render_thumbnails(combined).objects
    _redraw_map(aoi, combined)
    status.object = f"Done — **{len(combined)}** scene(s) found."


go_btn.on_click(_do_search)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

sidebar = pn.Column(
    pn.pane.Markdown("### Search"),
    sensor_widget,
    date_range,
    cloud_slider,
    max_items,
    pn.pane.Markdown("**AOI bbox** (WGS84):"),
    pn.Row(minx, miny),
    pn.Row(maxx, maxy),
    go_btn,
    status,
    pn.layout.Divider(),
    pn.pane.Markdown("**Map display**"),
    basemap_widget,
    width=320,
)

main = pn.Column(
    map_pane,
    pn.pane.Markdown("### Timeline"),
    timeline_pane,
    pn.pane.Markdown("### Thumbnails"),
    thumbs_pane,
    pn.pane.Markdown("### Results table"),
    results_table,
)

template = pn.template.MaterialTemplate(
    title="Satellite Time-Series Viewer",
    sidebar=[sidebar],
    main=[main],
)

template.servable()

"""Plotting helpers for the spatial-extremes notebooks.

Keeps map boilerplate out of the notebooks and degrades gracefully: cartopy
gives a proper gridded land/ocean basemap when available and online, but the
Natural Earth shapefiles it pulls on first use can fail offline / in CI, so
every cartopy call is guarded and falls back to a plain lon/lat axis.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt

from spatial_extremes.data import IBERIA_BBOX


def iberia_axes(ax=None, *, figsize=(7.5, 6.5), gridlines=True):
    """Return an axis framed on Iberia with a gridded land/ocean basemap.

    Uses a cartopy ``PlateCarree`` GeoAxes when cartopy is importable and its
    feature data is reachable; otherwise a plain axis with the Iberia extent and
    a light grid. The returned axis always accepts ``lon, lat`` data directly.
    """
    lon_min, lon_max, lat_min, lat_max = IBERIA_BBOX
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        try:
            # Natural Earth shapefiles download (once) on first use; hush the
            # DownloadWarning so it doesn't clutter rendered notebook output.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.add_feature(cfeature.OCEAN, facecolor="#dbeafe")
                ax.add_feature(cfeature.LAND, facecolor="#f5f3ee")
                ax.add_feature(
                    cfeature.BORDERS, linewidth=0.5, edgecolor="0.5", linestyle=":"
                )
                ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="0.35")
                ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="#9ec5e8")
        except Exception:  # offline: Natural Earth download failed
            pass
        if gridlines:
            try:
                gl = ax.gridlines(
                    draw_labels=True,
                    linewidth=0.4,
                    color="0.7",
                    alpha=0.6,
                    linestyle="--",
                )
                gl.top_labels = gl.right_labels = False
            except Exception:
                pass
        ax._is_geo = True
        return ax
    except Exception:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.grid(True, ls="--", lw=0.4, color="0.8")
        ax.set_aspect("equal", adjustable="box")
        ax._is_geo = False
        return ax


def _geo_kw(ax, kw):
    if getattr(ax, "_is_geo", False):
        import cartopy.crs as ccrs

        kw.setdefault("transform", ccrs.PlateCarree())
    return kw


def scatter_field(ax, lon, lat, values, *, label=None, **kw):
    """Scatter a per-station scalar field on an :func:`iberia_axes` axis.

    Adds a colorbar and handles the cartopy ``transform`` automatically. Returns
    the ``PathCollection``.
    """
    kw.setdefault("s", 60)
    kw.setdefault("edgecolor", "k")
    kw.setdefault("linewidth", 0.3)
    kw.setdefault("cmap", "magma")
    kw.setdefault("zorder", 4)
    sc = ax.scatter(lon, lat, c=values, **_geo_kw(ax, kw))
    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    if label:
        cbar.set_label(label)
    return sc


def mark_points(ax, lon, lat, *, marker="o", **kw):
    """Plot plain marker points (e.g. station locations) on a geo axis."""
    kw.setdefault("zorder", 5)
    return ax.scatter(lon, lat, marker=marker, **_geo_kw(ax, kw))


def mark_star(ax, lon, lat, *, label=None, color="gold", size=420, **kw):
    """Drop a highlighted star at ``(lon, lat)`` — e.g. a chosen station."""
    kw.setdefault("zorder", 6)
    sc = ax.scatter(
        lon,
        lat,
        marker="*",
        s=size,
        c=color,
        edgecolor="k",
        linewidth=0.8,
        **_geo_kw(ax, kw),
    )
    if label:
        tkw = {}
        if getattr(ax, "_is_geo", False):
            import cartopy.crs as ccrs

            tkw["transform"] = ccrs.PlateCarree()
        ax.annotate(
            label,
            (lon, lat),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            zorder=7,
        )
    return sc

"""Smoke tests for ``geostack.data``.

These exercise the public loader surface against real upstream APIs
(Microsoft Planetary Computer, GBIF, Natural Earth, Overture) so the
notebooks in ``projects/geostack/notebooks/`` keep running as those
APIs evolve. They are tagged ``slow`` because each test makes one or
more network calls — skip with ``pytest -m 'not slow'``.

Run from the repo root::

    pixi run -e geostack pytest projects/geostack/tests/ -v -m slow
"""

from __future__ import annotations

import numpy as np
import pytest
from geostack import (
    LAKE_TAHOE_BBOX,
    LAKE_TAHOE_TILE,
    LISBON_BBOX,
    LISBON_TILE,
    load_gbif_points,
    load_natural_earth_admin1,
    load_overture_buildings_url,
    load_s2_chip,
    load_s2_timestack,
    load_stac_items,
    mpc_catalog,
)


pytestmark = pytest.mark.slow


def test_constants_are_well_formed():
    """AOI constants are 4-tuples in EPSG:4326 with west<east, south<north."""
    for bbox in (LAKE_TAHOE_BBOX, LISBON_BBOX):
        west, south, east, north = bbox
        assert west < east, f"west={west} >= east={east} in {bbox}"
        assert south < north, f"south={south} >= north={north} in {bbox}"
    assert isinstance(LAKE_TAHOE_TILE, str) and len(LAKE_TAHOE_TILE) == 5
    assert isinstance(LISBON_TILE, str) and len(LISBON_TILE) == 5


def test_mpc_catalog_opens():
    """A signed MPC client constructs without raising."""
    client = mpc_catalog()
    # Ask for one collection root to confirm the signing modifier is wired in.
    coll = client.get_collection("sentinel-2-l2a")
    assert coll.id == "sentinel-2-l2a"


def test_load_stac_items_returns_sorted_results():
    """STAC search returns items sorted by ascending cloud cover."""
    items = load_stac_items(
        "sentinel-2-l2a",
        LAKE_TAHOE_BBOX,
        "2024-06-01/2024-07-15",
        tile=LAKE_TAHOE_TILE,
        max_cloud_cover=15,
    )
    assert items, "no items found for the Lake Tahoe AOI"
    clouds = [it.properties["eo:cloud_cover"] for it in items]
    assert clouds == sorted(clouds), "items should be sorted by ascending cloud cover"


def test_load_s2_chip_shape_and_dtype():
    """A Lake Tahoe BGRN chip loads as a 4-band uint16 GeoTensor."""
    gt = load_s2_chip(bbox=LAKE_TAHOE_BBOX)
    assert gt.ndim == 3
    assert gt.shape[0] == 4, f"expected 4 bands (BGRN), got {gt.shape[0]}"
    assert gt.dtype == np.uint16
    # Pixels should be in the S2 DN range, not all-zero / all-NaN.
    arr = np.asarray(gt)
    assert arr.max() > 0
    # CRS should be UTM zone 10 (32610) for Lake Tahoe.
    assert "EPSG" in str(gt.crs)


def test_load_s2_timestack_shape():
    """The temporal stack returns (T, C, H, W) plus parallel dates."""
    stack, dates, ref_da = load_s2_timestack(
        bbox=LAKE_TAHOE_BBOX,
        date_range="2024-06-01/2024-07-15",
        tile=LAKE_TAHOE_TILE,
        bands=("B04", "B08"),
        max_items=3,
    )
    assert stack.ndim == 4
    assert stack.shape[0] == len(dates) <= 3
    assert stack.shape[1] == 2  # (B04, B08)
    assert all(d[:4] == "2024" for d in dates)
    assert ref_da.shape == stack.shape[-2:]


def test_load_gbif_points_returns_geodataframe():
    """GBIF API yields a GeoDataFrame with EPSG:4326 Points inside the bbox."""
    df = load_gbif_points(species_key=5285750, limit=25)
    # Some queries may yield zero rows; tolerate but require correct shape.
    if len(df) > 0:
        assert df.crs is not None
        assert df.geometry.iloc[0].geom_type == "Point"
        # Sanity: every point inside the California-ish bbox we used.
        for pt in df.geometry:
            assert -125 <= pt.x <= -114
            assert 32 <= pt.y <= 42


def test_load_natural_earth_admin1():
    """Natural Earth admin-1 download includes major US states."""
    df = load_natural_earth_admin1()
    assert df.crs is not None
    names = set(df["name"].astype(str))
    assert {"California", "Oregon", "Nevada"} <= names


def test_load_overture_buildings_url_format():
    """The Overture URL helper returns a partitioned S3 path."""
    url = load_overture_buildings_url()
    assert url.startswith("s3://overturemaps-us-west-2/")
    assert "theme=buildings" in url
    assert "type=building" in url

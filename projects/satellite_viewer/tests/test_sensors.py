"""Static checks on the sensor registry."""

from __future__ import annotations

from satellite_viewer.sensors import SENSORS, SensorConfig


def test_registry_is_nonempty() -> None:
    assert len(SENSORS) >= 4


def test_user_requested_sensors_present() -> None:
    """The four polar/tasked sensor families in scope must all be reachable."""
    keys = list(SENSORS)
    assert any(k.startswith("sentinel") for k in keys)
    assert any(k.startswith("landsat") for k in keys)
    assert any(k.startswith("modis") for k in keys)
    assert any(k.startswith("emit") for k in keys)


def test_all_have_endpoint_and_collection() -> None:
    for key, cfg in SENSORS.items():
        assert cfg.stac_endpoint, f"{key}: stac_endpoint required"
        assert cfg.collection_id, f"{key}: collection_id required"


def test_key_matches_config_name() -> None:
    for key, cfg in SENSORS.items():
        assert isinstance(cfg, SensorConfig)
        assert cfg.name == key, f"registry key {key!r} != cfg.name {cfg.name!r}"

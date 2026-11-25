"""Fast unit tests for URL allowlist classification and policy enforcement.

Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §2.3, §3 (G1) and Finding C6.
"""

import pytest
from xvs.sim.allowlist import URLClassifier, match_host_pattern


def test_match_host_pattern():
    """Test wildcard and exact host pattern matching."""
    assert match_host_pattern("127.0.0.1", "127.0.0.1") is True
    assert match_host_pattern("localhost", "localhost") is True
    assert match_host_pattern("api.cesium.com", "*.cesium.com") is True
    assert match_host_pattern("assets.ion.cesium.com", "*.cesium.com") is True
    assert match_host_pattern("cesium.com", "*.cesium.com") is True
    assert match_host_pattern("othercesium.com", "*.cesium.com") is False
    assert match_host_pattern("tile.googleapis.com", "*.googleapis.com") is True
    assert match_host_pattern("example.com", "example.com") is True


def test_browser_url_classification():
    """URL classifier tests for browser URLs: allowed, forbidden, unknown."""
    clf = URLClassifier()

    # Allowed browser URLs
    verdict, _ = clf.classify_browser_url("http://127.0.0.1:8000/page/index.html")
    assert verdict == "ALLOWED"
    verdict, _ = clf.classify_browser_url("http://localhost:3000/cesium/Cesium.js")
    assert verdict == "ALLOWED"
    verdict, _ = clf.classify_browser_url("blob:http://127.0.0.1:8000/worker-uuid")
    assert verdict == "ALLOWED"
    verdict, _ = clf.classify_browser_url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...")
    assert verdict == "ALLOWED"

    # Forbidden browser URLs
    for forbidden_url in [
        "https://assets.ion.cesium.com/1",
        "https://api.cesium.com/v1/assets",
        "https://tile.googleapis.com/v1/3dtiles",
        "https://ecn.t0.tiles.virtualearth.net/tiles/r0.jpeg",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer",
        "https://tile.openstreetmap.org/0/0/0.png",
    ]:
        verdict, reason = clf.classify_browser_url(forbidden_url)
        assert verdict == "FORBIDDEN", f"Expected FORBIDDEN for {forbidden_url}"
        assert reason is not None

    # Unknown browser URLs (not in browser_allowed_hosts and not in forbidden)
    for unknown_url in [
        "https://example.com/index.html",
        "http://192.168.1.1:8080/data",
        "https://github.com/foo/bar",
    ]:
        verdict, reason = clf.classify_browser_url(unknown_url)
        assert verdict == "UNKNOWN", f"Expected UNKNOWN for {unknown_url}"


def test_upstream_url_classification():
    """URL classifier tests for upstream proxy requests."""
    clf = URLClassifier()

    # Allowed upstream requests
    usgs_url = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/10/395/162"
    verdict, reason, entry = clf.classify_upstream_url(usgs_url)
    assert verdict == "ALLOWED"
    assert entry is not None
    assert entry["id"] == "usgs_imagery_only"

    ourairports_url = "https://davidmegginson.github.io/ourairports-data/runways.csv"
    verdict, _, entry = clf.classify_upstream_url(ourairports_url)
    assert verdict == "ALLOWED"
    assert entry["id"] == "ourairports"

    ourairports_url2 = "https://ourairports.com/data/runways.csv"
    verdict, _, entry = clf.classify_upstream_url(ourairports_url2)
    assert verdict == "ALLOWED"

    # Forbidden upstream requests
    verdict, reason, _ = clf.classify_upstream_url("https://api.cesium.com/v1/assets")
    assert verdict == "FORBIDDEN"

    verdict, reason, _ = clf.classify_upstream_url("https://tile.googleapis.com/v1/3dtiles/root.json")
    assert verdict == "FORBIDDEN"

    # Unknown upstream requests (e.g. wrong path prefix or unknown host)
    wrong_path_url = "https://basemap.nationalmap.gov/unauthorized_path/secret"
    verdict, reason, _ = clf.classify_upstream_url(wrong_path_url)
    assert verdict == "UNKNOWN"

    unknown_host = "https://images.example.com/tiles/0/0/0.jpg"
    verdict, reason, _ = clf.classify_upstream_url(unknown_host)
    assert verdict == "UNKNOWN"

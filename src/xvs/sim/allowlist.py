"""Runtime data policy and allowlist enforcement module for OpenXVS.

Loads docs/xvs/spec/data_allowlist.json and classifies browser and upstream URLs.
Conforms strictly to docs/xvs/WP01_SIM_RIG.md v1.1 §2.3 and Finding C6.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import urllib.parse


def match_host_pattern(host: str, pattern: str) -> bool:
    """Check if host matches pattern, supporting *.domain wildcards."""
    host = host.lower().strip()
    pattern = pattern.lower().strip()
    if host == pattern:
        return True
    if pattern.startswith("*."):
        suffix = pattern[2:]
        if host == suffix or host.endswith("." + suffix):
            return True
    return False


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALLOWLIST_PATH = REPO_ROOT / "configs" / "xvs" / "spec" / "data_allowlist.json"


class URLClassifier:
    """Classifies URLs as ALLOWED, FORBIDDEN, or UNKNOWN based on data_allowlist.json."""

    def __init__(self, allowlist_path: str | Path = DEFAULT_ALLOWLIST_PATH):
        self.allowlist_path = Path(allowlist_path)
        with open(self.allowlist_path, "r", encoding="utf-8") as f:
            self.policy: dict[str, Any] = json.load(f)

        self.browser_allowed_hosts: list[str] = [
            h.lower().strip() for h in self.policy.get("browser_allowed_hosts", [])
        ]
        self.upstream_allowed: list[dict[str, Any]] = self.policy.get("upstream_allowed", [])
        self.forbidden: list[dict[str, Any]] = self.policy.get("forbidden", [])

    def classify_browser_url(self, url: str) -> tuple[str, str | None]:
        """Classify a browser/page URL as ('ALLOWED' | 'FORBIDDEN' | 'UNKNOWN', reason).

        Blob and data URLs are allowed (used by Cesium Web Workers).
        """
        if not url:
            return "UNKNOWN", "Empty URL"

        url_clean = url.strip()
        if url_clean.startswith("blob:") or url_clean.startswith("data:"):
            return "ALLOWED", None

        parsed = urllib.parse.urlparse(url_clean)
        hostname = (parsed.hostname or "").lower().strip()

        if not hostname:
            return "UNKNOWN", "Missing hostname"

        # Check forbidden first
        for item in self.forbidden:
            for pattern in item.get("hosts", []):
                if match_host_pattern(hostname, pattern):
                    return "FORBIDDEN", item.get("reason", "Host explicitly forbidden")

        # Check browser allowed hosts
        for allowed in self.browser_allowed_hosts:
            if match_host_pattern(hostname, allowed):
                return "ALLOWED", None

        return "UNKNOWN", f"Host '{hostname}' is not in browser_allowed_hosts"

    def is_browser_url_allowed(self, url: str) -> bool:
        verdict, _ = self.classify_browser_url(url)
        return verdict == "ALLOWED"

    def classify_upstream_url(self, url: str) -> tuple[str, str | None, dict[str, Any] | None]:
        """Classify an upstream proxy request as ('ALLOWED' | 'FORBIDDEN' | 'UNKNOWN', reason, entry)."""
        if not url:
            return "UNKNOWN", "Empty URL", None

        url_clean = url.strip()
        parsed = urllib.parse.urlparse(url_clean)
        hostname = (parsed.hostname or "").lower().strip()
        path = parsed.path or "/"

        if not hostname:
            return "UNKNOWN", "Missing hostname", None

        # Check forbidden first
        for item in self.forbidden:
            for pattern in item.get("hosts", []):
                if match_host_pattern(hostname, pattern):
                    return "FORBIDDEN", item.get("reason", "Host explicitly forbidden"), None

        # Check upstream allowed
        for entry in self.upstream_allowed:
            hosts = entry.get("host", [])
            if isinstance(hosts, str):
                hosts = [hosts]
            prefixes = entry.get("path_prefix", [])
            if isinstance(prefixes, str):
                prefixes = [prefixes]

            host_matched = any(match_host_pattern(hostname, h) for h in hosts)
            if host_matched:
                prefix_matched = any(path.startswith(p) for p in prefixes)
                if prefix_matched:
                    return "ALLOWED", None, entry

        return "UNKNOWN", f"URL '{url}' does not match any upstream_allowed rule", None

    def is_upstream_url_allowed(self, url: str) -> bool:
        verdict, _, _ = self.classify_upstream_url(url)
        return verdict == "ALLOWED"


# Global default instance
_DEFAULT_CLASSIFIER: URLClassifier | None = None


def get_default_classifier() -> URLClassifier:
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = URLClassifier()
    return _DEFAULT_CLASSIFIER


def setup_playwright_route_guard(context: Any, classifier: URLClassifier | None = None, server: Any = None) -> None:
    """Enforce allowlist routing on Playwright browser context.

    Conforms to docs/xvs/WP01_SIM_RIG.md v1.1 §2.3:
      Playwright context.route("**/*") aborts any request that is not allowed.
      blob: and data: URLs are allowed; Cesium starts its workers from blob:.
    """
    from datetime import datetime, timezone

    clf = classifier or get_default_classifier()

    def _route_handler(route: Any, request: Any) -> None:
        url = request.url
        verdict, reason = clf.classify_browser_url(url)
        if verdict == "ALLOWED":
            route.continue_()
        else:
            if server is not None and hasattr(server, "log_violation"):
                server.log_violation({
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "url": url,
                    "error": f"Browser request aborted by allowlist: {reason}",
                })
            route.abort("blockedbyclient")

    context.route("**/*", _route_handler)

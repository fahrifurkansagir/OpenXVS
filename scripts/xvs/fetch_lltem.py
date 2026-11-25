"""WP-02 G3b: download LLTEM v1.0 under decision D17 (docs/xvs/batches/BATCH_B2.md) and the owner's choice of
2026-09-24 (option 1): README and encounter-information CSV in full, plus the FIRST part of the binary trajectory
file through an HTTP Range request (the full file is 10 GB, above the D17 limit of 5 GB).

Rules enforced here (no silent fallback):
  - every URL, including a redirect target, must be ALLOWED by docs/xvs/spec/data_allowlist.json (entry lltem_v1);
  - the Range request must be answered with 206 Partial Content and the requested byte count, otherwise abort
    before reading the body (a 200 would be the whole 10 GB file);
  - the total downloaded size must stay <= 5 GB;
  - the README must contain the CC BY 4.0 licence sentence (quoted in the allowlist entry).
Files go to data/cache/lltem/ (gitignored). The sidecar data/reference/lltem/lltem_v1.meta.json records URL,
final URL, HTTP status, byte range, bytes, SHA-256 and UTC time of every file; it is committed.
usage: python scripts/xvs/fetch_lltem.py [--dat-bytes 104857600]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from xvs.sim.allowlist import URLClassifier  # noqa: E402

SOURCE_PAGE = "https://www.ll.mit.edu/r-d/datasets/unmanned-aircraft-terminal-area-encounters"
BASE = "http://archive.ll.mit.edu/datasets/lltem/"
FILES = [("README.txt", None), ("terminal_encounter_info_20200630.csv", None), ("terminal_encounters_20200630.dat", "prefix")]
LICENCE_SENTENCE = "This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0)."
MAX_TOTAL_BYTES = 5 * 10**9
USER_AGENT = "OpenXVS-research/0.1"
CACHE = REPO_ROOT / "data" / "cache" / "lltem"
SIDECAR = REPO_ROOT / "data" / "reference" / "lltem" / "lltem_v1.meta.json"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401 - urllib signature
        return None  # redirects are followed by hand, after an allowlist check


def fetch(url: str, dst: Path, clf: URLClassifier, byte_range: tuple[int, int] | None, budget: int) -> dict:
    opener = urllib.request.build_opener(_NoRedirect)
    hops = []
    for _ in range(5):
        verdict, reason, entry = clf.classify_upstream_url(url)
        if verdict != "ALLOWED" or entry.get("id") != "lltem_v1":
            raise RuntimeError(f"{url}: not allowed by the allowlist entry lltem_v1 ({verdict}: {reason})")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if byte_range is not None:
            req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
        try:
            resp = opener.open(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                hops.append({"url": url, "status": e.code})
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            raise
        break
    else:
        raise RuntimeError(f"too many redirects for {url}")
    status = resp.status
    length = int(resp.headers.get("Content-Length", "-1"))
    if byte_range is not None:
        want = byte_range[1] - byte_range[0] + 1
        if status != 206 or length != want:
            resp.close()
            raise RuntimeError(f"{url}: Range request answered with status {status}, Content-Length {length} (need 206, {want}); aborted")
    if length < 0 or length > budget:
        resp.close()
        raise RuntimeError(f"{url}: Content-Length {length} exceeds the remaining budget {budget} (D17: total <= 5 GB)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    h = hashlib.sha256()
    n = 0
    with open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
            if n > length:
                raise RuntimeError(f"{url}: more bytes than announced")
    resp.close()
    if n != length:
        raise RuntimeError(f"{url}: got {n} bytes, announced {length}")
    tmp.replace(dst)
    return {"url": hops[0]["url"] if hops else url, "redirects": hops, "final_url": url, "http_status": status,
            "content_range": resp.headers.get("Content-Range"), "byte_range": list(byte_range) if byte_range else None,
            "bytes": n, "sha256": h.hexdigest(), "path": dst.relative_to(REPO_ROOT).as_posix(),
            "downloaded_utc": datetime.now(timezone.utc).isoformat()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Download LLTEM v1.0 (D17, owner option 1)")
    ap.add_argument("--dat-bytes", type=int, default=100 * 2**20, help="length of the .dat prefix (bytes)")
    a = ap.parse_args()
    clf = URLClassifier()
    remaining = MAX_TOTAL_BYTES
    rows = []
    for name, mode in FILES:
        rng = (0, a.dat_bytes - 1) if mode == "prefix" else None
        row = fetch(BASE + name, CACHE / name, clf, rng, remaining)
        row["file"] = name
        row["complete_file"] = mode is None
        remaining -= row["bytes"]
        rows.append(row)
        print(f"  {name}: {row['bytes']:,} bytes, status {row['http_status']}, sha256 {row['sha256']}", flush=True)
    readme = (CACHE / "README.txt").read_text(encoding="utf-8", errors="replace")
    if LICENCE_SENTENCE not in readme:
        raise RuntimeError("README.txt does not contain the CC BY 4.0 licence sentence: D17 not met, stop G3b")
    meta = {
        "schema": "xvs.reference_sidecar/1", "id": "lltem_v1",
        "dataset": "MIT LL Terminal Encounter Model (LLTEM) V1.0 - Terminal Airspace Encounter Set V1.0 (June 30, 2020)",
        "source_page": SOURCE_PAGE, "host": "archive.ll.mit.edu", "path_prefix": "/datasets/lltem/",
        "license": "CC BY 4.0", "license_quote_readme": LICENCE_SENTENCE,
        "attribution": "MIT Lincoln Laboratory, Terminal Airspace Encounter Set V1.0 (LLTEM V1.0), June 30, 2020. "
                       "(c) 2020 Massachusetts Institute of Technology. Licensed under CC BY 4.0.",
        "decision": "D17 (docs/xvs/batches/BATCH_B2.md); owner option 1 of 2026-09-24: README and CSV in full, "
                    f"the first {a.dat_bytes:,} bytes of terminal_encounters_20200630.dat (Range request)",
        "files": rows, "total_bytes": sum(r["bytes"] for r in rows),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"sidecar {SIDECAR.relative_to(REPO_ROOT).as_posix()}: total {meta['total_bytes']:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

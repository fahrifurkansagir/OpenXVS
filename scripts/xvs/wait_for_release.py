"""Wait for RELEASE.json to appear per GEMINI_G-B13 P0-3."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# D19: the data root is XVS_DATA_ROOT or the repo's output folder (no machine path; merge of G-B13)
RELEASE_PATH = Path(os.environ.get("XVS_DATA_ROOT") or (REPO_ROOT / "output")) / "xvs" / "wp02" / "g3a_v01" / "RELEASE.json"
CHECK_INTERVAL_S = 300  # 5 minutes
MAX_WAIT_S = 6 * 3600   # 6 hours


def main() -> int:
    start_time = time.time()
    iteration = 0
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}] Starting P0-3 wait for {RELEASE_PATH}")
    print(f"Check interval: {CHECK_INTERVAL_S}s (5 min), max wait: {MAX_WAIT_S}s (6 h)")
    sys.stdout.flush()

    while True:
        iteration += 1
        elapsed = time.time() - start_time
        if RELEASE_PATH.is_file():
            size = RELEASE_PATH.stat().st_size
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}] Found RELEASE.json! Size: {size} bytes. Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            sys.stdout.flush()
            return 0

        if elapsed >= MAX_WAIT_S:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}] Timeout reached: 6 hours elapsed without RELEASE.json.")
            sys.stdout.flush()
            return 2

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}] Check #{iteration}: RELEASE.json not found yet. Elapsed: {elapsed/60:.1f} min. Sleeping {CHECK_INTERVAL_S}s...")
        sys.stdout.flush()
        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())

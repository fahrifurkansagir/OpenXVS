"""Generate ground truth labels for all 900 encounters under output/xvs/wp05/g0/truth_labels.json.

Conforms to GEMINI_G-B8.md A6 / F-B6-2.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import argparse
from xvs.paths import get_data_root
from xvs.sim.trajectory import t1_cruise_state
from xvs.traffic.encounter import EncounterSamplerV1
from xvs.tracking.evaluate import generate_truth_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ground truth labels for encounters")
    parser.add_argument("--data-root", default=None, help="Root path for data directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    sampler = EncounterSamplerV1()
    encs = sampler.generate(t1_cruise_state)
    out_path = data_root / "xvs" / "wp05" / "g0" / "truth_labels.json"
    p = generate_truth_labels(encs, out_path, max_workers=args.workers)
    print(f"Generated truth labels at {p}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.causal.gate import causal_gate_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-root", default="data/session_normalized")
    p.add_argument("--out-root", default="data/causally_gated_normalized")
    args = p.parse_args()
    report = causal_gate_root(args.in_root, args.out_root)
    print(report["status"])


if __name__ == "__main__":
    main()

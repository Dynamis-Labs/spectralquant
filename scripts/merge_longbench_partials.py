#!/usr/bin/env python3
"""Merge per-method LongBench partial JSON shards into a single result file.

This script is the recovery path for runs where one method (e.g. turboquant)
times out after earlier methods (fp16, spectralquant_v2) already finished.

Inputs:

  --partial-dir  Path to the directory holding ``method__<name>.json`` shards
                 produced by ``run_longbench.py`` after each method completes.
  --run-id       Run identifier to use in the merged payload.
  --model        Model id (e.g. ``Qwen/Qwen2.5-7B``).
  --avg-bits     Average-bits target for the quantized methods.
  --output       Path to write the merged JSON to.
  --paper-valid  If ALL three methods are present, mark the merged JSON
                 ``paper_valid: true`` (default: ``false``). Without this
                 flag the merged JSON is always tagged
                 ``paper_valid: false, partial: true`` so it cannot be
                 mistaken for canonical evidence.

Usage:

    python scripts/merge_longbench_partials.py \\
        --partial-dir /tmp/lb_partial/partial \\
        --run-id longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant \\
        --model Qwen/Qwen2.5-7B \\
        --avg-bits 3 \\
        --output results/v3/modal/longbench__merged.json

The merged file does NOT go through the canonical schema validator; it is
explicitly a recovery artifact. Use it for traceability and qualitative
comparison only. Any paper sentence cited from this artifact MUST disclose
that it is a merger of per-method shards from a partially-killed run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partial-dir", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--avg-bits", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--paper-valid", action="store_true",
                   help="Allow paper_valid=true if all 3 methods present")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    partial_dir: Path = args.partial_dir
    if not partial_dir.is_dir():
        print(f"[merge_longbench_partials] ERROR: not a directory: "
              f"{partial_dir}", file=sys.stderr)
        return 2

    methods: Dict[str, Dict[str, Any]] = {}
    seen: List[str] = []
    for shard in sorted(partial_dir.glob("method__*.json")):
        try:
            with shard.open("r") as fh:
                blob = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[merge_longbench_partials] WARN: skip {shard}: {exc}",
                  file=sys.stderr)
            continue
        m = blob.get("method")
        rec = blob.get("record")
        if not isinstance(m, str) or not isinstance(rec, dict):
            print(f"[merge_longbench_partials] WARN: malformed shard "
                  f"{shard}: missing method/record", file=sys.stderr)
            continue
        methods[m] = rec
        seen.append(m)

    if not methods:
        print("[merge_longbench_partials] ERROR: no usable shards found",
              file=sys.stderr)
        return 3

    have_three = {"fp16", "spectralquant_v2", "turboquant"}.issubset(methods)
    paper_valid = bool(args.paper_valid) and have_three

    payload = {
        "schema_version": "1",
        "family": "longbench",
        "mode": "full" if have_three else "full_partial",
        "model": {"id": args.model},
        "run_id": args.run_id,
        "repo": "niashwin/spectralquant-full",
        "timestamp": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "methods": methods,
        "paper_valid": bool(paper_valid),
        "partial": not have_three,
        "data": {
            "merged_from": "method partial shards",
            "shards_seen": list(seen),
            "shard_dir": str(partial_dir),
            "avg_bits_target": int(args.avg_bits),
        },
        "caveats": [
            "Merged from partial per-method JSON shards via "
            "scripts/merge_longbench_partials.py. NOT validated against the "
            "canonical longbench schema. Paper-valid only if --paper-valid "
            "was passed AND all three methods (fp16, spectralquant_v2, "
            "turboquant) were present.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, args.output)
    print(f"[merge_longbench_partials] wrote {args.output} "
          f"(methods={sorted(seen)}, paper_valid={paper_valid})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

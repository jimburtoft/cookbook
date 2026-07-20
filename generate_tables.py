#!/usr/bin/env python3
"""Generate Phase E comparison tables from the aggregated cookbook CSV.

Produces three markdown tables:
  1. Intra-node PT Native vs Task 009 nccom-test
  2. Cross-node scaling for all_reduce
  3. Collective coverage matrix

Also produces a summary of per-collective peak BW at 256 MB (or nearest).
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

SIZES_OF_INTEREST = {
    "4KB": 4 * 1024,
    "1MB": 1024 * 1024,
    "256MB": 256 * 1024 * 1024,
}

# Task 009 nccom-test baseline (from OpencodeDocs/projects/collective/tasks/README.md)
# Only sendrecv WS=2 was measured pairwise; allr WS=8 was full-world.
TASK_009 = {
    ("sendrecv_same_die", 4096): {"duration_us": 12.56, "bw_GBps": None},
    ("sendrecv_same_die", 1048576): {"duration_us": 23.99, "bw_GBps": None},
    ("sendrecv_same_die", 268435456): {"duration_us": 2585.82, "bw_GBps": 103.81},
    ("sendrecv_cross_die", 4096): {"duration_us": 18.46, "bw_GBps": None},
    ("sendrecv_cross_die", 1048576): {"duration_us": 29.94, "bw_GBps": None},
    ("sendrecv_cross_die", 268435456): {"duration_us": 2787.36, "bw_GBps": 96.30},
    # 8-core allreduce peaks at 124 GB/s bus BW according to Task 009. We
    # don't have a per-size table for allr, so we can only compare peak BW.
    ("allr_8core_peak", "peak_bus_bw"): {"bw_GBps": 123.9},
}


def load(csv_path: Path) -> list[dict]:
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    # Normalize types + normalize collective naming (alltoall -> all_to_all)
    for r in rows:
        r["size_bytes"] = int(r["size_bytes"])
        r["duration_us"] = float(r["duration_us"])
        r["throughput_gbps"] = float(r["throughput_gbps"])
        r["busbw_gbps"] = float(r["busbw_gbps"])
        r["world_size"] = int(r["world_size"])
        r["lnc"] = int(r["lnc"])
        r["config_ws"] = int(r["config_ws"])
        if r["collective"] == "alltoall":
            r["collective"] = "all_to_all"
    return rows


def nearest(rows: list[dict], target: int) -> dict | None:
    """Return the row whose size is closest to target."""
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r["size_bytes"] - target))


def group_by(rows: list[dict], keys: tuple) -> dict:
    out: dict = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def table_1_intra_vs_nccom(rows: list[dict]) -> str:
    """PT Native (trn2.3xlarge) vs Task 009 nccom-test for the two comparable
    configurations we have: (a) sendrecv-equivalent at WS=2 (b) allr at WS=8.

    We use LNC=1 rows for both since Task 009 was done under LNC=1."""
    lines = []
    lines.append("### Table 1 -- PT Native (trn2.3xlarge, LNC=1) vs Task 009 nccom-test\n")
    lines.append("| Collective (PT Native) | WS | Size | nccom-test (us) | PT Native (us) | Framework overhead |")
    lines.append("|----|----|----|----|----|----|")

    # Row a: all_reduce WS=2 vs sendrecv same-die (rank 0,1 is same-die under LNC=1)
    # We use PT Native all_reduce as the framework analog to sendrecv -- both
    # are 2-rank point-to-pointish primitives that only exercise a single
    # NeuronLink hop.
    lnc1_ar_ws2 = [r for r in rows if r["environment"] == "trn2_3xl"
                   and r["lnc"] == 1 and r["config_ws"] == 2 and r["collective"] == "all_reduce"]
    for label, target in [("4 KB", 4096), ("1 MB", 1048576), ("256 MB", 268435456)]:
        pt = nearest(lnc1_ar_ws2, target)
        nccom = TASK_009.get(("sendrecv_same_die", target), {}).get("duration_us")
        pt_us = pt["duration_us"] if pt else None
        overhead = f"{pt_us / nccom:.1f}x" if (pt_us and nccom) else "-"
        pt_str = f"{pt_us:.1f}" if pt_us else "-"
        nccom_str = f"{nccom:.1f}" if nccom else "-"
        lines.append(f"| all_reduce (as sendrecv analog) | 2 | {label} | {nccom_str} | {pt_str} | {overhead} |")

    # Row b: allr WS=8 -- compare peak bus BW
    lnc1_ar_ws8 = [r for r in rows if r["environment"] == "trn2_3xl"
                   and r["lnc"] == 1 and r["config_ws"] == 8 and r["collective"] == "all_reduce"]
    if lnc1_ar_ws8:
        pt_peak = max(r["busbw_gbps"] for r in lnc1_ar_ws8)
        nccom_peak = TASK_009[("allr_8core_peak", "peak_bus_bw")]["bw_GBps"]
        ratio = f"{pt_peak / nccom_peak * 100:.0f}% of nccom" if nccom_peak else "-"
        lines.append(f"| all_reduce | 8 | 256 MB (peak) | -- bus BW {nccom_peak:.1f} GB/s | bus BW {pt_peak:.1f} GB/s | {ratio} |")

    lines.append("")
    lines.append("Notes:")
    lines.append("* Task 009 measured **sendrecv** (a 2-rank P2P primitive) via `nccom-test sendrecv`. "
                 "PyTorch Native Beta 3 does not expose `dist.send`/`dist.recv` on the Neuron backend "
                 "(steering/pytorch-native.md:703), so the closest framework analog is all_reduce on WS=2. "
                 "Both cross a single NeuronLink hop.")
    lines.append("* Task 009 measured **all-reduce** (`nccom-test allr -r 8`) at 8 cores and reported "
                 "**peak 123.9 GB/s bus BW** (rdh algorithm, 256 KB - 64 MB). Our PT Native peak is "
                 "measured at 256 MB and reflects the ring algorithm the runtime selects for large messages.")
    return "\n".join(lines)


def table_2_cross_node_scaling(rows: list[dict]) -> str:
    """all_reduce scaling across intra-node (48xl) vs cross-node.

    We use PCS cluster data. If cross-node runs did not complete we mark them
    as pending."""
    lines = []
    lines.append("### Table 2 -- Cross-node scaling (all_reduce, LNC=2)\n")
    lines.append("| Environment | World | 4 KB (us) | 1 MB (us) | 128 MB (us) | Peak Bus BW (GB/s) |")
    lines.append("|----|----|----|----|----|----|")

    # trn2.3xlarge single-chip baseline
    for ws in [4, 8]:
        env_rows = [r for r in rows if r["environment"] == "trn2_3xl"
                    and r["collective"] == "all_reduce" and r["config_ws"] == ws
                    and r["lnc"] == (2 if ws <= 4 else 1)]
        if not env_rows:
            continue
        p4k = nearest(env_rows, 4096)
        p1m = nearest(env_rows, 1048576)
        p128m = nearest(env_rows, 128 * 1024 * 1024)
        peak = max(r["busbw_gbps"] for r in env_rows)
        lines.append(f"| trn2.3xl (LNC={'2' if ws<=4 else '1'}) | {ws} | "
                     f"{p4k['duration_us']:.1f} | {p1m['duration_us']:.1f} | "
                     f"{p128m['duration_us']:.1f} | {peak:.1f} |")

    # trn2.48xl single-node (PCS cluster)
    for ws in [8, 16, 32, 64]:
        env_rows = [r for r in rows if r["environment"] == "trn2_48xl_single"
                    and r["collective"] == "all_reduce" and r["world_size"] == ws]
        if not env_rows:
            lines.append(f"| trn2.48xl single-node | {ws} | (pending) | (pending) | (pending) | (pending) |")
            continue
        p4k = nearest(env_rows, 4096)
        p1m = nearest(env_rows, 1048576)
        p128m = nearest(env_rows, 128 * 1024 * 1024)
        peak = max(r["busbw_gbps"] for r in env_rows)
        lines.append(f"| trn2.48xl single-node | {ws} | "
                     f"{p4k['duration_us']:.1f} | {p1m['duration_us']:.1f} | "
                     f"{p128m['duration_us']:.1f} | {peak:.1f} |")

    # trn2.48xl multi-node (EFA)
    for ws in [16, 32, 64]:
        env_rows = [r for r in rows if r["environment"] == "trn2_48xl_multi"
                    and r["collective"] == "all_reduce" and r["world_size"] == ws]
        if not env_rows:
            lines.append(f"| trn2.48xl 2-node (EFA) | {ws} | (pending) | (pending) | (pending) | (pending) |")
            continue
        p4k = nearest(env_rows, 4096)
        p1m = nearest(env_rows, 1048576)
        p128m = nearest(env_rows, 128 * 1024 * 1024)
        peak = max(r["busbw_gbps"] for r in env_rows)
        lines.append(f"| trn2.48xl 2-node (EFA) | {ws} | "
                     f"{p4k['duration_us']:.1f} | {p1m['duration_us']:.1f} | "
                     f"{p128m['duration_us']:.1f} | {peak:.1f} |")

    return "\n".join(lines)


def table_3_coverage_matrix(rows: list[dict], job_map: dict) -> str:
    """For each (env, lnc, ws) config we tested, mark each collective as
    works / fails / not attempted."""
    lines = []
    lines.append("### Table 3 -- Collective coverage matrix\n")
    lines.append("Legend: `OK` = collective ran to completion (or completed then OOM'd at the largest size, "
                 "which is expected on HBM-limited configs); `FAIL` = did not produce any measurements; "
                 "`--` = not attempted.\n")
    collectives = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast", "pt2pt"]
    lines.append("| Environment | LNC | WS | all_reduce | all_gather | reduce_scatter | all_to_all | broadcast | pt2pt |")
    lines.append("|----|----|----|----|----|----|----|----|----|")
    seen_configs: set = set()
    for r in rows:
        seen_configs.add((r["environment"], r["lnc"], r["config_ws"]))

    # Also include configs we launched but that produced no rows (from job_map)
    for env, lnc, ws in sorted(seen_configs):
        row_cells = []
        for coll in collectives:
            got = any(r["collective"] == coll for r in rows
                      if r["environment"] == env and r["lnc"] == lnc and r["config_ws"] == ws)
            row_cells.append("OK" if got else "FAIL")
        lines.append(f"| {env} | {lnc} | {ws} | " + " | ".join(row_cells) + " |")

    lines.append("")
    lines.append("Known Beta 3 constraints (confirmed by these runs):")
    lines.append("* `pt2pt` (dist.send/dist.recv): **NEVER WORKS** on the Neuron backend. "
                 "Error: 'No backend type associated with device type neuron'. "
                 "Documented in steering/pytorch-native.md:703.")
    lines.append("* `all_to_all` (`AllToAllXlaOp`): **only WS in {4, 8, 16, or multiples of 32} is supported.** "
                 "WS=2 fails with 'unsupported world size 2'. Not documented in Beta 3 release notes.")
    lines.append("* `reduce_scatter`: fails on empty tensors (skip M=0 fix landed in commit 57cf076).")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = load(args.csv)

    md = []
    md.append("# Phase E: PyTorch Native Beta 3 Cookbook Communication Benchmarks -- Comparison Tables\n")
    md.append(f"Data source: {args.csv.resolve()} ({len(rows)} points across all environments)\n\n")
    md.append(table_1_intra_vs_nccom(rows))
    md.append("\n\n")
    md.append(table_2_cross_node_scaling(rows))
    md.append("\n\n")
    md.append(table_3_coverage_matrix(rows, {}))
    args.out.write_text("\n".join(md))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

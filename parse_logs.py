#!/usr/bin/env python3
"""Parse cookbook communication benchmark output logs into a single CSV.

Handles both the trn2.3xlarge Phase C layout:
    logs/phase_c/lnc{1,2}_ws{2,4,8}/<collective>.log
and the PCS cluster Phase D layout:
    logs/{single,multi}_node_<jobid>.out

Each log has 5-column rows:
    <size_bytes>  <numelxeltsize>  <duration_us>  <throughput>  <busbw>

where duration is in microseconds when --raw was passed and the last two are
GBps because --bw-unit=GBps.

Output rows: environment, collective, lnc, world_size, size_bytes, duration_us,
throughput_gbps, busbw_gbps, source_log
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterator

# 5-column line like:
#   4194304    1048576x4    172.352    48.671    36.504
DATA_ROW = re.compile(
    r"^(?P<size>\d+)\s+"
    r"(?P<desc>\d+x\d+)\s+"
    r"(?P<dur_us>[0-9.]+)\s+"
    r"(?P<tput>[0-9.]+)\s+"
    r"(?P<busbw>[0-9.]+)\s*$"
)

# The header of a per-collective section:
#   ---- Performance of all_reduce on 4 devices ---------
HEADER_ROW = re.compile(
    r"----\s*Performance of\s+(?P<coll>\w+)\s+on\s+(?P<devices>\d+)\s+devices"
)


def parse_log(path: Path) -> Iterator[dict]:
    """Yield one dict per data row in the log."""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    current_coll: str | None = None
    current_ws: int | None = None
    for line in lines:
        h = HEADER_ROW.search(line)
        if h:
            current_coll = h.group("coll")
            current_ws = int(h.group("devices"))
            continue
        d = DATA_ROW.match(line)
        if not d or current_coll is None:
            continue
        yield {
            "collective": current_coll,
            "world_size": current_ws,
            "size_bytes": int(d.group("size")),
            "duration_us": float(d.group("dur_us")),
            "throughput_gbps": float(d.group("tput")),
            "busbw_gbps": float(d.group("busbw")),
            "source_log": str(path),
        }


def parse_phase_c(root: Path) -> Iterator[dict]:
    """Walk trn2.3xlarge phase_c/<lnc?_ws?>/<coll>.log files."""
    for cfg_dir in sorted(root.glob("lnc*_ws*")):
        m = re.match(r"lnc(?P<lnc>\d+)_ws(?P<ws>\d+)", cfg_dir.name)
        if not m:
            continue
        lnc = int(m.group("lnc"))
        for log in sorted(cfg_dir.glob("*.log")):
            for row in parse_log(log):
                row["environment"] = "trn2_3xl"
                row["lnc"] = lnc
                # We trust the header's device count; keep the config's ws too.
                row["config_ws"] = int(m.group("ws"))
                yield row


def parse_phase_d(root: Path, job_map: Path | None = None) -> Iterator[dict]:
    """Walk PCS cluster {single,multi}_node_<jobid>.out files.

    job_map, if provided, associates each job id with metadata (nodes, WS,
    collective, LNC). If absent we fall back to what we can infer from the
    log header alone.
    """
    meta_by_jobid: dict[str, dict] = {}
    if job_map and job_map.exists():
        for line in job_map.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            jobid = parts[0]
            kv = {}
            kind = parts[1] if len(parts) > 1 else "unknown"
            for tok in parts[2:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            kv["_kind"] = kind
            meta_by_jobid[jobid] = kv

    for log in sorted(root.glob("*_node_*.out")):
        m = re.search(r"_node_(\d+)\.out$", log.name)
        if not m:
            continue
        jobid = m.group(1)
        meta = meta_by_jobid.get(jobid, {})
        env_type = "trn2_48xl_single" if log.name.startswith("single") else "trn2_48xl_multi"
        for row in parse_log(log):
            row["environment"] = env_type
            row["lnc"] = int(meta.get("lnc", 2))
            row["jobid"] = jobid
            row["config_ws"] = int(meta.get("ws", meta.get("nproc", row["world_size"])))
            yield row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase-c", type=Path, help="Path to phase_c/ directory (trn2.3xlarge logs)")
    p.add_argument("--phase-d", type=Path, help="Path to logs/ directory (PCS cluster)")
    p.add_argument("--job-map", type=Path, help="Path to Phase D job_map.txt")
    p.add_argument("--out", type=Path, default=Path("aggregated.csv"))
    args = p.parse_args()

    fieldnames = [
        "environment", "collective", "lnc", "world_size", "config_ws",
        "size_bytes", "duration_us", "throughput_gbps", "busbw_gbps",
        "jobid", "source_log",
    ]
    n = 0
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if args.phase_c:
            for r in parse_phase_c(args.phase_c):
                w.writerow(r)
                n += 1
        if args.phase_d:
            for r in parse_phase_d(args.phase_d, args.job_map):
                w.writerow(r)
                n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

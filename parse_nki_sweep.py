#!/usr/bin/env python3
"""Parse the framework-vs-NKI sweep output (from scripts/sweep_nki_vs_framework.py)
into a CSV compatible with the Phase E aggregator format.

The sweep script emits lines of the form:
  size_bytes per_rank_elem fw_us nki_us fw_tput_GBps nki_tput_GBps fw_busbw_GBps nki_busbw_GBps speedup

We extract those into rows keyed by (collective, world_size, path) where path is
either 'framework' or 'nki'.
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

# Header line printed by sweep_nki_vs_framework at start of each run
HDR = re.compile(r"===\s*coll=(?P<coll>\w+)\s+WS=(?P<ws>\d+)\s+LNC=(?P<lnc>\d+)\s+NKI_LNC=(?P<nki_lnc>\d+)")

# Data rows are 9 whitespace-separated numeric tokens
DATA = re.compile(
    r"^\s*(?P<size>\d+)\s+"
    r"(?P<numel>\d+)\s+"
    r"(?P<fw_us>[\d.]+)\s+"
    r"(?P<nki_us>[\d.]+)\s+"
    r"(?P<fw_tput>[\d.]+)\s+"
    r"(?P<nki_tput>[\d.]+)\s+"
    r"(?P<fw_busbw>[\d.]+)\s+"
    r"(?P<nki_busbw>[\d.]+)\s+"
    r"(?P<speedup>[\d.]+)\s*$"
)


def parse_file(path: Path):
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    coll = ws = lnc = nki_lnc = None
    rows = []
    for line in lines:
        h = HDR.search(line)
        if h:
            coll = h.group("coll")
            ws = int(h.group("ws"))
            lnc = int(h.group("lnc"))
            nki_lnc = int(h.group("nki_lnc"))
            continue
        d = DATA.match(line)
        if d and coll:
            rows.append({
                "collective": coll,
                "world_size": ws,
                "lnc": lnc,
                "nki_lnc": nki_lnc,
                "size_bytes": int(d.group("size")),
                "framework_us": float(d.group("fw_us")),
                "nki_us": float(d.group("nki_us")),
                "framework_tput_GBps": float(d.group("fw_tput")),
                "nki_tput_GBps": float(d.group("nki_tput")),
                "framework_busbw_GBps": float(d.group("fw_busbw")),
                "nki_busbw_GBps": float(d.group("nki_busbw")),
                "speedup": float(d.group("speedup")),
                "source_log": str(path),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    all_rows = []
    for f in sorted(args.logs_dir.glob("*.out")):
        all_rows.extend(parse_file(f))

    if not all_rows:
        print("no data rows found")
        return

    fields = list(all_rows[0].keys())
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

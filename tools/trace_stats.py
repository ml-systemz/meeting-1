#!/usr/bin/env python3
"""Summarise a GPU trace: how much time was spent computing, and how much in gaps.

Accepts an Nsight Systems report (.nsys-rep / .qdrep / .sqlite / a CSV already
produced by `nsys stats`) or a Chrome trace written by tools/torchprof.py.

The number to care about is the bubble fraction: wall time on the GPU timeline
that no kernel occupied. Fusing kernels is an attack on exactly that number.
"""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

TINY_US = 10.0


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def nsys_to_csv(path):
    """Ask nsys for a per-kernel trace in CSV form."""
    if not shutil.which("nsys"):
        die("nsys is not on PATH, so a .nsys-rep cannot be read here. "
            "Export it where nsys exists, or profile with tools/torchprof.py instead.")
    cmd = ["nsys", "stats", "--report", "cuda_gpu_trace", "--format", "csv",
           "--force-export", "true", "--output", "-", path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        die(f"nsys stats failed:\n{p.stderr.strip()[:800]}")
    return p.stdout


def pick(header, *wanted):
    """Find a column whose name contains all of `wanted`, case-insensitively."""
    for i, h in enumerate(header):
        low = h.lower()
        if all(w in low for w in wanted):
            return i
    return None


def parse_nsys_csv(text):
    rows = list(csv.reader(io.StringIO(text)))
    header = None
    for r in rows:
        if r and pick(r, "start") is not None and pick(r, "dur") is not None:
            header = r
            break
    if header is None:
        die("could not find Start/Duration columns in the nsys CSV")

    i_start = pick(header, "start")
    i_dur = pick(header, "dur")
    i_name = pick(header, "name") or pick(header, "kernel")
    if i_name is None:
        die("could not find a Name column in the nsys CSV")

    unit = 1e-3 if "ns" in header[i_start].lower() else 1.0
    out = []
    seen_header = False
    for r in rows:
        if r == header:
            seen_header = True
            continue
        if not seen_header or len(r) <= max(i_start, i_dur, i_name):
            continue
        try:
            start = float(r[i_start].replace(",", "")) * unit
            dur = float(r[i_dur].replace(",", "")) * unit
        except ValueError:
            continue
        name = r[i_name].strip()
        if name:
            out.append((start, dur, name))
    return out


def parse_chrome(path):
    with open(path) as f:
        blob = json.load(f)
    events = blob.get("traceEvents", blob if isinstance(blob, list) else [])
    out = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") != "kernel":
            continue
        try:
            out.append((float(e["ts"]), float(e.get("dur", 0.0)), e.get("name", "?")))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load(path):
    if not os.path.exists(path):
        die(f"no such file: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nsys-rep", ".qdrep", ".sqlite"):
        return parse_nsys_csv(nsys_to_csv(path))
    if ext == ".csv":
        with open(path) as f:
            return parse_nsys_csv(f.read())
    if ext in (".json", ".gz"):
        if ext == ".gz":
            die("gunzip the trace first")
        return parse_chrome(path)
    die(f"unrecognised trace type: {ext}")


def merge(intervals):
    """Union of [start, end) intervals, so overlapping streams are not double-counted."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def fmt(us):
    if us >= 1000:
        return f"{us / 1000:.3f} ms"
    return f"{us:.1f} us"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="path to .nsys-rep, .csv or a chrome trace .json")
    ap.add_argument("--top", type=int, default=10, help="how many kernels to list (default 10)")
    args = ap.parse_args()

    kernels = load(args.trace)
    if not kernels:
        die("no GPU kernels found in that trace")

    spans = [(s, s + d) for s, d, _ in kernels]
    busy_spans = merge(spans)
    busy = sum(e - s for s, e in busy_spans)
    first = min(s for s, _ in spans)
    last = max(e for _, e in spans)
    wall = last - first
    gaps = wall - busy
    tiny = [k for k in kernels if k[1] < TINY_US]

    by_name = defaultdict(lambda: [0, 0.0])
    for _, d, n in kernels:
        by_name[n][0] += 1
        by_name[n][1] += d

    print()
    print(f"  trace            {os.path.basename(args.trace)}")
    print(f"  kernel launches  {len(kernels)}")
    print(f"  distinct kernels {len(by_name)}")
    print()
    print(f"  timeline span    {fmt(wall)}")
    print(f"  gpu busy         {fmt(busy)}   ({busy / wall * 100:5.1f}%)")
    print(f"  gaps             {fmt(gaps)}   ({gaps / wall * 100:5.1f}%)  <- the bubble")
    print()
    if tiny:
        tiny_time = sum(d for _, d, _ in tiny)
        print(f"  kernels < {TINY_US:.0f}us    {len(tiny)} launches, {fmt(tiny_time)} of work")
        print(f"                   these are the fusion candidates")
        print()

    print(f"  top {args.top} by total time")
    ranked = sorted(by_name.items(), key=lambda kv: kv[1][1], reverse=True)[:args.top]
    width = max((len(n) for n, _ in ranked), default=10)
    width = min(width, 58)
    for name, (count, total) in ranked:
        label = name if len(name) <= width else name[:width - 1] + "…"
        print(f"    {label:<{width}}  {count:>5}x  {fmt(total):>11}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

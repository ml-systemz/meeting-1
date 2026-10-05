#!/usr/bin/env python3
"""Capture a GPU trace from inside Python, when nsys is unavailable.

Nsight Systems needs elevated permissions that many rented or containerised
boxes do not grant. The torch profiler needs none, and tools/trace_stats.py
reads what it writes.

    from tools.torchprof import trace

    for _ in range(5):
        step()                      # warm up outside the trace

    with trace("traces/baseline.json"):
        for _ in range(20):
            step()
"""

import contextlib
import os


@contextlib.contextmanager
def trace(path="traces/trace.json", cpu=True):
    """Profile the enclosed block and write a Chrome trace to `path`."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CUDA]
    if cpu:
        activities.append(ProfilerActivity.CPU)

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=False) as prof:
        yield prof
        torch.cuda.synchronize()

    prof.export_chrome_trace(path)
    print(f"wrote {path}")
    print(f"now run: python tools/trace_stats.py {path}")

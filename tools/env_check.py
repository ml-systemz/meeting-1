#!/usr/bin/env python3
"""Report whether this machine can build, run and profile CUDA kernels.

Run this first. It is cheap and it catches the failures that otherwise eat
GPU time: a missing profiler, locked-down counters, a torch built for the
wrong CUDA version.
"""

import os
import shutil
import subprocess
import sys

OK, WARN, BAD = "ok", "warn", "BAD"


def run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", "not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"


def line(status, label, detail=""):
    print(f"  [{status:>4}] {label}" + (f" — {detail}" if detail else ""))


def check_driver():
    rc, out, _ = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                      "--format=csv,noheader"])
    if rc != 0:
        line(BAD, "nvidia-smi", "no driver or no GPU visible")
        return None
    gpus = [g.strip() for g in out.splitlines() if g.strip()]
    for i, g in enumerate(gpus):
        line(OK, f"gpu {i}", g)
    return gpus


def check_sm90(gpus):
    if not gpus:
        return
    caps = [g.split(",")[-1].strip() for g in gpus]
    if any(c.startswith("9.") for c in caps):
        line(OK, "hopper", "SM90 present — WGMMA and TMA available")
    else:
        line(WARN, "hopper", f"compute capability {caps} — WGMMA/TMA are SM90+ only")


def check_toolchain():
    for tool, why in [("nvcc", "CUDA compiler"), ("cmake", "build"), ("ninja", "build")]:
        path = shutil.which(tool)
        if path:
            rc, out, _ = run([tool, "--version"])
            first = out.splitlines()[0] if out else ""
            line(OK, tool, first[:60])
        else:
            line(WARN if tool == "ninja" else BAD, tool, f"missing ({why})")


def check_profilers():
    for tool, why in [("nsys", "Nsight Systems — timelines and gaps"),
                      ("ncu", "Nsight Compute — per-kernel counters")]:
        if shutil.which(tool):
            rc, out, _ = run([tool, "--version"])
            ver = next((l for l in out.splitlines() if l.strip()), "")
            line(OK, tool, ver[:60])
        else:
            line(WARN, tool, f"missing — {why}; fall back to tools/torchprof.py")


def check_ncu_permission():
    if not shutil.which("ncu"):
        return
    path = "/proc/driver/nvidia/params"
    if os.path.exists(path):
        try:
            with open(path) as f:
                params = f.read()
            if "RestrictProfilingToAdminUsers: 0" in params:
                line(OK, "ncu perms", "counters readable by non-root")
            else:
                line(WARN, "ncu perms",
                     "RestrictProfilingToAdminUsers is set — ncu needs root "
                     "or NVreg_RestrictProfilingToAdminUsers=0")
        except OSError:
            line(WARN, "ncu perms", "could not read nvidia params")
    else:
        line(WARN, "ncu perms", "unknown — run tools/ncu_kernel.sh once to find out")


def check_torch():
    try:
        import torch
    except ImportError:
        line(WARN, "torch", "not installed — needed for the reference and for torchprof")
        return
    line(OK, "torch", torch.__version__)
    if not torch.cuda.is_available():
        line(BAD, "torch.cuda", "torch cannot see a GPU")
        return
    line(OK, "torch.cuda", f"built for CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s)")
    props = torch.cuda.get_device_properties(0)
    line(OK, "device 0", f"{props.name}, {props.multi_processor_count} SMs, "
                         f"{props.total_memory // (1024**3)} GiB")


def main():
    print("\nenvironment check\n")
    gpus = check_driver()
    check_sm90(gpus)
    print()
    check_toolchain()
    print()
    check_profilers()
    check_ncu_permission()
    print()
    check_torch()
    print("\nAnything marked BAD will waste GPU time. Fix it before you start.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Compare PyTorch device backends on the tensor shapes this project runs.

The recurring question is whether training should use the GPU. It is easy to
answer wrongly from intuition, because a GPU that is genuinely fast on large
tensors can still lose badly on small ones: accelerator dispatch has a fixed
per-operation cost, and if that cost exceeds the work being dispatched, more
hardware makes the run slower.

This script measures the shapes the simulation actually uses rather than
arguing about them:

- a per-step policy forward at **batch 1** — what the rollout does thousands
  of times per episode;
- a PPO minibatch forward, and a full forward + backward + optimiser step;
- connectome-scale tensors, the largest brain in the project;
- a deliberately large network as a **control**. If that row shows no
  accelerator speedup, the device or driver is at fault and the small-shape
  rows say nothing; if it does, the small-shape results are about shape, not
  hardware.

Absent devices are skipped, so the same command is meaningful on any host.

Usage::

    uv run ./scripts/benchmarks/bench_device_backends.py
    uv run ./scripts/benchmarks/bench_device_backends.py --repeats 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from statistics import median

import torch
from torch import nn

_MICROSECONDS = 1e6


@dataclass(frozen=True)
class Workload:
    """One benchmarked shape."""

    label: str
    batch: int
    in_features: int
    hidden: int
    iterations: int
    train: bool


# Shapes taken from the configs in use: the mlpppo C3 actor is 13 -> 64 -> 64
# -> 2 with a 256-step rollout split into 2 minibatches; the connectome brain
# carries 302 neurons.
WORKLOADS = (
    Workload("policy forward, batch=1 (per-step)", 1, 13, 64, 2000, train=False),
    Workload("policy forward, batch=128 (minibatch)", 128, 13, 64, 1000, train=False),
    Workload("policy fwd+bwd, batch=128 (PPO update)", 128, 13, 64, 500, train=True),
    Workload("connectome-scale forward, batch=1", 1, 302, 302, 1000, train=False),
    Workload("connectome-scale forward, batch=128", 128, 302, 302, 500, train=False),
    Workload("connectome-scale fwd+bwd, batch=128", 128, 302, 302, 300, train=True),
    Workload("control: fwd+bwd 1024x1024, batch=512", 512, 1024, 1024, 100, train=True),
)

_OUT_FEATURES = 2


def available_devices() -> list[str]:
    """Torch devices this host can actually provide, CPU first."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _synchronise(device: torch.device) -> None:
    """Wait for queued accelerator work; timing is meaningless without it."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _build_model(workload: Workload, device: torch.device) -> nn.Module:
    return nn.Sequential(
        nn.Linear(workload.in_features, workload.hidden),
        nn.ReLU(),
        nn.Linear(workload.hidden, workload.hidden),
        nn.ReLU(),
        nn.Linear(workload.hidden, _OUT_FEATURES),
    ).to(device)


def time_workload(workload: Workload, device_name: str, warmup: int) -> float:
    """Return microseconds per iteration for one workload on one device."""
    device = torch.device(device_name)
    model = _build_model(workload, device)
    inputs = (
        torch.randn(workload.batch, workload.in_features, device=device)
        if workload.batch > 1
        else torch.randn(workload.in_features, device=device)
    )

    if workload.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step() -> None:
            optimizer.zero_grad()
            model(inputs).pow(2).mean().backward()
            optimizer.step()
    else:

        def step() -> None:
            with torch.no_grad():
                model(inputs)

    for _ in range(warmup):
        step()
    _synchronise(device)

    started = time.perf_counter()
    for _ in range(workload.iterations):
        step()
    _synchronise(device)
    elapsed = time.perf_counter() - started

    return elapsed / workload.iterations * _MICROSECONDS


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed passes per cell; the median is reported (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Untimed iterations before each pass (default: 20).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the benchmark and print the table."""
    args = parse_arguments()
    devices = available_devices()

    print(f"torch {torch.__version__} | devices: {', '.join(devices)}")
    print(f"median of {args.repeats} passes, {args.warmup} warmup iterations\n")

    header = f"{'workload':<42}" + "".join(f"{device + ' (us)':>14}" for device in devices)
    if len(devices) > 1:
        header += f"{'best':>10}"
    print(header)
    print("-" * len(header))

    for workload in WORKLOADS:
        timings = {
            device: median(
                time_workload(workload, device, args.warmup) for _ in range(args.repeats)
            )
            for device in devices
        }
        row = f"{workload.label:<42}" + "".join(f"{timings[device]:>14.1f}" for device in devices)
        if len(devices) > 1:
            baseline = timings["cpu"]
            fastest = min(timings, key=lambda device: timings[device])
            speedup = baseline / timings[fastest]
            row += f"{fastest + f' {speedup:.2f}x':>10}" if fastest != "cpu" else f"{'cpu':>10}"
        print(row)

    if len(devices) == 1:
        print("\nOnly CPU is available on this host; no comparison to make.")
    else:
        print(
            "\nRead the control row first: if it shows no accelerator speedup, the device "
            "or driver is the problem and the other rows are uninformative.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

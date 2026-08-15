"""Benchmark: is CPU or MPS faster for a single Telemanom LSTM locally?

Answers a concrete question: local training of one channel on MPS takes ~1 hour
for a 2-year ESA series. Would CPU be faster?

The key observation that makes this cheap to answer: the CPU-vs-MPS question is
decided entirely by **per-optimizer-step wall time**, which does not depend on
how many steps an epoch has. Total time = steps/epoch x epochs x step_time, and
only the last factor is device-dependent. So there is no need to train on a
representative "sample" of the data -- a few hundred steps at the production
tensor shape gives the same ratio a full run would, in ~2 minutes instead of 2 hours.

Why the answer is not obvious a priori: TelemanomLSTM is tiny (hidden_dim=80,
input_size=1 -- ~53K params) but the sequence is long (window_size=250). An LSTM
cannot parallelize across time, so one forward pass is 250 x num_layers sequential
steps, each a trivially small matmul. That is a kernel-launch-bound workload, not
a FLOP-bound one, and launch overhead is exactly where an accelerator loses to CPU.
Larger batches amortize the launch overhead over more work, so the sweep also
reports where (if anywhere) MPS crosses over.

Two modes:

  micro (default) -- synthetic tensors, pure fwd/bwd/step timing. No I/O, no
      MLflow, no DataLoader. Isolates the device comparison. Sweeps batch size.
  e2e             -- real preprocessed Parquet through the real WindowedSequenceDataset
      and DataLoader, capped at --steps batches. Confirms the micro result survives
      dataloading overhead (which is device-independent and therefore dilutes,
      but cannot reverse, the ratio).

Note on CPU threads: model/training.py calls torch.set_num_threads(1), which is
correct under Ray fan-out (one process per channel -- unconstrained BLAS pools
would compound to 100+ threads). But it also applies when you train a single
channel locally, where the other 7 cores sit idle. This script times CPU at both
1 thread and the machine default so that cost is visible separately from the
CPU-vs-MPS question.

Usage:
    uv run python scripts/bench_train_device.py
    uv run python scripts/bench_train_device.py --batch-sizes 32,64,128,256,512
    uv run python scripts/bench_train_device.py --mode e2e --mission ESA-Mission1 --channel channel_22
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from spacecraft_telemetry.core.config import ModelConfig, Settings
from spacecraft_telemetry.model.architecture import build_model
from spacecraft_telemetry.model.dataset import (
    WindowedSequenceDataset,
    _build_window_index,
    load_series_parquet,
)


@dataclass(frozen=True)
class BenchResult:
    """Timing for one (device, threads, batch_size) cell."""

    label: str
    batch_size: int
    steps: int
    seconds: float

    @property
    def ms_per_step(self) -> float:
        return 1000.0 * self.seconds / self.steps

    @property
    def windows_per_s(self) -> float:
        return self.steps * self.batch_size / self.seconds


def _sync(device: torch.device) -> None:
    """Force completion of queued work so timers measure compute, not enqueue.

    MPS and CUDA dispatch asynchronously; without this the timer would record
    how fast Python can submit kernels, not how fast they run.
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _time_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    warmup: int,
    steps: int,
) -> float:
    """Run warmup + timed fwd/bwd/step passes over `batches`, return timed seconds.

    Mirrors the production train pass in model/training.py: zero_grad(set_to_none),
    forward, MSE, backward, step, and the per-step loss.item() (which is itself a
    synchronization point, so excluding it would flatter the accelerator unfairly).
    """
    model.train()
    n = len(batches)

    for i in range(warmup):
        x, y = batches[i % n]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        loss.item()
    _sync(device)

    t0 = time.perf_counter()
    for i in range(steps):
        x, y = batches[i % n]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x).squeeze(1), y)
        loss.backward()
        optimizer.step()
        loss.item()
    _sync(device)
    return time.perf_counter() - t0


def _make_synthetic_batches(
    batch_size: int, window: int, device: torch.device, count: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Preallocate resident batches so host->device transfer is out of the timed loop.

    Values are irrelevant to timing (LSTM cost is shape-determined), but a fixed
    seed keeps runs comparable and avoids denormals from an all-zeros input.
    """
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(batch_size, window, 1, generator=g).to(device),
            torch.randn(batch_size, generator=g).to(device),
        )
        for _ in range(count)
    ]


def _make_real_batches(
    settings: Settings,
    mission: str,
    channel: str,
    batch_size: int,
    device: torch.device,
    count: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize `count` real batches via the production Dataset + DataLoader.

    Uses the same window-index construction as make_dataloaders (segment-boundary
    and anomaly filtering), then takes the first count*batch_size windows.
    Materializing up front keeps DataLoader cost out of the timed loop -- it is
    device-independent, so including it would only compress the ratio.
    """
    from torch.utils.data import DataLoader

    cfg = settings.model
    values, segment_ids, is_anomaly, _ = load_series_parquet(
        settings.preprocess.processed_data_dir, mission, channel, "train"
    )
    indices = _build_window_index(
        segment_ids, is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=True,
    )
    needed = count * batch_size
    if len(indices) < needed:
        raise ValueError(
            f"channel {channel!r} has {len(indices)} valid windows, need {needed}. "
            f"Lower --steps or --batch-sizes."
        )
    ds = WindowedSequenceDataset(
        values, indices[:needed], cfg.window_size, cfg.prediction_horizon
    )
    loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return [(x.to(device), y.to(device)) for x, y in loader]


def _run_cell(
    label: str,
    device: torch.device,
    threads: int | None,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: ModelConfig,
    batch_size: int,
    warmup: int,
    steps: int,
) -> BenchResult:
    """Build a fresh model/optimizer and time one configuration."""
    if threads is not None:
        torch.set_num_threads(threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    seconds = _time_steps(
        model, optimizer, nn.MSELoss(), batches, device, warmup, steps
    )
    return BenchResult(label=label, batch_size=batch_size, steps=steps, seconds=seconds)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["micro", "e2e"], default="micro")
    p.add_argument("--steps", type=int, default=60, help="timed optimizer steps per cell")
    p.add_argument("--warmup", type=int, default=10, help="untimed steps before timing")
    p.add_argument("--batch-sizes", default="64", help="comma-separated, e.g. 32,64,256")
    p.add_argument("--window-size", type=int, default=None, help="override model.window_size")
    p.add_argument("--hidden-dim", type=int, default=None, help="override model.hidden_dim")
    p.add_argument("--mission", default="ESA-Mission1", help="e2e mode only")
    p.add_argument("--channel", default="channel_22", help="e2e mode only")
    p.add_argument(
        "--epoch-steps",
        type=int,
        default=None,
        help="steps/epoch to extrapolate a full-run estimate from (default: infer in e2e, "
             "skip in micro)",
    )
    p.add_argument("--epochs", type=int, default=20, help="epochs for the extrapolation")
    args = p.parse_args()

    settings = Settings()
    cfg = settings.model
    overrides: dict[str, int] = {}
    if args.window_size is not None:
        overrides["window_size"] = args.window_size
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    default_threads = torch.get_num_threads()

    # (label, device, threads). threads=None leaves the current setting alone.
    cells: list[tuple[str, torch.device, int | None]] = [
        ("cpu-1thread", torch.device("cpu"), 1),
        (f"cpu-{default_threads}thread", torch.device("cpu"), default_threads),
    ]
    if torch.backends.mps.is_available():
        cells.append(("mps", torch.device("mps"), None))

    print(f"torch {torch.__version__} | mode={args.mode} | "
          f"window_size={cfg.window_size} hidden_dim={cfg.hidden_dim} "
          f"num_layers={cfg.num_layers}")
    print(f"warmup={args.warmup} timed_steps={args.steps}\n")

    epoch_steps = args.epoch_steps
    results: dict[int, list[BenchResult]] = {}

    for bs in batch_sizes:
        n_batches = min(args.steps + args.warmup, 16)  # cycle a small resident set
        if args.mode == "micro":
            batches_cpu = _make_synthetic_batches(
                bs, cfg.window_size, torch.device("cpu"), n_batches, cfg.seed
            )
        else:
            batches_cpu = _make_real_batches(
                settings, args.mission, args.channel, bs, torch.device("cpu"), n_batches
            )
            if epoch_steps is None:
                values, seg, anom, _ = load_series_parquet(
                    settings.preprocess.processed_data_dir, args.mission, args.channel, "train"
                )
                n_win = len(_build_window_index(
                    seg, anom, cfg.window_size, cfg.prediction_horizon,
                    skip_anomalous_windows=True,
                ))
                epoch_steps = int(n_win * (1 - cfg.val_fraction)) // bs

        row: list[BenchResult] = []
        for label, device, threads in cells:
            batches = (
                batches_cpu if device.type == "cpu"
                else [(x.to(device), y.to(device)) for x, y in batches_cpu]
            )
            row.append(_run_cell(
                label, device, threads, batches, cfg, bs, args.warmup, args.steps
            ))
        results[bs] = row

    header = f"{'batch':>6}  " + "  ".join(f"{c[0]:>16}" for c in cells) + "   winner"
    print(header)
    print("-" * len(header))
    for bs, row in results.items():
        best = min(row, key=lambda r: r.ms_per_step)
        cellstr = "  ".join(f"{r.ms_per_step:>13.2f}ms" for r in row)
        slowest = max(row, key=lambda r: r.ms_per_step)
        speedup = slowest.ms_per_step / best.ms_per_step
        print(f"{bs:>6}  {cellstr}   {best.label} ({speedup:.2f}x vs slowest)")

    print("\nthroughput (windows/sec, higher is better)")
    print(header)
    print("-" * len(header))
    for bs, row in results.items():
        best = max(row, key=lambda r: r.windows_per_s)
        cellstr = "  ".join(f"{r.windows_per_s:>15.0f}" for r in row)
        print(f"{bs:>6}  {cellstr}   {best.label}")

    if epoch_steps:
        print(f"\nextrapolated full-run wall time "
              f"({epoch_steps} steps/epoch at batch={batch_sizes[0]}, {args.epochs} epochs):")
        for r in results[batch_sizes[0]]:
            hours = r.ms_per_step * epoch_steps * args.epochs / 1000 / 3600
            print(f"  {r.label:>16}: {hours:6.2f} h")


if __name__ == "__main__":
    main()

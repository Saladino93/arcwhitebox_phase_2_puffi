#!/usr/bin/env python3
"""Run one estimator on an explicit list of nets from the pinned Phase-2 dataset and report grader-style numbers.

Thin wrapper over the stock whestbench runner/scorer (no re-implementation of scoring): the only thing the stock
`whest run` cannot do is pick net indices, so this selects `--nets` from the `full` split, builds ContestData from
that slice and delegates to `whestbench.cli._run_estimator_with_runner` exactly as the CLI does.

    uv run python scripts/run_nets.py --estimator estimators/01_argon_l6_cold_336299/estimator.py \
        --nets 99 --out runs/net99_argon.json

Rows: net | raw (final-layer MSE) | GF | mult | adjusted | wall s | residual s | failed flags.
Pooled: raw = mean raw, adjusted = mean adjusted (the grader's aggregate), mult = adjusted / raw, GF = mean GF.
Memory: peak RSS summed over the worker process tree (children of this process; this process itself only for --runner local), sampled every 50 ms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "phase2" / "public"
DEFAULT_FLOP_BUDGET = 2_199_023_255_552  # 2**41, the Phase-2 budget


def parse_nets(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


class TreeMemorySampler:
    """Peak RSS of this process plus all descendants, in bytes."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> int:
        me = psutil.Process()
        kids = me.children(recursive=True)
        total = 0
        for p in (kids if kids else [me]):  # worker tree only; this process (dataset mmap) only when running in-process
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._sample())
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def take_peak(self) -> int:
        """Return the peak since the last take and reset it (per-net peaks)."""
        p, self.peak = self.peak, 0
        return p

    def __enter__(self) -> "TreeMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimator", required=True, help="path to estimator.py")
    ap.add_argument("--class", dest="class_name", default=None)
    ap.add_argument("--nets", required=True, help="e.g. '99' or '99,32-34,40' or '600-619' (indices in the split)")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--split", default="full")
    ap.add_argument("--runner", choices=["subprocess", "local"], default="subprocess",
                    help="subprocess = isolated worker with the grader's memory/time limits (default)")
    ap.add_argument("--flop-budget", type=int, default=DEFAULT_FLOP_BUDGET)
    ap.add_argument("--wall-time-limit", type=float, default=120.0)
    ap.add_argument("--residual-wall-time-limit", type=float, default=0.4)
    ap.add_argument("--no-residual-limit", action="store_true", help="diagnostics only: disable the residual gate")
    ap.add_argument("--memory-limit-mb", type=int, default=8192)
    ap.add_argument("--max-threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="write full JSON result here")
    ap.add_argument("--label", default=None, help="idea label for the NOTES.md row")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    if args.max_threads is not None:
        from whestbench.concurrency import apply_thread_limit

        apply_thread_limit(args.max_threads)

    import whestbench as wb
    from whestbench.cli import _run_estimator_with_runner
    from whestbench.runner import EstimatorEntrypoint, LocalRunner, SubprocessRunner
    from whestbench.scoring import ContestSpec, make_contest_from_dataset

    nets = parse_nets(args.nets)
    if not nets:
        print("no nets given", file=sys.stderr)
        return 2

    md = json.loads((Path(args.dataset) / "metadata.json").read_text())
    proto = md["seed_protocol"]["version"]
    ds = wb.load_dataset(args.dataset, split=args.split)
    if max(nets) >= len(ds):
        print(f"net index {max(nets)} out of range for split {args.split!r} (size {len(ds)})", file=sys.stderr)
        return 2
    sub = ds.select(nets)

    spec = ContestSpec(
        width=int(md["width"]),
        depth=int(md["depth"]),
        n_mlps=len(nets),
        flop_budget=args.flop_budget,
        ground_truth_samples=int(md["n_samples"]),
        memory_limit_mb=args.memory_limit_mb,
        wall_time_limit_s=args.wall_time_limit,
        residual_wall_time_limit_s=None if args.no_residual_limit else args.residual_wall_time_limit,
        seed=args.seed,
    )
    # The slice is unregistered (ds.select), so the seed protocol must be passed explicitly (protocol 3.0 needs no salt).
    data = make_contest_from_dataset(spec, sub, len(nets), seed_protocol_version=proto)

    entrypoint = EstimatorEntrypoint(file_path=Path(args.estimator).resolve(), class_name=args.class_name)
    runner = LocalRunner() if args.runner == "local" else SubprocessRunner()

    t0 = time.time()
    per_net_peak: List[int] = []
    with TreeMemorySampler() as mem:
        def _progress(ev):
            if ev.get("phase") == "scoring":
                per_net_peak.append(mem.take_peak())
        report = _run_estimator_with_runner(
            runner,
            entrypoint=entrypoint,
            contest_spec=spec,
            n_mlps=len(nets),
            profile=False,
            detail="full",
            output_format="json",
            contest_data=data,
            fail_fast=args.fail_fast,
            progress=_progress,
        )
    elapsed = time.time() - t0
    peak_all = max(per_net_peak + [mem.peak]) if per_net_peak else mem.peak

    results = report["results"]
    per = results.get("per_mlp") or results.get("per_mlp_results") or []
    rows: List[Dict[str, Any]] = []
    for net, r in zip(nets, per):
        flops = float(r.get("flops_used", 0))
        raw = r.get("final_layer_mse", r.get("final_mse"))
        failed = bool(r.get("budget_exhausted") or r.get("time_exhausted") or r.get("residual_wall_time_exhausted")
                      or r.get("error"))
        mult = 1.0 if failed else max(0.1, flops / args.flop_budget)
        rows.append({
            "net": net,
            "raw": raw,
            "GF": flops / 1e9,
            "mult": mult,
            "adjusted": r.get("adjusted_final_layer_score", (raw * mult) if raw is not None else None),
            "wall_s": r.get("wall_time_s"),
            "residual_s": r.get("residual_wall_time_s"),
            "failed": failed,
            "peak_rss_gb": (per_net_peak[len(rows)] / 1e9) if len(rows) < len(per_net_peak) else None,
            "flags": {k: r.get(k) for k in ("budget_exhausted", "time_exhausted", "residual_wall_time_exhausted", "error")
                      if r.get(k)},
        })

    ok = [r for r in rows if r["raw"] is not None]
    pooled_raw = sum(r["raw"] for r in ok) / len(ok) if ok else None
    pooled_adj = results.get("adjusted_final_layer_score")
    if pooled_adj is None and ok:
        pooled_adj = sum(r["adjusted"] for r in ok) / len(ok)
    pooled = {
        "nets": nets,
        "n": len(nets),
        "GF_mean": sum(r["GF"] for r in rows) / len(rows) if rows else None,
        "pooled_raw": pooled_raw,
        "pooled_adjusted": pooled_adj,
        "pooled_mult": (pooled_adj / pooled_raw) if (pooled_adj and pooled_raw) else None,
        "residual_max": max((r["residual_s"] or 0) for r in rows) if rows else None,
        "residual_min": min((r["residual_s"] or 0) for r in rows) if rows else None,
        "wall_max": max((r["wall_s"] or 0) for r in rows) if rows else None,
        "wall_min": min((r["wall_s"] or 0) for r in rows) if rows else None,
        "peak_tree_rss_gb": peak_all / 1e9,
        "peak_tree_rss_gb_min": (min(per_net_peak) / 1e9) if per_net_peak else None,
        "n_failed": sum(r["failed"] for r in rows),
        "elapsed_s": elapsed,
    }

    out = {
        "label": args.label,
        "estimator": str(Path(args.estimator).resolve()),
        "runner": args.runner,
        "spec": {k: getattr(spec, k) for k in ("flop_budget", "wall_time_limit_s", "residual_wall_time_limit_s",
                                                "memory_limit_mb", "seed")},
        "versions": {p: __import__("importlib.metadata").metadata.version(p) for p in ("whestbench", "flopscope")},
        "pooled": pooled,
        "rows": rows,
        "raw_report": report,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1, default=str))

    print(f"{'net':>5} {'raw':>11} {'GF':>7} {'mult':>6} {'adjusted':>11} {'wall':>7} {'resid':>6} {'memGB':>5} flags")
    for r in rows:
        raw_s = f"{r['raw']:.4e}" if r["raw"] is not None else "   n/a"
        adj_s = f"{r['adjusted']:.4e}" if r["adjusted"] is not None else "   n/a"
        print(f"{r['net']:>5} {raw_s:>11} {r['GF']:7.1f} {r['mult']:6.3f} {adj_s:>11} "
              f"{(r['wall_s'] or 0):7.1f} {(r['residual_s'] or 0):6.3f} {(r['peak_rss_gb'] or 0):5.2f} {r['flags'] or ''}")
    p = pooled
    fmt = lambda x, f: (f % x) if x is not None else "n/a"  # noqa: E731
    print("\nPOOLED  GF %s | mult %s | raw %s | adjusted %s | resid %s/%s | wall %s/%s | mem %s GB | failed %d | %ss" % (
        fmt(p["GF_mean"], "%.1f"), fmt(p["pooled_mult"], "%.4f"), fmt(p["pooled_raw"], "%.4e"),
        fmt(p["pooled_adjusted"], "%.4e"), fmt(p["residual_max"], "%.3f"), fmt(p["residual_min"], "%.3f"),
        fmt(p["wall_max"], "%.1f"), fmt(p["wall_min"], "%.1f"), fmt(p["peak_tree_rss_gb"], "%.2f") + "/" + fmt(p["peak_tree_rss_gb_min"], "%.2f"),
        p["n_failed"], fmt(p["elapsed_s"], "%.0f")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

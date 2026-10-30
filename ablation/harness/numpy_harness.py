#!/usr/bin/env python3
"""Unmetered numpy+BLAS screening harness (user-requested, 2026-10-20): run an estimator root's physics with plain numpy
instead of flopscope and report the final-layer MSE per net. Used to reject candidates quickly (MSE worse than the GF saving
allows) before porting them to the metered path. NOT a scoring tool: no FLOPs, no residual, no limits.

How it works: whestbench + the dataset are imported first (they need the real flopscope); then `flopscope`,
`flopscope.numpy` and `flopscope.stats` are replaced in sys.modules by numpy shims, the root's estimator.py is imported from
its directory, and `fmm.levels_for` is forced to 0 so every fmm entry point takes its plain-product (BLAS) fallback.
Results differ from the metered run only by floating-point association (Strassen vs plain products).

    uv run python scripts/numpy_harness.py --root estimators/01_argon_l6_cold_336299 --nets 99
    uv run python scripts/numpy_harness.py --root candidates/foo --nets 99,32,33,34,40 --profile
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "phase2" / "public"
DEFAULT_FLOP_BUDGET = 2_199_023_255_552


def parse_nets(spec):
    out = []
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


def install_numpy_shims():
    import numpy as np

    class _Linalg(types.ModuleType):
        def __getattr__(self, name):
            return getattr(np.linalg, name)

        @staticmethod
        def svd(a, full_matrices=True, compute_uv=True, hermitian=False, k=None):
            res = np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
            if k is None or not compute_uv:
                return res if k is None else res[..., :k]
            u, s, vh = res
            return u[..., :, :k], s[..., :k], vh[..., :k, :]

    class _XP(types.ModuleType):
        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def einsum(*operands, **kw):
            subs = operands[0] if isinstance(operands[0], str) else None
            if subs == "mkli,kmij,lmij->j":
                # pk4.fconsumer: numpy's path executor hands the Hadamard pair to c_einsum (10 s per call); do it with
                # a broadcast product and one BLAS matvec instead (same terms).
                qc, a1, a2 = operands[1:4]
                inter = a1[:, None] * a2[None, :]                     # (k, l, m, i, j)
                K = inter.shape[0] * inter.shape[1] * inter.shape[2] * inter.shape[3]
                left = np.transpose(qc, (1, 2, 0, 3)).reshape(K)       # (k, l, m, i)
                return left @ inter.reshape(K, inter.shape[4])
            opt = kw.get("optimize")
            if isinstance(opt, (list, tuple)) and opt and isinstance(opt[0], tuple):   # flopscope accepts a bare path list
                kw["optimize"] = ["einsum_path", *opt]
            elif opt is None and len(operands) > 3:      # >= 3 array operands: numpy's default (no path) runs one nested C loop
                kw["optimize"] = "greedy"                # over every index (pk4 fconsumer "mki,kmij,mpj->j": ~130 s/net); flopscope
            return np.einsum(*operands, **kw)            # contracts pairwise, so a greedy path matches it up to association

    xp = _XP("flopscope.numpy")
    xp.linalg = _Linalg("flopscope.numpy.linalg")
    xp.__file__ = "<numpy shim>"

    class NoBudgetContextError(RuntimeError):
        pass

    class BudgetExhaustedError(RuntimeError):
        def __init__(self, *a, **k):
            super().__init__(str(a[0]) if a else "budget exhausted")

    fs = types.ModuleType("flopscope")
    fs.numpy = xp
    fs.NoBudgetContextError = NoBudgetContextError
    fs.BudgetExhaustedError = BudgetExhaustedError
    fs.current_budget = lambda: (_ for _ in ()).throw(NoBudgetContextError("numpy harness: no budget context"))
    fs.budget_summary_dict = lambda **k: {"flops_used": 0}
    fs.namespace = lambda name: _NullCtx()

    _erf = np.vectorize(math.erf, otypes=[np.float64])

    class _Norm:
        @staticmethod
        def cdf(a):
            a = np.asarray(a)
            return (0.5 * (1.0 + _erf(a.astype(np.float64) / math.sqrt(2.0)))).astype(a.dtype if a.dtype.kind == "f" else np.float64)

        @staticmethod
        def pdf(a):
            a = np.asarray(a)
            return (np.exp(-0.5 * a.astype(np.float64) ** 2) / math.sqrt(2 * math.pi)).astype(a.dtype if a.dtype.kind == "f" else np.float64)

    st = types.ModuleType("flopscope.stats")
    st.norm = _Norm()
    fs.stats = st

    sys.modules["flopscope"] = fs
    sys.modules["flopscope.numpy"] = xp
    sys.modules["flopscope.stats"] = st
    return np


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def load_root(root: Path):
    root = root.resolve()
    sys.path.insert(0, str(root))
    for name in ("fmm", "bank", "pk4", "estimator"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("estimator", root / "estimator.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["estimator"] = mod
    spec.loader.exec_module(mod)
    fmm = sys.modules.get("fmm")
    if fmm is not None:
        fmm.levels_for = lambda a, b: 0          # every fmm entry point falls back to the plain BLAS product
        if hasattr(fmm, "HOST_LEVELS"):
            fmm.HOST_LEVELS = 0
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="estimator root directory (contains estimator.py)")
    ap.add_argument("--class", dest="class_name", default="Estimator")
    ap.add_argument("--nets", required=True)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--split", default="full")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--profile", action="store_true", help="cProfile the predict calls (top 25 cumulative)")
    ap.add_argument("--threads", type=int, default=None, help="BLAS threads (set before numpy import)")
    args = ap.parse_args()

    if args.threads:
        import os
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[v] = str(args.threads)

    import whestbench as wb  # real flopscope is imported here, before the shims
    nets = parse_nets(args.nets)
    ds = wb.load_dataset(args.dataset, split=args.split)
    rows_in = [(i, wb.mlp_at(ds, i), ds[i]) for i in nets]

    np = install_numpy_shims()
    mod = load_root(Path(args.root))
    Est = getattr(mod, args.class_name)

    results = []
    prof = None
    if args.profile:
        import cProfile
        prof = cProfile.Profile()
    for net, mlp, row in rows_in:
        weights = [np.asarray(w, dtype=np.float32) for w in mlp.weights]
        mlp_np = types.SimpleNamespace(weights=weights, width=mlp.width, depth=mlp.depth, validate=lambda: None)
        est = Est()
        t0 = time.perf_counter()
        if prof is not None:
            prof.enable()
        pred = est.predict(mlp_np, DEFAULT_FLOP_BUDGET)
        if prof is not None:
            prof.disable()
        dt = time.perf_counter() - t0
        pred = np.asarray(pred, dtype=np.float32)
        final = np.asarray(row["final_means"], dtype=np.float32)
        alll = np.asarray(row["all_layer_means"], dtype=np.float32)
        mse = float(np.mean((pred[-1] - final) ** 2))
        mse_all = float(np.mean((pred - alll) ** 2))
        per_layer = [float(np.mean((pred[l] - alll[l]) ** 2)) for l in range(pred.shape[0])]
        results.append({"net": net, "raw": mse, "all_layers_mse": mse_all, "per_layer_mse": per_layer, "wall_s": dt,
                        "fallback": getattr(est, "_fallback_reason", None)})
        print(f"net {net:4d}  raw {mse:.4e}  all-layers {mse_all:.3e}  wall {dt:5.1f}s  fallback={results[-1]['fallback']}", flush=True)

    pooled = sum(r["raw"] for r in results) / len(results)
    print(f"\nPOOLED raw {pooled:.4e} over {len(results)} nets  ({args.label or args.root})")
    if prof is not None:
        import pstats
        pstats.Stats(prof).sort_stats("cumulative").print_stats(25)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"label": args.label, "root": str(Path(args.root).resolve()), "nets": nets,
                                              "pooled_raw": pooled, "rows": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

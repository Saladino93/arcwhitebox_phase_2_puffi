# Phase 2 submission of the Puffi team — ARC White-Box Estimation Challenge 2026

Estimators for the Gaussian-input means of width-1024, depth-16 random bias-free ReLU networks, given their weights.
Score = raw final-layer MSE x max(0.1, FLOPs / 2^41), averaged over networks. No sampling: everything is a deterministic
moment closure (mean, covariance, third and fourth cumulants propagated layer by layer and passed through the ReLU with
Edgeworth and Mehler expansions). Nothing is fitted to Monte Carlo targets.

## Selected submissions

Two submissions were selected on the AIcrowd platform for the official Phase 2 evaluation (both graded, 100/100
networks, no failures):

| selected submission | official score (adjusted) | raw final-layer MSE | FLOP budget used | why it was selected |
|---|---:|---:|---:|---|
| **[337631](submissions/337631)** | **1.8608e-9** | 1.2106e-8 | 15.36 % | **our best adjusted score**: the cheapest member of the family (elder-basis cap 256 and the pointwise final Strassen combine) |
| **[337577](submissions/337577)** | 1.8671e-9 | **1.1741e-8** | 15.89 % | **our most accurate graded estimate**: it keeps the elder-basis cap at 320, which costs 0.5 % more FLOP budget and buys 3 % lower raw MSE |

The two differ by exactly one line of `bank.py` (the rank cap of the shared elder basis, 320 in 337577 and 256 in 337631);
`estimator.py`, `pk4.py` and `fmm.py` are byte-identical between them. 337631 is the better score under the competition
metric, raw MSE x max(0.1, FLOPs / 2^41); 337577 is the better estimator if accuracy alone is judged. The distilled program
in [`distilled_submission/`](distilled_submission/) reproduces the statistics of this family (it was built from 337479, the
base the two selected submissions descend from, with the cap set to 320, i.e. the 337577 statistics).

This repository has two parts.

| folder | what it is | lines of Python |
|---|---|---:|
| [`submissions/`](submissions/) | the exact source of our graded Phase 2 submissions, as uploaded | ~3,800 each |
| [`distilled_submission/`](distilled_submission/) | the same estimator rewritten as one readable numpy program, plus the organizers' K=3 estimator in numpy for comparison | 807 + 271 |

## `submissions/`: the graded code

Four graded uploads of one estimator family; the two selected ones are in bold. `estimator.py` and `pk4.py` are byte-identical across the four; they differ
only in `bank.py` (the shared elder-basis rank cap) and `fmm.py` (the final Strassen combine).

| folder | submission | official score | raw final-layer MSE | FLOP budget used | change |
|---|---|---:|---:|---:|---|
| [`337479`](submissions/337479) | 337479 | 1.8841e-9 | 1.1742e-8 | 16.04 % | base: shared elder basis (cap 320), slice row repair, exact switch restructuring |
| [`337563`](submissions/337563) | 337563 | 1.8783e-9 | 1.2106e-8 | 15.50 % | elder cap 320 -> 256 (one line) |
| **[`337577`](submissions/337577)** | **337577** | **1.8671e-9** | **1.1741e-8** | **15.89 %** | **pointwise Strassen 7-to-4 combine (10 lines); SELECTED (most accurate)** |
| **[`337631`](submissions/337631)** | **337631** | **1.8608e-9** | **1.2106e-8** | **15.36 %** | **both edits; SELECTED (best official score)** |

All four completed 100/100 networks with no failures, max wall 96.8-98.0 s, max residual 0.37 s. Each folder has its own
README with the full audit, `manifest.json`, `SHA256.txt` and the platform receipt. Graded on whestbench 0.16.x with
FlopScope 0.12.3.

Each submission is four files: `estimator.py` (the moment engine), `bank.py` (the third-cumulant source bank and its
fourth-cumulant reads), `pk4.py` (tree-level fourth cumulants) and `fmm.py` (an alternative-basis Strassen-Winograd
matrix-multiplication engine with call-priced routing). About 80 % of the FLOPs are in `bank.py`.

## `distilled_submission/`: the same logic without the tricks

The graded code is hard to read because most of it exists to save FLOPs or client round trips, not to compute the
estimate: a fast matrix-multiplication engine, neuron thinning plans padded to Strassen quanta, a shared low-rank basis
for old third-cumulant sources, rank-1 and rank-2 truncations, a grouped Horner recurrence, buffer reuse and cached views.

**`distilled_submission/estimator.py` (807 lines, numpy only, plain `@` products) deliberately uses none of these
FLOP-saving tricks.** Its one goal is to preserve the logic of the base submission (337479): the same closure orders, the
same row structure of the pair law, the same trace core, the same correction schedule, the same hub rules at a birth. What
it removes is only economy:

- no fast matrix multiplication: every product is a dense `@`, routed through one `mm()` helper that only counts FLOPs;
- every neuron is kept and every third-cumulant source stays dense at every layer: no thinning, no padding, no shared
  elder basis, no randomized or Krylov SVD;
- the completed square and its rank-1 "negative packet" are replaced by an exact three-leg form of the same source.

Two low-rank pieces of submission 337479 remain because their exact versions failed, and both are stated in the code: the rank-2 repair
of a birth (the exact version costs multiplier 1.05) and the rank-4 image inside the trace correction (the exact version
produced NaN on one development network).

Because nothing is cropped, the FLOP count is the same on every network: 1677 GF, multiplier 0.763. That is five times the
graded submission's cost and would not be competitive; it is not meant to be. It is meant to be read.

### Does it preserve the estimate?

Measured with plain numpy against the graded submission 337479 statistics on 20 held-out public networks (indices 600-619 of the
Phase 2 public split), final-layer MSE:

| | distilled | submission 337479 statistics | ratio |
|---|---:|---:|---:|
| pooled raw MSE, 20 networks | 1.2253e-8 | 1.2153e-8 | 1.008 |
| per-network ratio min / median / max | | | 0.941 / 1.008 / 1.091 |
| organizers' K=3 estimator (`k3_reference.py`), pooled | 3.5548e-8 | | 2.93 |

Within 1 % pooled. The per-network spread is the difference between cropped and dense history, not a bias. Peak RSS is
2.2-2.9 GiB; wall about 17 s per network on a laptop with 4 BLAS threads.

### `k3_reference.py`: the organizers' K=3 estimator in numpy

A 271-line float64 implementation of ARC's own K=3 SIMPLE estimator (Wu et al., arXiv 2605.05179) in factored form: mean,
covariance, third cumulant as symmetric CP factors (rank grows by 2n per layer) and one radial fourth-cumulant scalar.
Validated to 1e-14 against a dense width-64 oracle (`docs/k3_dense_oracle_width64.py`). Multiplier 0.925, pooled raw MSE
3.55e-8 on the same 20 networks, so the distilled submission 337479 is about 2.9x more accurate at a lower FLOP count.

### Reading order

1. [`distilled_submission/README.md`](distilled_submission/README.md): the model in plain words, the three ideas that
   carry the deep part, what is kept and dropped, the memory census, every measured decision.
2. [`distilled_submission/estimator.py`](distilled_submission/estimator.py): one function per mechanism, formulas in comments.
3. [`distilled_submission/docs/SPEC.md`](distilled_submission/docs/SPEC.md): every equation in loop order, cross-referenced to
   the source lines of the graded code it was read from. `CENSUS.md` is the FLOP and memory census; `REVIEW.md` the
   line-by-line review of the program against the spec; `DISCREPANCIES.md` the places where program, spec and graded source
   disagree and which one was followed; `ABLATION_EVIDENCE.md` what each mechanism is worth; `RESULTS.md` all measurements.

### Running

Both programs expose the whestbench estimator contract: `class Estimator` with a no-argument constructor and
`predict(mlp, budget)` returning a `(16, 1024)` array, where `mlp.weights` is the list of 16 float32 `(1024, 1024)` matrices
applied as `h = relu(h @ W)` to `x ~ N(0, I)`. They need only numpy. The targets and the metered grader are the
organizers' `whestbench` / `flopscope` packages; the numbers above were obtained with unmetered numpy (BLAS), which only
changes floating-point association.

```python
import numpy as np, importlib.util, types
spec = importlib.util.spec_from_file_location("est", "distilled_submission/estimator.py")
est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)
mlp = types.SimpleNamespace(weights=weights, width=1024, depth=16)   # weights: list of 16 float32 (1024, 1024) arrays
pred = est.Estimator().predict(mlp, 2**41)                            # (16, 1024) predicted post-ReLU means
```

## Puffi Team
- https://github.com/Eppie
- https://github.com/lhd23
- https://github.com/Saladino93

## License

MIT (see `LICENSE`; the submission folders carry the AIcrowd MIT license they were uploaded with).

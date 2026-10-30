# Switch-off ledger of submission 337631 (Team Puffi, ARC White-Box Estimation Challenge 2026, Phase 2)

Measured 2026-10-30 on the exact uploaded files of submission **337631** (`submissions/337631/`), one declared edit per row, nothing retrained
(there is nothing to retrain: the estimator has no fitted parameters). Companion to the write-up's 5-network stacked ablation; this table is
leave-one-component-out on the shipped file over 30 fresh networks.

## Method

- **Networks:** IDs 600-629 of the `full` split of `aicrowd/arc-whestbench-public-2026` (revision `v2-phase2`), none used during development
  (the development set was IDs 99, 32-36, 40). Targets: the dataset's `all_layer_means` (10^9-sample labels), final layer only.
- **Raw MSE (30 nets):** the submitted `estimator.py` / `bank.py` / `pk4.py` executed with plain numpy products in place of the FlopScope
  client (`numpy_harness.py`: same code path, every fast-multiplication entry point falls back to the plain BLAS product). The only
  difference from the graded run is floating-point association (Strassen vs plain products): the unmodified file reproduces our stored
  2026-10-27 receipt of 337631 on these 30 nets within **0.28 % on every net**, and the cap-320 variant reproduces the stored receipt of
  337479 within 0.36 %. Control: mean raw 1.2750e-08 (official 100-net graded raw of 337631: 1.2106e-8).
- **FLOPs:** the same variant files run under the pinned grader stack (whestbench 0.16.2, FlopScope 0.12.3, `run_nets.py`, subprocess
  runner) on nets 600-602; GF = billed FLOPs / 1e9, mean of the 3 nets (control: 335.7 GF = 0.1527 of the budget;
  official mean for 337631: 15.36 %). Timing columns of those runs are not reported (Mac, unisolated).
- **MSE ratio:** geometric mean over the 30 nets of (variant raw / control raw), 95 % CI from the per-net log-ratios; **worse** = nets whose
  raw MSE rose. **Implied score ratio** = MSE ratio x GF ratio (score = raw x max(0.1, GF/2199); both sides are above the 0.1 floor).
- **Reproduce:** `python build_variants.py <dir> submissions/337631` builds every variant; `run_numpy30.sh` and `run_metered3.sh` run
  them; `ledger.py` prints the table. The two harness scripts (`harness/`) need `whestbench` 0.16.2 + `flopscope` 0.12.3 and the pinned dataset at `data/phase2/public`; the driver scripts point at our research checkout (`W=` line), adjust the path. Per-net results are in `results/np30/*.json` (raw, per-layer MSE) and `results/met3/*.json` (raw, GF).

## Ledger (switch one component off, keep everything else)

| component switched off (or changed) | kind | MSE ratio | 95% CI | worse | GF (600-602) | dGF | implied score ratio |
|---|---|---|---|---|---|---|---|
| Spherical-radial input law -> Gaussian input (single-site kappa4 scale 2, zero layer-0 core, radius factor 1) | mechanism | **x1.056** | [1.052, 1.061] | 30/30 | 336.1 | +0.3 | x1.057 |
| Trace contraction TC (hidden + terminal) off; edge-block schedule kept at layers < 4 | mechanism | **x1.193** | [1.173, 1.214] | 30/30 | 318.5 | -17.2 | x1.132 |
| TC Horner history (3 generations) -> none | mechanism | **x1.094** | [1.080, 1.108] | 30/30 | 325.9 | -9.8 | x1.062 |
| Tree-level fourth cumulants PK4 (single-site + legs) off | mechanism | **x1.059** | [1.046, 1.072] | 29/30 | 330.2 | -5.5 | x1.042 |
| TC slice fold + history fold off | mechanism | **x1.049** | [1.040, 1.058] | 29/30 | 334.0 | -1.7 | x1.044 |
| E6 directional response (E_TU, E_PROP) off | mechanism | **x1.042** | [1.025, 1.059] | 24/30 | 332.4 | -3.3 | x1.032 |
| PK4 fold response (star + u closure in every deposit) off | mechanism | **x1.025** | [1.019, 1.032] | 28/30 | 332.6 | -3.2 | x1.016 |
| Birth repair rank 2 -> 1 | cost/accuracy trade | **x1.005** | [1.002, 1.009] | 22/30 | 334.2 | -1.5 | x1.001 |
| Unit-prevariance gauge -> identity gauge | mechanism (invariance) | **x0.993** | [0.989, 0.997] | 10/30 | 335.0 | -0.7 | x0.991 |
| Terminal z3^2/72 Edgeworth term off | mechanism | **x0.997** | [0.995, 0.999] | 9/30 | 335.7 | +0.0 | x0.997 |
| Elder-basis rank cap 256 -> 320 (= submission 337577 / 337479 statistics) | cost/accuracy trade | **x0.969** | [0.963, 0.975] | 0/30 | 345.2 | +9.5 | x0.996 |
| No elder compression: every birth keeps dense legs | cost mechanism | **x1.020** | [1.007, 1.033] | 22/30 | 377.0 | +41.2 | x1.145 |
| Lane L1 saturated-row drop + rank-2 refill off (dense slice rows) | cost mechanism | **x0.985** | [0.980, 0.990] | 5/30 | 352.4 | +16.6 | x1.034 |
| Gate thinning off (COL / OUT / hub-row thresholds -> 0) | cost mechanism | **x0.971** | [0.958, 0.985] | 8/30 | 428.2 | +92.5 | x1.239 |
| Off-diagonal core edge block on every layer (default: layers < 4) | cost mechanism | **x0.998** | [0.991, 1.006] | 15/30 | 339.6 | +3.9 | x1.010 |

Reading the table: the first ten rows are accuracy mechanisms (switching them off saves few FLOPs and costs MSE); the last five are
FLOP-economy mechanisms (switching them off costs FLOPs and changes MSE by less than the FLOPs are worth). Rows are single switch-offs on
a co-dependent estimator, so they do not add up: e.g. the TC history row is contained in the TC row.

## Notes on specific rows

- **Spherical input (x1.056):** the Gaussian-input variant sets the first-layer single-site fourth-cumulant scale to 2 (sphere: 2 - 4/(n+2)),
  zeroes the layer-0 fourth-cumulant core (sphere: -W^T W/(n+2)) and drops the exact radius factor E[chi_n]/sqrt(n). A 2-edit version that
  keeps the layer-0 core is inconsistent (x7.7 from layer 1) and is not reported.
- **Unit-prevariance gauge (x0.993):** on benchmark networks the gauge costs 0.7 % MSE; its value is invariance under exact ReLU-preserving
  rescalings of hidden neurons (write-up Sec. 3.2: ungauged x5,000 under log-uniform rescaling, gauged unchanged). We report it as measured.
- **Terminal z3^2/72 term (x0.997):** neutral to slightly harmful; kept in the submission.
- **Elder compression (dense legs x1.020, +41 GF):** the shared elder basis is not only cheaper, it is slightly more accurate than dense
  transport of old births on these nets.
- **Thinning off (x0.971, +92 GF):** the gate thresholds sit at the score optimum: removing all thinning buys 2.9 % MSE for 27.6 % FLOPs.
- **Not switchable in this file:** the n x n fourth-cumulant trace core A = T + sI (no plain off switch; the archived rank-2 replacement
  measured x2.01 on net 99, see `distilled_submission/docs/ABLATION_EVIDENCE.md` row 14), the dense (2,1) slice and the cubic source bank
  (same document, rows 3 and 10). Two constants that look like switches are dead in this version: `SLICE_UROW` (unused; output bitwise
  identical) and `STAR_START` / `STAR_TERM` (no-op: PK4 owns every star age).

## Exact edits

| variant | file: exact edit |
|---|---|
| sphere_off | `estimator.py`: `stfscale = _scalar(xp.subtract(2.0, xp.divide(_scalar(4.0), dim["n+2"])), dtype)` -> `stfscale = _scalar(xp.subtract(2.0, xp.divide(_scalar(0.0), dim["n+2"])), dtype)`<br>`estimator.py`: `if self.layer == 0: gm = pc / self.K.negative_n2; return gm, pc, _diag(gm), v` -> `if self.layer == 0: gm = pc * 0.0; return gm, pc, _diag(gm), v`<br>`estimator.py`: `return radius_mean(n).astype(self.dtype) * xp.stack(rows)` -> `return xp.stack(rows)` |
| tc_off | `bank.py`: `TCP_START = 4` -> `TCP_START = 99`<br>`bank.py`: `TERM_TC, TERM_TC_TAU = True, 0.3` -> `TERM_TC, TERM_TC_TAU = False, 0.3`<br>`estimator.py`: `TERM_TC_DQ, EDGE_END, GM_OFFDIAG_LAYERS = True, None, None` -> `TERM_TC_DQ, EDGE_END, GM_OFFDIAG_LAYERS = True, 4, None` |
| tch0_no_tc_history | `bank.py`: `TCH = 3  #` -> `TCH = 0  #` |
| pk4_off | `estimator.py`: `PK4_SINGLE, PK4_LEGS, PK4_CORESUB, PK4_TERM = True, True, "lite", True` -> `PK4_SINGLE, PK4_LEGS, PK4_CORESUB, PK4_TERM = False, False, "lite", True` |
| folds_off | `bank.py`: `TC_SLICE_FOLD = True` -> `TC_SLICE_FOLD = False`<br>`bank.py`: `HIST_FOLD = True` -> `HIST_FOLD = False` |
| e6_off | `bank.py`: `E_TU, E_RADIAL = True, 'mehler1'` -> `E_TU, E_RADIAL = False, 'mehler1'`<br>`bank.py`: `E_PROP = True` -> `E_PROP = False` |
| pk4_fold_response_off | `pk4.py`: `FOLD = True   #` -> `FOLD = False   #` |
| repair_rank1 | `bank.py`: `REPAIR_RANK, AGE_SWITCH_MAX_BIRTH = 2, 7` -> `REPAIR_RANK, AGE_SWITCH_MAX_BIRTH = 1, 7` |
| gauge_off | `estimator.py`: `GAUGE_POLICY = "unit_prevariance"` -> `GAUGE_POLICY = "identity"` |
| terminal_z3sq_term_off | `estimator.py`: `(z3t * z3t / 72.0) * w16` -> `(z3t * z3t * 0.0) * w16` |
| cap320_eq_337577 | `bank.py`: `kmax = max(1, min(ncol, int(M.shape[0]), n // 4))` -> `kmax = max(1, min(ncol, int(M.shape[0]), 5 * n // 16))` |
| elders_dense_no_switch | `bank.py`: `REPAIR_RANK, AGE_SWITCH_MAX_BIRTH = 2, 7` -> `REPAIR_RANK, AGE_SWITCH_MAX_BIRTH = 2, -1` |
| slice_rows_dense | `bank.py`: `SLICE_ROWS, SLICE_SAT, SLICE_DEAD, SLICE_ROWS_TC = True, 0.015, 0.0, "rows"` -> `SLICE_ROWS, SLICE_SAT, SLICE_DEAD, SLICE_ROWS_TC = False, 0.015, 0.0, "rows"` |
| thinning_off | `bank.py`: `THIN_TAU, THIN_OUT, ROW_THIN, V_SAT, TERM_KEY = 0.04, 0.015, 0.01, 0.003, 0.01` -> `THIN_TAU, THIN_OUT, ROW_THIN, V_SAT, TERM_KEY = 0.0, 0.0, 0.0, 0.003, 0.01` |
| edges_all_layers | `estimator.py`: `TERM_TC_DQ, EDGE_END, GM_OFFDIAG_LAYERS = True, None, None` -> `TERM_TC_DQ, EDGE_END, GM_OFFDIAG_LAYERS = True, 16, None` |

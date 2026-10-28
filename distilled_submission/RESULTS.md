# RESULTS: clean dense numpy rewrite of ex_34 (Fable A)

Date 2026-10-27. Machine: Apple M4 Pro, numpy 2.2.6 (Accelerate), Python 3.10; host busy (load 7-9), wall times are single
observations. Harness: `scripts/numpy_harness.py` shims; isolated runs through `clean/tools/run_net.py` (same code path as
`isolated_rss.py`, net exported to npz so the process RSS is the estimator's). Reference numbers on net 99: plain cap-320
(= ex_34 statistics) 1.1051331e-8; ex_34 graded 1.1064921e-8.

Program: `clean/estimator.py`, **807 lines** (one file, numpy only, every product through `mm`; README.md holds the explanation).
Final form D3b: three dense legs per source (u = C dv, X = extra, P), rank-2 repair + conservation column kept, rank-4
repeated-index image in the TC kept (exact SVD), ex_34's hub rules kept as statistics (gate > 0.01 to be a hub, gate < 0.997
to carry a square, TC hub rows gate > 0.04, r/beta of saturated hubs from the full rows), all other crops / plans / padding /
elder basis / block-Krylov SVD removed; dtype: legs and coefficient rows float32, vectors and the cross-source accumulators
(slice, z3, rt, dq, PK4) float64.

## 1. FLOP estimate (dense design, n = 1024, L = 16, S = 15 sources; P = one 2 n^3 product = 2.147 GF) and what was measured

| design | estimate | measured (mm tally) |
| --- | ---: | ---: |
| D2: legs H, P + rank-1 packet + rank-2 repair, rank-4 repeated index (= plain code without crops) | 1230 GF (0.56) | 1194 GF (0.543) |
| D3b: legs u, X, P (exact F (x) F removal), rank-2 repair, rank-4 repeated index | 1680 GF (0.76) | 1677 GF (0.763) |
| D3b + exact repeated-index image (rejected, see section 3) | 1820 GF (0.83) | 1818 GF (0.827) |
| D4: D3b with a dense R leg as well (CENSUS: +483 GF) | 2300 GF (1.05) | not built (over budget) |

Breakdown of the final bill (1677 GF, multiplier 0.763): transports sum_l (3l + 2) = 345 P = 741 GF; slice deposits sum_t 3t =
315 P = 676 GF; Horner steps and terminal query ~50 P = 105 GF; covariance / core / terminal products 60 P = 129 GF; edge block
4 P = 9 GF; radial solves 22 GF; SVDs (15 repair at k = 2, 12 TC repeated-index at k = 4) at FlopScope's truncated-SVD tariff
4 m n k = 0.33 GF (LAPACK's true cost ~12 n^3 each = 350 GF is not tallied); untallied O(n^2) elementwise work ~15 GF.
Correction to my first estimate (CENSUS.md section 5): in the Horner / pre-lift form the TC's old-repair term for a dense leg
is the deposit product already counted, so a dense R or X leg costs +483 GF, not +1503 GF.

## 2. Validation against the plain code

Two scratch copies of `source/estimator_plain` were run with `clean/tools/dump_plain_states.py` on net 99 and compared with
`clean/tools/compare_states.py` (relative L2 per layer of premean, z3, extra = TC + PK4, q4, rt, radial v, ga, T, s, b, Cov(h),
m, dv, ov, k3, k4, and the TC intermediates qt, A B, stash B, a1, dq, cpf, ch):

- cap 320 (ex_34 statistics): 1.1051e-8. The crops make every state differ at 1e-3..1e-1 from layer 1 on (the core output crop
  at gate 0.3 alone changes the off-diagonal of T_1 by 18 %), so this comparison only bounds gross errors.
- fully uncropped (every threshold -1, no saturation, no elders, LAPACK SVD): 1.1273e-8. My H-form program (D2, diagonal
  gram) agrees with it to 1e-5..1e-6 in every state through layer 9, including the TC intermediates (dq 1e-5..1e-4); from
  layer 10 the PK4 read 'extra' diverges (1e-3 -> 5e-1) while its inputs still agree to 1e-4, and the final means differ by
  2e-5 relative (D2 raw 1.1138e-8, i.e. 1.2 % below the uncropped plain). The divergence sits in the dressed leg
  g' = (H - H_aa P)/sqrt(e) of hubs with tiny curvature, which both programs form in float32 (DISCREPANCIES.md 6).
- Two implementation bugs were found and fixed this way: the edge block's self-term removal used `edge_left` instead of
  `left` (T_1 off by 2.6 % in ga), and the three-leg TC leg content (DISCREPANCIES.md 1).

## 3. Measured changes (net 99; "ratio" = raw / raw of the base it changes; +-1 % is noise)

| change | base | raw | ratio | GF | decision |
| --- | --- | ---: | ---: | ---: | --- |
| D2 all-dense H-form, dense gram W^T W (manager's first plan) | - | 1.5034e-8 | 1.360 vs cap-320 | 1194 | gram rejected |
| D2 all-dense H-form, diagonal gram | cap-320 1.1051e-8 | 1.1138e-8 | 1.0079 | 1194 (0.543) | all-dense history: consistent with ABLATION #7 (x1.04) |
| D3b three legs, exact F (x) F removal, TC leg M = e u, rank-4 repeated index | D2 1.1138e-8 | 1.1182e-8 | 1.0040 | 1677 (0.763) | accepted (<= +1 %) |
| D3b with source 0's F (x) F kept (ex_34's exception) | D3b 1.1182e-8 | 1.1180e-8 | 0.9998 | 1678 | noise; exact removal kept |
| D3b with the SPEC 2.1 reading M = e u + X/2 | D3b | 2.5825e-8 | 2.31 | 1678 | rejected; SPEC to be corrected |
| D3b + exact repeated-index image (full off-diagonal B, two products) | D3b 1.1182e-8 | 1.1155e-8 | 0.9975 | 1818 (0.827) | **rejected**: diverges on net 32 (NaN from layer 13; nets 33/34/40 fine: 1.0427e-8, 1.1933e-8, 1.0095e-8). The rank-4 truncation (ex_34's statistic) is kept. |
| D3b + dense gram W^T W | D3b 1.1182e-8 | 1.5013e-8 | 1.343 | 1708 | rejected |
| dense R leg (exact residual pair power) | - | not measured | - | +483 GF -> 1.05 | over budget with D3b; not built |
| D3b with every neuron a hub (manager's plan) vs with ex_34's hub rules (gate > 0.01 admits, gate >= 0.997 -> P only) | see section 4 | 99: 1.1182e-8 vs 1.0979e-8; 32: 8.421e-9 vs 7.975e-9; 33: 1.68e-3 (terminal divergence) vs 1.0599e-8; 34: 1.2105e-8 vs 1.1796e-8; 40: 1.0217e-8 vs 9.984e-9 | all-hubs is 2.2-5.6 % worse where finite and diverges on net 33 | 1677 both | **hub rules adopted** (final) |

| REVIEW S1: saturated hubs' r and beta from the FULL u, X rows (ex_34's hybrid) vs from the represented rows only (self-consistent) | self-consistent 1.0979e-8 | hybrid 1.0871e-8 | 0.990 | 1677 | **hybrid adopted** (faithful, not worse) |
| TC hub rows: dv > 0.04 (ex_34's COL plan) vs dv > 0.01 | 0.01: 1.0979e-8 | 0.04: 1.1007e-8 | 1.0025 (noise) | 1677 | **0.04 adopted** (faithful) |
| both together (the final statistics) | 1.0979e-8 | 1.0893e-8 | 0.992 | 1677 | final |

Why the hub rules are statistics, not economy: the PK4 dressed leg g' = (H - H_aa P)/sqrt(e) = u + (X - coef P)/(2e) of a hub
whose Edgeworth-corrected curvature e nearly cancels (dv = 1.0, e ~ 1e-9 on net 33) reaches |g'| ~ 230 (sources 12-14 at the
terminal) and the terminal q4 correction 10; hubs with 0.01 < dv < 0.997 stay below 0.22. ex_34 never evaluates such hubs:
saturated hubs keep only P (no H row, PK4 live mask 0) and dead hubs are not admitted. The "fully uncropped" plain copy used
for validation still dropped 64 hubs of sources 13-14 on net 33 through `_hub_masks` (RESULTS section 2 / DISCREPANCIES 5),
which is why its terminal PK4 read stayed at 1e-3. The exact repeated-index image's divergence on net 32 (row above) had the
same origin (large stashed-slice entries from those hubs); with the hub rules it was not re-measured.

Final program vs references on net 99: 1.0894e-8 / 1.1051e-8 (cap-320) = **0.9858**; / 1.1065e-8 (ex_34 graded) = 0.9846.

### 3b. Memory churn and cuts (statistics unchanged)

- Churn fixes (REVIEW c): pair_power through one preallocated float64 scratch (`np.multiply(..., out=)`), the deposit's undo
  copies in a preallocated (3, S, n, n) float32 scratch (+180 MiB persistent, no churn; the free-looking `legs[-1]` cannot be
  used: generation 3 is read by the Horner query), the birth's float64 temporaries deleted before the transport. Net 99
  predictions bitwise identical before/after; peak RSS 3.63 GiB -> 2.94 GiB (default threads), 2.00-2.16 GiB at OMP_NUM_THREADS=4.
- Cuts (REVIEW d, e): 993 -> 880 lines by structure (debug hook, Frame class, core_diagonal, PK4.products, tc_pending/consume
  merge, inline right scales, per-source sums for the repair slice and the old-repair read, theta once per deposit, banners),
  then 880 -> 807 by moving the long docstrings to README.md. 807 vs 880: bitwise identical predictions on net 99. 880 vs the
  pre-cut file: max relative difference 1.0e-7 (the repair slice and old-repair read as per-source float64 sums instead of one
  flat float32 product), raw 1.089435e-8 vs 1.089330e-8 (+0.01 %).

## 4. Final table

### 4a. Dev nets (final 807-line file, OMP_NUM_THREADS=4, isolated process via tools/run_net.py; net 99 also via isolated_rss.py)

| net | clean raw | plain cap-320 raw | ratio | GF | mult | wall (s) | peak RSS (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 1.08944e-8 | 1.10513e-8 | 0.9858 | 1677.4 | 0.763 | 18.8 (isolated_rss.py) / 24.5 | 2.16 (isolated_rss.py) / 2.00 |
| 32 | 8.08401e-9 | 7.67821e-9 | 1.0528 | 1677.4 | 0.763 | 18.1 | 2.74 |
| 33 | 1.10109e-8 | 1.03616e-8 | 1.0627 | 1677.4 | 0.763 | 18.0 | 2.77 |
| 34 | 1.19170e-8 | 1.25771e-8 | 0.9475 | 1677.4 | 0.763 | 17.0 | 2.71 |
| 40 | 1.02939e-8 | 9.98562e-9 | 1.0309 | 1677.4 | 0.763 | 16.8 | 2.73 |
| **pooled** | **1.04391e-8** | 1.03308e-8 | **1.0105** | 1677.4 | 0.763 | | |

(The earlier self-consistent-S1 / TC 0.01 build gave pooled 1.02666e-8 = x0.994 on the same nets; the faithful S1/TC choices
cost +1.7 % pooled on these five nets, within the +-2 % noise of five nets, and are kept for faithfulness.)
GF = tallied FLOPs (matmul 1655.6 + solve 21.5 + SVD at FlopScope tariff 0.33); the 27 LAPACK SVDs (~12 n^3 each, ~350 GF)
and elementwise O(n^2) work (~15 GF) are not tallied. Peak RSS = ru_maxrss of the isolated process.

### 4b. Final 20-net table (nets 600-619)

Final 807-line file, OMP_NUM_THREADS=4, one net at a time through tools/run_net.py (isolated process). plain cap-320 =
`source/estimator_plain` with the elder cap at 5n/16 (= ex_34's statistics); k3 reference = clean/k3_reference.py (Fable B,
factored ARC K=3, float64, mult 0.925). GF = 1677.4 (mult 0.763) on every net.

| net | clean raw | plain cap-320 raw | ratio | k3 reference raw | clean wall (s) | clean peak RSS (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 1.23161e-08 | 1.26180e-08 | 0.9761 | 3.88959e-08 | 17.0 | 2.83 |
| 601 | 1.11950e-08 | 1.08244e-08 | 1.0342 | 3.59376e-08 | 17.9 | 2.87 |
| 602 | 1.07543e-08 | 1.14318e-08 | 0.9407 | 3.32881e-08 | 17.0 | 2.83 |
| 603 | 1.30634e-08 | 1.24638e-08 | 1.0481 | 3.40075e-08 | 17.2 | 2.74 |
| 604 | 1.15616e-08 | 1.09329e-08 | 1.0575 | 3.31981e-08 | 16.3 | 2.71 |
| 605 | 1.38354e-08 | 1.38453e-08 | 0.9993 | 3.11054e-08 | 16.1 | 2.70 |
| 606 | 1.30588e-08 | 1.32696e-08 | 0.9841 | 3.94277e-08 | 16.2 | 2.85 |
| 607 | 1.25957e-08 | 1.25766e-08 | 1.0015 | 3.92359e-08 | 16.5 | 2.87 |
| 608 | 1.29713e-08 | 1.28686e-08 | 1.0080 | 3.55709e-08 | 16.5 | 2.87 |
| 609 | 1.16124e-08 | 1.14156e-08 | 1.0172 | 3.44836e-08 | 17.0 | 2.76 |
| 610 | 1.21948e-08 | 1.24491e-08 | 0.9796 | 3.02662e-08 | 16.5 | 2.85 |
| 611 | 1.19033e-08 | 1.21189e-08 | 0.9822 | 3.12700e-08 | 17.0 | 2.76 |
| 612 | 1.21183e-08 | 1.11091e-08 | 1.0908 | 2.89615e-08 | 17.0 | 2.84 |
| 613 | 1.36698e-08 | 1.35467e-08 | 1.0091 | 4.16944e-08 | 17.9 | 2.73 |
| 614 | 1.15962e-08 | 1.08276e-08 | 1.0710 | 3.10681e-08 | 16.9 | 2.74 |
| 615 | 1.56280e-08 | 1.55050e-08 | 1.0079 | 3.90801e-08 | 17.4 | 2.79 |
| 616 | 1.09044e-08 | 1.11926e-08 | 0.9743 | 3.91575e-08 | 16.6 | 2.73 |
| 617 | 1.15883e-08 | 1.13312e-08 | 1.0227 | 4.04476e-08 | 17.0 | 2.72 |
| 618 | 1.15190e-08 | 1.19433e-08 | 0.9645 | 4.11355e-08 | 17.4 | 2.87 |
| 619 | 1.09744e-08 | 1.07822e-08 | 1.0178 | 3.27205e-08 | 17.2 | 2.73 |
| **pooled (20)** | **1.22530e-08** | 1.21526e-08 | **1.0083** | 3.55476e-08 | 16.1-17.9 | 2.70-2.87 |

Per-net ratio: min 0.941, median 1.008, max 1.091; 12 of 20 nets above 1. Pooled ratio to the k3 reference: 0.345.

**Verdict.** MSE is preserved: the pooled raw of the dense program is 1.2253e-8 against 1.2153e-8 for ex_34's statistics
(cap-320 plain), a pooled ratio of **1.0083**, inside the +-2 % band, and the all-dense history is confirmed to be roughly
lossless against the elder/cropped version (ABLATION_EVIDENCE #7 recorded x1.04 on one net). The per-net spread (0.94-1.09)
is the crop-versus-dense difference acting on individual nets, not a systematic bias. At 1677 GF (multiplier 0.763) the program
sits inside the allowed 0.5-1.0 band; it is 2.9x more accurate than the factored ARC K=3 reference at a comparable bill.


## 5. Open questions

1. Length: 807 lines (target <= 820). The explanation lives in README.md.
2. The float32 cancellation in the PK4 dressed leg for small-curvature hubs (DISCREPANCIES 6) remains inside the admitted
   range 0.01 < dv < 0.997; it is ex_34's statistic, but a float64 X leg (or storing X/(2e) directly) would be the clean fix.
3. (closed) TC hub rows: 0.04 vs 0.01 measured x1.0025 (noise); ex_34's 0.04 used.
4. Dense R leg: not built (multiplier 1.05 with D3b); recorded effect of rank 2 -> 256 elsewhere is -2.9 %.

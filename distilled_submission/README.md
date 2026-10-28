# estimator.py: submission 337479 as one dense numpy program

`estimator.py` reproduces the statistics of submission 337479 (ARC white-box challenge, submission 337479) in one numpy file with plain
`@` products, every FLOP economy removed. This README holds the explanation the code refers to; `RESULTS.md` the
measurements; `docs/DISCREPANCIES.md` the places where the plain source, SPEC.md and this program disagree.

## What is estimated

A bias-free ReLU network z_0 = x W_0, h_l = relu(z_l), z_l = h_(l-1) W_l (row vectors, x ~ N(0, I_n)); we want E[h_l] for
every layer. The estimator is a deterministic moment closure. For every hidden layer it carries the pre-activation mean mu
and covariance pc, the third cumulant through its diagonal z3_i = kappa3(z_i), its (2,1) slice b_ij = kappa3(z_i, z_i, z_j)
and the partial trace rt_j = sum_i b_ij, and the fourth cumulant through a trace core A = T + s I (the member of the family
Sym6(A (x) I) whose partial trace sum_k kappa4(h_i, h_j, h_k, h_k) is exact) plus per-neuron corrections to its diagonal
q4_i = kappa4(z_i). The ReLU is passed with Gaussian integrals corrected by the first Edgeworth terms (z3 and q4); pairs of
neurons use a bivariate Mehler (Gram-Charlier) series with 21 rows (the row set and the ROW_FOLD reuse of rows 17/18/20 are
submission 337479's statistics). The terminal layer forms only its mean, h + z3 f/6 + q4 ds4/24 + z3^2 ds6/72.

## The three ideas that carry the deep part

1. **Spherical-radial factor.** x = (r / sqrt n) u with r ~ chi_n independent of u uniform on the sphere; by homogeneity
   E[h_l(x)] = E[r / sqrt n] E[h_l(u)]. The sphere's law is propagated (no third cumulant, isotropic fourth cumulant
   A = -I/(n+2) at layer 0) and every mean is multiplied by E[chi_n] / sqrt n at the end.
2. **Unit-prevariance gauge.** Before a hidden layer is processed its weight columns are rescaled so every Var(z_i) = 1 and
   the next weight's rows are divided by the same scales (exact by homogeneity); stored means are divided back. The isotropic
   parts of the closure (the trace core's s I, the thresholds) are gauge-dependent, so the gauge is part of the statistics.
3. **The cubic source bank.** Each hidden layer's ReLU creates a third cumulant, a "source" born at that layer, stored over its
   birth-layer neurons a ("hubs") as three (n, n) legs with hub rows: u_a = (pc dv)_a (the Gaussian leg), X_a = extra_a (the
   coupling of the incoming slice, the Mehler second order and Q22 into the square) and P_a (the birth direction, delta_a at
   birth):
       B_s = sum_a [ e_a Sym(P_a, u_a, u_a) + (1/2) Sym6(P_a, u_a, X_a) + Sym(P_a, P_a, R_a) + beta_a P_a^(x3) ].
   This is submission 337479's completed square Sym(P, H, H), H = sqrt(e) u + X / (2 sqrt e), with its spurious F (x) F part (F = X/(2 sqrt e))
   removed exactly (SPEC 1.6(ii)(B); submission 337479 removed it at rank one with a negative packet -Sym(K, G, G) and not at all for
   source 0). R_a is a rank-2 repair of the remaining (2,1) pair power plus one conservation column that keeps every column
   sum exact; beta_a is the hub's residual diagonal. Legs are transported one layer at a time by the first-order gate law
   kappa3(h) ~ D^(x3) kappa3(z), Leg -> (Leg o dv) W, and read at every later layer:
   - the **deposit**: z3, the (2,1) slice b (coefficient rows C_u = 2 e P o u + P o X, C_X = P o u, C_P = e u o u + u o X +
     2 P o R + beta P o P, so b_ij = sum_s sum_a sum_leg C_leg_ai Leg_aj) and the partial trace rt;
   - the **trace correction TC** (consumers 4..15): a kappa4 term produced when the newest source's quadratic response meets
     the third cumulant it inherited from the older sources, dq = 12 sum_i Zp Qt, delivered to q4 and, folded with theta =
     -mu/(3v) into the slice, as b + diag(theta) Q31; the TCs of the two previous consumers are re-read through the nested
     history query (submission 337479's Horner form; with no elders it equals the direct sum of SPEC 2.10 exactly);
   - the **directional response E6**: the (2,1,1) contraction T(u) along the variance-response direction u = ga1, beyond its
     radial image in the core, chained through the core by rho = 2 kappa/(n+4) and folded into the slice for one deposit;
   - **PK4**: tree-level fourth cumulants (star, u and (3,1) families) read elementwise from the legs minus their trace-core
     images, plus a single-site correction replacing the core's own model of kappa4(h_a) by the exact one, and a persistent
     response fold of the star + u quartic into the coefficient rows.
   A covariance-star component Sym3(v (x) C) of kappa3 rides on the covariance as one vector v (transported exactly); the
   radial solve (2C + tr(C) I) v = rt fits it to the delivered partial trace.

## What is kept and what is dropped

Kept (submission 337479's statistics): Edgeworth and Mehler orders, the 21-row structure with ROW_FOLD, the trace core A = T + s I,
TCP_START = 4, TRACE_HISTORY_DEPTH = 3, the terminal TC, PK4 families star/u/(3,1) read at every age, the PK4 fold persistent
versus the directional fold undone (submission 337479's asymmetry: the restore leaves the PK4 fold only on the newest source's C_u, C_X
rows), the z3^2/72 terminal term, the edge block only before layer 4, the TC repeated-index image at rank n/256 = 4, the
rank-2 repair + conservation column, the diagonal metric gram after layer 0, and the two hub rules of a birth (a neuron is a
hub if its corrected gate exceeds 0.01; a hub with gate >= 0.997 keeps only its P leg, no square, no PK4) together with the
TC's hub rows (gate > 0.04). The hub rules are also the closure's numerical guard: without them the PK4 dressed leg
(X - coef P)/(2e) of hubs whose Edgeworth-corrected curvature nearly cancels reaches 1e2 and the terminal q4 diverges.

Dropped (FLOP economies of submission 337479): every crop and thinning plan (COL, S, OUT, row plan and its rank-2 fill, terminal column
plan, core and covariance crops), the 64-padding, the shared elder basis and the switch schedule, the block-Krylov SVD
(exact LAPACK SVDs instead), the rank-one negative packet (exact removal instead), fused and batched product forms.

Code conventions: one function per mechanism, formulas in comments; `mm`, `svd_top`, `solve` tally FLOPs (SVDs at
FlopScope's truncated-SVD tariff 4 m n k); vectors float64, (n, n) legs and coefficient rows float32, float64 accumulators
for the cross-source sums (slice, z3, rt, dq, PK4); `col(v)` / `row(v)` scale the rows / columns of a float32 matrix.

## Memory census (n = 1024, L = 16, S = L - 1 = 15 sources)

| persistent array | shape, dtype | MiB |
| --- | --- | ---: |
| Bank.legs (3 legs x 4 generations) | 12 x (S, n, n) float32 | 720 |
| Bank.coef (C_u, C_X, C_P) | 3 x (S, n, n) float32 | 180 |
| Bank.coef_saved (undo copies of the deposit's temporary fold) | 3 x (S, n, n) float32 | 180 |
| Bank.R, Bank.PP | 2 x (S, n, n) float32 | 120 |
| LEFT/RIGHT/RREAD/rread_saved/beta/e, TC chain, PK4 tables | | < 12 |
| engine n x n per layer: pc, Cov(h), T, b, 16 Mehler matrices float32; float64 cov/post/pv3/u/half/trace/block | | ~ 160 |
| pair-power scratch (one float64 (n, n)) | | 8 |
| weights raw + gauged | 2 x 16 x (n, n) float32 | 128 |
| **total persistent** | | **~1.5 GiB** |

Transient per layer: the birth's float64 extra/u/X/own/R (5 x 8 MiB, deleted before the transport), the TC's float64
Q/Y/effc/Zp/Qt/completion (~10 x 8 MiB), per-source (n, n) float32 products in the folds and PK4 reads. No object with n^3
entries is ever formed; every product over sources is a per-source loop. Measured on net 99: tracemalloc 1.17 GiB live /
1.52 GiB peak before the churn fixes; process peak RSS 3.63 GiB before and 2.94-3.01 GiB after them with default BLAS threads,
2.0-2.8 GiB at OMP_NUM_THREADS=4 (macOS keeps freed large temporaries; the plain reference shows 4.8-6.1 GiB in the same runner).

## Measured decisions (details and numbers in RESULTS.md)

- dense metric gram W^T W instead of its diagonal: x1.34 on net 99, rejected;
- exact F (x) F removal including source 0: x1.004 vs the completed-square form, accepted; keeping source 0's term x0.9998;
- the TC's newborn leg content M = e u (the exact limit of submission 337479's sqrt(e)(H - L G)); SPEC's first reading e u + X/2 gave x2.3;
- exact repeated-index image (full off-diagonal slice): x0.9975 on net 99 but divergent on net 32, rank 4 kept;
- every neuron a hub: 2-6 % worse on four dev nets and divergent on net 33; submission 337479's hub rules kept;
- saturated hubs' r/beta from the full rows (submission 337479's hybrid) vs the self-consistent form: x0.990, hybrid kept;
- TC hub rows dv > 0.04 (submission 337479's COL) vs 0.01: x1.0025 (noise), 0.04 kept;
- dense residual leg R (exact repair): not built, multiplier 1.05 with the three-leg form.

## Running

`tutorial.ipynb` is a runnable example: one public network, four estimators (K=1, K=2, the K=3 reference and this program),
the per-layer MSE plot `tutorial_mse_per_layer.png` and a table of final-layer MSE, FLOPs and wall time.

`Estimator().predict(mlp, budget)` returns the (16, 1024) predicted means; `mlp.weights` is the list of 16 float32 (1024, 1024)
matrices applied as `h = relu(h @ W)` to `x ~ N(0, I)`. Only numpy is needed. The run prints the FLOP tally (GF and
multiplier). The development tools that exported networks and compared per-layer states against the graded code live in
the team's research repository and are not part of this release.

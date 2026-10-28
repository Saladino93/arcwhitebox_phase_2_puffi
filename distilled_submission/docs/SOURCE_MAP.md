# SOURCE_MAP: ex_37 plain estimator (`source/estimator_plain/`)

This map covers the non-FMM rewrite `source/estimator_plain/` (5,412 lines). It is the source for the 500-800 line
numpy+BLAS distillation of ex_34 (submission 337479). Paths below are relative to `source/estimator_plain/`. Line
ranges are whole functions, from an AST scan of the copied files. Constants are listed with their values at
width 1024 and depth 16.

## 0. Provenance, and how the plain version relates to ex_34

- The graded ex_34 package is `source/ex_34/{estimator,bank,fmm,pk4}.py` (3,813 lines, FMM).
- ex_37 is ex_34 with two changes. In `bank.py:649` the elder-basis SVD cap goes from `5 * n // 16` (320) to
  `n // 4` (256). In `fmm.py` a ten-line final-combine shortcut is deleted, which is FMM arithmetic only. Checked
  with `diff`. `estimator.py` and `pk4.py` are byte-identical between ex_34 and ex_37.
- `estimator_plain` is the ex_37 refactor rewritten without FMM, caches or round-trip machinery. Its README claims
  rounding-level equivalence. Its cap is `n // 4` at `bank.py:1340`, so **its statistics are ex_34 except for the
  elder cap (256 instead of 320).**
- Measured on net 99 with `scripts/numpy_harness.py` (unmetered; FMM forced to plain BLAS); details in
  RUN_PLAIN_NET99.md:

  | variant | final-layer MSE |
  | --- | ---: |
  | ex_34 as graded | 1.10649e-8 |
  | ex_34 with the cap set to 256 (scratch copy) | 1.11861e-8 |
  | plain (cap 256) | 1.11783e-8 |
  | plain with the cap set to 320 (scratch copy) | 1.10513e-8 |

  plain / (ex_34 with cap 256) = 0.9993 and (plain with cap 320) / ex_34 = 0.9988, both rounding level. The cap alone
  accounts for +1.09 % MSE on net 99. **To reproduce ex_34's statistics, set `bank.py:1340` to `5 * n // 16`.**
- Version note: `requirements-dev.txt` pins numpy 2.4.6. The local venv has numpy 2.2.6 (Accelerate),
  flopscope 0.12.3 and whestbench 0.16.2.

### ex_34 flag names and their plain equivalents

| ex_34 (`source/ex_34`) | plain | value |
| --- | --- | --- |
| `SWITCH_AGE`, `EARLY`, `AGE_SWITCH_MAX_BIRTH` | `ELDER_SWITCH_AGES`, `EARLY_BIRTH_COUNT`, `LAST_ELDER_BIRTH` (bank.py:307,333) | (6, 7), 4, 7 |
| `REPAIR_RANK` (KSVD repair) | `REPAIR_RANK` (bank.py:307) + conservation column: `Bank.rank = 3` | 2 (+1) |
| `KSVD, KSVD_P, KSVD_MAXK` | `_svd_topk`, `KRYLOV_OVERSAMPLING`, `KRYLOV_MAX_RANK` (bank.py:668-669) | on, 2, 8 |
| `THIN_TAU, THIN_OUT, ROW_THIN, V_SAT, TERM_KEY` | `THIN_TAU, THIN_OUT, HUB_GATE_THRESHOLD, SATURATION_TOLERANCE, TERMINAL_COLUMN_THRESHOLD` (bank.py:313) | 0.04, 0.015, 0.01, 0.003, 0.01 |
| `OUT_THIN_TAU` | `TRANSPORT_OUTPUT_THRESHOLD` (bank.py:358) | 1e-3 |
| `SLICE_ROWS, SLICE_SAT, SLICE_UROW="fill"` (Lane L1) | `_slice_row_plan` + `_fill_omitted_rows`; `SLICE_SATURATION_TOLERANCE` (bank.py:368) | on, 0.015 |
| `TCP_START`, `TERM_TC_TAU`, `TCH` | `TCP_START`, `TERM_TC_TAU`, `TRACE_HISTORY_DEPTH` (bank.py:343,346,351) | 4, 0.3, 3 |
| `TC_SLICE_FOLD`, `HIST_FOLD`, `HF_LITE_ROWS` | `_deposit_folded_slice`, `_history_fold_coefficients`, `_history_fold_lite` | always on |
| `SB, _SB_TAU2, _SB_COND`; cap `5n/16` | shared elder basis; `SHARED_BASIS_TAIL_TOLERANCE`, `SHARED_BASIS_CONDITION_LIMIT` (bank.py:375,380); cap `n//4` (bank.py:1340) | 2^-8, 2^11.5; **256 (ex_34: 320)** |
| `SB_E1`, `SB_E2` | exact reassociations kept inline (bank.py:1352-1374, 1424-1436) | always on |
| `HUB_EPS` | `HUB_DROP_TOLERANCE` (bank.py:2770) | 1e-3 |
| `E_TU, E_RADIAL='mehler1', E_PROP`; E_GA | `_prepare_directional_response`, `_read_directional_response`, `_apply_temporary_directional_fold`; `response_corrections` | always on |
| `GM_TAU_A, GM_OUT`; `GM_TAU_A_LATE, GM_OUT_LATE`; `TERM_GA`; `PC_TAU_A, PC_OUT` | `CORE_INPUT/OUTPUT_THRESHOLD`; `LATE_CORE_*`; `TERMINAL_CORE_THRESHOLD`; `COVARIANCE_INPUT/OUTPUT_THRESHOLD` (moments.py:77-89) | 0.1/0.3; 0.015/0.05; 1e-4; 1e-3/1e-3 |
| `TF_OUT, TF_K` (edge product) | `EDGE_OUTPUT_GATE_THRESHOLD, EDGE_CONTRACTION_THRESHOLD` (moments.py:71) | 0.03, 0.1 |
| `PK4_FAMS, PK4_AGES, PK4_TERM_AGES, PK4_CORESUB="lite"`, pk4 `FOLD` | `FourthCumulantCorrections(hidden_ages=(1,16), terminal_ages=(1,64))` (pk4.py:179-180): star/u/(3,1) families minus the trace-core image; `apply_persistent_response_fold` | every age is read at depth 16 |
| `NFOLD = 21`, `ROWS` (Mehler coefficient buffer, "CB") | `FOLDED_PAIR_COUNT = 21`, `PAIR_ORDERS` (moment_formulas.py:215-248); `MehlerRows.stack` | 21 matrices out of 26 rows |
| `GAUGE_POLICY = "unit_prevariance"` | `_normalize_layer` (always) | |
| `STAR_AGES/STAR_TERM` Gaussian star history | gone: PK4's star family covers every age in ex_34 too (`_star_owned`) | |

Names in the brief that do not exist as such:
- **`PK4_MAX_AGE = 5`** is not a constant in ex_34 or in the plain version. PK4 reads every source age at every
  consumer: hidden window (1, 16) widened to every young source, terminal window (1, 64).
- **"NegativeBank rank n/32"**: the negative packet has rank one per source (`_svd_topk(completion, 1)`,
  moments.py:343). The n/32 figure appears only in the screening history of the TC repeated-index rank, which is
  n/256 (bank.py:759-761).
- ex_34's `RECOMP_AGE = 4` / `n // 8` recompression and its per-source `n/4` / `3n/16` bases (ex_34 bank.py:17,
  516, 1304) belong to the per-source elder path. The shared basis replaces that path, and the plain version keeps
  only the shared one.

## 1. Mechanism table

Abbreviations: E = `estimator.py`, M = `moments.py` (class `MomentEngine` unless noted), MM = `moment_math.py`,
MF = `moment_formulas.py`, B = `bank.py` (class `Bank` unless noted), K = `pk4.py` (class
`FourthCumulantCorrections`), SC = `state_contracts.py`.

### 1a. Driver, input law and gauge

| Mechanism | Where | What it computes | Constants / flags |
| --- | --- | --- | --- |
| Entry, routing, final clip | E:`Estimator.predict` 270-335 | Chooses a route (point mass, depth 1, width <= 2, engine) and clips the result at 0. | `_gaussian_reserve` E:99-104 |
| Per-layer driver | M:`predict_weights` 464-533 | Per layer: star transport, `bank.deposit`, star images, `tc_pending` + PK4 into `extra_fourth`, `_advance_hidden_moments`, birth, `_complete_layer_transition`. The terminal layer runs `_terminal_mean`. | sequence documented in `ex_34_refactored_README.md` §Per-layer pipeline |
| Phase boundary | M:`_complete_layer_transition` 405-462 | Order: TC stash; prepare next frame; directional context; `bank.accumulate`; `pk4.record_birth`. Statement order is part of the numerics. | TC stash when `TCP_START <= l+1 <= L-1` (M:421-422) |
| Spherical-radial factorization | MM:`radius_mean` 226-244; applied M:533 | E[h(x)] = E[r/sqrt n] E[h(u)]. The engine propagates the uniform-sphere law and multiplies every layer mean by E[chi_n]/sqrt n (log-recurrence). | none |
| Sphere at layer 0 | M:`_transport_fourth_core` 582-584; M:`_initialize` 226-237 | Layer-0 core A = -I/(n+2), so gm = -pc/(n+2) and gram = pc. Layer-0 stf scale is 2 - 4/(n+2). Initial z3 = b = 0. | `first_stf_scale`, `core_trace_coefficient` = -1/(2(n+2)(n+4)) |
| Unit-prevariance gauge | M:`_normalize_layer` 1104-1145 | Divides W rows by the previous scale; pc = W^T Cov(h) W; then rescales columns by 1/sqrt(diag pc), so diag pc = 1. Means are divided back (M:523). The terminal layer is not gauged (M:`_prepare_next_layer` 384-403). Exact by homogeneity before closure, but the isotropic `I` in the trace core and the thresholds are gauge-dependent, so keep it. | none |
| Covariance transport W^T C W | M:`_propagate_covariance` 661-715 (hidden); M:`_terminal_variance` 617-628 (terminal diagonal only) | Contracts only the previous-layer rows with corrected gate > 1e-3; dropped rows add only their diagonal C_aa W^2. Off-diagonal entries only on columns with new Gaussian gate > 1e-3. Builds the Gaussian gate cache and `gate_plans`. | `COVARIANCE_INPUT_THRESHOLD = COVARIANCE_OUTPUT_THRESHOLD = 1e-3` (M:89) |
| Gate frame cache | M:`live_gate` 737-757; SC:`GateFrame` 20-32 | Caches (sd, z, Phi(z), v) of the current frame, keyed by array identity. | none |

### 1b. Unary and pair laws

| Mechanism | Where | What it computes | Constants / flags |
| --- | --- | --- | --- |
| Gaussian ReLU basis | MF:`gaussian_relu_basis` 88-106 | h, raw2..raw4 and the mu-derivatives d, e, f, ds4..ds7 as sd^p (P Phi + Q phi). | none |
| Unary Edgeworth closure | MF:`edgeworth_moments` 109-127; called M:799-806 | Corrected m, m2, m3, m4, dv, ev, fv = Gaussian row + z3/6 * 3rd derivative + q4/24 * 4th derivative. dv is clipped to [0, 1] and ev at 0. q4 = 6 ga gv + `extra_fourth` (TC + PK4). | none |
| Unary statistics / families | MF:`unary_statistics` 134-209 | ov, k3, k4 of relu; centred-power families f, g, c (orders 0..5); a1, a2, ga1, ga2, stf, astar, ddv, fv2, a1tc, dz, m7; edge keys and weights (key, lc, lr, oqc, oqr). | `FAMILY_ORDERS = 6` |
| Bivariate Mehler rows | MF:`PAIR_ORDERS` 215-248 (26 rows (r, s, k)); M:`_mehler_rows` 843-899 | Builds the 21 coefficient matrices: 1, x, x^2/2, x^3/6, Q22/4, x Q22/4, b/2, x b/2, b^T/2, x b^T/2, Q31/6, x Q31/6, Q31^T/6, x Q31^T/6, (x^2/2) b/2, (x^2/2) b^T/2, z3_j/6 (k = 0..2), q4_j/24 (k = 0..1). Q22/4 = outer(ga, gv) part + diag; Q31/6 = gvh * gm. | `FOLDED_PAIR_COUNT = 21` (MF:244); x = pc with zero diagonal |
| ROW_FOLD | M:`_contract_pair_powers` inner `left_factors` 912-921 | Folds rows (3,0,k) and (4,0,k) into the left vectors of rows (0,0,k). **Not purely an algebraic fold:** rows 17, 20 and 18 reuse the folded left vectors, which adds (3,3), (4,3), (3,4), (4,4) cross products (XXX at M:919). Reproduce exactly. | none |
| Pair powers cov / post / pv3 | M:`_contract_pair_powers` 901-953 | Each is sum_q M_q[i,j] left_q[i] right_q[j]: Cov(h) (f,f), the next source's (2,1) slice `post` (g,f) and the (3,1) pair power (c,f). Diagonals are set to ov and k3. `cubic_self_coefficient = left_g[1]` feeds the birth. | none |
| U_ROWSUM | M:944-952 | The (2,2) pair power u (g,g) is read only through its row sums (the next trace diagonal). Row 0 is handled analytically against ov ov^T to avoid float32 cancellation. Exact; the n x n matrix is never formed. | none |

### 1c. Third cumulant: covariance star and source bank

| Mechanism | Where | What it computes | Constants / flags |
| --- | --- | --- | --- |
| Covariance-star vector v (`cubic_vector`) | M:480 (transport `pre_vector = v W`); M:494-498 (images); MM:`covariance_star_repeated` 154-160, `covariance_star_trace` 163-169 | A kappa3 component Sym3(v ⊗ C) carried as a single vector. It is transported exactly with the covariance (no history). Its (2,1)-slice C_ii v_j + 2 v_i C_ij, diagonal 3 v C_jj and trace tr(C) v + 2 C v are added to the bank's deliveries. | none |
| Radial solve (Chebyshev) | MM:`_chebyshev_star_vector` 96-121, `radial_parts` 124-138, `_radial_vector` 141-151 | Solves (2C + tr C I) v = t with a Chebyshev series in X = C/\|C\|_F (matvecs only), stopped at eps32. Falls back to float64, then to a direct `linalg.solve`. Used for the trace-core radial image (M:1032) and the star birth. | eps 2^-23 / 2^-52 |
| Star birth | MM:`covariance_star_birth` 172-199; called M:367 | Replaces the fresh core star Sym3(astar ⊗ d gm d) and the gated inherited star by one star in metric Cov(h) with the same partial trace. Returns the (2,1)-slice delta, which is charged to the repair and to beta. | astar = ev gv / 2 |
| Source birth | M:`_create_cubic_source` 302-382 | `extra` (M:319-324) couples the incoming b, C∘C and Q22 into the square. H = sqrt(e) C d + extra/(2 sqrt e) (M:332). Hub diagonal repair (MM:`repair_completed_square_diagonal` 202-223). Negative packet = rank-1 SVD of F (M:343). Repair r = post - images of b, H, star + negative add-back (M:355-362). beta = k3 - dv dz - 3 H_ii^2 + 3 (LG)_ii^2 (M:365). The repair diagonal is moved into beta. | rank 1 (negative); e = 0 guard |
| Hub admission | B:`_admit_source` 2616-2661; B:`_hub_masks` 2773-2799 | Keeps hubs with corrected gate > 0.01. Saturated hubs (gate >= 1 - 0.003) keep only P. Saturated hubs whose kappa3 bound is in the smallest 1e-3 tail are dropped. Rows are reordered (live unsaturated first) and padded to 64. Registers the negative packet. | `HUB_GATE_THRESHOLD = 0.01`, `SATURATION_TOLERANCE = 0.003`, `HUB_DROP_TOLERANCE = 1e-3`, `PLAN_QUANTUM = 64` |
| Repair fit (KSVD = 2) + conservation | B:`_fit_source_repair` 2703-2725; B:`_repair_conservation` 699-739; B:`_svd_topk` 672-696 | Rank-2 truncated SVD of r (deterministic block-Krylov, 3 powers, k + 2 seed rows) plus one analytic `ones ⊗ v` column, so every column sum of the off-diagonal repair is exact. The spurious hub diagonal goes into beta. LEFT is fixed at birth; RIGHT is transported. | `REPAIR_RANK = 2` (rank stored 3); `KRYLOV_OVERSAMPLING = 2`, `KRYLOV_MAX_RANK = 8` |
| Negative packet bank | B:`NegativeSquareBank` 858-1007 (`append` 899-909, `advance` 911-955, `block` 984-1007, `hist_block` 962-977) | Removes the spurious Sym(P, F, F): K = sum L_a^2 P_a, rows [K; G] transported and gated. Slice = -(2 K_i G_i G_j + G_i^2 K_j); diagonal -3 G^2 K. The first source has no packet (XXX B:869). | rank 1 per source |
| Young-source transport (LEG2) | B:`_transport_dense_legs` 2727-2764; B:`_transport` 1171-1183; B:`_gate_history` 2663-2701; fresh P leg B:2477-2484 | Gates every young H/P leg by the source layer's corrected gate dv, then multiplies by W over the COL rows (dv > 0.04) and forms only the S columns (next Gaussian gate > 1e-3). The fresh P leg is just the retained rows of W. Dominant cost. | `THIN_TAU = 0.04`, `TRANSPORT_OUTPUT_THRESHOLD = 1e-3`; LEG2 = `TRACE_HISTORY_DEPTH + 1 = 4` generations of (2(L-1), n, n) |
| Repair transport | B:`accumulate` 2490-2503 | RIGHT (s, 3, n) for every source, elders included, is gated and multiplied by the full W. RECON = LEFT @ RIGHT (s, n, n). | none |
| Elder shared basis G | B:`_gate_history` 2685-2700 (switch rule); B:`_switch_to_shared_basis` 1312-1378; `_factor_shared_basis` 1380-1392; `_reorthogonalize_shared_basis` 1394-1411; `_transport_shared_basis` 1413-1439; `_shared_slice_coefficients` 1441-1450; `_elder_slice_families` 1251-1276 | At age 6 (births < 4) or 7 (later), only for births <= 7 and layer <= L-3, a source's legs become coordinates Y on one shared basis G (d x n). The first entrant sets G by truncated SVD with a 2^-8 tail-energy rule, capped and 64-padded. Later entrants get least-squares Y = M Q R^-T (SB_E1). Only G is transported, and the elders' dense legs are rebuilt as Y @ G (on S) because the readouts are elementwise. | `EARLY_BIRTH_COUNT = 4`, `ELDER_SWITCH_AGES = (6, 7)`, `LAST_ELDER_BIRTH = 7`, cap `n//4` = 256 (ex_34: 320) at B:1340, `SHARED_BASIS_TAIL_TOLERANCE = 2^-8`, `SHARED_BASIS_CONDITION_LIMIT = 2^11.5` |
| Accumulate (z3 diagonal, trace, pending) | B:`accumulate` 2382-2607 | Admit, directional pre-read, gate + switch, install the fresh source + repair, plans, transport legs / P / G / repair / negative, then `_write_slice_coefficients` (CU = 2 P∘H, CP = H∘H + 2 P∘R + beta P∘P, B:2609-2614). z3_j = 3 sum(P H^2 + P^2 R) + beta P^3 - negdiag (B:2505-2519). Partial trace from COMB2 row sums + repair + elders (B:2573-2593), net of the represented diagonal. Leaves `pending`. Terminal: adds z3 on the terminal plan and returns before the LEG2 rotation. | `TERMINAL_COLUMN_THRESHOLD = 0.01` (terminal plan, weight \|z\| exp(0.5 - z^2/2)) |
| Deposit ((2,1) slice) | B:`deposit` 2037-2058; B:`_deposit_slice` 2122-2169 | Applies the temporary directional fold, then the PK4 persistent fold. The slice block on OUT (Gaussian gate > 0.015) = families C^T @ legs + elders C^T Y @ G + repair rread @ rbf - negative packets, minus the directional covariance-star part. Its diagonal is set to z3. Writes `repeated_bank[l]`, `third_bank[l]`, `trace_override[l]`. Undoes the directional fold. | `THIN_OUT = 0.015` |
| Lane L1 row plan + fill (SLICE_ROWS) | B:`_slice_row_plan` 1222-1249; B:`_omitted_row_functionals` 2171-2222; B:`_fill_omitted_rows` 393-429; B:`_deposit_cropped_slice` 2224-2264; B:`_scatter2` 432-456 | Forms slice rows only for OUT rows with gate < 1 - 0.015. Omitted saturated rows get the minimum-norm rank-2 fill that reproduces the two Mehler functionals that urow reads (wr = d - t phi, wc = t(1 - d)), computed exactly from the factored bank. Without the fill, final MSE roughly doubles. | `SLICE_SATURATION_TOLERANCE = 0.015` |
| Trace correction TC (newest) | B:`tc_stash_engine` 1452-1461; B:`_prepare_trace_correction` 1463-1547; B:`_factor_repeated_query` 750-768; B:`_repeated_query_residual` 846-855; B:`_completion_coefficients` 783-823; B:`_complete_bilinear` 826-843 | dq_j = 12 sum_i Zp_ij Qt_ij over COL rows. Q = un-gated old-source query + old-repair read. Zp = W∘effc - isotropic image. Qt = Q minus its repeated-index image, using the source slice B factored at rank n/256 = 4 with its exact diagonal. Also builds the fold coefficients cpf / ch (with distinct-index completion). | `TCP_START = 4` (consumers 4..L-1); rank `n // 256` (B:1519) |
| Deposit with TC fold (TC_SLICE_FOLD) | B:`_deposit_trace_corrected_slice` 1856-1900; B:`_deposit_folded_slice` 1739-1854; `_folded_transport_product` 2802-2823; `_fresh_rows_outside_transport` 1283-1310 | On TC layers the old-source query is built before the lift by W, then lifted with the fresh source's coefficients plus theta x the TC fold, so every reader sees b + diag(theta) Q31, theta = -mu/(3v). From TCP_START on, this is why the engine drops its edge block. | theta = -mu/(3v) |
| Grouped Horner history query + HIST_FOLD | B:`_history_query` 1902-2035; B:`_history_fold_coefficients` 1651-1712; B:`_history_trace_correction` 1714-1737; B:`_consume_folded_trace_correction` 1549-1572 | The query is nested in Horner form, acc_s = (acc_(s-1) + entries) @ (W_s ∘ gate_s) over COL, so that each intermediate is the query for the TC born at t-g (g = 2..3). The history TCs are re-read and their fold coefficients enter the families. On row-plan deposits the "lite" fold omits the completion term. | `TRACE_HISTORY_DEPTH = 3` (chain of 2); lite = row plan in use |
| tc_pending / tc_terminal | B:`tc_pending` 1637-1649; B:`tc_terminal` 1574-1591; B:`_tc_terminal` 1593-1635 | Consumes the newest TC context + history into a scalar dq added to q4. The terminal TC is evaluated only on terminal-plan columns with \|z^2 - 1\| exp(-z^2/2) > 0.3; no slice is formed. | `TERM_TC_TAU = 0.3` |

### 1d. Fourth cumulant

| Mechanism | Where | What it computes | Constants / flags |
| --- | --- | --- | --- |
| Trace core push (affine_fourth) | M:`_transport_fourth_core` 562-615; M:`_alive_rows` 535-548; M:`_core_thresholds` 550-560 | gm = W^T A W with A = T + s I (`core_trace`, `core_isotropic`), over previous-layer rows with dv > 0.1 (late layers 10..14: 0.015). Off-diagonal only on columns with gate > 0.3 (late: 0.05). ga = diag gm on every column. The metric gram = W^T W is kept diagonal (gv) after layer 0. | `CORE_INPUT_THRESHOLD = 0.1`, `CORE_OUTPUT_THRESHOLD = 0.3`, `LATE_CORE_INPUT_THRESHOLD = 0.015`, `LATE_CORE_OUTPUT_THRESHOLD = 0.05` |
| q4 diagonal | M:799-801 | q4 = 6 ga gv + `extra_fourth` (TC dq + PK4). E_GA has already added to ga (M:793-795). | none |
| Core update (core_t / core_s) | M:`_update_fourth_core` 1006-1089; MF:`core_weights` 251-315; M:`_prepare_edge_trace` 955-1004 | Mixed fourth trace t_ij = sum_k kappa4(h_i, h_j, h_k, h_k), built as a half trace Z: Mehler rows 0..10 with g/r weights, pc, gm, radial (x xc xr), outer terms, cubic pair, -3 ov Cov(h), and the edge block before TCP_START. Then trace = Z + Z^T with diagonal = urow row sums, T = t/(n+4), s = -tr t/(2(n+2)(n+4)). Also sets `radial_image` and `core_propagation` (E_PROP kappa). | `EDGE_OUTPUT_GATE_THRESHOLD = 0.03`, `EDGE_CONTRACTION_THRESHOLD = 0.1`; edge block only for layer < `TCP_START` |
| PK4 single site | K:`single_site` 377-403 | Replaces the core's model 6 A_aa of kappa4(h_a) by the exact kappa4(h_a) through diagonal powers p and reads sum_a p W^4 + the core change. | dimension ratios (K:134-153) |
| PK4 leg families (star / u / t31) | K:`record_birth` 206-305; K:`read` 307-375 | Per source: coefficients [isq, de], quartic [[4 c31, 4 f3], [-p_u, 6 V]] and image rows. Read: sum over sources of 4 c31 P^3 g' + 4 f3 P g'^3 + 6 V P^2 g'^2 - p_u P^4 minus the trace-core images (CORESUB "lite"), with g' = isq H - de P. Hidden: LEG2[0], every young source + window; terminal: LEG2[-1]. | `hidden_ages = (1, 16)`, `terminal_ages = (1, 64)` (every age) |
| PK4 response fold (fold_response) | K:`apply_persistent_response_fold` 405-498; called B:2135-2136 | Adds theta x the quarter-gradients of the star + u residual quartic into the COMB2 rows of young sources before each deposit. It is not undone: the next accumulate's directional pre-read sees it. | theta = -mu/(3v); young sources only (XXX K:423) |
| PK4 call site | M:`_fourth_correction` 248-266 | single_site + read (None at layer 0). | none |
| E6 directional response: pre-read (`_e_pre`, E_TU) | B:`_prepare_directional_response` 2266-2332; context set at M:429-439 | Hub coefficients x2 = (H·u, P·u), xr = R·u (u = ga1) of young sources, chained with the previous level by rho = 2 kappa/(n+4) (E_PROP). T(u)_aa is read from the previous COMB2, PP and the negative cache. | rho from M:1053-1055 |
| E6 directional read (`_e_read`, E_GA) | B:`_read_directional_response` 2334-2380; consumed M:793-795 (hidden) and M:283-286 (terminal) | dga = -(dvec @ W^2 - read)/(n+4) net of the radial image (Mehler first order, E_RADIAL = 'mehler1'). Added to the next layer's core diagonal ga. Zero off S. | none |
| E6 directional fold (`_g_fold`) | B:`_apply_temporary_directional_fold` 2060-2120 | Adds the theta-weighted (3,1) image of T(u) to CU, CP, RREAD and the newest CP row for this deposit only, and undoes it afterwards. The newest-source part is excluded from the TC query (B:1764-1775). | s = 3 theta gv/(n+4) |

### 1e. Plans, terminal readout and fallbacks

| Mechanism | Where | What it computes | Constants / flags |
| --- | --- | --- | --- |
| Gate thinning plans | B:`_plan` 544-557; B:`GatePlans` 575-625; B:`_sub_plan` 560-572; B:`_pad` 539-541; B:`_scatter` 635-658; MM:`plan_mask` 75-84; M:`_prepare_gate_plans` 717-735; `vertex_plans` M:808-817 | Each plan keeps the largest gates, max(64, pad64(#gate > tau)). Its order is the argsort; `_scatter` restores natural order with zeros. `gate_plans` is built on the Gaussian gate (thresholds 1e-3, 0.015, 0.03, 0.3, 0.05) and `vertex_plans` on the corrected gate (1e-3, 0.1, 0.04, 0.015). Plans named in the bank: COL (THIN_TAU), S (`_output_transport_plan` B:1196-1220), OUT (THIN_OUT), row plan (B:1222-1249). | `PLAN_QUANTUM = 64` (padding changes which neurons are kept, so it is part of the approximation) |
| Terminal readout | M:`_terminal_mean` 268-300; M:`_terminal_core_diagonal` 630-659; M:`_terminal_variance` 617-628; terminal branch of M:`_prepare_next_layer` 395-403 | Mean = h + z3 f/6 + q4 ds4/24 + (z3^2/72) ds6 (M:300), Gaussian basis in network coordinates. z3 = bank terminal diagonal + 3 v C_jj. q4 = 6 (ga + E_GA) gv + PK4 + terminal TC. ga is read only where \|(z^2 - 1) exp(-z^2/2)\| > 1e-4 (`terminal_ga`). | `TERMINAL_CORE_THRESHOLD = 1e-4`, `TERM_TC_TAU = 0.3`, `TERMINAL_COLUMN_THRESHOLD = 0.01` |
| DegenerateMoments fallback | B:`DegenerateMoments` 459-464; raised in M:1091-1102, B:1010-1034, B:1384-1392, B:803-815, MM:110-111, B:2630-2631; caught E:315-322; E:`_fallback` 248-268 → E:`_gaussian_fallback` 159-195 | Unsupported states (non-finite or non-positive variance, singular shared QR, degenerate completion, empty source) switch the whole network to a first-order Gaussian closure. Recognized LinAlgError messages do the same (E:`_is_solver_failure` 218-231). Not triggered on net 99 (`fallback=None`). | none |

## 2. Economy and safety versus statistics

The plain version's FLOPs are 27.8 % of 2^41 (multiplier 0.278). At multiplier <= 1.0 there are about 1,590 GF of
headroom. Its README states that every approximation of the FMM version is kept. The classes below tell the
distiller what can go.

### 2a. Exact economy: drop or rewrite freely (rounding-level change only)

- **Batching and fused forms.** `_sum_families` / `_equal_source_groups` (B:504-536) form one batched matmul per
  equal-row run. `_folded_transport_product` (B:2802-2823) reuses the transported fresh H. `_fresh_rows_outside_transport`
  concatenation (B:1818-1830). The reassociation of `_repeated_query_residual` against `_contract_repeated_query`. The
  Horner nesting of the *newest* query in `_history_query`; the history queries' content still depends on it, see 2c.
- **SB_E1 / SB_E2 reassociations** (B:1352-1374, 1424-1436): Y = U_d S_d; Y = M (Q R^-T); G columns reused from the
  transport product.
- **U_ROWSUM** (M:944-952): the u pair power could be formed as an n x n matrix, but keep the ov ov^T cancellation or
  use float64. **ROW_FOLD** (M:912-921) is an exact fold for rows 21-25, **but** the reuse of the folded left vector
  in rows 17/18/20 is part of the statistics.
- **DEAD_DIAG** skips (M:612, 855, 867, 1066).
- **Plan bookkeeping.** `GatePlans` sharing one argsort (identical plans); `_scatter` via `put_along_axis` versus
  pad + take; `_write_transport_columns` (zeroing persistent buffers); `_scatter2`; the hub-order permutations
  (`_kept`, `_rank`, `hub_order`). All of these disappear if the crops are dropped.
- **Chebyshev radial solve** (MM:96-151) equals `linalg.solve(2C + tr C I, t)` to eps32.
- **Persistent buffers.** LEG2 / COMB2 / RECON / PP / RREAD rotation and preallocation; only the generations needed
  by the history query matter (`TRACE_HISTORY_DEPTH + 1`).
- **Metering-only code.** `scalar` / `_scalar` / `_ratio` / `_dimension_constants` (metered float64 constants),
  `_structural_constants` (K:147-153), the float32 cast in `normal_cdf`, and dtype-by-name checks.
- **Explicit einsum paths** in PK4 `read` (K:358-370) and the binary-only contractions in
  `_omitted_row_functionals`.

### 2b. Economy approximations: affordable at multiplier <= 1.0, but dropping them changes MSE

The recorded effects are from the docstrings: single-setting screens on 1024 x 16 development networks. They are not
re-measured here. Dropping all of them at once has not been measured.

| Item | Where | Recorded effect / note |
| --- | --- | --- |
| COL crop `THIN_TAU = 0.04` (transports and TC rows) | B:313-331, 2422 | 0.05 costs 16 % MSE per point saved. The outside-COL fresh P rows are kept exactly. |
| S crop `TRANSPORT_OUTPUT_THRESHOLD = 1e-3` | B:352-358, 1196-1220 | 5e-3: +6.9 % MSE; 5e-4: no MSE gain. Not exact for z3, the trace and the directional read. |
| OUT crop `THIN_OUT = 0.015` | B:2137 | 0.01 gave x0.997 (noise) |
| Row plan + rank-2 fill `SLICE_SATURATION_TOLERANCE = 0.015` | B:1222-1249, 2171-2264 | Saves ~22 GF for +0.9 to 1.6 % MSE. If the row crop is dropped, the fill goes too (the full rows are exact). |
| Hub crops `HUB_GATE_THRESHOLD = 0.01`, `SATURATION_TOLERANCE = 0.003` (P-only saturated hubs: H rows zeroed), `HUB_DROP_TOLERANCE = 1e-3` | B:2616-2661, 2773-2799 | 1e-2 drop tolerance: +6.5 % MSE; 1e-4 adds cost, no gain. Saturation 0.01 costs 35 %/pt. |
| Terminal crops `TERMINAL_COLUMN_THRESHOLD = 0.01`, `TERM_TC_TAU = 0.3`, `TERMINAL_CORE_THRESHOLD = 1e-4` | B:2448-2459, 1606; M:630-659 | Noise-level in screens |
| Engine crops: covariance 1e-3/1e-3, core 0.1/0.3, late core 0.015/0.05 (layers 10..14), edge 0.03/0.1 | M:62-89 | Core 0.2/0.5: +8.3 % MSE. Diagonal-only core push: MSE nearly doubles. |
| `PLAN_QUANTUM = 64` padding | B:337 | Changes which neurons are kept; irrelevant once the crops are dropped |
| Diagonal metric gram = diag(gv) after layer 0 | M:574-576, 589-590 | x1.002 raw MSE for -5.5 GF |
| Shared elder basis (switch ages 6/7, cap 256, tail 2^-8, `LAST_ELDER_BIRTH = 7`, layer <= L-3) | B:2685-2700, 1312-1439 | Ages (7, 8): -2.8 % MSE. Cap 320 against 256: -1.08 % MSE on net 99 (measured here). With every source dense ("no elders") the MSE effect is unmeasured. |
| Block-Krylov `_svd_topk` instead of an exact truncated SVD | B:672-696 | Introduced for server wall time; it is an approximation (depth 2) |
| Lite history fold on row plan (completion term omitted) | B:1688-1692 | Saves 0.8 GF; MSE unchanged to 2e-4 on one network |

### 2c. Statistics truncations: keep to reproduce MSE (economy-motivated knobs)

These define the estimator's statistics. Changing them changes MSE in the recorded direction.

| Item | Value | Recorded effect |
| --- | --- | --- |
| Repair rank (+ conservation column) | 2 | Rank 3: -0.03 % MSE |
| Negative packet | rank 1 per source; none for source 0 | Not screened |
| TC repeated-index rank | n/256 = 4 | Lower ranks scored better (adjusted) |
| `TCP_START` | 4 | Starting at 3: -2.3 % raw MSE |
| `TRACE_HISTORY_DEPTH` | 3 | 4: about -1 % MSE |
| Terminal TC on | yes | Dropping it: +1.8 % MSE |
| PK4 families | star, u, (3,1) only; path family excluded | Path family: -1 % MSE for 9.7 GF |
| PK4 response fold | young sources only | Elders excluded, rationale not recorded |
| Edge block | only before `TCP_START` | Replaced by the TC slice fold |
| Directional fold | temporary (undone); PK4 fold persistent | Asymmetry must be preserved |
| Horner history queries | omit sources whose representation changed after t-g | Declared approximation |

Open XXX items to keep bit-for-bit or to decide on deliberately:
- Elder repair rows double-gated in the TC old-repair term (B:1490-1497).
- Rows 16/19 not dressed (M:919).
- The `layer <= depth - 3` switch cutoff (B:2680).
- The reason for the condition limit (B:378).
- The completion tolerance 1e-10 (B:796).

### 2d. Safety and guards: droppable for a research distillation (not active on net 99)

- **Estimator routing (E:74-268).** `_PredictionAllowance`, `_gaussian_reserve`, `_budget_exhausted`,
  `_session_flops_used`, `_point_mass_fallback`, `_gaussian_fallback`, `_is_solver_failure`, the depth-1 and
  width <= 2 routes, the non-finite re-route, and the dtype / budget ValueErrors. Keep `xp.maximum(result, 0)`
  (E:331): it is cheap and cannot increase error.
- **`DegenerateMoments` raises.** `_require_positive_variance` / `_require_finite` (M:1091-1102), bank
  `_require_finite` / `_validate_shared_qr` (B:1010-1034), `_factor_shared_basis` width check, completion checks
  (B:803-815; the history fold swallows them, B:1689-1692), the Chebyshev degree check (MM:110-111), and the
  empty-source check.
- **Finiteness preflight in `_fill_omitted_rows`** (B:409-429), falling back to the full-row deposit.
- **Conditioning and denominators.** `SHARED_BASIS_CONDITION_LIMIT` re-orthonormalization (never triggered),
  `_gate_div` (inf for zero gates, B:742-747), the `+1e-30` in Krylov, the float64 retry and direct solve of the
  radial solve, and the e = 0 guards in the birth and in PK4 (these define the values at e = 0: keep their
  semantics).
- **Edgeworth clips.** `dv` to [0, 1] and `ev >= 0` (MF:124-125). These are statistics when they bind: keep them.
- **Plumbing.** `_require_phase_value`, asserts, `close()` / reference breaking, typed handoff records (SC, the
  dataclasses at M:93-161, the TypedDicts at B:118-250).

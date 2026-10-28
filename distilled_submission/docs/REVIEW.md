# REVIEW: clean/estimator.py (981 lines, Fable A) against SPEC.md and source/estimator_plain/

Reviewer: Fable B, 2026-10-28. Line numbers are estimator.py's (E:) unless prefixed (plain: M/MF/MM/B/K as in SPEC).
Verdict in one line: the program is a faithful dense rewrite; I found no error that changes the statistics beyond the
decisions A already recorded, two places where SPEC.md was wrong and the code right (now corrected in SPEC.md), one
convention for saturated hubs where the code is self-consistent and submission 337479 is not (A should know it is a change), and
one set of memory/length cleanups that remove no statistics.

## (a) Line-by-line check

Checked mechanism by mechanism (SPEC section in brackets). "same" = same statistic as the plain code up to rounding.

| E: lines | mechanism | finding |
| --- | --- | --- |
| 119-180 | Gaussian basis, Edgeworth rows, unary statistics [5.2.1-5.2.3] | same; coefficients match MF:88-209 term by term (D1/D2/D3 orders, a1, a2, ga1, ga2, u2, u3, stf with 2-4/(n+2) at layer 0, lc/lr, m7, a1tc, dz). |
| 188-222 | Mehler rows [5.2.4] | same. LEFT_ORDER/RIGHT_ORDER equal r+k, s+k of PAIR_ORDERS for all 21 rows. Rows 16-20 scale the right vector by z3_j/6, q4_j/24 instead of the matrix (M:892-896): identical. ROW_FOLD and the row 17/20/18 reuse (E:206-208) match M:915-920. |
| 376-380 | Q22/4 and Q31/6 | same for gram = diag(gv): gm o gram puts ga gv on the diagonal (M:859); q316 = gvh gm + gah gram adds a diagonal gah gv that the plain code omits past layer 0 (M:871-872). That diagonal is dead: rows 10-13 reach only output diagonals that are overwritten (cov, post, trace) or excluded (u row sums, E:391), and x o q316 has zero diagonal. No change. |
| 385-392 | pair powers, (2,2) row sums [5.2.5] | same; float64 accumulation makes M:944-951's analytic row-0 trick unnecessary; the diagonal exclusions agree (E:391 vs M:946, 951). |
| 398-434 | core update [5.3.2] | same; g_w, r_w, gmc, grc, cc, oc/or, kappa, radial parts match MF:274-315 and M:1036-1067. gram o (grc d^T) (E:423) is added at every layer, M:1065-1067 only at layer 0: past layer 0 it is diagonal (gram diagonal) and overwritten by urow (E:430). No change. p = x ga1 (E:412) as in M:1040 (SPEC D3). |
| 425-428 | edge block [5.3.3] | same as M:983-1001, 1070-1071 without crops (hv = v). |
| 349-369 | core push, E_GA, q4 [5.3.1] | same; layer 0 sphere values (E:357-358) match M:582-584; ga = diag(W^T T W) + s gv, E_GA added before the diagonal fill (M:793-795). |
| 436-462 | birth [1.2, 1.6(ii)B] | three-leg form correct: `own` (E:454) is (1.18) at P = I (hub a = i: (2 e d + X_aa) u_ij + d X_ij; hub a = j: e_j u_ji^2 + u_ji X_ji); beta (E:456) = k3 - dv^3 z3 - (3 e d^2 + 3 d X_aa) is the exact-removal form of (1.11) (3 H_ii^2 - 3 F_ii^2 = 3 e d^2 + 3 d X_aa). Star birth and its charge to R, beta as M:367-371. **Convention difference, saturated hubs (adm and not sq):** the code drops their u/X rows AND leaves them out of `own` and of the 3 e d^2 + 3 d X_aa term, so the represented (2,1) slice at birth equals post on those rows (self-consistent). submission 337479 computes r and beta in M:355-369 with the FULL H row (incl. the huge F part), then zeroes the H row in B:2643-2645 and masks the packet to H-carrying hubs (B:905-906), so its r/beta subtract an image that is no longer represented. The code's choice is the consistent one and avoids r rows of size |X|/(2e); it is a statistics change on the saturated hubs only, unrecorded. Recommend: keep, record in DISCREPANCIES as item 9. |
| 299-347 | loop order [0] | same as M:476-531/405-462: deposit -> star images -> TC + PK4 -> advance -> birth -> TC stash (b before the star image, E:313/332 = M:493) -> next frame -> response_input -> accumulate -> record_birth. Radius via lgamma (E:346) equals MM:226-244 to rounding. |
| 464-483 | terminal [5.5] | same as M:268-300 (ga on every column, E_GA, PK4 single+legs on the terminal legs = legs[0] after the terminal rotation, TC, z3^2/72 term). |
| 578-630 | accumulate [1.3] | same: gating of old legs and RIGHT by dv (E:591-593, B:2688-2692), newborn ungated, P = W rows on hubs (E:601 = B:2479-2482), RIGHT transport, R = LEFT RIGHT, coefficient rows (E:610-612) = (1.18) + 2 P o R + beta P o P, z3 (E:615) = (1.13) in exact-removal form, trace (E:616-622) = (1.14). |
| 632-644 | repair fit [1.3(c)] | same as B:699-739, 2714-2725 with the hub indicator as conservation column (ones on retained rows); v on hub columns (colsum - delta)/(m-1), elsewhere colsum/m; beta += 3 (delta - v). |
| 647-673 | deposit [1.4] | same, including the undo: copies are taken before the directional fold (E:731-732) and restored after the PK4 fold (E:670-671), so the PK4 fold survives only on the newest source's C_u and C_X rows (the split of submission 337479's CU row). This matches B:2052-2058 + 2134-2136 (the plain undo list covers COMB2 rows of all old sources and the newest CP row; only the newest CU row keeps the PK4 fold). **SPEC 1.4 and 3.3 said "persistent on all rows": SPEC was wrong, corrected now.** |
| 676-752 | E6 [4] | same: x_leg = Leg.u for u, X, P and xr = LEFT (RIGHT u) (B:2287-2294); chain with rho, the newest row of xr gets -rho dvec_gross (B:2309, prev[3] = gross dvec) and the fold uses dvec_net (B:2102, prev[4]); read, radial image and dga (E:702-718 = B:2346-2375). The three-leg fold (E:738-741) is the mixed contraction (., u, .) of e Sym(P,u,u) + (1/2) Sym6(P,u,X) + Sym(P,P,R) + beta P^3: C_u += e (xP u + xu P) + (xX P + xP X)/2, C_X += (xu P + xP u)/2, C_P += e xu u + (xX u + xu X)/2 + xP R + (beta xP + xr) P, RREAD += (xP P)^T LEFT; checked against B:2092-2100, 2116 with H = sqrt(e) u + X/(2 sqrt e). Correct. |
| 755-795 | newest TC [2.1-2.4] | same up to the two recorded decisions. M = e u (E:785): correct, see (b)1. cv = e v d - a1/2 (E:784) = B:1516 with _tc_fdiag -> 0. Zp, Qt, dq, cp, s, cpf, ch (E:787-794) = B:1517-1529 with the fold on the u leg only (ch = 3 e W Qt against u^(t) = (C d) W). Hub rows = adm (dv > 0.01) where the plain uses COL (dv > 0.04): A's open question 3; the rows in between carry Zp ~ dv so the effect is second order. qt_tc = qt - dcf^T (E:762) = B:1769-1776. |
| 797-829 | Horner history query [2.5] | same as B:1946-2024 for dense sources: sources older than the window enter at generation A with legs[A] (gated legs at layer t-A), the P identity of source t-g-1 enters at the step to generation g, lifted by W_(t-g) o dv_(t-g) (E:821 = B:2003), H entries via legs[g]; qh[g-2] is the query of the TC whose newborn is source t-g (E:825-828: chain[g-2]["k_in"] = t-g, checked). |
| 831-850 | history reads and folds [2.5] | same as B:1651-1737 with (W, MW) -> (P^(b->t), e u^(b->t)); theta3 = 3 theta as B:1973; the completion term always on (no row plan). |
| 852-882 | consume / terminal TC [2.3, 2.6] | same as B:1549-1572, 1593-1635 (history (Zp, Qt) from the fold when present, else recomputed; chain length 2; terminal query in history form without fold = B:1610-1611). |
| 888-961 | PK4 [3] | same as K:206-498: live = (e > 0) & sq (K:257), g' = u + (X - coef P)/(2e) = isq H - de P, cn' with the self entry zeroed, 6V/4c31/4f3 (K:286-288, incl. `ov - m^2` as in K:287), powers and image rows (K:293-304), read (K:358-371), single site (K:398-403), fold gradients and the chain rule to (u, X, P) (K:449-497 with isq -> (live, inv2e)). |
| 967-981 | entry | fine; the stderr print and FLOPS reset are tooling. |

Items that change statistics relative to submission 337479 (all intended and recorded except the first):
  S1 saturated-hub r/beta convention (E:451-456; see table) - unrecorded, recommend recording as DISCREPANCIES 9.
  S2 exact F (x) F removal incl. source 0 (measured x0.9998).  S3 no crops/plans/elders; exact SVDs.
  S4 TC hub rows dv > 0.01 instead of COL dv > 0.04 (open question 3).  S5 float64 accumulators (rounding only).

## (b) Verdict on A's DISCREPANCIES 1-8

1. Correct; SPEC 2.1 and 2.4 were wrong. sqrt(e)(H - L G) = e u + sqrt(e)(F - F1); the exact limit F1 -> F is e u,
   _tc_fdiag -> 0, cv = e v d - a1/2, fold on the u leg only. SPEC 2.1, 2.4 and 1.6(ii) corrected (2026-10-28).
2. Agree: noise, keep the removal (the model's reading); D1 stands as a recorded exception of submission 337479.
3. Partly agree. A's dense-gram wiring (Q22/4, Q31/6, gra, grc term "as layer 0") is not the plain code's pre-economy
   form either: with a dense gram the s I part of A pushes to s W^T W OFF the diagonal (M:612-613 says exactly this),
   and MF:297-299's gmc term a2 gv d exists only in the diagonal-gram specialisation. So x1.343 measured a hybrid; the
   recorded x1.002 (M:574-577, "-5.5 GF" is not even one full W^T W per layer) belongs to an earlier wiring that cannot
   be reconstructed from the plain source. The diagonal gram is submission 337479's statistic and ships; SPEC 6 updated.
4. Agree (CENSUS section 5).
5. Agree; useful for anyone building an "uncropped" plain reference. Not a property of the clean program.
6. Agree it is an observation. The clean fix is a float64 X leg or storing X/(2e) (A's open question 2); not now.
7. Agree to keep rank 4 given the net-32 divergence. Note the divergence was measured without the hub rules; with them
   it is unmeasured (A says so). Rank 4 is submission 337479's statistic either way.
8. Agree, with the correction to SPEC 1.7 made now: in completed-square form the H row of a saturated hub is F =
   X/(2 sqrt e), huge, and submission 337479's P-only rule is a numerical necessity, not an economy; in the three-leg form the X
   row is finite but the PK4 dressed leg divides by e, so PK4's live mask must exclude saturated hubs (as it does,
   E:900). Keeping the X row of saturated hubs in the bank (but not in PK4) would be a measurable variant, not required.

## (c) Memory census check (E:35-44 vs the code)

Persistent, verified from the allocations: legs 4 x 3 x (15, n, n) f32 = 720 MiB (E:564); coef 3 x 60 = 180 (E:565);
R, PP 120 (E:566); LEFT/RIGHT/RREAD/beta/e/adm < 10 MiB; W_raw + W_at gauged 2 x 64 = 128 MiB (E:288, 569); T (f32),
and per layer pc, Cov(h) (f32), b (f32), 16 Mehler matrices f32 (64 MiB), float64 cov/post/pv3/u/half/trace/block
(8 MiB each) ~60-100 MiB. Census total 1.25 GiB agrees with tracemalloc 1.17 live / 1.52 peak.
Missing from the census: the deposit's undo copies (E:731-732) 3 x old x 4 MiB + RREAD, up to 170 MiB at t = 14, and
the birth's float64 extra/u/X/own/R (5 x 8 MiB). Neither is n^3 or (sources, n, n, k): every batched product is a
per-source loop (E:598-601, 604-620, 663-665, 685-688, 696-697, 707-708, 733-741, 815-823, 926-931, 948-961); the
only batched products are the thin RIGHT (E:602) and RREAD @ RIGHT (E:666) reshapes, O(S n r). No growing
concatenation exists: every bank buffer is preallocated at full size (E:563-568). The 3.5 GiB RSS vs 1.52 GiB tracked is
therefore allocator retention of churned temporaries, not a growth pattern: per layer the pair law creates ~100 float64
(n, n) temporaries (E:217-222: `mats[q] * np.outer(left, right)` allocates an f64 outer and an f64 product per row, 21
rows x 5 pair powers), the PK4 read/fold ~10 (n, n) f32 per source, the folds ~10. The plain reference shows 4.8-6.1
GiB in the same runner, so this is the environment's behaviour; it is under the 4 GiB rule but not by much. Cheap
reductions that change no statistics: (i) pair_power with two preallocated f64 scratch buffers
(`np.multiply(left[:, None], right[None, :], out=tmp); np.multiply(mats[q], tmp, out=tmp); out += tmp`); (ii) apply the
directional fold as stored increments and undo by subtraction, or keep the undo copies in one preallocated (3, S, n, n)
f32 scratch (submission 337479's undo also copies, B:2097, 2104); (iii) `del` the birth's f64 temporaries before accumulate.
Expected effect: tracemalloc peak ~1.3 GiB, RSS noticeably lower (allocator churn is the driver), no MSE change.

## (d) Cuts toward 800 lines that remove no statistics and no explanatory comments

| lines | what | saving |
| --- | --- | ---: |
| 292-297, 372, 433, 482, 970, 976 | `rec` debug hook and the `debug` plumbing (tools/ can monkeypatch instead) | 12 |
| 71-92, 974-980 | three tally wrappers + FLOPS dict + stderr print: one `tally(kind, flops)` called from `mm`, `svd_top`, `solve`; move the report to tools/ | 12 |
| 55-68 vs 1-45 | constants block comments repeat the module docstring's decisions (keep one copy, in the docstring) | 8 |
| 255-280 | `Frame` + `prepare_frame`: fold the sd/z/gate lines into prepare_frame returning a small namespace | 6 |
| 546-558 | Bank class docstring repeats the module docstring's model and the three-leg formula of E:610-612 | 8 |
| 770-776, 755-759 | `prepare_tc` and `tc_deposit` docstrings both explain theta and the fold; keep the formulas in one | 6 |
| 924-932, 942-961 | PK4.read and PK4.fold both form P, gp, P2, g2, pg: one `products(bank, s)` helper | 5 |
| 464-483, 356-368 | `ga = diag(W^T T W) + s gv` + E_GA appears in `advance` and `terminal_mean`: one `core_diagonal(W, l)` | 6 |
| 811-818 | `horner_query`: initialise `acc` before the loop (g = A branch) and drop the branch | 4 |
| 852-866, 876-882 | `consume_tc` + `tc_terminal`: merge the chain-advance bookkeeping; `tc_pending` can call consume | 5 |
| 116-118, 183-185, 225-227, 252-254, 486-488, 885-887, 964-966 | 3-line section banners -> 1 line each | 14 |
| 94-102 | `col`/`row` are used ~40 times: keep (they are the plain reader's friend) | 0 |
| 212-214, 189 | `right_factors` + MEHLER_RIGHT_SCALE: scale right vectors inline in `pair_power` for rows 16-20 | 4 |
| **total** | | **~90 -> ~890** |

Reaching 800 needs ~90 more lines and cannot be done without touching docstrings: the file has ~130 lines of
docstring beyond the first line of each function. Dropping only the sentences that restate SPEC.md formulas already
cited by section number (and citing the section instead) would get there without losing explanation that is not
available elsewhere. I would not do it before the 20-net run; the 981 lines carry no dead branch and no duplicated
mechanism (the two `complete_bilinear`/`completion_coefficients` functions are the plain code's, needed once).

## (e) Formulas that could be written more plainly

- E:666 `mm(RREAD[:k].transpose(1, 0, 2).reshape(n, -1), RIGHT[:k].reshape(-1, n))` -> `for s: block += mm(RREAD[s], RIGHT[s])`
  (same FLOPs, says "repair slice = sum over sources of RREAD_s RIGHT_s"). Same for E:779-780, 808, 835 (`A`, `B`, `Bt`):
  write the old-repair term as `sum_s mm(RIGHT_gated[s][:, i].T, RREAD[s][j])` or keep but name it `old_repair_read`.
- theta appears as `-premean/(3 v)` (E:764, 947), `-premean/v` (E:807, "3 theta") and `premean/v * gvw * (-1/(n+4))`
  (E:728, "3 theta gv/(n+4)"): define `theta = -fr.premean / (3 fr.v)` once per deposit and write `3*theta*gv/(n+4)`
  and `3*theta`, as SPEC (2.4) and (4.3) do.
- E:781 `gdc = where(d > 0, d, inf)`: say "divide by dv; rows with dv = 0 (clipped Edgeworth gate) become 0" in a comment.
- E:454 `own`: name the two hub placements in the comment as "(a = i) and (a = j) terms of SPEC (1.18)" - it does, keep.
- E:542 RepeatedImage.residual: write the three terms on three lines with the SPEC (2.2) names (k = l, i = k or i = l, i = k = l).
- E:217-222 pair_power: the MEHLER_MATRIX indirection (16 matrices for 21 rows) is the one matrix trick a reader must
  decode; a one-line comment "rows 16-20 reuse matrices 0, 1, 2, 0, 1 with z3_j/6 or q4_j/24 on the right vector" exists
  at E:186-187; fine.
- E:615 z3: the comment gives 3 e P u^2 + 3 P u X + 3 P^2 R; add "+ beta P^3" (the term is there, the comment omits it).
- E:821 `acc = mm(pb, W) * gate`: comment "transport the query to layer s and gate it, = legs stored gated".

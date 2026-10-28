# DISCREPANCIES (clean program vs SPEC.md / plain source), Fable A, 2026-10-27

1. **SPEC 2.1 / 2.4, the TC's newborn leg content M with the three-leg form.** SPEC reads the exact form as
   M = e u + X/2 ("no L G0 term") and consequently a TC fold ch_X = (3/2) W o Qt on the X leg. The plain code uses
   sqrt(e) (H - L G) = e u + sqrt(e) (F - F1) where F1 = L G is the best rank-one approximation of F = X / (2 sqrt e), and
   _tc_fdiag = X_aa/2 - sqrt(e) (LG)_aa. The exact limit of what the code does (F1 -> F) is M = e u, cv = e v d - a1/2, the
   fold only on the u leg (ch = 3 e W o Qt), nothing on X. Measured on net 99: M = e u + X/2 gives raw 2.58e-8 (x2.3, blow-up
   from layer 12), M = e u gives 1.118e-8 (within 0.4 % of the H-form program that reproduces the plain code). The program
   uses M = e u. SPEC.md section 2.1 and the `ch_X` sentence of 2.4 should be corrected.
2. **SPEC D1 / 1.6(ii), source 0's F (x) F term.** Keeping it (submission 337479's exception) vs removing it exactly: raw 1.11799e-8 vs
   1.11824e-8 on net 99 (x0.9998, noise). The program removes it (the model's reading); the measurement is recorded.
3. **SOURCE_MAP 2b "diagonal metric gram = diag(gv) after layer 0: x1.002 raw MSE for -5.5 GF".** On this dense program the
   dense gram W^T W (entering Q22/4, Q31/6, gra and the grc half-trace term exactly as layer 0 does) measured x1.343 on
   net 99 (1.5013e-8 vs 1.1182e-8). The recorded x1.002 cannot be the same change; the diagonal gram is kept.
4. **CENSUS.md section 5 vs RESULTS.md section 1 (first version).** CENSUS is right: in the pre-lift (Horner) form a dense R or
   X leg's TC query product is the deposit product already counted, so a dense leg costs +483 GF, not +1503 GF. RESULTS.md
   section 1 is corrected accordingly.
5. **Plain-dense reference and exactly-zero/one gates.** A scratch copy of the plain estimator with every threshold set to 0
   still crops: hubs with dv == 0 (Edgeworth clip) fall outside the COL plan (which drops their columns from the ungated
   fresh H transport), gate >= 1 - 0 counts as saturated, and `d < 1 - 0` drops rows with Gaussian gate exactly 1. Thresholds
   of -1 (and `sat0 = gb >= 2`) are needed for a truly uncropped reference. Recorded in RESULTS.md section 2.
6. **Not a discrepancy, an observation.** The PK4 dressed leg g' = (H - H_aa P)/sqrt(e) = u + (X - coef P)/(2e) amplifies
   float32 rounding for hubs with tiny curvature e; between my program and the uncropped plain copy the PK4 read ('extra')
   differs by 1e-3..5e-1 relative from layer 9 on while every other state agrees to 1e-5..1e-4, and the final means differ by
   2e-5 relative (1.2 % in MSE). Both programs share the instability; which rounding is "right" is moot.
7. **SPEC 2.2 recommendation "implement (2.2) exactly".** Measured: the exact repeated-index image is x0.9975 on net 99 but
   diverges on net 32 (q4 corrections jump from 0.08 to 5 at layer 13, NaN at the terminal) while the rank-4 truncation is
   stable on all five nets. The truncation acts as a regulariser against large entries of the stashed slice (which carries the
   PK4 fold of tiny-curvature hubs); the program keeps rank 4.
8. **SPEC 1.7 / SOURCE_MAP 2b: hub admission and saturation are economy.** They are also the closure's numerical guard: with
   every neuron a hub, saturated hubs (dv = 1.0, Edgeworth-corrected e ~ 1e-9) give PK4 dressed legs |g'| ~ 2e2 and the
   terminal q4 diverges (net 33: raw 1.7e-3; net 32 with the exact repeated index: NaN). SPEC 1.7's dense reading "keep the H
   row (its e is tiny, so H ~ F)" is numerically wrong: F = X/(2 sqrt e) is huge, not small. The program keeps submission 337479's two rules
   (gate > 0.01 to be a hub; gate >= 1 - 0.003 -> P only, no square, no PK4) as statistics; the HUB_DROP_TOLERANCE crop and
   the COL/S/OUT/row plans stay removed. With the rules the five dev nets are within -6 %..+4 % of the cap-320 plain run
   (pooled x0.994).
9. **REVIEW S1: saturated hubs' repair and beta.** submission 337479 forms r and beta (M:355-369) from the FULL completed-square row of a
   saturated hub (gate >= 0.997, the H row includes the huge F part), then zeroes that H row (B:2643-2645) and masks the
   packet, so r and beta subtract an image that is no longer represented; my first version subtracted only the represented
   (u, X dropped) image, which is self-consistent. Measured on net 99 (three-leg program, hub rules on): submission 337479's hybrid
   1.08706e-8 vs self-consistent 1.09794e-8 (x0.990). The hybrid is faithful and not worse, so the program now uses it
   (birth: `own`, beta from the full rows, then u, X rows of saturated hubs dropped). Also measured there: TC hub rows
   dv > 0.04 (submission 337479's COL plan) vs dv > 0.01: 1.10067e-8 vs 1.09794e-8 (x1.0025, noise); the faithful 0.04 is used. With
   both changes net 99 is 1.08933e-8.

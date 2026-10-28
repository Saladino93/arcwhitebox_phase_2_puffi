# submission 337479 mechanism ablation evidence (recorded measurements only)

Compiled 2026-10-27 from existing receipts in whest-ship and arcwhitebox. No new runs were made. Every number is
quoted from the cited file:line (line numbers were re-checked against the files on 2026-10-27).

Path conventions: paths without a prefix are relative to `whest-ship/`. `AWB/` = `arcwhitebox/`. `N:` =
`whest-ship/NOTES.md`, `I:` = `whest-ship/INVENTORY.md`, `AN:` = `arcwhitebox/NOTES.md`, `PROF` =
`research/SUBMISSION337479_PROFILE.md` (submission 337479's own metered component census, net 99), `CENSUS` =
`AWB/lab/census/RESULTS.md`, `X18` = `AWB/eppie_ideas/ex18_timing/PROGRESS.md`, `X29R`/`X29A` =
`AWB/research/submission 336271/research/removal_screen_500_507.md` / `addition_screen_500_507.md`.

## Bases and how to read the numbers

Most ablations were run on members of the lineage, not on submission 337479 itself. A ratio is ablated raw final-layer MSE divided
by the same-run control raw (>1 = worse). "Score" or "adjusted" ratios include the FLOP change. "dev-5" = nets
99,32,33,34,40. NumPy = unmetered `scripts/numpy_harness.py`; "metered" = FlopScope.

| base label | what it is | control raw |
|---|---|---|
| **submission 337577 / ex36_base** | `candidates/ex36_base/{bank,estimator,pk4}.py` are byte-identical to `candidates/ex34_reference/*` and to `research/digest_ex34/source/submission 337479/*` (checked with `cmp` on 2026-10-27); only `fmm.py` differs. **These are direct ablations of submission 337479's statistics.** | net 99 1.1049e-8 NumPy, dev-5 pooled 1.03481e-8 (`research/GOAL15_TERMINAL_PROPOSAL.md:4`, N:2279); net 99 metered 1.1052e-8 at 356.461 GF (AN:5) |
| submission 337479 metered | submission 337479 on its own backend | net 99 1.1030e-8 at 359.678 GF (PROF:149) |
| ARGON01 | `estimators/01_argon_l6_cold_336299` (submission 336299 chassis; submission 337479 = this + shared basis + L1 + L2) | net 99 1.0743e-8 at 368.2 GF (N:27); NumPy 1.0741e-8 (N:105) |
| tdir_b | `packages/argon_l6_tdir_b` (ARGON + shared propagator frame) | net 99 1.0923e-8 NumPy (`research/ON_L2D_RESULTS.md:16`) |
| v5 | `packages/k3_boost_120_v5_legal_f32` (later ARGON descendant) | net 99 1.0467e-8 NumPy (`research/READS_NEEDED_ABLATION.md:32`) |
| submission 335982 | arcwhitebox submission 335982 (same bank/core/TC/PK4 architecture, older settings) | oracle nets 0-4, net 0 1.2079e-8, metered (CENSUS:64); public 500-507 1.2672e-8 (X29R:3) |
| internal version 18/submission 335292 | earlier arcwhitebox chassis with the same bank design | 5- or 20-net panels |
| external | 504aldo's public factorized-K3 estimator (same bank architecture, different code) | `research/ALDO_WRITEUP_REVIEW.md` |

submission 337479's metered bill on net 99 is **359.678 GF** (mult 0.1636, PROF:137); the cubic history bank is **298.632 GF =
83.0 %** of it (PROF:161). Break-even at this bill is ~3.6 GF per 1 % MSE (CENSUS:67): a cut pays only if it costs
less than ~0.28 % MSE per GF removed.

## Table

| # | mechanism | effect on MSE when removed / weakened / replaced | FLOP cost (submission 337479 net 99 unless stated) | net(s), base | evidence |
|---|---|---|---|---|---|
| 1 | Spherical-radial factor (`radius_mean`, sphere input law) | Replace by Gaussian input throughout: **x1.074** (+7.4 %); per-layer ratio rises from 1.017 at L3 to 1.074 at L16. With exact cumulants the closure alone differs little (9.33e-11 sphere vs 9.70e-11 Gaussian control): the loss is in propagation | ~0 (exact identity); inside "fresh sources / radius / basis / gates" 1.373 GF | 99, bundle_b (ARGON-class), NumPy; oracle 0-9 closure probe (submission 335982) | `research/GAUSS_INPUT_RESULTS.md:15-19,21-27`; `AWB/lab/attrib/RESULTS.md:11-15,82-83`; cost PROF:134 |
| 2 | Unit-prevariance gauge | Its introduction (G6D6 + online pre-activation variance normalization): mean raw 1.6526e-8 vs 1.6864e-8 without (gauge worth **~-2.0 %** raw; removal ~x1.020), adjusted -1.98 %, FLOPs +0.037 %. Other gauges: post-activation inverse-std gauge x1.401 (diverged at L11) | +0.14 GF per net; covariance/gauge transport bucket 26.461 GF | Puffi 450/451, internal version 19 G6D6 lineage, metered (2 nets) | `AWB/research/internal version 19/validation/review_2026-09-30/online_prevariance/REPORT.md:3-5,17-20`, `.../comparison.json:17`; `.../audits/ONLINE_GAUGE_TWO_NET_AUDIT.md:9-14`; `.../audits/ONLINE_VARIANCE_GAUGE_REVIEW.md:71`; cost PROF:126 |
| 3 | Cubic source bank at all | None (Gaussian fallback): **x296** (3.1828e-6). Keep only newest K ages: K=8 x3.13, K=5 x13.7, K=3 x39.8, K=1 x94.9. History-free (no-K3) estimator x242 at 58 GF. Old births removed from all slice reads: age>=2 x137.8, >=4 x42.9, >=8 x4.09; from the diagonal (>=4) x54.8. Retire switched births (no elders): x8.4-11.8. Drop sources older than 12/10/8 layers: +5 %/+57 %/+280 % (submission 335292). Every source killed after age 1: x130. Bank-free compact ARC rewrite (B1): x45.3. internal version 10 family `no_history`: 3.197e-6 vs 1.177e-8 (x272) at mult 0.128 vs 3.621. External: memoryless closure x45 | **298.632 GF = 83.0 %** of the bill | 99 ARGON01 NumPy; 99/613/601 v5; submission 335292/internal version 18 5-20-net panels; 99 B1; internal version 10 net 140 | N:107-115; I:953; `research/READS_NEEDED_ABLATION.md:76-83`; N:1936, N:1939; `AWB/research/submission 335292/README.md:82-83`; X18:286-288; `research/ARC_CHEAP_RESULTS.md:16,22`; `AWB/research/internal version 10/README.md:104-112`; `research/ALDO_WRITEUP_REVIEW.md:22-23`; cost PROF:161 |
| 4 | Completed-square decomposition (H = sqrt(e) u + F, F = extra/(2 sqrt e)) | **No plain removal.** On the internal copy of submission 337479's statistics, net 99 metered: replacing the fresh H leg by a covariance bridge **x7.68** raw (8.49e-8) for -10.5 GF; with SVD repair x1.42; replacing H by a P-anchor x13.3, with conservation x5.49. Zeroing the kappa3-slice feedback rows of `extra` (part of F): dev-5 **x1.498** (99 x1.367 ... 40 x1.669), a lower bound. Truncating fresh H to rank n/4 from layer 5: +24 % (internal version 18); 0.5 % of H energy outside rank n/4 carries ~18 % of MSE | zero marginal (H is transported anyway); birth admission / repairs / negative / E reads / diagonal 8.748 GF | 99 metered and dev-5 NumPy, **the internal copy of submission 337479's statistics**; internal version 18 5-net panel | AN:10, AN:17, AN:22-23; `AWB/research/goal20_20261021/common_proposals.md:28-33,55-61`; `research/GOAL15_FORUM_PROPOSAL.md:42-57,58-66`; X18:272-276; cost PROF:119 |
| 5 | Rank-2 SVD repair + conservation vector | Raise rank 2 -> 256: raw -2.9 % but +53 GF, adjusted +11 % (ARGON01). Rank 3: x1.0036 score, MSE -0.03 % (submission 335982). Rank 32 all births: MSE -1.72 % at +1.15 budget points, score +4.7 % (internal version 18). Lower to rank 1: x0.998 (99), pooled x1.0079 (v5). **Conservation column: no isolated removal** ("ALREADY in tdir_b"; submission 335924's trace-conserving repair was a bundled result) | inside the 8.748 GF admission bucket | 99 ARGON01 NumPy; submission 335982 500-507; internal version 18; v5 | N:433, `research/2026-10-20_round2_math.md:109`; X29A:15; X18:185-186; N:2148, N:2226; N:1604 |
| 6 | NegativeBank (rank-1 -Sym(K,G,G)) | **No removal measured.** Richer debits lose: four-mode F^2 subtraction MSE +0.09 %, adjusted +1.84 % (net 600, submission 335400); rank-2 F debit MSE +0.50 % (internal version 18, net 800). Residual beyond the rank-1 debit + rank-2 repair: 0.65-0.80 % of ||H||^2 (diagnostic) | 0.351 GF | submission 335400, internal version 18; ARGON | `runs/submission337479_profile_20261023/net99.json:708`; `AWB/eppie_ideas/research/ex21_square_remainder/FINDINGS.md:3-5`; `AWB/eppie_ideas/ex18_negative_rank2/experiment.json:67`; N:1239 |
| 7 | Young/elder split, shared elder basis (cap 320 vs 256, switch ages 6/7) | Never switch (all births dense): x1.040, i.e. compression is lossless. No elders at all: **x8.43** (99); 600-609 x11.8 raw for -53.5 GF. **Cap 256 vs 320 (submission 337563 vs submission 337479, official): MSE x1.031** (1.2106e-8 vs 1.1742e-8) at 15.50 % vs 16.04 % FLOPs (~-11.7 GF), adjusted -0.31 %. Shared vs per-birth bases x1.0031; shared rank 128 x2.24 (submission 335400). Elder rank on submission 335982: 0.75x ranks x1.086, 1.5x x0.976 (+24.5 GF); all elders at n/8 MSE +69 % (score x1.60, submission 335982). Every elder n/4, never recompressed: x1.0047 (+8-12 GF est., submission 337577). Earlier switch (6,7)->(5,6): **x1.058** for -13.4 GF (submission 335982); ages 4/5: MSE +21.5 % (submission 335982), +10.4 % (internal version 18); v5 age 5 x1.101, age 4 x1.280. submission 337479 bundle (shared basis + L1 + L2) vs 337428: MSE x1.016 for -1.12 budget points | switch/QR 13.852 + transport/Y@G rebuild 28.660 + C^T Y 15.890 = **58.40 GF** | 99 ARGON01; 600-609; official 100 nets; submission 335982 oracle 0-4 and 500-507; dev-5 submission 337577; v5 | N:300; N:1936, N:1939; `AWB/research/submission 337563/README.md:4-6,18-24`; `research/IMPORT_ATTRIBUTION_99.md:20`; `AWB/eppie_ideas/research/ex21_shared_elder_basis/FINDINGS.md:3-5`; CENSUS:310-312; X29R:7-8,24; N:2281; CENSUS:80; X18:376; N:2134-2135; `research/digest_ex34/source/submission 337479/README.md:94`; cost PROF:115,116,121 |
| 8 | SB_E1 / SB_E2 switch restructuring | Exact algebra: **-2.48 GF, MSE x0.9999** (net 32); SB_E2 bit-identical values | saves ~2.5 GF | 32, submission 337479 | `research/digest_ex34/source/submission 337479/bank.py:88-94`; `research/EX34_20261021_ASSESSMENT.md:41` |
| 9 | Neuron thinning (THIN_TAU, THIN_OUT, ROW_THIN) and 64-padding | **No "thinning off" run.** Tighter: THIN_TAU 0.04->0.08 x1.159 (-9.9 GF); THIN_OUT 0.015->0.03 x1.081 (-5.5 GF) (submission 335982); OUT tau 0.0015 MSE +56 % (submission 335982 500-507). Looser: THIN_TAU 0.02 x0.977 (+8.35 GF, break-even), THIN_OUT 0.0075 x0.996 (+3.9 GF). v6 bankthin: +3.9 % raw for -1.6 % GF (600-609). v5: THIN 0.08 x1.177, 0.12 x1.530; ROW_THIN .03 x1.115, .05 x1.568; OUT 1e-2 x1.227. 0.1-floor schedule (216 GF): raw x8.2, crops x140. Padding: bank quantum 0 -> 64 *improved* score -2.67 % (internal version 18; the gain is six-level FMM products); removing exact zero padding (`algebra_ragged`) saves 0.92 GF, raw x1.0016 (submission 337577 metered) | thinning is a saving (it shrinks bank products) | submission 335982 oracle 0-4 and 500-507; 600-609 v6; 99 v5 NumPy; internal version 18 700-717; 99 submission 337577 metered | CENSUS:78-79,154-155; X29R:10; N:1941; N:2161-2162; N:2155-2156; N:2153; I:964; `AWB/eppie_ideas/ex18_bank64/experiment.json:781`; AN:27 |
| 10 | (2,1) slice deposit (incl. saturated rows) | Dense slice removed (diagonal + trace kept): **x120-x15000** (internal version 18); x100 (submission 335292). Slice removed from the Mehler pair law only x180; from the kappa4 core update only +83 %. On the internal copy of submission 337479's statistics (net 99 metered): slice replaced by a covariance-star completion (TC, folds, L1 also removed) **x53.4** raw for -36.7 % GF. Rows dropped without fill (submission 335982 SLICE_ROWS): **MSE +109 %** (score x1.99); tighter saturation MSE +468 %. Old births removed from slice reads: x137.8 / x42.9 / x4.09 (age >=2/4/8). Read-side rank-r slice: r=256 x2.26, r=128 x5.07. External: D21 must be right to ~2 % | slice family products **93.514 GF** + other slice/TC/fold 16.840 GF | internal version 18 5-net; submission 335292; 99 submission 337577 metered; submission 335982 500-507; v5 99/613/601; ARGON01 99 oracle | X18:196-197, X18:380-381; `AWB/research/submission 335292/README.md:86-87`; AN:8, `AWB/research/goal20_20261021/reader_proposals.md:57`; X29R:12-13; `research/READS_NEEDED_ABLATION.md:76-83`; N:206; `research/ALDO_WRITEUP_REVIEW.md:69`; cost PROF:122,124 |
| 11 | Trace contraction TC + Horner 3-generation history | TC off: **x1.299** metered (-13.4 GF, adjusted x1.249), x1.295 NumPy (tdir_b); x1.232 (-13.2 GF, submission 335982); adjusted x1.050 (submission 335971); +7.1 % (internal version 18). TCH 3->0 (no history TCs): **x1.121** (-6.5 GF, adjusted x1.100). v5: TC dq removed x1.266. Adding ages 2-3 Horner history to G6D6: raw -9.5 %/-6.9 %, adjusted -6.48 %. Terminal TC off x1.028 (tdir_b), +1.8 % (submission 335982) | TC's own work ~13 GF; Horner sum_families 68.93 GF is the slice deposit, paid with or without TC (only 13 of 120 GF TC-layer Horner is TC's own, submission 335982); terminal TC 7.276 GF | 99 tdir_b metered + NumPy; submission 335982; submission 335971; Puffi 450/451; v5 | `research/ON_L2_CONTRACT.md:25-26`; `research/ON_L2D_RESULTS.md:17-18,23`; CENSUS:71,100; `AWB/research/submission 335971/README.md:28-29`; X18:343; `research/READS_NEEDED_ABLATION.md:83`; `AWB/research/internal version 19/validation/review_2026-09-30/tc_history/candidate/REPORT.md:3-5,22-24`; X29R:5; cost `runs/submission337479_profile_20261023/net99.json:64`, PROF:125 |
| 12 | TC_SLICE_FOLD / HIST_FOLD | Both off: **x1.055** (L1-L5 unchanged, tdir_b). All folds off (incl. G_Q31, PK4 fold): x1.100 (bundle_b). v5 HIST_FOLD off x1.021. Fold replaced by an explicit separate Q31: MSE +0.18 %, but +0.69 budget points, score +4.46 % (submission 335400) - the fold is also a FLOP saving. Recursive history fold added: score -1.00 % (submission 335400, 24 nets) | inside "other slice / TC / fold" 16.840 GF | 99 tdir_b NumPy; 99 bundle_b; v5; net 600 and 880-903 (submission 335400) | `research/ON_L2D_RESULTS.md:21`; `research/W2_L1_RESULTS.md:30`; N:2144; `AWB/eppie_ideas/research/ex21_tc_separate/FINDINGS.md:40-43`; `AWB/eppie_ideas/research/ex21_recursive_tc/FINDINGS.md:70-73` |
| 13 | Lane L1 row fill | **FLOP economy:** ported to r11, dev-5 metered: 364.19 -> 344.00 GF (**-20 GF, -5.5 %**), raw x0.996, adjusted x0.940. Without the fill (rows simply dropped) MSE ~x2 (+109 % on submission 335982; "roughly doubles" per submission 337479 README) | 1.080 GF read | dev-5 r11 metered; submission 335982 500-507 | `research/GOAL15_L1_RESULT.md:26-35`; N:2282; X29R:12; `research/digest_ex34/source/submission 337479/README.md:30-33`; cost PROF:120 |
| 14 | Fourth-cumulant trace core A = T + sI | **No plain core-off run.** On the internal copy of submission 337479's statistics (net 99 metered): both core products replaced by a mean-action rank-2 completion **x2.01** raw for -19.25 GF (-5.4 %). Second core product replaced by alpha*pc + rank 2: +27.7 % (internal version 18). Low-rank core: svd32/64 diag x1.47-1.80, offdiag32 x1.38. Core read collar GM_TAU_A 0.2 / GM_OUT 0.5: x1.057 (-3.0 GF); looser: x0.986 (+2.45 GF). Dropping the Gram product ('nogram', already in submission 337479): x1.002 for -5.5 GF. internal version 2 H2 machine: removing the off-diagonal fourth sector x3. External: memoryless kappa4 regeneration took raw 4.2e-8 -> 2.2e-8 | affine transport/read 19.320 + update 5.514 = **24.83 GF** | 99 submission 337577 metered; internal version 18 700; 32/40 argon280 lane; submission 335982 oracle 0-4; internal version 2 | AN:14, `AWB/research/goal20_20261021/other_proposals.md:54,59`; `AWB/eppie_ideas/ex18_observed_fourth_transport/experiment.json:45`; N:2089; CENSUS:76,151; `AWB/research/submission 337479/estimator.py:390`; `AWB/research/internal version 2/README.md:41-43`; `research/ALDO_WRITEUP_REVIEW.md:67`; cost PROF:127-128 |
| 15 | PK4 single-site | **No isolated measurement** (catalogue estimate: saves 0.004 points). PK4_SINGLE + PK4_LEGS off: x1.027 (tdir_b); all PK4 off: x1.044, -5.4 GF (submission 335982); v5 x1.060. All per-neuron kappa4 reads incl. single-site delivered x0.978 raw at +0.3 % FLOPs | 0.079 GF | 99 tdir_b; submission 335982 oracle 0-4; v5 | `research/ON_L2D_RESULTS.md:22`; CENSUS:72; N:2147; `AWB/eppie_ideas/estimators.1/estimators/08_pk4_opsfrugal/evidence/historical_notes.md:29-30`; cost PROF:132 |
| 16 | PK4 star / u / t31 families, fold_response | **No family-level removal.** Adding fold_response (submission 335882 vs submission 335875): raw -2.61 %, adjusted -1.84 %, +0.12 budget points. PK4 ages (1,1): +0.7 % raw, (1,2): +0.3 %; shorter ages x1.0044 (v5). Adding the t31 quarter-gradient: x1.0016 at +1.33 GF | fold_response 3.156 + leg read 2.804 GF | dev 40-80 (submission 335882); v5; L1 lineage | `AWB/research/submission 335882/RESULTS.md:7-10`; `.../08_pk4_opsfrugal/evidence/historical_notes.md:30`; N:2146; N:2913; cost PROF:123,131 |
| 17 | E6 directional response (E_TU / E_PROP / E_GA) | E_TU+E_PROP off: x1.008 NumPy, metered -3.1 GF, raw +1.0 %, adjusted x1.001 (tdir_b). E_PROP off x1.022 (tdir_b), x1.003 (ARGON01), x1.090 (v5). E_TU off x1.093 metered (v5). E_TU/E_PROP/G_Q31 off on the E6 base: x1.030 (5 nets). Adding E_TU+E_PROP to submission 336271 (= submission 336552): official MSE x0.959, +0.51 points, adjusted x0.990 | e_pre 0.962 + e_read 0.740 GF (~3.1 GF incl. reads on tdir_b) | 99 tdir_b, ARGON01, v5; 40-80 E6; official 100 nets (submission 336552) | `research/ON_L2D_RESULTS.md:19-20`; `research/ON_L2_CONTRACT.md:24`; `research/R1_LITE_RESULTS.md:27`; N:2125, N:2140; `AWB/eppie_ideas/research/cipo_e6_cubic_core_isolation/receipt.json:53-60`; `AWB/research/submission 336552/README.md:32-36`; cost `runs/submission337479_profile_20261023/net99.json:537,601` |
| 18 | z3^2/72 terminal term | On the internal copy of submission 337479's statistics, removing it: pooled **x0.9971** (99 0.9939, 34 1.0026, 32 0.9927), i.e. neutral. Enabling it on internal version 18: score +0.9 % / +1.12 % worse; submission 335292 "+0.7 ... +1.3 %" for second-order terms incl. this one. RMS of the term 2.3e-5 vs RMS error 1.0e-4. With exact cumulants (closure only): 9.39e-11 -> 6.0e-11 | ~0 GF (about 10 n-vector ops) | 99/34/32 submission 337577 NumPy; internal version 18 5-net; submission 335292; oracle 0-4 | `research/GOAL15_TERMINAL_PROPOSAL.md:49,53,60-61,88`; X18:189-190, X18:386; `AWB/research/submission 335292/README.md:84`; `AWB/lab/closure_probe/RESULT.txt:1-2` |
| 19 | Off-diagonal core edge block omitted from layer 4 on | Edges on every layer (the internal copy of submission 337479's statistics, EDGE_END 16): dev-5 x0.9861 (99 .987, 32 1.079, 33 .957, 34 .957, 40 .981), est +1-4 GF: mixed / within noise. No edge block on any layer (C_noedge): -14.3 GF, raw +0.7 % dev / -1.2 % fresh (x0.9884 5/5). On v5 (edges on layers <8 by default), stopping at 4: x1.022. Edge thinning x1.022 (-2.0 GF, submission 335982); rank-1 edge mediator +1.52 % (internal version 18) | one n x n x n product per layer where kept (~14 GF over all layers on dev net 40; 3.0 % of the submission 335982 bill) | dev-5 submission 337577 NumPy; fresh 40-44; v5 99; submission 335982; internal version 18 700-701 | N:2279; `research/LEAN_SCAN_RESEARCH.md:19`; `research/LEAN_SCAN_EXPERIMENTS.md:74`; N:2141; CENSUS:77; `AWB/eppie_ideas/ex18_edge_rankone/experiment.json:158`; `estimators/10_kappa3_C_noedge/estimator.py:32-35` |

## submission 337479 component census (metered, net 99, PROF:113-137)

| component | GF | | component | GF |
|---|---:|---|---|---:|
| young H/P transport | 108.295 | | covariance / gauge transport | 26.461 |
| slice delivery family products | 93.514 | | core affine transport / read | 19.320 |
| elder transport + Y@G rebuild | 28.660 | | core update | 5.514 |
| other slice / TC / fold | 16.840 | | Mehler / Edgeworth | 3.384 |
| elder C^T Y | 15.890 | | PK4 leg / single / birth | 2.804 / 0.079 / 0.071 |
| elder switch / QR | 13.852 | | fresh sources / radius / gates | 1.373 |
| birth admission / repairs / negative / E / diag | 8.748 | | terminal mean | 1.415 |
| terminal TC | 7.276 | | unassigned | 0.625 |
| PK4 fold response | 3.156 | | | |
| young P-only tail / L1 dropped-row read | 1.322 / 1.080 | | **total** | **359.678** |

Bank subtotal 298.632 GF (83.0 %), chassis + PK4 ~61 GF. The submission 335982 census (CENSUS:26-49) has the same shape: bank
78.5 %, of which Horner/slice deposit 35 %, young transport 28 %, elders 15 %.

## Where the error comes from (attribution, submission 335982, oracle nets 0-9)

- L16 MSE splits as mu 78 %, sigma^2 12 %, kappa3 6 %, kappa4 4 %; 92 % of the L16 error is injected at layers 9-16
  (`AWB/lab/PLAN.md:47-48`).
- Per-layer local injection: sigma^2 53 %, kappa3 31 %, kappa4 17 % (`AWB/lab/PLAN.md:49`).
- Given exact layer-16 pre-activation cumulants the terminal closure leaves 9.4e-11 MSE (Gaussian alone 1.83e-7,
  +kappa3 1.93e-8): the readout is not the bottleneck, the propagated statistics are (`AWB/lab/PLAN.md:13-15`,
  `AWB/lab/closure_probe/RESULT.txt:1-2`).
- Teacher-forcing kappa3/kappa4 diagonals at one layer barely helps: diagonal errors are regenerated each layer from the
  transported tensors (`AWB/lab/PLAN.md:52-53`).

## What must be kept to preserve MSE (evidence only)

Each item has a recorded removal or weakening cost of at least ~5 % raw:

1. **The full-depth cubic history bank, all ages** (83 % of the bill). None x296; only the newest 8 ages x3.1; sources
   older than 12 layers dropped +5 %; old births removed from slice reads x4-138; bank-free compact rewrite x45.
2. **Elder information** (compressed old births): retiring elders x8.4-11.8. The compression is lossless (all-dense x1.04),
   but each notch of lower rank or earlier switch costs 6-60 % (0.75x ranks x1.086, all n/8 MSE +69 %, switch one layer earlier
   x1.058, two layers x1.2-1.28). Cap 256 instead of 320 costs +3.1 % raw officially (but pays in adjusted).
3. **The dense (2,1) slice from every live source**, including saturated rows or the L1 fill: slice removed x100-x15000;
   replaced by a covariance-star completion x53; rows dropped without fill x2; low-rank slice reads x2.3-5.1.
4. **The fresh completed-square H leg at full rank**, with its kappa3-slice feedback: covariance-bridge replacement x7.7;
   P-anchor x5.5-13.3; feedback rows zeroed x1.50; rank n/4 truncation +24 %.
5. **TC and its 3-generation Horner history:** TC off x1.23-1.30; history off x1.12.
6. **The fourth-cumulant trace core** (both products): rank-2 replacement x2.01; low-rank cores x1.38-1.80; tighter read
   collar x1.057.
7. **Spherical-radial factor:** x1.074 when replaced by Gaussian input (exact, no cost).
8. **Thinning thresholds at their current values** (they sit at break-even): doubling THIN_TAU x1.16; ROW_THIN 0.05 x1.57.
9. **TC/history slice folds:** x1.055-1.10.
10. Smaller, base-dependent (2-9 %): PK4 off x1.03-1.06; E6 off x1.008 (tdir_b) to x1.09 (v5); unit-prevariance gauge
    ~2 % (two nets, older lineage).

## What exists only for FLOP economy (evidence only)

1. **SB_E1 / SB_E2:** exact restructuring, -2.48 GF, MSE x0.9999.
2. **Lane L1 row fill:** -20 GF per net at raw x0.996. Its only accuracy role is to undo the row drop (x2 without it).
3. **Young/elder compression and the shared elder basis:** all-dense history is not more accurate (x1.04); shared vs
   per-birth bases x1.003. The elder machinery costs 58.4 GF and replaces dense transport of old births.
4. **Thinning plans and 64-padding:** they shrink or shape products (quantum 64 gained -2.7 % score via six-level FMM);
   exact removal of the padding gains < 1 GF.
5. **Omitting the core edge block from layer 4 on:** removes about one n^3 product per later layer; MSE effect within
   noise on submission 337479 (x0.986 dev-5 with edges everywhere, mixed per net).
6. **Rank-2 repair instead of a higher rank:** rank 256 buys -2.9 % for +53 GF; rank 32 -1.7 % for +1.15 points.
7. **TC_SLICE_FOLD** partly: an explicit separate Q31 has the same MSE (+0.18 %) and costs +0.69 points.
8. **'nogram'** (core metric as its diagonal only): x1.002 for -5.5 GF.

Accuracy-neutral on the evidence (removable): **z3^2/72 terminal term** (x0.997 when removed on submission 337479 itself; 0 GF).

No evidence either way: NegativeBank removal (only richer debits were tested, and lost), the conservation vector,
individual PK4 families, PK4 single-site in isolation.

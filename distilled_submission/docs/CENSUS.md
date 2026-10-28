# CENSUS: FLOPs and memory of the dense design (n = 1024, L = 16)

Author: Fable B. Conventions: one "product" = an (n x n) @ (n x n) matmul billed 2 n^3 = 2.147 GF (FlopScope bills
m n (2k-1)); n^2 elementwise work is 1 MF per pass and is counted only where it accumulates. 2^41 = 2199 GF.
Source counts: accumulate(l) transports k = l+1 sources (l = 0..14): sum = 120 source-layer transports; deposit(t)
reads t sources (t = 1..14): sum = 105 source reads; the terminal (t = 15) reads legs only elementwise (PK4, z3) and
through the TC query.

## 1. Leg inventory per source (what is stored and transported)

| design | legs transported per source per layer | thin rows (3 x n or 2 x n) | slice products per source per deposit |
| --- | --- | --- | --- |
| D2: plain statistics made dense (H, P; rank-3 repair RIGHT; rank-1 packet K, G) | 2 (fresh source: 1, its P is W rows) | RIGHT 3, packet 2 | 2 (+ thin) |
| D3a: D2 with dense R (SPEC 1.6(i)) | 3 | packet 2 | 3 |
| D3b: D2 with exact F (x) F removal, three-leg form u, X, P (SPEC 1.6(ii)B) | 3 | RIGHT 3 | 3 |
| D4: both (u, X, P, R) | 4 | none | 4 |

## 2. FLOP census

Bank, per leg: transports 120 x 2.147 = 258 GF per leg; deposits 105 x 2.147 = 225 GF per leg.

| item | D2 | D3a / D3b | D4 | source line |
| --- | ---: | ---: | ---: | --- |
| leg transports (1.4) | 515 | 773 | 1031 | B:2727-2764 |
| slice deposit products (1.15) | 451 | 676 | 902 | B:2152-2160, 1879-1900 |
| thin transports: RIGHT (3k x n) @ W, packets, RECON, RREAD, repair slice | ~4 | ~3 | 0 | B:2490-2503, 2514, 2160 |
| TC Horner lifts (2 per TC consumer t = 5..14 beyond the deposit's own, 1 at t = 4, 2 at terminal) | 45 | 45 | 45 | B:1983-2008 |
| TC repeated-index subtraction, exact (2 products per TC evaluation, 33 evaluations) | 142 | 142 | 142 | B:846-855 |
|   (same with rank 4 as in the plain code) | ~0 | ~0 | ~0 | B:750-768 |
| TC A B old-repair term, completion, Zp, folds (O(n^2) per TC evaluation) | ~3 | ~3 | ~3 | B:1498-1529 |
| engine covariance W^T C W (2 products x 14) + terminal diagonal (1) | 62 | 62 | 62 | M:661-715, 617-628 |
| engine core push W^T T W (2 x 14) + terminal (1) | 62 | 62 | 62 | M:562-615, 630-659 |
| edge block (1 product, layers 0..3) | 9 | 9 | 9 | M:955-1004 |
| Mehler rows + pair powers + core update + birth elementwise (~150 n^2 per layer) | ~2 | ~2 | ~2 | M:843-1089 |
| radial solves (Chebyshev matvecs; or linalg.solve ~0.7 GF float64 each x 30) | <1 (21) | <1 (21) | <1 (21) | MM:96-151 |
| PK4 read + fold + single site (elementwise over sources x n^2) | ~4 | ~4 | ~4 | K:307-498 |
| E6 pre-read / read (elementwise) | <1 | <1 | <1 | B:2266-2380 |
| SVDs (repair rank 2, packet rank 1; B_s rank 4 if kept): billed ~0, wall 0.2-0.4 s each | 0 | 0 (D3b: half) | 0 | B:672-696 |
| **total** (exact repeated index) | **1300** | **1783** | **2267** | |
| **multiplier** | **0.59** | **0.81** | **1.03** | |
| total with rank-4 repeated index | 1158 (0.53) | 1641 (0.75) | 2125 (0.97) | |

Notes.
- The Horner form of the history query is not optional: the direct form (2.10) recomputes the queries at t-2 and
  t-3 from scratch, about 4t products per TC consumer, +770 GF over t = 6..14. The Horner nesting is exact for
  dense sources (SPEC 2.5).
- The pre-lift form of the TC deposit costs the same as the per-source deposit (same count of products), so a TC
  layer costs its two extra Horner lifts only.
- D4 is just above multiplier 1.0 (2267 / 2199 = 1.03 exact, 0.97 with rank 4). Ranking of the remaining economies
  by GF saved per MSE cost, from SOURCE_MAP 2b/2c numbers:
    1. keep the rank-3 repair instead of dense R: saves 483 GF (258 transport + 225 deposit), recorded cost of the
       rank cut 0.03 % MSE (B:304, rank 3 vs 2) -> best ratio by far; this is D3b.
    2. keep the rank-4 repeated-index SVD instead of the exact (2.2): saves 142 GF; the recorded adjusted-score
       direction favoured lower ranks (B:759-761), raw effect unrecorded.
    3. keep the rank-1 negative packet instead of the exact removal: saves 483 GF; effect never screened.
    4. everything else (crops, elders, plans) is < 60 GF each and is what the clean version exists to remove.
  Recommendation: D3b (u, X, P legs; thin repair kept; exact repeated index), 1783 GF, multiplier 0.81. If a cheaper
  first build is wanted: D2 at 0.59, then measure D3b and the exact (2.2) separately (PLAN rule 2).

## 3. Memory census (persistent arrays), n = 1024

Per (n x n) array: 4 MiB float32, 8 MiB float64. s = L-1 = 15 sources.

| array | shape | count | float32 | float64 | needed because |
| --- | --- | --- | ---: | ---: | --- |
| LEG2 legs, D2 (H, P) | (2 s, n, n) x 4 generations | 4 | 480 MiB | 960 MiB | Horner history needs legs at t, t-1, t-2, t-3 (SPEC 2.5) |
| LEG2 legs, D3 (3 legs) | (3 s, n, n) x 4 | 4 | 720 MiB | 1440 MiB | |
| COMB2 coefficients at the current layer, D2 / D3 | (2 s or 3 s, n, n) | 1 | 120 / 180 MiB | 240 / 360 MiB | slice, TC query, directional pre-read, PK4 fold |
| RECON (R = LEFT RIGHT) | (s, n, n) | 1 | 60 MiB | 120 MiB | CP = ... + 2 P o R; directional fold; (D3a: R is a leg instead) |
| PP = P o P | (s, n, n) | 1 | 60 MiB | 120 MiB | directional pre-read T(u) (can be recomputed per source: optional) |
| LEFT, RIGHT x 2, RREAD, beta, packets | O(s n) | | < 2 MiB | < 4 MiB | |
| engine: pc, Cov(h), T (core), b slice, star-free B stash, Q22/4 | (n, n) | 6 | 24 MiB | 48 MiB | |
| Mehler stack (plain: 21 matrices) | (21, n, n) | 1 | 84 MiB | 168 MiB | avoid: accumulate the four pair powers row by row with two n x n temporaries |
| TC chain (per history TC): A (n x 3b), Bo factors or Bo itself (exact: n x n), vectors | 2 x (n, n) | 2 | 8 MiB | 16 MiB | exact (2.2) keeps Bo for two consumers |
| PK4 coefficient buffers | (s, 7, n) | | < 1 MiB | | |
| **state total D2** | | | **~0.84 GiB** | **~1.7 GiB** | |
| **state total D3b** | | | **~1.1 GiB** | **~2.3 GiB** | |

Peak RSS = state + temporaries. Temporaries per step: a batched product output (n x n per source in the batch; chunk
per source: 4-8 MiB), einsum temporaries of the pair law (a few n x n), the birth's extra/H/r (3-4 n x n). Budget
0.3 GiB float32 / 0.6 GiB float64 with per-source chunking. Expected peak: D3b float32 1.4 GiB; D3b float64 2.9 GiB.

dtype recommendation. The plain code is float32 throughout and reproduces submission 337479, so float32 legs reproduce the
statistic. Keep LEG2, COMB2, RECON, PP in float32 (1.1 GiB state for D3b; a float64 history at 2.3 GiB leaves
< 1.7 GiB for everything else under the 4 GiB peak rule and doubles BLAS time). Use float64 for: all (n,) per-neuron
algebra (vertices, Edgeworth rows, families, core weights, radial solve, completion coefficients), the engine's n x n
objects (pc, Cov(h), T, b, Q22/4: 48 MiB), and the ACCUMULATION of sums over sources (slice block, z3, rt, TC dq,
PK4 read): form each per-source product in float32 BLAS and add it into a float64 accumulator (n^2 per source, 1 MF).
Reason: the (2,2) row-sum cancellation (M:935-951) and the TC/PK4 sums over up to 15 sources are the places where the
plain code already works around float32 cancellation. This keeps state <= 1.2 GiB and peak RSS <= ~1.6 GiB, under the
2.5 GB / 4 GB constraints with margin. If the comparison against the plain version demands bitwise-close numbers, the
first build may stay float32 everywhere and switch the accumulators afterwards (measure alone).

## 4. Wall-time note

1783 GF of mostly (n x n)^3 BLAS at the plain run's observed rate (~60 GF/s effective on the busy M4 Pro, RUN_PLAIN_NET99)
would be ~30 s; large dense products run several times faster than the plain code's mix of small ops, so 15-30 s is
the expectation, well under 100 s. The SVDs (30 full 1024 x 1024 SVDs in D2, 15 in D3b, 0 in D4) cost 0.2-0.4 s each.

## 5. Note on Fable A's RESULTS.md section 1 (dense R / dense F at "+700 P")

RESULTS.md adds, for the dense R leg, "+265 P" for the TC's old-repair term A @ B becoming one product per old
source per TC evaluation. In the Horner/pre-lift form that product IS the deposit's pre-lift product for the R leg:
with R a dense leg, the old-source query (2.10) is sum over sources of [CU^T H + CP^T P + (P o P)^T R] at the entry
generation and the Horner lifts carry all three legs together, so each source pays one R product per consumer, the
+225 GF already counted in section 2 (the thin A @ B term disappears). The history TCs reuse the same Horner
intermediates. So dense R costs +483 GF total (D3a = 1783 GF, 0.81), not +1503 GF. The same holds for a dense F or
X leg: the packet terms in the queries vanish and the X leg enters the query like H. D4 (both) is 2267 GF (1.03).

## 6. k3_reference.py: FLOP reconciliation (net 99 tally 3001 GF, multiplier 1.365)

Per layer l = 1..16 (`layer_`), the bank holds J_l = 2n (l-1) columns when the layer starts; one product (m x k)(k x n)
is billed m n (2k-1) ~ 2 m k n. Counts in units of n^3 (1.0737 GF):

| step | per layer | sum over l |
| --- | --- | --- |
| bank transport, 3 legs: W^T (n x n) @ (n x J_l) | 3 x 2 n^2 J_l = 12 n^3 (l-1) | 12 x 120 = 1440 (l = 1..16) |
| face read, 3 products (n x J_l)(J_l x n) [not at the last layer: t3 only] | 12 n^3 (l-1) | 12 x 105 = 1260 (l = 1..15) |
| engine congruences: W^T C W (2 products) + W^T W | 6 n^3 | 96 |
| total | | 2796 n^3 = 3002 GF -> multiplier 1.365 |

So the tally is exactly the as-built arithmetic; nothing is read twice. My "~1.6 TF" after removing the second face read
was a slip: the removed read had never been in my earlier estimate (which already said 24 n^3 l per layer), so halving it
was wrong; the true before/after is 4056 -> 2796 n^3.

Remaining genuine redundancy (NOT implemented; manager to decide): every block of the bank has two EQUAL legs. The
tree block is (3 e_i delta_i, h_i, h_i) with b = c, the repair block is (delta_i, delta_i, r_i) with a = b, and every
later step applies the same W^T and the same gate d to all three legs, so the equality is preserved forever. Hence
(ab) c^T = (ac) b^T on tree columns and (ac) b^T = (bc) a^T on repair columns: one of the three face products per block
is a duplicate, and one of the three stored/transported legs is a duplicate. Storing two legs per block (x, y, with
the Sym_avg weights 1/3 [2 (x o y) y^T + (y o y) x^T] on tree blocks and 1/3 [(x o x) y^T + 2 (x o y) x^T] on repair
blocks) gives: transport 8 n^3 (l-1) -> 960, faces 8 n^3 (l-1) -> 840, congruences 96: 1896 n^3 = 2036 GF
(multiplier 0.926), bank 480 MiB instead of 720 MiB, identical results in exact arithmetic (rounding differs only in
the order of the summed products). About 40 changed lines in k3_reference.py.

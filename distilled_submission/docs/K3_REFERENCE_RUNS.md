# K=3 reference (clean/k3_reference.py) runs, 2026-10-27

Program: ARC's own K=3 SIMPLE estimator (Wu et al., arXiv 2605.05179) in factored form, numpy float64, 271 lines (two-leg storage, exact),
written by Fable B from the dense oracle k3_original/dense_simple.py (agreement <= 8e-15 at widths 8-64).
Runs: isolated process (isolated_rss.py), OMP_NUM_THREADS=4, numpy harness shims, nets from the `full` split.

| net | raw final-layer MSE | wall | peak RSS | GF | mult |
|---|---:|---:|---:|---:|---:|
| 99 | 3.1397e-8 | 6.6 s | 1.86 GiB | 2034.8 | 0.925 |
| 32 | 3.1790e-8 | 6.4 s | 1.77 GiB | 2034.8 | 0.925 |
| 33 | 3.4091e-8 | 6.8 s | 1.66 GiB | 2034.8 | 0.925 |
| 34 | 3.1561e-8 | 6.4 s | 1.71 GiB | 2034.8 | 0.925 |
| 40 | 3.1669e-8 | 6.7 s | 1.71 GiB | 2034.8 | 0.925 |

Reference on net 99: plain ex_34 statistics (elder cap 320) 1.1051e-8, so K=3 is 2.84x worse in raw MSE there.
The paper's own FLOP formula gives 3.6-4.1 x 2^41; this program meters 0.925 x 2^41 (FlopScope-style m n (2k-1)). The FLOP count is data-independent (no gating of the bank size).

History: the first version peaked at 13.04 GiB RSS on net 99 (same MSE): not an n^3 object but ~7 live copies of the
growing (n, 30n) factor bank per layer plus allocator fragmentation. Fixed by one preallocated (3, n, 2n(L-1)) bank
transported in place in 16 MiB column chunks; peak fell to 1.92 GiB and wall from 15.1 s to 8.3 s.

Second revision (two-leg storage): every factor block had two identical legs (tree (3e delta, h, h), repair (delta, delta, r)),
so one of three transported legs and one of three face products were duplicates. Storing two legs per block with the
Sym_avg weights is exact arithmetic (validated <= 8e-15 vs dense_simple at widths 8-64); MSE identical to every printed digit
on all five nets, GF 3001 -> 2035, wall 8.3 -> 6.6 s, bank 720 -> 480 MiB. Earlier three-leg rows: net 99 3.1397e-8 / 8.3 s /
1.92 GiB; nets 32/33/34/40 peak RSS 2.0-2.1 GiB, same MSEs.

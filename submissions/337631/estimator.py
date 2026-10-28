"""SQ1 Trace-M estimator (lineage FreshCPFoldCommonW -> PKG_C1VC -> F8), call-fused form. Per layer the engine carries the
pre-activation mean mu and covariance pc, a quadratic fourth-cumulant core A (pre-activation kappa4 as A-weighted symmetrised
products of pc) and the repeated third-cumulant slice b(i, j) = kappa3(i, i, j) delivered by bank.Bank. Post-activation moments
are the Gaussian ReLU integrals with an Edgeworth correction in (kappa3 diag, kappa4 diag); pair powers to order 3 of the Mehler
expansion give the next covariance and the (2,1), (2,2), (3,1) pair slices; the core is updated from this layer's mixed fourth
trace. Angular moments are propagated on the sphere; the exact radial mean E[chi_n / sqrt n] is restored. Inherited structurally tuned comparison arm; no targets are read.

Same physics as F8; only the NUMBER of logged array calls is reduced (the residual cap counts calls):
  * every per-neuron vector is a row of a few stacked arrays: the Gaussian basis B (one einsum over powers of z and sd), the
    Edgeworth-corrected moments MV (one einsum) and two sparse vector programs (_Prog: take, prod, take, einsum);
  * the Mehler coefficient buffer CB holds the matrices the core update needs (x, x^2/2, x^3/6, q22/4, b/2 and their products)
    as contiguous rows, so the core trace is one non-symmetrised sum Z of matrix * (column x row) terms, symmetrised once;
  * the stored core is T = A - s I = trace / (n + 4) (the form every consumer uses); the radial M is kept as (s, v, Ca).
  * round trips: on the official runtime every view x[...] costs like a call, so rows of the stacked arrays come from one
    unstack / split, views of the persistent coefficient buffer are made once, and column broadcasts go through transposed views.
Only the production configuration VERT = "AB" without the second-order (E2) rows is implemented."""
import flopscope.numpy as xp
from flopscope.stats import norm
from flopscope import current_budget, budget_summary_dict, BudgetExhaustedError, NoBudgetContextError
from whestbench import BaseEstimator, MLP
import bank
import fmm
import pk4
# TERM_CORE_CROP (lossless switch): every core T is read only on the alive block a x a (a = GM_TAU_A plan of the same vertex
#   plans: the next affine_fourth, or terminal_ga after the last hidden layer) and on its diagonal (the (2,2) row sums urow); so
#   its edge product G and block H are formed on (TF_OUT plan) n a instead of the TF_OUT plan at every layer (was: last only).
TERM_CORE_CROP = True
# MEHLER_SHARE (lossless up to rounding): the pair-power einsum's outputs cov, post and pv3 share their right factors (the f
#   family), so CB o f_j is formed once and contracted with the three left families (26 + 3 x 51 instead of 3 x 78 n^2 per
#   layer); u (right family g) keeps the 3-operand form. Same terms, reassociated products: not bitwise.
MEHLER_SHARE = True
LOSSLESS_DIAG, DIAG = False, {}   # research only: arrays kept for the offline harness when True
VERT = "AB"  # gate (A) and birth vertices (B) use the Edgeworth-corrected law; the core keeps the Gaussian vertices

# ---- PK4 research switches (2026-10-07, arm_pk4; all OFF = src_ct byte for byte; see pk4.py and ../ALGEBRA.md) -----------
# PK4_SINGLE: single-site sum_a W^4 kappa4(h_a) replacing the core's own diagonal (powers fit), O(n^2) per layer.
# PK4_LEGS:   per-birth reads on the bank legs: star / u-term (ages PK4_AGES) + Gaussian path / A1-path (age 1 only); the
#             chassis Gaussian-star consumer is REPLACED on the ages PK4 covers (no double count).
# PK4_CORESUB: subtract each added family's own trace-core image: "lite" (O(n^2)) or "exact" (+2 cropped products at age 1).
# PK4_TERM:   the same reads at the terminal mean (ages PK4_TERM_AGES).  PK4_RECORD: research dict receiving the components.
PK4_SINGLE, PK4_LEGS, PK4_CORESUB, PK4_TERM = True, True, "lite", True
PK4_FAMS, PK4_AGES, PK4_TERM_AGES = ("star", "u", "t31"), (1, 16), (1, 64)
PK4_HTAU, PK4_CTAU, PK4_PTAU = 0.1, 1e-3, 0.02   # crops: age-1 hubs e + |A1| > HTAU max; columns min(g, 1-g) > PTAU (product); CTAU only in the unfused path
# PK4 AGES (lane3_late_zone): the hidden consumers read every birth whose dense H / P legs are live (not yet switched to an
#   elder basis), on every column; PK4_AGES[1] is then only the minimum window. Measured: the age-window and column-band
#   truncations inject errors whose removal costs O(n^2) per age (+0.16 % FLOPs at width 1024 / depth 16).
PK4_RECORD = None
# ---- Line D graft (2026-10-17) from the PREMAP v7_sub8 lineage; both False = 08a1_argon_sub8 bit for bit --------------
# GRAFT_TAU2: second-order unary Edgeworth term (kappa3^2 / 72) f_p^(6) in the hidden-layer moments m, m2, m3, m4 and the
#   corrected gate dv (PREMAP _raw_moments); the terminal already has it (z3t^2 / 72 w16). One more (n,) column in the MV einsum.
# GRAFT_K31C: connected (3,1) closure q31(a,a,a,b) += 0.25 (kappa3_a / Sigma_aa) b(a,b) (PREMAP _advance_layer_body
#   fourth_triple += 0.25 (td / v) off_third), applied on the q31 / 6 coefficient row: one (n, n) multiply-add per layer.
GRAFT_TAU2, GRAFT_K31C = False, False   # coordinator: line D patch OFF (net-harmful on the panel); it came in through the L3 root

TF_OUT, TF_K = 0.03, 0.1                                    # edge product: output-row / contraction thinning
# ---- Lane L3 kappa4-sector economy knobs (2026-10-17); defaults = argon_min bit for bit ----------------------------------
# TERM_TC_DQ: False skips the terminal TC scalar dq (bank.tc_terminal, the Horner old-source query on the terminal columns);
#   the terminal accumulate still forms its coefficients for the E_TU read. EDGE_END: hidden layers < EDGE_END form the
#   off-diagonal edge block H of the core trace (None = bank.TCP_START, the chassis). GM_OFFDIAG_LAYERS: (lo, hi) = hidden
#   layers lo <= l < hi whose core push gm = W^T A W keeps only its exact diagonal ga (no symmetric off-diagonal product).
TERM_TC_DQ, EDGE_END, GM_OFFDIAG_LAYERS = True, None, None
GM_LOWRANK = 0   # with GM_OFFDIAG_LAYERS: 0 = no off-diagonal; r > 0 = rank-r off-diagonal from the core block's top-r SVD
# L3 port of sibling M11 (2026-10-13): core-read collar on the last depth // 3 hidden layers (structurally tuned, sibling dev)
GM_TAU_A_LATE, GM_OUT_LATE = 0.015, 0.05
GM_TAU_A, GM_OUT, TERM_GA, PC_TAU_A, PC_OUT = 0.1, 0.3, 0.0001, 0.001, 0.001   # core / terminal / covariance thinning (ex19_trims: GM was 0.03, 0.1)

def ndtr(a): return norm.cdf(a).astype(a.dtype)
def _diag(x): return x.diagonal() if getattr(x, "ndim", 0) == 2 else xp.diag(x)   # method form is unlogged locally
def _mask(plan, n, kept):  # 0/1 float mask (natural order) of the indices a plan kept / dropped, from its inverse permutation
    m = int(plan[0].shape[0]); return (plan[1] >= n - m if kept else plan[1] < n - m).astype(xp.float32)

# ---- constant tables (nested Python lists; uploaded once per prediction by _Tables) ------------------------------------
def _nest(shape, fill=0.0): return [_nest(shape[1:], fill) for _ in range(shape[0])] if shape else fill

# Fixed Gaussian basis data; exact source-bound generator lives outside this package.
_HIDDEN_BASIS = ([[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 1.0, 0.0, 0.0],
   [3.0, 0.0, 6.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
   [0.0, 5.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, -1.0, 0.0, 0.0],
   [3.0, 0.0, -6.0, 0.0, 1.0, 0.0], [0.0, -15.0, 0.0, 10.0, 0.0, -1.0]]],
 [[1.0], [2.0], [3.0], [4.0], [0.0], [-1.0], [-2.0], [-3.0], [-4.0], [-5.0], [-6.0]],
 [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
_TERMINAL_BASIS = ([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
  [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 1.0]]],
 [[1.0], [0.0], [-1.0], [-2.0], [-3.0]], [[0.0], [1.0], [2.0]])

BN = ["h", "raw2", "raw3", "raw4", "d", "e", "f", "ds4", "ds5", "ds6", "ds7"]      # hidden basis rows (ds1..ds3 = d, e, f)
MN = ["m", "m2", "m3", "m4", "dv", "ev", "fv", "t3"]                               # corrected moments, vertices, 3 h - m
INF = float("inf"); MV_LO, MV_HI = [[-INF]] * 4 + [[0.0], [0.0], [-INF], [-INF]], [[INF]] * 4 + [[1.0], [INF], [INF], [INF]]   # guards of dv, ev

_MV_LITERAL = [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.041666666666666664, 0.0, 0.0, 0.0]],
 [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0]],
 [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]],
 [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
 [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.041666666666666664, 0.0, 0.0]],
 [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.041666666666666664, 0.0]],
 [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.041666666666666664]],
 [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.16666666666666666, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.041666666666666664, 0.0, 0.0, 0.0]]]

# Mehler rows of the coefficient buffer: (r, s, k) = coefficient of orders (r, s) times cross^k / k!. Rows 1..10 are the
# matrices of the core trace; the order only permutes the terms of the pair-power sum.
# ROW_FOLD: the left-scaled rows (3, 0, k) = z36_i (0, 0, k) and (4, 0, k) = q424_i (0, 0, k) come last (rows 21..25); they share
# their coefficient matrix and right member with row k, so the pair-power sums fold them into row k's left vectors.
_MV_TAU2 = [[9, 1.0 / 72], [8, 1.0 / 36], [6, 1.0 / 12], [6, 1.0 / 3], [10, 1.0 / 72], None, None, [9, -1.0 / 72]]   # (BN index, coef) of the z3^2 column per MN row: m, m2, m3, m4, dv, ev, fv, t3

ROWS = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (2, 2, 0), (2, 2, 1), (2, 1, 0), (2, 1, 1), (1, 2, 0), (1, 2, 1), (3, 1, 0),
        (3, 1, 1), (1, 3, 0), (1, 3, 1), (2, 1, 2), (1, 2, 2), (0, 3, 0), (0, 3, 1), (0, 3, 2), (0, 4, 0), (0, 4, 1),
        (3, 0, 0), (3, 0, 1), (3, 0, 2), (4, 0, 0), (4, 0, 1)]
NFOLD = 21   # rows 0..20 carry the folded sums
assert all(ROWS[NFOLD + m] == (3, 0, m) and ROWS[m] == (0, 0, m) for m in range(3)) and all(ROWS[NFOLD + 3 + m] == (4, 0, m) for m in range(2))
assert sorted(ROWS) == sorted((r, s, k) for r, s in ((0, 0), (3, 0), (0, 3), (2, 1), (1, 2), (4, 0), (0, 4), (3, 1), (1, 3), (2, 2))
                              for k in range((3 if r + s == 0 else max(0, 3 - (r + s) + 2)) + 1))

class _Prog:
    """Sparse vector program OUT[k] = sum_g coef[k, g] * prod_j PV[idx[t(k, g), j]] over stacked primitive rows PV: four
    logged calls (take, prod, take, einsum) for any number of output vectors. Terms are (coef, factor names...); a string
    coef names a per-layer scalar (dyn), supplied at run time as the vector of those scalars."""
    def __init__(self, prims, outputs, dyn=(), factors=3, group=3):
        index, products = {name: i for i, name in enumerate(prims + ["1"])}, {(): 0}; assert len(index) == len(prims) + 1, "duplicate primitive"
        self.names, self.dyn, K = [name for name, _ in outputs], list(dyn), len(outputs)
        self.row = {name: i for i, name in enumerate(self.names)}; assert len(self.row) == K, "duplicate output"
        gidx, coef = _nest((K, group), 0), _nest((1 + len(self.dyn), K, group))
        for k, (name, terms) in enumerate(outputs):
            assert 0 < len(terms) <= group, name
            for g, (c, *fs) in enumerate(terms):
                key = tuple(sorted(index[f] for f in fs)); assert 0 < len(key) <= factors, (name, fs)
                gidx[k][g] = products.setdefault(key, len(products))
                if isinstance(c, str): coef[1 + self.dyn.index(c.lstrip("-"))][k][g] = -1.0 if c.startswith("-") else 1.0
                else: coef[0][k][g] = float(c)
        idx = [list(key) + [index["1"]] * (factors - len(key)) for key in products]   # insertion order = product number
        self.l_idx, self.l_gidx, self.l_coef, self.n_prims = idx, gidx, coef, len(prims)
    def bind(self, up):
        self.idx, self.gidx = up(self.l_idx, xp.int64), up(self.l_gidx, xp.int64)
        self.coef = up(self.l_coef if self.dyn else self.l_coef[0]); return self
    def run(self, pv, scalars=None):  # pv: (primitives + ones row, n); scalars: (1 + len(dyn),) with a leading one
        assert int(pv.shape[0]) == self.n_prims + 1
        coef = self.coef if scalars is None else xp.einsum("s,skg->kg", scalars, self.coef)
        return xp.einsum("kg,kgn->kn", coef, xp.take(xp.prod(xp.take(pv, self.idx, axis=0), axis=1), self.gidx, axis=0))

D1 = ["h", "d", "e", "f", "ds4", "ds5", "ds6", "ds7"]                                   # k-th mu-derivative of E relu
D2 = [(1, "raw2"), (2, "h")] + [(2, x) for x in D1[1:7]]                                # ... of E relu^2
_NEG3_D2 = (-3, -6, -6, -6, -6, -6, -6, -6)
D3 = [(1, "raw3"), (3, "raw2"), (6, "h")] + [(6, x) for x in D1[1:6]]                   # ... of E relu^3
PB = BN + MN + ["mm", "omd", "v", "sd", "gv", "ga", "z3", "q4"]                       # primitives of program B
BSING = ["ov", "k3", "k4a", "k4b", "a1", "a2", "ga1", "ga2", "stf", "dh", "ov3", "gah", "gvh", "z36", "q424", "astar", "astar2",
         "ddv", "fv2", "a1tc", "dmv"]                                                          # rows handed out as 1-D vectors
BBLOCK = [("key", 3), ("lc", 4), ("lr", 4), ("oqc", 2), ("oqr", 2)]                    # rows handed out as blocks

def _program_b(first):
    """Every vector that is a polynomial of the basis: single rows (BSING), blocks (BBLOCK), the centered families f, g,
    cube (rows fam0 + q * 8 + k) and rows that only feed program C."""
    fam = [("f%d" % k, [(1, D1[k])] + ([(-1, "m")] if k == 0 else [])) for k in range(8)]
    fam += [("g%d" % k, [D2[k], (-2, "m", D1[k])] + ([(1, "mm")] if k == 0 else [])) for k in range(8)]
    fam += [("c%d" % k, [D3[k], (_NEG3_D2[k], "m", D2[k][1]), (3, "mm", D1[k]) if k else (1, "mm", "t3")]) for k in range(8)]
    u3, u2 = [(1, "e"), (-1, "h", "f"), (-1, "d", "e")], [(2, "d"), (-2, "h", "e"), (-2, "d", "d")]
    stf = [("stfscale" if first else 2.0, "d"), ("-stfscale" if first else -2.0, "m", "e"), (-2, "d", "d")]
    one = dict(ov=[(1, "m2"), (-1, "mm")], k3=[(1, "m3"), (-3, "m", "m2"), (2, "mm", "m")],
               k4a=[(1, "m4"), (-4, "m", "m3"), (12, "mm", "m2")], k4b=[(-6, "mm", "mm"), (-3, "m2", "m2")],
               a1=dict(fam)["g1"], a2=dict(fam)["g2"], ga1=[(2, "h", "omd")], ga2=[(2, "d"), (-2, "h", "e")], stf=stf,
               dh=[(0.5, "d")], ov3=[(3, "m2"), (-3, "mm")], gah=[(0.5, "ga")], gvh=[(0.5, "gv")], z36=[(0.16666666666666666, "z3")],
               q424=[(0.041666666666666664, "q4")], astar=[(0.5, "ev", "gv")], astar2=[(1, "ev", "gv")], ddv=[(1, "dv", "dv")],
               fv2=[(2, "fv")], a1tc=[(2, "h"), (-2, "m", "dv")], dmv=[(1, "dv", "dv", "ga")])
    out = [(name, one[name]) for name in BSING]
    out += [("key0", stf), ("key1", u2), ("key2", [(c, *fs, "sd") for c, *fs in u3])]                         # plan key rows
    out += [("lc0", [(2, "e")]), ("lc1", [(2, "e")]), ("lc2", [(1, "e")]), ("lc3", [(2, "d")])]              # left: columns
    out += [("lr0", u3), ("lr1", u3), ("lr2", u2), ("lr3", u3)]                                               # left: rows
    out += [("oqc0", one["gah"]), ("oqc1", one["gvh"]), ("oqr0", one["gvh"]), ("oqr1", one["gah"])]           # q22 outer part
    assert [name for name, _ in out[len(BSING):]] == ["%s%d" % (b, i) for b, m in BBLOCK for i in range(m)]
    out += fam + [("a2md", [(2, "d", "d"), (-2, "m", "e", "d"), (-2, "d", "d", "d")]),
                  ("a1d", [(2, "h", "d"), (-2, "m", "d", "d")]), ("a1e", [(2, "h", "e"), (-2, "m", "d", "e")]),
                  ("m7", [(-6, "d", "d"), (12, "m", "d", "e"), (-6, "h", "e")]),                             # -3 (a2 d + a1 e)
                  ("dz", [(1, "dv", "dv", "z3")])]
    return _Prog(PB, out, dyn=["stfscale"] if first else ())

SN = ["ca1", "gma", "gra", "p", "q2", "bg", "btg", "qg", "vv", "ca", "v", "gv", "ga", "mk"]    # per-layer singles of program C
CBLOCK = [("g", 11), ("r", 11), ("cc", 2), ("oc", 2), ("or", 2)]                               # blocks of program C
CTAIL = ["gmc", "grc", "xc", "xr", "beta0", "hd", "hv"]                                       # its 1-D rows

def _program_c(bnames, first):
    """Column / row weights of the core trace Z and the vector parts of the birth: products of program-B rows, the basis
    and the matrix-vector products. Rows g0..g10 / r0..r10 weight the coefficient-buffer rows 0..10."""
    g = [[(1, "vv", "d")], [(1, "f", "w3"), (1, "e", "btg")], [(-2, "a1d"), (1, "f", "p"), ("sr", "e")], [(-3, "f")],
         [(-2, "a1d"), (1, "f", "p")], [(-1, "f")], [(-2, "a1d"), (1, "f", "p")], [(-3, "d", "ga2"), (-3, "e", "ga1")],
         [(1, "e", "p")], [(1, "d", "ga1")], [(1, "m7")]]        # ones (radial v (x) Ca), x, x^2/2, x^3/6, q22/4, x q22/4, b/2, x b/2, b^T/2, x b^T/2, q31/6
    rho = [(1, "d", "ga2"), (1, "e", "ga1")]
    r = [[(1, "ca", "d")], [(1, "d")], [(1, "e")], rho, [(1, "e")], rho, [(1, "d")], [(1, "e")], [(1, "e")], [(-2, "f")], [(1, "d")]]
    out = [("g%d" % i, t) for i, t in enumerate(g)] + [("r%d" % i, t) for i, t in enumerate(r)]
    out += [("cc0", [(1, "e", "ca1"), (-1, "v", "a2md"), (-1, "v", "a1e")]), ("cc1", [("hsr", "d")]),          # pc (two parts)
            ("oc0", [(0.5, "e", "ga")]), ("oc1", [(0.5, "e", "gv")]), ("or0", [(1, "d", "gra")]), ("or1", [(1, "d", "gma")])]
    assert [name for name, _ in out] == ["%s%d" % (b, i) for b, m in CBLOCK for i in range(m)]
    tail = dict(gmc=[("hs2", "d"), (1, "e", "gra")] + ([] if first else [(1, "a2", "gv", "d")]),                # W^T A W
                grc=[("hs1", "d"), (1, "e", "gma")],                                                           # W^T W
                xc=[(1, "vv", "e")], xr=[(1, "ca", "e")],                                                     # x: radial v (x) Ca on e e
                beta0=[(1, "k3"), (-1, "dv", "dz"), (-3, "astar", "dmv")],
                hd=[(1, "d")], hv=[(1, "v", "mk")])                                                           # edge block: d, masked v
    return _Prog(bnames + BN + MN + ["w3"] + SN, out + [(name, tail[name]) for name in CTAIL], dyn=["sr", "hsr", "hs1", "hs2"])

_PB_SPEC = [_program_b(first) for first in (False, True)]
_PC_SPEC = [_program_c(_PB_SPEC[0].names, first) for first in (False, True)]
def _cuts(sizes):  # split points after each block size
    out, at = [], 0
    for m in sizes: at += m; out.append(at)
    return out

class _Tables:
    """Per-prediction constants (uploaded inside the caller's BudgetContext)."""
    def __init__(self, n, dtype, dimension_constants=None):
        up = self.up = lambda a, dt=dtype: xp.asarray(a, dtype=dt)
        self.bC, self.bexp, self.bk = [up(t) for t in _HIDDEN_BASIS]
        self.tC, self.texp, self.tk = [up(t) for t in _TERMINAL_BASIS]
        mv = _MV_LITERAL
        if GRAFT_TAU2:   # fourth coefficient column (z3^2): f_p^(6) / 72 per moment row (BN index, coefficient)
            mv = [[list(c) for c in r] + [[0.0] * len(BN)] for r in _MV_LITERAL]
            for r, t in zip(mv, _MV_TAU2):
                if t is not None: r[3][t[0]] = t[1]
        self.mv, self.lo, self.hi = up(mv), up(MV_LO), up(MV_HI)
        self.mhalf, self.half, self.two, self.six, self.one = up(-0.5), up(0.5), up(2.0), up(6.0), up([1.0])
        dim = bank._dimension_constants(n) if dimension_constants is None else dimension_constants
        self.n4 = dim["n+4"].astype(dtype)
        self.negative_n2 = xp.negative(dim["n+2"].astype(dtype))
        self.core_coef = xp.divide(_scalar(-1.0), dim["2(n+2)(n+4)"])   # metered dimension products, float64 coefficient
        self.ones = xp.ones(n, dtype=dtype); self.ones1 = xp.ones((1, n), dtype=dtype)
        self.mones = xp.negative(self.ones)   # FUSE: constant -1 column of the fused core-trace einsum
        self.dynm = up([[1.0, 0, 0], [0.5, 0, 0], [0, 0.25, 0], [0, 0, 0.25]])            # sr, hsr, hs1, hs2 from (s_r, E1, E2)
        self.pb = [p.bind(up) for p in _PB_SPEC]
        stfscale = _scalar(xp.subtract(2.0, xp.divide(_scalar(4.0), dim["n+2"])), dtype)
        self.pb[1].coef = xp.einsum("s,skg->kg", xp.stack([up(1.0), stfscale]), self.pb[1].coef)
        self.pc = [p.bind(up) for p in _PC_SPEC]
        self.bcuts, self.ccuts = _cuts([len(BSING)] + [m for _, m in BBLOCK]), _cuts([m for _, m in CBLOCK])
        QL, QR, f0 = (0, 1, 1, 2), (0, 0, 1, 0), self.pb[0].row["f0"]   # pair powers: left families (f, g, g, cube), right (f, f, g, f)
        self.il = up([[f0 + q * 8 + r + k for r, s, k in ROWS] for q in QL], xp.int64); self.ir = up([[f0 + q * 8 + s + k for r, s, k in ROWS] for q in QR], xp.int64)
        if MEHLER_SHARE:  # outputs (cov, post, pv3) share the right family f (QR 0); u alone uses g: left rows in order (0, 1, 3, 2)
            assert QR[0] == QR[1] == QR[3] != QR[2]
            self.il_s = up([[f0 + QL[p] * 8 + r + k for r, s, k in ROWS] for p in (0, 1, 3, 2)], xp.int64)
            self.ir_s = up([[f0 + QR[p] * 8 + s + k for r, s, k in ROWS] for p in (0, 2)], xp.int64)

def _scalar(value, dtype=xp.float64):
    """Numerical scalar operations must use public xp, including reductions."""
    return xp.asarray(value, dtype=dtype)


def _radial_parts_f64(covariance, a, trace, work_dtype=xp.float64, eps=2.220446049250313e-16):
    """Float64 radial solve and recurrence, with every scalar step metered (used by radial_parts for the direct-solve branch)."""
    c, rhs = covariance.astype(work_dtype), trace.astype(work_dtype)
    tau = _scalar(xp.einsum("ii->", c), work_dtype)
    frob = xp.sqrt(_scalar(xp.einsum("ij,ij->", c, c), work_dtype))
    halfwidth = xp.multiply(2.0, frob)
    if not bool(xp.greater(xp.subtract(tau, halfwidth), 0.0)):
        v = xp.linalg.solve(2.0 * c + tau * xp.eye(int(c.shape[0]), dtype=work_dtype), rhs)
    else:
        aa = xp.divide(tau, halfwidth)
        root = xp.sqrt(xp.subtract(xp.multiply(aa, aa), 1.0))
        rho = xp.subtract(aa, root)
        numerator = xp.divide(xp.multiply(xp.multiply(eps, xp.subtract(1.0, rho)), root), 2.0)
        degree_value = xp.ceil(xp.divide(xp.log(numerator), xp.log(rho)))
        if not bool(xp.isfinite(degree_value)):
            raise bank.DegenerateMoments('nonfinite radial series degree')
        degree = max(2, int(degree_value))
        m2 = c * xp.divide(2.0, frob)
        rows = [rhs, xp.matmul(m2, rhs)]
        xp.multiply(rows[1], 0.5, out=rows[1])
        for k in range(1, degree):
            nxt = xp.matmul(m2, rows[k])
            rows.append(xp.subtract(nxt, rows[k - 1], out=nxt))
        coeff = xp.divide(xp.divide(xp.multiply(2.0, xp.power(xp.negative(rho), xp.arange(degree + 1, dtype=work_dtype))), root), halfwidth)
        xp.copyto(coeff[:1], xp.divide(xp.divide(1.0, root), halfwidth))
        v = xp.matmul(coeff, xp.stack(rows))
    v = v.astype(covariance.dtype)
    return _scalar(xp.einsum("i,i->", a, v), covariance.dtype), v, xp.matmul(covariance, a)

_EPS32 = 1.1920928955078125e-07   # float32 machine epsilon 2^-23 (the float64 form uses 2^-52)
_M10_AR = {}   # M10: per-prediction cache of the radial-series power ranges (cleared by _predict_gauged)


def radial_parts(covariance, a, trace):
    """Radial solve v = (2C + tau I)^-1 trace: the Chebyshev series runs in the covariance dtype (float32) to its machine
    epsilon (the float64 form's outputs are cast back to float32 anyway); the rare direct-solve branch (tau <= 2 |C|_F) is
    delegated to the float64 form, where conditioning can be large. Knob-free (working dtype's machine epsilon)."""
    dt = covariance.dtype; c, rhs = covariance, trace
    tau = xp.einsum("ii->", c)                       # FUSE: already a dt scalar (the asarray wrapper was a logged no-op)
    frob = xp.sqrt(xp.einsum("ij,ij->", c, c))
    halfwidth = xp.multiply(2.0, frob)
    if not bool(xp.greater(tau, halfwidth)):         # FUSE: tau - hw > 0 <=> tau > hw in IEEE arithmetic
        return _radial_parts_f64(covariance, a, trace)
    aa = xp.divide(tau, halfwidth)
    root = xp.sqrt(xp.subtract(xp.multiply(aa, aa), 1.0))
    rho = xp.subtract(aa, root)
    numerator = xp.multiply(xp.multiply(5.960464477539063e-08, xp.subtract(1.0, rho)), root)   # FUSE: /2 folded into eps (exact power of 2)
    degree_value = xp.ceil(xp.divide(xp.log(numerator), xp.log(rho)))
    if not bool(xp.isfinite(degree_value)):
        raise bank.DegenerateMoments('nonfinite radial series degree')
    degree = max(2, int(degree_value))
    m2 = c * xp.divide(2.0, frob)
    rows = [rhs, xp.matmul(m2, rhs)]
    xp.multiply(rows[1], 0.5, out=rows[1])
    for k in range(1, degree):
        nxt = xp.matmul(m2, rows[k])
        rows.append(xp.subtract(nxt, rows[k - 1], out=nxt))
    ar = _M10_AR.get(akey := (degree + 1, str(dt)))   # M10: one upload per (degree, dtype) per prediction
    if ar is None: ar = _M10_AR[akey] = xp.arange(degree + 1, dtype=dt)
    coeff = xp.divide(xp.divide(xp.multiply(2.0, xp.power(xp.negative(rho), ar)), root), halfwidth)
    xp.copyto(coeff[:1], xp.divide(xp.divide(1.0, root), halfwidth))
    v = xp.matmul(coeff, xp.stack(rows))
    return xp.einsum("i,i->", a, v), v, xp.matmul(covariance, a)

def covariance_star_repeated(vector, covariance):
    return xp.outer(_diag(covariance), vector) + 2.0 * vector[:, None] * covariance


def covariance_star_trace(vector, covariance):
    tau = _scalar(xp.einsum("ii->", covariance), covariance.dtype)
    return xp.multiply(tau, vector) + 2.0 * xp.matmul(covariance, vector)


def covariance_star_birth(a, matrix, covariance, pre_vector, pre_covariance, gate):
    """Trace-project the combined inherited and fresh stars into the current metric.

    This preserves their total cubic trace before repeated repair compression.
    Distinct-index tensors remain approximate; all existing ranks and readers
    are unchanged. The repair compensates new minus inherited repeated ownership.
    """
    inherited = gate * pre_vector
    old_metric = (pre_covariance * gate[:, None]) * gate[None, :]
    trace = covariance_star_trace(a, matrix) + covariance_star_trace(inherited, old_metric)
    _, vector, _ = radial_parts(covariance, xp.zeros_like(a), trace)
    delta = covariance_star_repeated(vector, covariance) - covariance_star_repeated(inherited, old_metric)
    return vector, delta


def pcx_first_order_coefficient(g1, g4, g5, z3, q4):
    """Own pair-power first-order (2,1) coefficient; named program-B rows."""
    return g1 + z3 * g4 / 6.0 + q4 * g5 / 24.0


def pcx_self_entry(hh, fdiag, coefficient, sq, pos, denom):
    """Replace only the completed-square hub diagonal; keep its TC debit aligned.

    Existing later repeated/beta compensation and refreshed dh own the change.
    The negative ff source is untouched. Cropping and rank fit remain approximations.
    """
    diagonal = _diag(hh)
    delta = xp.where(pos, coefficient / denom - diagonal, 0.0)
    xp.fill_diagonal(hh, diagonal + delta)
    return hh, fdiag + sq * delta


class _Views:
    """Views of the persistent coefficient buffer CB (rows = ROWS), made once per prediction: single rows, row blocks
    rA_B(_STEP) = CB[A:B:STEP] and transposed views (suffix T: matrices transposed) for column-vector coefficients."""
    def __init__(self, CB):
        R = xp.unstack(CB); self.ones, self.x, self.x2, self.x3, self.qq4, self.bbh, self.q316, self.r11 = R[0], R[1], R[2], R[3], R[4], R[6], R[10], R[11]
        self.bbhT, self.q316T = R[6].T, R[10].T
        for a, b, step in ((0, 2, 1), (0, 3, 1), (0, 11, 1), (4, 8, 2), (5, 9, 2), (6, 10, 2), (2, 10, 2), (8, 10, 1), (12, 14, 1), (14, 16, 1),
                           (16, 19, 1), (19, 21, 1), (21, 24, 1), (24, 26, 1)):
            setattr(self, "r%d_%d" % (a, b) + ("_%d" % step if step > 1 else ""), CB[a:b:step])
        for name in ("r0_2", "r0_3", "r21_24", "r24_26"): setattr(self, name + "T", getattr(self, name).transpose(0, 2, 1))
        self.cb_fold, self.cb_tail = CB[:NFOLD], CB[1:NFOLD]   # ROW_FOLD rows; U_ROWSUM: those past the constant row 0 (ones)

GAUGE_POLICY = "unit_prevariance"  # fixed candidate; "identity" schedule-only control; original/off loop archived

class TraceM:
    """Layer moment recurrence and its scalar / core / slice consumers."""
    def __init__(self): self.cdf, self.dtype, self.layer_coverage, self.diag_on, self.diag, self.l1_drop = ndtr, xp.dtype(xp.float32), [], LOSSLESS_DIAG, DIAG, None
    def _alive_rows(self, w, matrix, tau):  # previous-layer thinning: (plan, rows of w, matrix on the previous gates > tau)
        if (pa := self.vertex_plans.plan(tau)) is None: return None, w, matrix
        return pa, xp.take(w, pa[0], axis=0), xp.take(xp.take(matrix, pa[0], axis=0), pa[0], axis=1)
    def _gm_taus(self, layer):   # L3/M11: (GM_TAU_A, GM_OUT) of the core read at `layer` (terminal: chassis values)
        depth = self.bank.depth
        return (GM_TAU_A_LATE, GM_OUT_LATE) if depth - 1 - depth // 3 <= layer <= depth - 2 else (GM_TAU_A, GM_OUT)
    def affine_fourth(self, w, ww, gv, pc, d, v):
        """(gm, gram, diag gm, diag gram) with gm = W^T A W, A = T + s I (T over previous gates > GM_TAU_A, off-diagonals only on
        current gates > GM_OUT, exact diagonal) and the spherical metric W^T W carried as its diagonal gv only (no Gram
        product: 'nogram', measured x1.002 raw for -5.5 GF at width 1024). self.core_t is T itself."""
        if self.layer == 0: gm = pc / self.K.negative_n2; return gm, pc, _diag(gm), v
        gta, gout = self._gm_taus(self.layer)
        n, s = int(w.shape[0]), self.core_s; po = self.gate_plans.plan(gout)
        gram = xp.zeros((n, n), dtype=w.dtype); xp.fill_diagonal(gram, gv)
        pa, wa, taa = self._alive_rows(w, self.core_t, gta); wta = fmm.matmul(wa.T, taa, self.levels)
        tdiag = xp.einsum("ja,aj->j", wta, wa)
        if pa is not None: tdiag = tdiag + xp.matmul(xp.einsum("ii,i->i", self.core_t, _mask(pa, n, False)), ww)
        ga = s * gv + tdiag
        # DEAD_DIAG: s I (= s * gram's only entries) sits on the diagonal, which fill_diagonal(gm, ga) overwrites
        if GM_OFFDIAG_LAYERS is not None and GM_OFFDIAG_LAYERS[0] <= self.layer < GM_OFFDIAG_LAYERS[1]:   # L3: diagonal-only core push
            if GM_LOWRANK:   # off-diagonal from the top-GM_LOWRANK singular triplets of the cropped core block (metered block Krylov)
                u, sv, vh = bank._svd_topk(taa, GM_LOWRANK); gm = xp.matmul(xp.matmul(wa.T, u) * sv, xp.matmul(vh, wa))
            else: gm = xp.zeros((n, n), dtype=w.dtype)
        elif po is None: gm = fmm.matmul_sym(wta, wa, self.levels)
        else:
            io, inv_o, _ = po
            gm = bank._scatter(fmm.matmul_sym(xp.take(wta, io, axis=0), xp.take(wa, io, axis=1), self.levels), inv_o, (0, 1), io)
        xp.fill_diagonal(gm, ga); return gm, gram, ga, gv
    def terminal_variance(self, wn, cov, ov, layer):
        """Terminal v_j = W_j^T C W_j with C the post-activation covariance of the last hidden layer: the rows / columns of
        neurons whose previous gate is <= PC_TAU_A (the exact-alive tolerance thin_pc already uses) enter through their
        exact diagonal C_aa W_aj^2 only; the rest is one cropped product."""
        pa = self.vertex_plans.plan(PC_TAU_A)
        if pa is None: return xp.einsum("aj,aj->j", wn, fmm.gen(cov, wn, b_key=("Wterm", layer + 1)))
        _, wa, caa = self._alive_rows(wn, cov, PC_TAU_A)
        v = xp.einsum("aj,aj->j", wa, fmm.matmul(caa, wa, self.levels))
        return v + xp.matmul(ov * _mask(pa, int(wn.shape[0]), False), wn * wn)
    def terminal_ga(self, w, mu, v, gv):
        """Terminal ga_j = w_j^T A w_j only where the q4 weight |(z^2 - 1) phi(z)| > TERM_GA (else s |w_j|^2)."""
        n, s = int(w.shape[0]), self.core_s
        z = mu / xp.sqrt(v); zz = z * z; pb = bank._plan(xp.abs((zz - 1) * xp.exp(zz * -0.5)), TERM_GA, n, self.quantum)
        if bank.TERM_U16: pb = bank._sub_plan(pb, self.u16_keep(z), n, self.quantum)   # q4 is not read off U16
        _, wa, taa = self._alive_rows(w, self.core_t, GM_TAU_A)
        if pb is None: return xp.einsum("aj,aj->j", wa, fmm.matmul(taa, wa, self.levels)) + s * gv
        ib, inv_b, mb = pb[0], pb[1], int(pb[0].shape[0])
        wab = xp.take(wa, ib, axis=1); tb = xp.einsum("aj,aj->j", wab, fmm.matmul(taa, wab, self.levels))
        return xp.take(xp.concatenate([xp.zeros(n - mb, dtype=gv.dtype), tb]), inv_b) + s * gv
    def u16_keep(self, z):  # TERM_U16 read set: |alpha| < TERM_U16_ALPHA or inside the transported terminal plan (exact there)
        keep, tp = xp.abs(z) < bank.TERM_U16_ALPHA, self.bank.term_plan
        if tp is not None: keep = keep | (tp[1] >= int(z.shape[0]) - int(tp[0].shape[0]))
        self._u16 = keep; return keep
    def thin_pc(self, w, ww, cov, ov, mu, check_variance=False):  # pc = W^T C W over previous gates > PC_TAU_A; off-diagonal only on current gates > PC_OUT
        n, (pa, wa, caa) = int(w.shape[0]), self._alive_rows(w, cov, PC_TAU_A)   # ov = the diagonal of cov
        wtc = fmm.matmul(wa.T, caa, self.levels); vd = xp.einsum("ja,aj->j", wtc, wa)
        if pa is not None: vd = vd + xp.matmul(ov * _mask(pa, n, False), w * w if ww is None else ww)   # ww only when read
        if check_variance: self._check_prevariance(vd)
        sd = xp.sqrt(vd); z = mu / sd; d = self.cdf(z); po = self._gates(d, n).plan(PC_OUT)
        if po is None: pc = fmm.matmul_sym(wtc, wa, self.levels)
        else: pc = bank._scatter(fmm.matmul_sym(xp.take(wtc, po[0], axis=0), xp.take(wa, po[0], axis=1), self.levels), po[1], (0, 1), po[0])
        xp.fill_diagonal(pc, vd); self._gate_cache = (mu, pc, sd, z, d, vd); return pc   # the gate of this pc, computed once
    def _gates(self, d, n):  # the plans of the current Gaussian gate d share one argsort (covariance, deposit, core, edge)
        self.gate_plans = bank.GatePlans(d, (PC_OUT, bank.THIN_OUT, TF_OUT, GM_OUT, GM_OUT_LATE), n, self.quantum, (), self._thr_cache); return self.gate_plans
    def live_gate(self, mu, pc):  # (sd, z, Phi(z), v) of the current layer, computed once, shared by the bank deposit and the update
        cache = self._gate_cache
        if cache is None or cache[0] is not mu or cache[1] is not pc:
            v = _diag(pc); sd = xp.sqrt(v); z = mu / sd; cache = self._gate_cache = (mu, pc, sd, z, self.cdf(z), v); self._gates(cache[4], len(mu))
        return cache[2:]
    def basis(self, sd, z, d, C, exps, ks):  # stacked Gaussian ReLU integrals: rows sd^exp (P(z) Phi + Q(z) phi)
        phi = xp.exp(xp.einsum("i,i,->i", z, z, self.K.mhalf)) * 0.3989422804014327
        return xp.einsum("bn,cbk,kn,cn->bn", xp.power(sd, exps), C, xp.power(z, ks), xp.stack([d, phi]))
    def extended(self, mu, pc, z3, b, w, ww, gvw):
        """Gates, Edgeworth-corrected moments, centered pair powers, the core update and the vector parts of the birth.
        Returns the outgoing mean, covariance, (2,1) pair power, the program-B rows and the program-C rows."""
        K, L, n, V = self.K, self.layer, len(mu), self.V; first = int(L == 0)
        sd, z, d, v = self.live_gate(mu, pc); B = self.basis(sd, z, d, K.bC, K.bexp, K.bk)
        trace = self.bank.trace_override.pop(L, None); rt = xp.sum(b, axis=0) if trace is None else trace
        gm, gram, ga, gv = self.affine_fourth(w, ww, gvw, pc, d, v)
        if (dga := self.bank.e_dq.pop(L, None)) is not None:   # E_GA: the T(u) image enters the core diagonal read itself (q4, q22, ...)
            ga = ga + dga; xp.fill_diagonal(gm, ga)
        self.input_core = (gm, gram)
        q4 = xp.einsum("i,i,->i", ga, gv, K.six)
        if self.tc_extra is not None: q4 = q4 + self.tc_extra
        # Edgeworth-corrected moments m_p and vertices E[relu^(k)(z)] under the (z3, q4)-corrected law (same order as the mean);
        # validity guards on two rows: the gate is a probability, the density is non-negative
        MV = xp.clip(xp.einsum("rcb,cn,bn->rn", K.mv, xp.stack([K.ones, z3, q4, z3 * z3] if GRAFT_TAU2 else [K.ones, z3, q4]), B), K.lo, K.hi)
        m, dv, ev, fv = MV[0], MV[4], MV[5], MV[6]; self.vert = dv, ev, fv; self.forecast_gates[L] = dv   # M10: views
        self.vertex_plans = bank.GatePlans(dv, (PC_TAU_A, GM_TAU_A, bank.THIN_TAU, GM_TAU_A_LATE), n, self.quantum, (), self._thr_cache)  # corrected gate: transport / previous-layer plans
        SB = K.pb[first].run(xp.vstack([B, MV, m * m, 1 - d, v, sd, gv, ga, z3, q4, K.ones1]))   # M10: one vstack
        bc = K.bcuts; sing, key, lc, lr, oqc, oqr = SB[:bc[0]], SB[bc[0]:bc[1]], SB[bc[1]:bc[2]], SB[bc[2]:bc[3]], SB[bc[3]:bc[4]], SB[bc[4]:bc[5]]   # M10: views
        S = dict(zip(BSING, xp.unstack(sing)))
        ov, k3, k4 = S["ov"], S["k3"], S["k4a"] + S["k4b"]; S["_m10_k4"] = k4   # M10: shared with the PK4 single-site read
        # coefficient buffer: rows 0..3 = cross^k / k!, then the coefficient matrices (zero or irrelevant diagonals) times them
        x = V.x; xp.copyto(x, pc); xp.fill_diagonal(x, 0)
        xp.multiply(x, x, out=V.x2); xp.multiply(V.x2, 0.5, out=V.x2)
        xp.multiply(V.x2, x, out=V.x3); xp.true_divide(V.x3, 3, out=V.x3)
        if L:   # DEAD_DIAG: past layer 0 gram = diag(gv), so gm * gram = diag(ga gv): added on the diagonal only (same values)
            self.q224 = q224 = xp.einsum("ki,kj->ij", oqc, oqr); xp.fill_diagonal(q224, _diag(q224) + ga * gv)
        else: self.q224 = q224 = xp.einsum("ki,kj->ij", oqc, oqr) + gm * gram                                   # q22 / 4
        xp.copyto(V.qq4, q224); xp.fill_diagonal(V.qq4, 0)
        xp.multiply(b, 0.5, out=V.bbh); xp.fill_diagonal(V.bbh, 0)                                              # off(b) / 2
        # q31 / 6. DEAD_DIAG: past layer 0 the gah gram part is diagonal-only, and the coefficient-buffer diagonals reach only
        # output diagonals that are overwritten (cov <- ov, post <- k3, u <- k4) or dead (pv3 and Z: trace diagonal <- urow)
        if L: xp.multiply(S["gvh"][:, None], gm, out=V.q316)   # FUSE: a ufunc with out= (einsum out= logs an extra copyto)
        else: xp.add(xp.einsum("i,ij->ij", S["gah"], gram), xp.einsum("i,ij->ij", S["gvh"], gm), out=V.q316)
        if GRAFT_K31C: xp.add(V.q316, xp.einsum("i,ij->ij", z3 / (12.0 * v), V.bbh), out=V.q316)   # q31 += 0.25 (z3 / v) off(b); q316 = q31 / 6, bbh = off(b) / 2
        xp.multiply(V.r4_8_2, x, out=V.r5_9_2); xp.multiply(V.bbhT, V.r0_2, out=V.r8_10)
        xp.multiply(V.q316, x, out=V.r11); xp.multiply(V.q316T, V.r0_2, out=V.r12_14); xp.multiply(V.r6_10_2, V.x2, out=V.r14_16)
        xp.multiply(S["z36"], V.r0_3, out=V.r16_19); xp.multiply(S["q424"], V.r0_2, out=V.r19_21)               # row vectors (0, 3), (0, 4)
        if not MEHLER_SHARE:   # column vectors (3, 0), (4, 0): ROW_FOLD folds them into the left vectors instead
            xp.multiply(S["z36"], V.r0_3T, out=V.r21_24T); xp.multiply(S["q424"], V.r0_2T, out=V.r24_26T)
        path = ["einsum_path", (0, 1, 2)] if xp.__name__ == "numpy" else [(0, 1, 2)]
        if MEHLER_SHARE:   # the shared right family scales the coefficient rows once (NFOLD n^2), then three 2-operand contractions
            Ls, Rs = xp.take(SB, K.il_s, axis=0), xp.take(SB, K.ir_s, axis=0)
            # ROW_FOLD: row (3, 0, k) = z36_i CB_k with right member k (as row k): its term CB_k o ((z36 L_{21+k}) R_k^T) joins row k's
            # left vector; likewise (4, 0, k) with q424. Same terms, 21 coefficient rows instead of 26 (rows 21..25 are not built).
            F = NFOLD; xp.add(Ls[:, 0:3], xp.einsum("pkn,n->pkn", Ls[:, F:F + 3], S["z36"]), out=Ls[:, 0:3])
            xp.add(Ls[:, 0:2], xp.einsum("pkn,n->pkn", Ls[:, F + 3:F + 5], S["q424"]), out=Ls[:, 0:2]); Ls, Rs = Ls[:, :F], Rs[:, :F]
            xp.copyto(Ls[:, 17], Ls[:, 1]); xp.copyto(Ls[:, 20], Ls[:, 1]); xp.copyto(Ls[:, 18], Ls[:, 2])
            self.pcx_coefficient = Ls[1, 1]
            cov, post, pv3 = xp.unstack(xp.einsum("kij,pki->pij", xp.einsum("kij,kj->kij", V.cb_fold, Rs[0]), Ls[:3]))
            # U_ROWSUM: the (2,2) pair power u = M - ov ov^T - 2 cov o cov (diagonal <- k4) is read only through its row sums
            # (the core trace diagonal urow), so the n x n matrix is never formed. M = sum_k CB_k o (g_k h_k^T); its row-0 term
            # (CB_0 = ones) is the large g_0 h_0^T ~ ov ov^T, taken analytically against ov ov^T so that no large row sum cancels:
            # urow = sum_{j != i} [M - ov ov^T - 2 cov o cov]_ij + k4 (cov with its diagonal <- ov)
            g, h, u3, r1 = Ls[3, 0], Rs[1, 0], Ls[3, 1:], Rs[1, 1:]; xp.fill_diagonal(cov, ov); xp.fill_diagonal(post, k3); dg, dh = g - ov, h - ov
            urow = (xp.einsum("kij,ki,kj->i", V.cb_tail, u3, r1) - xp.einsum("kii,ki,ki->i", V.cb_tail, u3, r1) + k4
                    + g * xp.sum(dh) + dg * xp.sum(ov) - (g * h - ov * ov) - 2 * (xp.einsum("ij,ij->i", cov, cov) - ov * ov))
            if (dr := getattr(self, "l1_drop", None)) is not None:   # lane L1 SLICE_UROW: the dropped saturated rows' CB-row 6 / 8 image
                urow = urow + 0.5 * (Ls[3, 8] * dr[0] + Ls[3, 6] * dr[1]); self.l1_drop = None
        else:
            pr = K.pb[first].row
            self.pcx_coefficient = pcx_first_order_coefficient(SB[pr["g1"]], SB[pr["g4"]], SB[pr["g5"]], z3, q4)
            pv = xp.einsum("kij,pki,pkj->pij", self._coef_buf, xp.take(SB, K.il, axis=0), xp.take(SB, K.ir, axis=0), optimize=path)
            cov, post, u, pv3 = xp.unstack(pv)
            xp.subtract(u, xp.outer(ov, ov), out=u); xp.subtract(u, xp.einsum("ij,ij,->ij", cov, cov, K.two), out=u)
            xp.fill_diagonal(u, k4); xp.fill_diagonal(cov, ov); xp.fill_diagonal(post, k3); urow = xp.sum(u, axis=1)
        SC = self.update_core(pc, x, B, MV, SB, S, (key, lc, lr), d, v, gm, gram, gv, ga, rt, urow, pv3, cov)
        return m, cov, post, S, SC
    def update_core(self, c, x, B, MV, SB, S, blocks, d, v, gm, gram, gv, ga, rt, urow, pv3, cov):
        """Next core T' = trace / (n + 4), s' = -tr(trace) / (2 (n + 2) (n + 4)) from this layer's mixed fourth trace: radial
        zz and M dd, inherited W^T A W terms, repeated kappa3/kappa4 image subtractions, the (3,1) pair power and the merged
        edge products. trace = Z + Z^T with its diagonal replaced by the (2,2) row sums; Z is accumulated term by term.
        Returns the 1-D program-C rows (CTAIL)."""
        K, n, V, first, (key, lc, lr) = self.K, len(v), self.V, int(self.layer == 0), blocks
        a1, a2, ga1, ga2 = S["a1"], S["a2"], S["ga1"], S["ga2"]
        sr, vv, ca = radial_parts(c, ga1, rt)
        self._e_rad = (vv, ca, sr, ga1)   # E_TU: the radial image of the cubic trace, for the bank's exact-minus-radial read
        edge = self.layer < (bank.TCP_START if EDGE_END is None else EDGE_END)   # C_noedge_ge4: skip off-diagonal edge block H once TC_SLICE_FOLD folds theta Q31 into b
        if edge:
            po = self.gate_plans.plan(TF_OUT); pk = bank._plan(xp.max(xp.abs(key), axis=0), TF_K, n, self.quantum)
            if TERM_CORE_CROP and (pa := self.vertex_plans.plan(self._gm_taus(self.layer + 1)[0])) is not None:   # every core: only its [a, a] block is read
                po = bank._sub_plan(po, pa[1] >= n - int(pa[0].shape[0]), n, self.quantum)
            left = xp.einsum("kij,ki,kj->ij", V.r2_10_2, lc, lr)   # merged edge product: left over the rows x^2/2, q22/4, b/2, b^T/2
            lt = left + xp.einsum("ij,i,j->ij", c, S["dh"], S["stf"])
            if po is None and pk is None: G = fmm.gen(lt, c)
            else:
                A, Bm = (lt, c) if po is None else (xp.take(lt, po[0], axis=0), xp.take(c, po[0], axis=1))
                if pk is not None: A, Bm = xp.take(A, pk[0], axis=1), xp.take(Bm, pk[0], axis=0)
                G = fmm.matmul(A, Bm, self.levels)
        else:
            po = pk = left = G = None
        q2, bg, qg = V.x2 @ ga2, V.bbh @ ga1, V.qq4 @ ga2
        singles = [q2 + bg + qg, c @ a1, gm @ a1, gv * a1 if self.layer else gram @ a1, x @ ga1, q2, bg, V.bbhT @ ga2, qg, vv, ca, v, gv, ga,
                   K.ones if pk is None else _mask(pk, n, True)]   # w3 first
        scal = xp.einsum("s,ds->d", xp.stack([sr, xp.einsum("i,i->", a2, ga), xp.einsum("i,i->", a2, gv)]), K.dynm)
        SC = K.pc[first].run(xp.vstack([SB, B, MV] + singles + [K.ones1]), xp.concatenate([K.one, scal]))   # M10: one vstack
        cc_ = K.ccuts; cols, rows, cc, oc, orr, tail = SC[:cc_[0]], SC[cc_[0]:cc_[1]], SC[cc_[1]:cc_[2]], SC[cc_[2]:cc_[3]], SC[cc_[3]:cc_[4]], SC[cc_[4]:]   # M10: views
        T = dict(zip(CTAIL, xp.unstack(tail)))
        eff = T["gmc"] + S["gvh"] * SB[K.pb[first].row["m7"]]
        self._e_kappa = xp.einsum("i,i->", eff, d) / xp.einsum("i,i->", d, d)   # E_PROP: core propagation (gmc + gvh m7) ~ kappa d
        Z = xp.einsum("kij,ki,kj->ij", V.r0_11, cols, rows)
        if self.layer:   # FUSE (lane5): the six matrix o (column x row) terms as ONE stacked einsum (same terms, reassociated sum)
            M6 = xp.stack([c, gm, x, cov, V.ones, V.ones])
            U6 = xp.stack([xp.sum(cc, axis=0), T["gmc"], T["xc"], S["ov3"], oc[0], oc[1]])
            xp.add(Z, xp.einsum("kij,ki,kj->ij", M6, U6, xp.stack([d, d, T["xr"], K.mones, orr[0], orr[1]])), out=Z)
            xp.add(Z, pv3, out=Z)
        else:
            xp.add(Z, xp.einsum("ij,ki,j->ij", c, cc, d), out=Z)
            xp.add(Z, xp.einsum("ij,i,j->ij", gm, T["gmc"], d), out=Z)
            xp.add(Z, xp.einsum("ij,i,j->ij", gram, T["grc"], d), out=Z)   # DEAD_DIAG past layer 0 (gram diagonal)
            xp.add(Z, xp.einsum("ij,i,j->ij", x, T["xc"], T["xr"]), out=Z)
            xp.add(Z, xp.einsum("ki,kj->ij", oc, orr), out=Z)
            xp.add(Z, pv3, out=Z); xp.subtract(Z, xp.einsum("i,ij->ij", S["ov3"], cov), out=Z)
        if edge:
            # edge block H = (G - left v^T) d^T on the kept rows / columns
            if po is None: H = xp.einsum("ij,j->ij", G - xp.einsum("ij,j->ij", left, T["hv"]), T["hd"])
            else:
                lo = xp.take(xp.take(left, po[0], axis=0), po[0], axis=1)
                H = bank._scatter(xp.einsum("ij,j->ij", G - xp.einsum("ij,j->ij", lo, xp.take(T["hv"], po[0])), xp.take(T["hd"], po[0])), po[1], (0, 1), po[0])
            xp.add(Z, H, out=Z); trace = xp.add(Z, Z.T, out=H); xp.fill_diagonal(trace, urow)
        else:
            trace = Z + Z.T; xp.fill_diagonal(trace, urow)
        coefficient = K.core_coef   # FUSE: per-prediction constant -1 / (2 (n+2)(n+4)) (float64), made once in _Tables
        self.core_s = _scalar(xp.multiply(xp.trace(trace), coefficient), trace.dtype)   # M10: no redundant inner cast
        self.core_t = xp.true_divide(trace, K.n4, out=trace)
        return T
    # ---- PK4 (research; only reached when a PK4 switch or PK4_RECORD is set) ---------------------------------------------
    def _pk4_init(self, n, depth):
        rec = PK4_RECORD is not None
        if not (PK4_SINGLE or PK4_LEGS or rec): return None
        cs = "exact" if PK4_CORESUB is True else PK4_CORESUB
        if cs not in (False, "lite", "exact"): raise ValueError("PK4_CORESUB must be False, 'lite' or 'exact'")
        self._pk4_cfg = cfg = dict(single=PK4_SINGLE or rec, legs=PK4_LEGS or rec, fams=tuple(PK4_FAMS), coresub=cs or ("exact" if rec else False),
                                   ages=tuple(PK4_AGES), term_ages=tuple(PK4_TERM_AGES), term=PK4_TERM or rec, htau=PK4_HTAU, ctau=PK4_CTAU, ptau=PK4_PTAU,
                                   record=rec)
        return pk4.PK4(self, n, depth, self.quantum, cfg)
    def _star_dq(self, t, terminal):
        """Chassis Gaussian-star dq (bank.GaussianStars) restricted to the ages PK4 does not own (None if nothing is left)."""
        st, bk = self.bank.stars, self.bank; ages = bank.STAR_TERM if terminal else bank.STAR_AGES
        own = (PK4_TERM_AGES if terminal else PK4_AGES) if PK4_LEGS and (PK4_TERM or not terminal) and "star" in PK4_FAMS else None
        if not terminal and not (bank.STAR_START <= t <= bk.depth - 2): return None
        lo, hi = max(0, t - ages[1]), min(len(st.births), t - ages[0] + 1)
        runs = [(lo, hi)] if own is None else [(lo, min(hi, t - own[1])), (max(lo, t - own[0] + 1), hi)]
        runs = [(a, b) for a, b in runs if b > a]
        if not terminal and own is None: return st.consumer(bk, t)
        if terminal and own is None: return st.terminal(bk, t)
        if not runs: return None   # (terminal: the caller adds nothing; was an explicit zero vector)
        if terminal:
            plan = bk.term_plan; mt = bk.n if plan is None else int(plan[0].shape[0]); L = bk.LEG2[-1]
            out = sum(st._part(L[2 * a:2 * b:2, :, :mt], L[2 * a + 1:2 * b:2, :, :mt], a, b) for a, b in runs)
            return out if plan is None else bank._scatter(out, plan[1])
        L = bk.LEG2[0]
        return sum(st._part(L[2 * a:2 * b:2], L[2 * a + 1:2 * b:2], a, b) for a, b in runs)
    def _pk4_parts(self, t, w, ww, gv, terminal):
        c, n, parts = self._pk4_cfg, len(gv), {}
        if c["single"] and t >= 1:
            parts["single"] = pk4.single_site(self._pk4_k4h, _diag(self.core_t) + self.core_s, w, ww, gv, n)
        if c["legs"] and t >= 1 and (not terminal or c["term"]): parts.update(self.pk4.consumer(self.bank, t, terminal))
        return parts
    def _pk4_apply(self, t, parts, dq):
        """dq (chassis star, possibly restricted) plus the switched-on PK4 components; records everything if asked."""
        if PK4_RECORD is not None: PK4_RECORD.setdefault(t, {}).update({k: v for k, v in parts.items()})
        use = {k: v for k, v in parts.items() if (PK4_TERM or t < self.bank.depth - 1) and ((k == "single" and PK4_SINGLE)
               or ("@" in k and not k.startswith("x:") and PK4_LEGS and (PK4_CORESUB or "_img@" not in k)))}
        add = pk4.total(use) if use else None
        if add is None: return dq
        add = add.astype(xp.float32)
        return add if dq is None else dq + add
    def _star_owned(self):
        """True when PK4 replaces the chassis Gaussian star at every hidden and terminal age (its births are then never read)."""
        own = lambda ages, mine: mine[0] <= ages[0] and ages[1] <= mine[1]
        return (self.pk4 is not None and PK4_RECORD is None and PK4_LEGS and PK4_TERM and "star" in PK4_FAMS
                and own(bank.STAR_AGES, PK4_AGES) and own(bank.STAR_TERM, PK4_TERM_AGES))
    def _pk4_fused(self, t, w, ww, gv, terminal):
        """Production PK4 (pk4.fused): single site + every elementwise read minus its image as ONE vector, plus the chassis
        star restricted to the ages PK4 does not own. Same terms as _pk4_parts / _pk4_apply (summation order differs)."""
        dq = self._star_dq(t, terminal)
        if t == 0 or (terminal and not PK4_TERM): return dq
        P, add = self.pk4, None
        if PK4_SINGLE: add = P.fsingle(self._pk4_k4h, self.core_t, self.core_s, ww, gv)
        if PK4_LEGS and (legs := P.fconsumer(self.bank, t, terminal)) is not None: add = legs if add is None else add + legs
        return add if dq is None else dq if add is None else dq + add
    def _pk4_hidden(self, t, w, ww, gv):
        if self.pk4.fused and PK4_RECORD is None: return self._pk4_fused(t, w, ww, gv, False)
        dq = self._star_dq(t, False)
        if PK4_RECORD is not None: PK4_RECORD.setdefault(t, {})["star_chassis"] = self.bank.stars.consumer(self.bank, t)
        if t == 0: return dq
        return self._pk4_apply(t, self._pk4_parts(t, w, ww, gv, False), dq)
    def _pk4_terminal(self, t, w, gv):
        if self.pk4.fused and PK4_RECORD is None: return self._pk4_fused(t, w, w * w, gv, True)
        dq = self._star_dq(t, True)
        if PK4_RECORD is not None: PK4_RECORD.setdefault(t, {})["star_chassis"] = self.bank.stars.terminal(self.bank, t)
        return self._pk4_apply(t, self._pk4_parts(t, w, w * w, gv, True), dq)
    def predict_weights(self, weights):
        if GAUGE_POLICY not in ("identity", "unit_prevariance"): raise ValueError("unknown GAUGE_POLICY")
        if len(weights) < 2: raise ValueError("online gauge requires at least two layers")
        return self._predict_gauged(weights)
    def _check_prevariance(self, variance):  # fail before sqrt/division; no clipping, retry, or alternative scale
        if int(xp.count_nonzero(xp.isfinite(variance) & (variance > 0))) != int(variance.shape[0]):
            raise bank.DegenerateMoments("online gauge requires finite positive prevariance/scale")
    def _check_finite(self, array):
        if int(xp.count_nonzero(xp.isfinite(array))) != int(array.size):
            raise bank.DegenerateMoments("online gauge produced nonfinite effective weight/mean")
    def _prepare_gauge_frame(self, original, previous_scale, mean, cov, ov, first=False):
        """Choose this receiving frame before bank transport; return immutable effective W, mean, PC and output scale.
        Identity is a schedule-only control. Unit mode normalizes raw physical prevariance and rebuilds the live gate cache."""
        w = original if previous_scale is None else original / previous_scale[:, None]
        if GAUGE_POLICY == "unit_prevariance": self._check_finite(w)
        mu = mean @ w
        if GAUGE_POLICY == "unit_prevariance": self._check_finite(mu)
        pc = fmm.sym(w.T, w) if first else self.thin_pc(w, None, cov, ov, mu, check_variance=GAUGE_POLICY == "unit_prevariance")
        if GAUGE_POLICY == "identity": return w, mu, pc, None
        # opsfrugal: diag(pc) of thin_pc IS the vd it checked (fill_diagonal), so only the first frame re-checks it; for a
        # finite positive float32 variance v, scale = 1/sqrt(v) lies in [5e-20, 3e22] and diag((pc s) s) = (v s) s ~ 1, so
        # the two later prevariance checks can never fire once this one passed (same fallback behaviour, fewer calls)
        variance = _diag(pc)
        if first: self._check_prevariance(variance)
        scale = 1.0 / xp.sqrt(variance)
        w, mu = w * scale, mu * scale   # M10: trailing-axis broadcast without a [None, :] view
        self._check_finite(w); self._check_finite(mu)
        pc = xp.einsum("ij,i->ij", pc, scale) * scale   # M10: same products, no broadcast views
        self._gate_cache = None; self.live_gate(mu, pc)
        return w, mu, pc, scale
    def _predict_gauged(self, weights):  # weights: sequence of (n, n) float32 layers -> the (depth, n) layer means
        originals, weights = list(weights), list(weights)
        fmm.configure(); _M10_AR.clear(); self.levels = fmm.levels_for(weights[0], weights[0]); self.quantum = min(int(weights[0].shape[0]), fmm.QUANTUM)
        self.bank = bank.Bank(self, weights, self.levels)
        stars, (depth, n), self._gate_cache = self.bank.stars, (len(weights), int(weights[0].shape[0])), None
        K = self.K = _Tables(n, xp.float32, self.bank._dim); self.forecast_gates = [None] * depth   # corrected gate of every layer
        self._thr_cache = {}; self.pk4 = self._pk4_init(n, depth); star_owned = self._star_owned()
        # incoming repeated slice and its diagonal per layer (set by the bank deposit; zero at the first layer)
        self.repeated_bank = [xp.zeros((n, n), dtype=self.dtype)] + [None] * (depth - 2) + [xp.zeros(n, dtype=self.dtype)]
        self.third_bank = [xp.zeros(n, dtype=self.dtype)] + [None] * (depth - 1)
        self.core_t, mean, cov, ov, rows = xp.zeros((n, n), dtype=self.dtype), xp.zeros(n, dtype=self.dtype), None, None, []
        self.cubic_vector = xp.zeros(n, dtype=self.dtype)
        self._coef_buf = xp.empty((len(ROWS), n, n), dtype=self.dtype)
        self.V = _Views(self._coef_buf)   # views made once
        xp.copyto(self.V.ones, 1.0)   # row 0: cross^0
        w0, mu0, pc0, scale0 = self._prepare_gauge_frame(originals[0], None, mean, cov, ov, first=True)
        weights[0], prepared = w0, (mu0, pc0, scale0)
        for layer in range(depth):
            self.layer_coverage.append(layer)
            self.layer, w = layer, weights[layer]
            pre_vector = xp.matmul(self.cubic_vector, w)
            if layer == depth - 1:  # terminal: mean only, with the star history's kappa4 diagonal dq
                mu = mean @ w
                if GAUGE_POLICY == "unit_prevariance": self._check_finite(mu)
                gv = xp.einsum("aj,aj->j", w, w); ga = self.terminal_ga(w, mu, self._term_v, gv); sd = xp.sqrt(self._term_v)
                if (dga := self.bank.e_dq.pop(layer, None)) is not None: ga = ga + dga   # E_GA (terminal)
                z = mu / sd; ds = self.basis(sd, z, self.cdf(z), K.tC, K.texp, K.tk)   # rows ds0, ds1, ds2, ... (M10: row views)
                dq, z3t = stars.terminal(self.bank, layer) if self.pk4 is None else self._pk4_terminal(layer, w, gv), self.repeated_bank[layer]
                z3t = z3t + 3.0 * pre_vector * self._term_v
                q4 = xp.einsum("i,i,->i", ga, gv, K.six)
                if dq is not None: q4 = q4 + dq
                if bank.TERM_TC and TERM_TC_DQ and (tcq := self.bank.tc_terminal()) is not None: q4 = q4 + tcq   # TERM_TC: terminal TC dq
                if bank.TERM_U16:  # outside U16 (and outside the transported plan, where z3t = 0): the Gaussian value
                    q4 = q4 * self._u16
                    if LOSSLESS_DIAG: DIAG["u16_keep"], DIAG["u16_z"] = self._u16, z
                zz = z * z; w16 = ((zz - 6.0) * zz + 3.0) * ds[2] / (self._term_v * self._term_v)
                rows.append(ds[0] + z3t * ds[3] / 6 + q4 * ds[4] / 24 + (z3t * z3t / 72.0) * w16); break
            mu, pc, scale = prepared
            ww = xp.multiply(w, w) if layer else None; gvw = xp.sum(ww, axis=0) if layer else None
            self.bank.deposit(layer, mu, pc, gvw)
            rep_bank = self.repeated_bank[layer]   # LANE F: the bank slice before the covariance-star image joins it
            if layer:
                self.repeated_bank[layer] = self.repeated_bank[layer] + covariance_star_repeated(pre_vector, pc)
                self.third_bank[layer] = self.third_bank[layer] + 3.0 * pre_vector * _diag(pc)
                self.bank.trace_override[layer] = self.bank.trace_override[layer] + covariance_star_trace(pre_vector, pc)
            tc = self.bank.tc_pending()
            dq = stars.consumer(self.bank, layer) if self.pk4 is None else self._pk4_hidden(layer, w, ww, gvw)
            ex = dq if tc is None else tc if dq is None else tc + dq   # float32 already: no astype copy
            self.tc_extra = ex if ex is None or str(ex.dtype) == "float32" else ex.astype(xp.float32)   # M10: by name on both runtimes
            repeated, third = self.repeated_bank[layer], self.third_bank[layer]
            mean, cov, post, S, T = self.extended(mu, pc, third, repeated, w, ww, gvw); rows.append(mean if scale is None else mean / scale); ov = S["ov"]
            if self.pk4 is not None and self._pk4_cfg["single"]: self._pk4_k4h = S["_m10_k4"]   # kappa4(h) of this layer (M10: formed once)
            d, e, f = self.vert; gm = self.input_core[0]   # corrected gate / density / third vertex: the transport gate is d
            # fresh kappa3 source: u = C d, v = e u + extra (incoming b, C^2 and Q22 C coupled into the cubic source); the
            # completed square H = sqrt(e) u + F, F = extra / (2 sqrt e); its rank-one part goes to the negative bank (L, G)
            # FUSE (lane5): the four (matrix o column x row) terms of `extra` as one stacked einsum (reassociated sum)
            extra = xp.einsum("kij,ki,kj->ij", xp.stack([repeated, repeated.T, pc * pc, self.q224]), xp.stack([f, e, f, S["fv2"]]),
                              xp.stack([d, e, e, e]))
            sq = xp.sqrt(e); pos = sq > 0; denom = xp.where(pos, 2 * sq, 1.0); ff = extra / denom[:, None]
            fdiag = xp.where(pos, xp.einsum("ii,->i", extra, K.half), 0.0); hh = xp.add(xp.einsum("ij,i,j->ij", pc, sq, d), ff, out=extra)
            hh, fdiag = pcx_self_entry(hh, fdiag, self.pcx_coefficient, sq, pos, denom)
            left, s, right = bank._svd_topk(ff, 1); left = left * s   # (n, 1), (1, n): L = U s
            lr, ll, rr = xp.einsum("ia,ai->i", left, right), xp.einsum("ia,ia->i", left, left), right + right
            # star (a, M) = (e diag(W^T W) / 2, d (W^T A W) d); repair = pair power minus the repeated images of the incoming
            # slice and completed square and (added back) its rank-one part; then compensate the covariance-star replacement.
            star_a, star_m = S["astar"], xp.einsum("ij,i,j->ij", gm, d, d)
            r = xp.subtract(post, xp.einsum("ij,i,j->ij", repeated, S["ddv"], d), out=post)
            dh = _diag(hh); xp.subtract(r, xp.einsum("i,ij->ij", dh + dh, hh), out=r)
            xp.subtract(r, xp.einsum("ji,ji->ij", hh, hh), out=r)
            xp.add(r, xp.einsum("i,ai,bj->ij", ll, right, rr), out=r); xp.add(r, xp.outer(right * right, ll), out=r)
            beta = T["beta0"] + 3 * (lr * lr - dh * dh + star_a * S["dmv"])
            self.cubic_vector, star_delta = covariance_star_birth(star_a, star_m, cov, pre_vector, pc, d)
            xp.subtract(r, star_delta, out=r); beta = beta - _diag(star_delta)
            xp.fill_diagonal(r, 0)
            fresh = dict(u=hh, d=d, e=e, r=r, beta=beta, star=(star_a, star_m), _negL=left, _negG=right, _tc_fdiag=fdiag - sq * lr, _dh=dh)
            self.repeated_bank[layer] = None
            # Save the CURRENT source cache before next-PC preparation replaces it; current vertex plans remain live.
            if bank.TCP_START <= layer + 1 <= depth - (1 if bank.TERM_TC else 2): self.bank.tc_stash_engine(self._gate_cache[5], d, e, S["a1tc"], rep_bank)   # LANE F: star-free TC image slice
            if layer == depth - 2:  # terminal D=I: preserve the native diagonal-only covariance shortcut
                self.V = None
                self._coef_buf = None
                wn = originals[layer + 1] if scale is None else originals[layer + 1] / scale[:, None]
                if GAUGE_POLICY == "unit_prevariance": self._check_finite(wn)
                weights[layer + 1] = wn
                self._term_v = self.terminal_variance(wn, cov, ov, layer)
                if GAUGE_POLICY == "unit_prevariance": self._check_prevariance(self._term_v)
                terminal_mu = mean @ wn
                if GAUGE_POLICY == "unit_prevariance": self._check_finite(terminal_mu)
                self.terminal_z = terminal_mu / xp.sqrt(self._term_v)
            else:
                wn, next_mu, next_pc, next_scale = self._prepare_gauge_frame(originals[layer + 1], scale, mean, cov, ov)
                weights[layer + 1], prepared = wn, (next_mu, next_pc, next_scale)
            if bank.E_TU and layer:   # E_TU: layer-l context (u = ga1, star vector, pc, radial image, next prevariance, Var h)
                self.bank.e_ctx = dict(u=self._e_rad[3], pv=pre_vector, pc=pc, rad=self._e_rad, ov=ov, rho=2.0 * self._e_kappa / K.n4, g=self.forecast_gates[layer],
                                       vnext=self._term_v if layer == depth - 2 else _diag(prepared[1]))
            self.bank.accumulate(fresh, layer, weights)
            if self.pk4 is not None: self.pk4.birth(layer, fresh, mean, d, e, f, pc, self.bank.nrow[-1], self.bank.nvrow[-1], ov)
            if layer + 2 <= depth - 1 and not star_owned: stars.birth(fresh, f, self.bank.nrow[-1], self.bank.nvrow[-1])  # age >= 2 consumer
        return radius_mean(n).astype(self.dtype) * xp.stack(rows)

def radius_mean(n):
    """Exact E[chi_n/sqrt(n)] through a public metered logarithmic recurrence."""
    first = 1 if n % 2 else 2
    base = 0.797884560802865355879892119868763737 if first == 1 else 1.253314137315500251207882642405522627
    j = xp.arange(first, n - 1, 2, dtype=xp.float64)
    logs = xp.log1p(xp.divide(_scalar(1.0), j))
    product = xp.exp(_scalar(xp.sum(logs)))
    return _scalar(xp.divide(xp.multiply(_scalar(base), product), xp.sqrt(_scalar(n))))


def gaussian_fallback(weights, coverage):
    """Metered first-Hermite Gaussian closure, with deterministic zero variance.

    This is the declared general fallback, not the historical accuracy arm.
    Earlier attempted operations stay charged in the same caller context.
    """
    n, dtype = int(weights[0].shape[0]), weights[0].dtype
    mean, cov, rows = xp.zeros(n, dtype=dtype), xp.eye(n, dtype=dtype), []
    for layer, w in enumerate(weights):
        coverage.append(layer)
        mu = xp.matmul(mean, w); pc = xp.matmul(w.T, xp.matmul(cov, w))
        v = xp.maximum(xp.diag(pc), 0.0); sd = xp.sqrt(v); active = v > 0
        z = mu / xp.where(active, sd, xp.ones_like(sd))
        gate = xp.where(active, ndtr(z), (mu > 0).astype(dtype))
        phi = xp.exp(-0.5 * z * z) / 2.5066282746310002
        mean = xp.where(active, mu * gate + sd * phi, xp.maximum(mu, 0.0)); rows.append(mean)
        var = xp.maximum((mu * mu + v) * gate + mu * sd * phi - mean * mean, gate * gate * v)
        cov = gate[:, None] * pc * gate[None, :]; xp.fill_diagonal(cov, var)
    return xp.stack(rows)


def _solver_failure(exc):
    reasons = ("SVD did not converge", "Eigenvalues did not converge", "Singular matrix")
    kind, message = type(exc).__name__, str(exc)
    if kind == "FlopscopeServerError" and message.startswith("internal server error: LinAlgError:"):
        return message[len("internal server error: LinAlgError:"):].strip() in reasons
    return kind in ("LinAlgError", "FlopscopeServerError", "RuntimeError") and message in reasons


def _budget_exhausted(operation, cost, remaining):
    # Native and official-client 0.12.x expose different constructors for
    # the same public exception. This adapts diagnostic metadata only.
    try:
        return BudgetExhaustedError(operation, flop_cost=cost, flops_remaining=remaining)
    except TypeError:
        return BudgetExhaustedError(f'{operation} would cost {cost} FLOPs but only {remaining} remain')

def _session_flops_used():
    # This PUBLIC API reports cumulative use across recorded contexts. Only
    # differences within one uninterrupted prediction are meaningful here;
    # its absolute remaining allowance is not the active grader allowance.
    used = budget_summary_dict()['flops_used']
    if not isinstance(used, int) or isinstance(used, bool) or used < 0:
        raise ArithmeticError('invalid cumulative FLOP counter')
    return used

class Estimator(BaseEstimator):
    """Inherited tuned ex19 arm, under the caller-owned passed allowance."""
    def setup(self, ctx) -> None:
        fmm.prepare_static_views(int(ctx.width))

    def predict(self, mlp: MLP, budget: int) -> xp.ndarray:
        _M10_AR.clear(); DIAG.clear()
        bank._M10_IV.clear(); bank.RESEARCH_LOG.clear()
        self._layer_coverage = []
        self._fallback_reason = None
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError('budget must be a positive integer')
        mlp.validate()
        weights = list(mlp.weights)
        if any(str(w.dtype) != 'float32' for w in weights):
            raise ValueError('ex19 requires float32 MLP weights')
        try:
            snapshot = current_budget()
        except NoBudgetContextError:
            # A grader-owned server session has no client-local snapshot.
            # The caller still owns and enforces the passed allowance.
            # Participant code must never open, close, or reset that session.
            snapshot = None
        cumulative_start = _session_flops_used() if snapshot is None else None
        n, depth = mlp.width, mlp.depth
        available = budget if snapshot is None else min(budget, snapshot.flops_remaining)
        gaussian_reserve = depth * (4 * n**3 + 64 * n**2 + 512 * n)
        if (snapshot is not None and snapshot.flops_remaining > budget) or available < gaussian_reserve:
            # A mismatched larger caller meter cannot enforce our passed limit.
            # Use a completely bounded numerical circuit; do not weaken the meter.
            # Same route applies when even the inherited Gaussian reserve cannot fit.
            # Float32 tariff: first mean 2*n*n+n, each later matvec/ReLU
            # 2*n*n, stack depth*n, isfinite/all 2*depth*n-1. Reserve one
            # additional FLOP; array views are free, but stack is billed.
            point_mass_bound = 2 * depth * n * n + (3 * depth + 1) * n
            if available < point_mass_bound:
                raise _budget_exhausted('analytic point-mass closure reserve', point_mass_bound, available)
            self._fallback_reason = 'budget_point_mass_closure'
            mean = xp.sqrt(xp.sum(weights[0] * weights[0], axis=0)) / 2.5066282746310002
            rows = [mean]
            self._layer_coverage.append(0)
            for index in range(1, depth):
                mean = xp.maximum(mean @ weights[index], 0.0)
                rows.append(mean)
                self._layer_coverage.append(index)
            raw = xp.stack(rows)
            if not bool(xp.all(xp.isfinite(raw))):
                raise ArithmeticError('nonfinite point-mass closure')
            return raw
        start = None if snapshot is None else snapshot.flops_used
        engine = TraceM()
        self._fallback_reason = None

        def fallback(reason):
            self._fallback_reason = reason
            engine.bank = None
            if cumulative_start is not None:
                # The grader owns a serial, uninterrupted server session.
                # A cumulative-counter difference measures our own spend;
                # prior contexts cancel. The server additionally enforces any
                # precharged or smaller actual active allowance.
                spent = _session_flops_used() - cumulative_start
                if spent < 0:
                    raise ArithmeticError('cumulative FLOP counter decreased during prediction')
                remaining = budget - spent
            else:
                now = current_budget()
                remaining = min(now.flops_remaining, budget - now.flops_used + start)
            n, depth = mlp.width, mlp.depth
            reserve = depth * (4 * n**3 + 64 * n**2 + 512 * n)
            if remaining < reserve:
                raise _budget_exhausted('Gaussian fallback reserve', reserve, remaining)
            engine.layer_coverage.clear()
            return gaussian_fallback(weights, engine.layer_coverage)

        try:
            if mlp.depth == 1:
                engine.layer_coverage.append(0)
                return xp.sqrt(xp.sum(weights[0] * weights[0], axis=0))[None, :] / 2.5066282746310002
            if mlp.width <= 2:
                raw = fallback('low_dimensional_star_projector')
            else:
                try:
                    raw = engine.predict_weights(weights)
                except bank.DegenerateMoments as exc:
                    raw = fallback(str(exc))
                except Exception as exc:
                    if not _solver_failure(exc): raise
                    raw = fallback(str(exc))
            if not bool(xp.all(xp.isfinite(raw))):
                if self._fallback_reason is not None:
                    raise ArithmeticError('Gaussian fallback produced nonfinite result')
                raw = fallback('nonfinite_result')
                if not bool(xp.all(xp.isfinite(raw))):
                    raise ArithmeticError('Gaussian fallback produced nonfinite result')
            # ReLU means are nonnegative; Euclidean projection cannot increase MSE.
            return xp.maximum(raw, 0.0)
        finally:
            self._layer_coverage = list(engine.layer_coverage)
            engine.bank = None
            if getattr(engine, "pk4", None) is not None:
                engine.pk4.engine = None
                engine.pk4 = None
            _M10_AR.clear()
            bank._M10_IV.clear()

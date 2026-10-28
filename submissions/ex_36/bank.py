# Covariance-star carrier port: separate cubic (a,M) transport removed.
# All inherited rank/crop/elder/TC/PK4 choices are retained from P5e6.
"""Third-cumulant history bank (completed square) plus the Gaussian star history, for the estimator engine. A source born at
layer b represents the transported kappa3 tensor 3 sum_i (P_i o H_i o H_i + P_i o P_i o R_i) + beta_i P_i^3 over its kept hub
rows i: P = transported birth direction, H = sqrt(e) u + F (completed square), R = rank-2 birth repair (LEFT fixed, RIGHT
transported). A rank-one negative-square bank -3 sum K G^2 carries the square's cross term; the engine owns a covariance-star vector,
with its repeated image and trace added after deposit. Young sources transport dense P/H legs (gated, COL-thinned); elders switch at age 5/6 to a
right basis G (rank n/4 for births < EARLY, later n/8; else 3n/16), rebuilt as Y @ G. accumulate() transports and reads out
the kappa3 diagonal/trace; deposit() delivers the delayed (2,1) slice on the OUT block once the next gate is known and, on TC
layers, builds the old-source query that tc_pending() turns into the scalar kappa4 correction dq. No targets, no fits."""
# Ages2-3 Horner TC: frozen exp724 test562/c1 bank, SHA10da86c3 (see DESIGN.md).
import flopscope.numpy as xp
import fmm

REPAIR_RANK, AGE_SWITCH_MAX_BIRTH = 2, 7                    # repair rank; births > 7 never switch to an elder basis
THIN_TAU, THIN_OUT, ROW_THIN, V_SAT, TERM_KEY = 0.04, 0.015, 0.01, 0.003, 0.01   # COL, OUT, hub-row, saturation, terminal
EARLY, SWITCH_AGE, RECOMP_AGE = 4, (6, 7), 4                # births < 4: switch at age 6 to n/4, recompress to n/8 4 layers on
TCP_START = 4                                       # TC consumer layers (scalar dq into q4); package_d5: 5 (was 8)
# TERM_TC (agents/early_accuracy): the TC consumer schedule also covers the terminal layer. The terminal has no slice deposit to
#   share the TC query with, so tc_terminal builds the same query (Horner form with the TC history chain) on the transported
#   terminal columns whose kappa4 weight |z^2 - 1| exp(-z^2 / 2) of the terminal mean exceeds TERM_TC_TAU (declared structural crop,
#   selected on the 15 selection networks among 0 / 0.2 / 0.3) and adds the scalar dq to the terminal q4.
TERM_TC, TERM_TC_TAU = True, 0.3
STAR_AGES, STAR_START, STAR_TERM = (2, 3), 1, (2, 6)   # Gaussian star history: hidden / terminal consumer ages (ex19_trims: was (2, 8) / (2, 4))

TCH = 3  # fixed selected prior; 0 is the archived-bank control, other values rejected

# ---- research switches (2026-10-06 fresh2e9, arm ex19_outthin); all OFF = the frozen ex19-cheap bank, byte for byte ------
# OUT_THIN: hidden-layer bank transports (young H/P legs, elder G and Y @ G rebuild, star M W) produce only the next-layer
#   columns S = top-gate plan of the NEXT layer's Gaussian gate at OUT_THIN_TAU (+ OUT_THIN_PAD quanta of margin for the
#   still-unknown next COL plan); the other columns are exact zeros. S contains the deposit's OUT plan (same argsort,
#   lower threshold). NOT exact: the kappa3 diagonal, the radial trace and the Gaussian-star dq4 read every column.
# SLICE_ROWS: the delivered (2,1) slice b(i, j) keeps only repeated-index rows i in OUT with Gaussian gate in
#   (SLICE_DEAD, 1 - SLICE_SAT) (padded with the most uncertain of the rest); columns stay on OUT. SLICE_ROWS_TC picks the
#   TC-layer form: "rows" restricts the whole Horner query (and so the TC dq rows), "block" only the final slice product.
# STAR_ROWS: the hidden Gaussian-star consumer dq4 only on the columns with gate in (STAR_DEAD, 1 - STAR_SAT).
OUT_THIN, OUT_THIN_TAU, OUT_THIN_PAD, OUT_THIN_CHECK = True, 1e-3, 0, False   # ex19_trims: ON at 1e-3 (research default False, 2e-4)
OUT_THIN_ABL = None   # research ablations without thinning: "diag" zeroes only the delivered kappa3 diagonal off S;
#   "rowshare" drops the non-S rows' share from the radial trace column sums; "trace" also zeroes the trace off S
SLICE_ROWS, SLICE_SAT, SLICE_DEAD, SLICE_ROWS_TC = True, 0.015, 0.0, "rows"   # lane L1: ON (with SLICE_UROW)
# SLICE_UROW (lane L1, 2026-10-17): the dropped saturated rows are NOT irrelevant: the (2,2) pair-power row sums urow (the
#   diagonal of the next core trace, hence the next kappa4 diagonal) read them with O(1) weights (g'' = 2 (d - m e) ~ 2 on a
#   saturated row, and the saturated row's own g'' x g' read). The deposit delivers the two linear functionals urow needs from
#   the dropped rows (factored form, O(sources x hubs x n)); estimator.extended adds their CB-row 6 / 8 image to urow.
# SLICE_FOLD_ROWS (line I): keep the HIST_FOLD of the TC history chain on the row plan (rank-one right factor on OUT).
# SLICE_UROW = "fill" (default): the dropped rows of the delivered slice are filled with the minimum-norm rank-2 matrix that
#   reproduces both functionals (wr^T F = a, F wc = c), so every consumer (urow, cov, post / repair, pv3, TC image) reads it;
#   "urow": only urow gets the functionals (estimator l1_drop); False: zero rows (original SLICE_ROWS).
SLICE_UROW, SLICE_FOLD_ROWS = "fill", True
HF_LITE_ROWS = True   # L1: HIST_FOLD on the row plan without the history distinct-completion (LANE F) term: -1.3k calls, deviation +11 % of a 1 % effect
STAR_ROWS, STAR_SAT, STAR_DEAD = False, 2e-4, 2e-4
RESEARCH_LOG = []   # research diagnostics (only appended when a switch is on; OUT_THIN "S" entries only when RESEARCH)
RESEARCH = False    # opsfrugal F7: the production path no longer grows RESEARCH_LOG every layer
# ---- lossless-stack switches (2026-10-07 lossless lane); all False = src_ct (ex19_trims + fmm views B) byte for byte ------
# TERM_U16: terminal-aware crop. The transport into the terminal layer (young legs, elder G / rebuild, star, P tails,
#   repair, negative packets) and the terminal Gaussian-star read use the TERM_KEY plan restricted to U16 = {|alpha| <
#   TERM_U16_ALPHA} (alpha = terminal Gaussian z); estimator.py zeroes q4 on the columns outside U16 and outside that plan,
#   so those means are the Gaussian value s (alpha Phi + phi) with the same (m, s) (kappa3 is already 0 off the plan).
TERM_U16, TERM_U16_ALPHA = False, 3.0
# SCATTER_PUT: plan scatters (undo a top-gate plan) write the kept block into zeros with put_along_axis (billed 1 per element
#   written, cost-model "scatter at a given index") instead of pad + take(inverse) per axis (pad 1 + take 4 per OUTPUT
#   element); same values bit for bit; one more logged call per two-axis scatter.
# HYG_OPS: drop calls that only move data: the star merge Gram / combination act on the (k, n, n) stack directly (no
#   reshape to (k, n^2) and back), the star read operand W[:, OUT]^T is one take of the transposed view (no empty + copyto)
#   and a [None] view (no broadcast_to), put_along_axis indices broadcast through singleton views (no broadcast_to).
SCATTER_PUT, HYG_OPS = True, True

# Approximate newest-TC closure: deliver B + diag(theta) Q31 to all B consumers.
# This is not an exact full-F8 rewrite. The physical third diagonal and radial
# trace stay on their existing independent paths. OFF is an offline control.
TC_SLICE_FOLD = True
# HIST_FOLD (lane 6): the newest-TC (3,1) slice fold extended to the TC history chain (TCs born at t-2, t-3 read at t), fused
#   into the deposit's Horner families (no added products); the fold's (Zp, Qt) also give the history dq in tc_pending.
HIST_FOLD = True
FOLD_DIAGNOSTIC = None  # offline harness hook; no extra xp work when None
# E_TU (lane E, 2026-10-14 moonshot wave 2): the chassis core carries the (2,1,1) contraction T(u)_ab = sum_c kappa3(z)_abc u_c
#   (u = ga1, the variance response) only through its radial image Sym(v, pc)(., ., u); premap carries it exactly. Deliver the
#   next-layer kappa4-diagonal read of (exact - radial) for EVERY birth from the legs already transported: O(n * hubs) per birth.
#   E_RADIAL: 'exact' (radial (v.u) pc term read through one pc W~ product), 'mehler1' (that term from the next prevariance,
#   first Mehler order), 'none' (no radial subtraction, 059 convention).
E_TU, E_RADIAL = True, 'mehler1'
# E_PROP: propagate the delivered T(u) image through the core as hub-coefficient chains X^m = x^m + rho_m X^{m-1}
E_PROP = True
SB_DENSE_REPAIR = True
SB, SB_AGE, _SB_TAU2 = True, None, 0.00390625
_SB_COND = 2896.309375740099
# ---- lane L2 exact changes (2026-10-17; rounding-level / bit-identical, verified at n = 512 and 1024) ----------------------
# SB_E1: SB-T switch products re-associated. First entrant: Y = U_d S_d from the SVD it already computes (was M V_d, one
#   (2 nr x ncol) @ (ncol x d) plain product). Later entrants: Y = M (Q R^-T), the R^-T solve on the d x ncol side (was
#   (M Q) R^-T with 2 nr right-hand sides). Same math. Measured -2.48 GF on net 32, MSE x0.9999.
# SB_E2: the elder rebuild Y @ G[:, S] reads the S columns of the basis straight from its transport product (bit-identical to
#   the xp.take it replaces, one per equal-nrow run) and shares one B-side Strassen transform across the runs (per-layer
#   b_key). Bit-identical values; fewer calls and B transforms.
SB_E1, SB_E2 = True, True


def _key_plan(key, count, n, q):
    """Top-key plan (idx, inv) keeping max(q, pad(count)) largest keys as the suffix of argsort(key); None = keep all."""
    if (mp := min(n, max(q, _pad(count, q)))) == n: return None
    order = xp.argsort(key); return order[n - mp:], xp.argsort(order)


def _l1_put(n, S, v, dtype):  # lane L1: zeros (n,) with v at the indices S
    out = xp.zeros(n, dtype=dtype); xp.put_along_axis(out, S, v, axis=0); return out

def _l1_fill(a, c, wr, wc):
    """Return a complete finite rank-two fill, or select the original full-row route.

    These are raw factored-slice functionals with Gaussian response weights.
    Diagonal replacement and the separate TC scalar remain approximations.
    No state is consumed here; every attempted operation remains metered.
    """
    if not bool(xp.all(xp.isfinite(xp.stack([a, c, wr, wc])))): return None
    nr2, nc2 = xp.einsum("i,i->", wr, wr), xp.einsum("i,i->", wc, wc)
    norms = xp.stack([nr2, nc2])
    if not bool(xp.all(xp.isfinite(norms) & (norms > 0))): return None
    cr = c - wr * (xp.einsum("i,i->", a, wc) / nr2)
    factors = xp.stack([wr, cr, a / nr2, wc / nc2])
    if not bool(xp.all(xp.isfinite(factors))): return None
    fill = xp.matmul(factors[:2].T, factors[2:])
    return fill if bool(xp.all(xp.isfinite(fill))) else None


def _scatter2(values, inv_r, inv_c):  # undo a row plan (inv_r) and a column plan (inv_c); None = that axis is complete
    n = int((inv_r if inv_r is not None else inv_c).shape[0])
    full = xp.pad(values, [[0 if inv_r is None else n - int(values.shape[0]), 0], [0 if inv_c is None else n - int(values.shape[1]), 0]])
    if inv_r is not None: full = xp.take(full, inv_r, axis=0)
    if inv_c is not None: full = xp.take(full, inv_c, axis=1)
    return full


class DegenerateMoments(ArithmeticError):
    """An explicitly unsupported closure state, before unsafe divisions."""


def _scalar(value, dtype=xp.float64):
    return xp.asarray(value, dtype=dtype)


def _ratio(numerator, denominator, dtype):
    return _scalar(xp.divide(_scalar(numerator), _scalar(denominator)), dtype)


def _dimension_constants(n):
    """Numerical dimension expressions, evaluated in metered float64.

    Callers own this table for one prediction; Python only supplies the shape n
    and literal offsets/indices. In particular, denominator products are not
    precomputed by the interpreter.
    """
    base = xp.add(_scalar(n), xp.asarray([4.0, -2.0, -1.0, 2.0], dtype=xp.float64))
    products = xp.multiply(xp.take(base, [1, 2, 3]), xp.take(base, [2, 0, 0]))
    twice = xp.multiply(xp.concatenate([base[0:1], products]), 2.0)
    def item(v, i): return xp.squeeze(v[i:i + 1])
    return {"n+4": item(base, 0), "n-2": item(base, 1), "n-1": item(base, 2), "n+2": item(base, 3),
            "(n-2)(n-1)": item(products, 0), "(n-1)(n+4)": item(products, 1), "(n+2)(n+4)": item(products, 2),
            "2(n+4)": item(twice, 0), "2(n-2)(n-1)": item(twice, 1),
            "2(n-1)(n+4)": item(twice, 2), "2(n+2)(n+4)": item(twice, 3)}


def _equal_groups(lo, hi, counts):  # consecutive half-open runs [first, stop) of sources sharing one value of counts
    out, first = [], lo
    for stop in range(lo + 1, hi + 1):
        if stop == hi or counts[stop] != counts[first]: out.append((first, stop, counts[first])); first = stop
    return out

def _pad(count, q): return -(-int(count) // q) * q

def _plan(gate, tau, n, q):
    """Top-gate plan (idx, inverse, order): idx = the max(q, pad(#gate > tau)) largest gates as the suffix of order =
    argsort(gate); None when everything is kept. Shared by every bank and engine thinning plan."""
    if (mp := min(n, max(q, _pad(xp.count_nonzero(gate > tau), q)))) == n: return None
    order = xp.argsort(gate); return order[n - mp:], xp.argsort(order), order

def _sub_plan(plan, allowed, n, q):
    """The columns of `plan` (None = all) that are also `allowed` (bool), as a _plan padded first with the other columns
    of `plan` (whose reads are then the uncropped ones), then with arbitrary others."""
    if plan is None: return _plan(allowed.astype(xp.float32), 0.5, n, q)
    inside = plan[1] >= n - int(plan[0].shape[0])
    return _plan((allowed & inside).astype(xp.float32) + inside.astype(xp.float32), 1.5, n, q)

class GatePlans:
    """The _plan of ONE gate vector at several thresholds: one threshold count for all of them and one argsort pair (made
    when the first thinned plan is asked for). Same comparisons and the same order as _plan, so the plans are identical.
    extra: further thresholds whose raw counts #(gate > t) ride on the same comparison and host sync (count()); cache: a
    per-prediction dict holding the uploaded threshold column per (thresholds, dtype) (one upload instead of one per gate)."""
    def __init__(self, gate, taus, n, q, extra=(), cache=None):
        self.gate, self.taus, self.n, self.q, self.order, self.mp = gate, taus, n, q, None, None
        self.extra, self.cache, self.counts = tuple(extra), cache, None
    def count(self, t):
        if self.counts is None: self._count()
        return self.counts[t]
    def _count(self):
        ts = tuple(self.taus) + self.extra; key = (ts, str(self.gate.dtype))
        thresholds = None if self.cache is None else self.cache.get(key)
        if thresholds is None:
            thresholds = xp.asarray([[t] for t in ts], dtype=self.gate.dtype)
            if self.cache is not None: self.cache[key] = thresholds
        self.counts = dict(zip(ts, xp.sum(self.gate > thresholds, axis=1).tolist()))
    def plan(self, tau):
        if self.mp is None:
            if self.counts is None: self._count()
            self.mp = {t: min(self.n, max(self.q, _pad(self.counts[t], self.q))) for t in self.taus}
        if (mp := self.mp[tau]) == self.n: return None
        if self.order is None: self.order = xp.argsort(self.gate); self.inv = xp.argsort(self.order); self.idx = {}
        if mp not in self.idx: self.idx[mp] = self.order[self.n - mp:]
        return self.idx[mp], self.inv, self.order

_M10_IV = {}   # M10: broadcast index views of plan indices for put_along_axis, per prediction (cleared by Bank())


def _m10_iview(idx, a, nd):
    if nd == 1: return idx
    hit = _M10_IV.get(key := (id(idx), a, nd))
    if hit is None or hit[0] is not idx: hit = _M10_IV[key] = (idx, idx[tuple(slice(None) if d == a else None for d in range(nd))])
    return hit[1]

def _scatter(values, inv, axes=(0,), idx=None):  # undo a plan along `axes`: values fill the trailing slots of zeros, inv restores order
    n = int(inv.shape[0])
    if SCATTER_PUT and idx is not None:   # idx = the plan's kept indices (order[n - m:]): full[idx] = values, zeros elsewhere
        full, nd = values, len(values.shape)
        for a in axes:
            z = xp.zeros(tuple(n if d == a else int(x) for d, x in enumerate(full.shape)), dtype=values.dtype)
            xp.put_along_axis(z, _m10_iview(idx, a, nd), full, axis=a); full = z
        return full
    full = xp.pad(values, [[n - int(s), 0] if a in axes else [0, 0] for a, s in enumerate(values.shape)])
    for a in axes: full = xp.take(full, inv, axis=a)
    return full


# M lane (2026-10-13) wall rule KSVD: flopscope's linalg.svd(a, k=...) bills the truncated cost 4 m n k but its server runs
# the FULL economy SVD of a (O(m n min(m, n)) wall whatever k: 0.2-0.4 s per 1024 x 1024 call on x86, ~25 % of the d4
# predict wall on dev ID 40). The small-k sites use instead a metered deterministic block-Krylov Rayleigh-Ritz: start block
# = the b = k + KSVD_P rows of a with the largest norms (a's row space; no sampling), Krylov block [Y0, (a a^T) Y0,
# (a a^T)^2 Y0] (scaled by 1 / |a|_F^2 per power), one QR, the exact SVD of the small projected (3b x n) matrix. Returns
# (u, s, vh) with the shapes of linalg.svd(..., full_matrices=False, k=k); u and vh carry the same joint sign freedom.
# Approximation (Krylov depth 2): structurally tuned constants KSVD_P, KSVD_MAXK; used only when the Krylov block
# 3 b <= min(m, n) / 4 (else the exact flopscope svd).
KSVD, KSVD_P, KSVD_MAXK = True, 2, 8


def _svd_topk(a, k):
    m, n = int(a.shape[0]), int(a.shape[1]); k = int(k); b = k + KSVD_P
    if not KSVD or k > KSVD_MAXK or 12 * b > min(m, n): return xp.linalg.svd(a, full_matrices=False, k=k)
    rn = xp.einsum("ij,ij->i", a, a); idx = xp.argpartition(rn, m - b)[m - b:]   # the b largest row norms (any order)
    at = xp.transpose(a) * (1.0 / (xp.sum(rn) + 1e-30))   # a^T / |a|_F^2, built once (KSVD v2: fewer logged calls)
    y0 = xp.matmul(xp.take(a, idx, axis=0), at).T          # a a_idx^T / |a|_F^2: bounded by the row scale
    y1 = xp.matmul(a, xp.matmul(at, y0)); y2 = xp.matmul(a, xp.matmul(at, y1))   # (a a^T / |a|_F^2)^j y0, norm <= |y0|
    q, _ = xp.linalg.qr(xp.concatenate([y0, y1, y2], axis=1))
    ub, s, vh = xp.linalg.svd(xp.matmul(xp.transpose(q), a), full_matrices=False, k=k)
    return xp.matmul(q, ub), s, vh


def _repair_conservation(matrix, left, right, kept=None, coefficient_cache=None):
    """Restore cropped repair tensor trace/diagonal with ONE analytic factor.

    matrix has m kept hub rows and n natural columns, with zero natural diagonal;
    left @ right is the unchanged inherited rank-2 approximation. Natural hub j is kept[row].
    Return v, beta_delta for repair(ones(m) outer v) + Diag(beta_delta on hubs).
    """
    m = int(matrix.shape[0])
    hubs_right = right if kept is None else xp.take(right, kept, axis=1)
    delta_diag = -xp.einsum("ir,ri->i", left, hubs_right)
    colsum = xp.sum(matrix, axis=0) - xp.matmul(xp.sum(left, axis=0), right)
    key = (m, str(colsum.dtype))
    ratios = None if coefficient_cache is None else coefficient_cache.get(key)
    if ratios is None:
        den = xp.subtract(_scalar(m), xp.asarray([0.0, 1.0 if m > 1 else 0.0], dtype=xp.float64))
        ratios = xp.divide(1.0, den).astype(colsum.dtype)
        if coefficient_cache is not None: coefficient_cache[key] = ratios
    v = colsum * xp.squeeze(ratios[0:1])
    if m == 1:
        hub_v = xp.zeros_like(delta_diag)
    else:
        col_hubs = colsum if kept is None else xp.take(colsum, kept)
        hub_v = (col_hubs - delta_diag) * xp.squeeze(ratios[1:2])
    if kept is None: v = hub_v
    else: xp.put_along_axis(v, kept, hub_v, axis=0)
    return v, 3.0 * (delta_diag - hub_v)


def _repeated_scalar(B, I, Y, Z, rank):
    """TC repeated-index subtraction from the rank-r off-diagonal SVD of B[I, I] (exact diagonal kept, the truncation's
    spurious diagonal removed analytically): the three repeated-index contractions with Y and Z."""
    bi = xp.take(xp.take(B, I, axis=0), I, axis=1); diagonal = xp.copy(xp.diag(bi)); xp.fill_diagonal(bi, 0)
    u, s, vh = _svd_topk(bi, min(max(1, int(rank)), int(bi.shape[0])))   # exactly k columns / rows (M lane KSVD)
    right, ut, y2 = xp.einsum("r,rj->rj", s, vh), u.T, Y * Y; dd = diagonal - 3 * xp.einsum("ir,ri->i", u, right)
    uy2, ry = xp.matmul(ut, y2), xp.matmul(right, Y)
    return xp.sum(uy2 * xp.matmul(right, Z) + 2 * xp.matmul(ut, Z * Y) * ry, axis=0) + xp.einsum("ij,i,ij->j", Z, dd, y2)

def _gate_div(d):  # divisor of a gated query: an exactly-zero gate (clipped corrected gate) gives 0 / inf = 0, not 0 / 0
    return xp.where(d > 0, d, float("inf"))

# LANE F (TC_REP_MODE): the TC repeated-index image of the query. "svd" = the base rank-n/256 off-diagonal SVD; "exact" =
#   the full off-diagonal slice B[I, I] (two products per TC read, as premap's fresh tc source); "none" = no subtraction.
TC_REP_MODE = 'svd'
def _rep_exact(B, I, rank):
    bi = xp.take(xp.take(B, I, axis=0), I, axis=1); diagonal = xp.copy(xp.diag(bi)); xp.fill_diagonal(bi, 0)
    return ("X", bi, diagonal)
def _rep_svd(B, I, rank):
    if TC_REP_MODE == "exact": return _rep_exact(B, I, rank)
    if TC_REP_MODE == "none": return ("0",)
    """TC repeated-index subtraction, SVD part: rank-r off-diagonal SVD of B[I, I] (exact diagonal kept, the truncation's
    spurious diagonal removed analytically); kept for the next consumer's history."""
    bi = xp.take(xp.take(B, I, axis=0), I, axis=1); diagonal = xp.copy(xp.diag(bi)); xp.fill_diagonal(bi, 0)
    u, s, vh = _svd_topk(bi, min(max(1, int(rank)), int(bi.shape[0])))   # exactly k columns / rows (M lane KSVD)
    right = xp.einsum("r,rj->rj", s, vh); return u.T, right, diagonal - 3 * xp.einsum("ir,ri->i", u, right), u   # M10: + u

def _rep_apply(svd, Y, Z):  # the three repeated-index contractions with Y and Z
    if isinstance(svd[0], str): return xp.einsum("ij,ij->j", Z, _rep_residual(svd, Y))
    ut, right, dd = svd[:3]; y2 = Y * Y; uy2, ry = xp.matmul(ut, y2), xp.matmul(right, Y)
    return xp.sum(uy2 * xp.matmul(right, Z) + 2 * xp.matmul(ut, Z * Y) * ry, axis=0) + xp.einsum("ij,i,ij->j", Z, dd, y2)


def _cb_coefficients(quadratic, x):
    """Distinct-index completion of one quadratic query per direction. FUSE (lane5): every column sum of a product is one
    multi-operand einsum (no product temporaries), the diagonal is Horner-factored and the degeneracy tests read one minimum
    (same arithmetic up to float32 rounding of the reassociated sums). Returns (a, e, norm, e2)."""
    norm = xp.sqrt(xp.einsum("ij,ij->j", x, x)); safe = xp.where(norm > 0, norm, 1.0)
    e = x / safe; q = quadratic / (safe * safe); e2 = e * e
    diagonal = e2 * (6 * e2 - 4) + (1 - xp.einsum("ij,ij->j", e2, e2))
    if float(xp.min(xp.abs(diagonal))) < 1e-10: raise DegenerateMoments('degenerate distinct-completion diagonal')
    inverse = 1 / diagonal; first = inverse * q; e2i = e2 * inverse
    g00 = xp.sum(e2i, axis=0); g01 = xp.einsum("ij,ij->j", e2, e2i) - 0.5
    g11 = xp.einsum("ij,ij,ij->j", e2, e2, e2i) - 0.5
    b0 = xp.einsum("ij,ij->j", e, first); b1 = xp.einsum("ij,ij,ij->j", e, e2, first); det = g00 * g11 - g01 * g01
    if float(xp.min(xp.abs(det))) < 1e-10: raise DegenerateMoments('singular distinct-completion normal update')
    l0 = (g11 * b0 - g01 * b1) / det; l1 = (g00 * b1 - g01 * b0) / det
    return first - inverse * (xp.einsum("ij,j->ij", e, l0) + xp.einsum("ij,ij,j->ij", e, e2, l1)), e, safe, e2


def _complete_bilinear(quadratic, x, z):
    """Columnwise completed kappa3(., x, z); only O(rows*directions) work. FUSE (lane5): e2 reused from the coefficients,
    the column sums as multi-operand einsums."""
    a, e, norm, e2 = _cb_coefficients(quadratic, x); ez = e * z
    dot, third = xp.sum(ez, axis=0), xp.einsum("ij,ij->j", e2, ez)
    ae, az, ae2z = xp.einsum("ij,ij->j", a, e), xp.einsum("ij,ij->j", a, z), xp.einsum("ij,ij,ij->j", a, e2, z)
    out = a * ((1 - 2 * e2) * dot - 2 * ez + 6 * e2 * ez - third) + e * ((dot - ez) * ae + (1 - e2) * az - 2 * ae2z)
    return out * norm


def _rep_residual(svd, Y):
    """Repeated-index query R, with sum(Z*R) equal to _rep_apply(svd,Y,Z)."""
    if isinstance(svd[0], str):
        if svd[0] == "0": return xp.zeros_like(Y)
        bi, dg = svd[1], svd[2]; y2 = Y * Y
        return fmm.gen(xp.transpose(bi), y2) + 2 * Y * fmm.gen(bi, Y) + xp.einsum("i,ij->ij", dg, y2)
    ut, right, dd = svd[:3]; y2 = Y * Y
    uy2, ry = xp.matmul(ut, y2), xp.matmul(right, Y)
    return xp.matmul(right.T, uy2) + 2 * Y * xp.matmul(svd[3], ry) + xp.einsum("i,ij->ij", dd, y2)   # M10: u = (u.T).T

class NegativeBank:
    """Rank-one negative-square carrier -3 sum_j K_j G_j^2 (K = L^2 from the completed square's rank-one part). K and G share
    the old-source COL map; the first source carries no packet. Packets are rows of one stacked array kg = [K rows; G rows];
    cache / tc_cache: (a, b) = ([G^2; 2 G K], [K; G]) row stacks for the deposit / TC slices (a^T b)."""
    def __init__(self):
        self.count, self.kg, self.fresh, self.cache, self.tc_cache, self.tc_enabled = 0, None, None, None, None, False
        self.tc_old, self.tc_hist = None, []
    def append(self, left, g, mask):  # left (n, 1), g (1, n); mask: 0/1 rows that keep the completed square (None = all)
        if self.count and mask is not None: left = xp.einsum("ia,i->ia", left, mask)
        self.last_left = left; self.fresh = ((left * left).T, g) if self.count else None; self.count += 1
    def advance(self, gate, w, thin):  # gate and transport every packet; return the negative diagonal and trace
        if TCH: self.tc_hist = ([self.tc_old] + self.tc_hist)[:TCH - 1]
        self.tc_old = None
        n = int(w.shape[1]); self.cache = self.tc_cache = self._last = None
        if self.count < 2: return xp.zeros(n, dtype=w.dtype), xp.zeros(n, dtype=w.dtype)
        (ids, wg), (fresh, g), c, old = ((None, w) if thin is None else thin), self.fresh, self.count - 2, None
        if ids is not None: g = xp.take(g, ids, axis=1)
        if c:
            old = self.kg * gate
            if ids is not None: old = xp.take(old, ids, axis=1)
            g = xp.concatenate([old, g], axis=0)
        moved = xp.matmul(g, wg); parts = [moved[:c], moved[c:]] if c else [None, moved]   # M10: views, no split call
        kg = self.kg = xp.concatenate(([parts[0]] if c else []) + [xp.matmul(fresh, w), parts[1]], axis=0)
        hk = int(kg.shape[0]) // 2; k, g = kg[:hk], kg[hk:]; weighted, h = g * g, g * k   # M10: views, no split call
        a = xp.concatenate([weighted, h + h], axis=0); self.cache = (a, kg)
        if self.tc_enabled and c: self.tc_cache = (xp.take(a, list(range(c)) + list(range(c + 1, 2 * c + 1)), axis=0), old)
        if TCH and self.tc_enabled and c: self.tc_old = old
        return 3 * xp.einsum("cn,cn->n", weighted, k), xp.matmul(xp.sum(a, axis=1), kg)
    @property
    def last_g(self):  # the newest transported G row (1, n), for the TC query
        if self._last is None: self._last = self.kg[-1:]
        return self._last
    def g_row(self, b): p = self.count - 1; return self.kg[p + b - 1:p + b]   # transported G row (1, n) of source b >= 1
    def hist_block(self, old, r0, idx):
        """OUT-block negative slice of the packets r0..c-1 of an older generation: current a rows, that generation's
        pre-transport [K; G] rows (old, c packets)."""
        c, p = int(old.shape[0]) // 2, self.count - 1
        a = xp.take(self.cache[0], list(range(r0, c)) + list(range(p + r0, p + c)), axis=0)
        return self.block((a, old if r0 == 0 else xp.take(old, list(range(r0, c)) + list(range(c + r0, 2 * c)), axis=0)), idx, pre=True)
    # (hist_block's idx is the repeated-index (row) side only, so SLICE_ROWS passes its row plan there unchanged)
    def fresh_cache(self): c = self.count - 2; return tuple(x[c::c + 1] for x in self.cache)  # the newest packet only
    def block(self, cache, idx, pre=False, ridx=False):  # the OUT-block negative slice a^T b of a cache (None before the first packet)
        if cache is None: return None
        a, b = cache
        if ridx is not False:  # SLICE_ROWS: rows on ridx, columns on idx (None = complete)
            if ridx is not None: a = xp.take(a, ridx, axis=1)
            if idx is not None and not pre: b = xp.take(b, idx, axis=1)
            return xp.matmul(a.T, b)
        if idx is not None: a, b = xp.take(a, idx, axis=1), b if pre else xp.take(b, idx, axis=1)
        return xp.matmul(a.T, b)

# Fixed age-switch coefficient data; generated offline from the frozen donor.
_SWQ_LITERAL = ([[[[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]],
   [[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
  [[[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
   [[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]],
 [[[0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  [[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
 [[0.0, 0.7071067811865475, 0.0, 0.0, 1.414213562373095, 0.0, 0.7071067811865475, 0.0],
  [0.0, 0.0, 0.0, 1.414213562373095, 0.0, 0.0, 0.0, 0.0], [0.7071067811865475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

def _sb_finite(value):
    """Reject invalid shared factors through the existing whole-estimator fallback."""
    if not bool(xp.all(xp.isfinite(value))): raise DegenerateMoments("nonfinite shared-basis factor")
    return value


def _sb_validate_qr(q, r, d):
    if tuple(r.shape) != (d, d) or int(q.shape[-1]) != d:
        raise DegenerateMoments("shared-basis QR is not square")
    _sb_finite(q); _sb_finite(r)
    if not bool(xp.all(xp.abs(xp.diag(r)) > 0)):
        raise DegenerateMoments("zero shared-basis QR pivot")
    return q, r


class Bank:
    """Ping-pong source storage. LEG2 rows are (H, P) per source (row 2j, 2j + 1), COMB2 the slice coefficients (C_H, C_P); LEFT is fixed at
    birth, RIGHT transported, RECON = LEFT @ RIGHT. Elder levels hold fixed Y/H coefficients and two right-basis buffers G.
    The cubic covariance-star vector is owned and read by the estimator engine."""
    def __init__(self, engine, weights, levels):
        self.engine, self.levels, self.depth, self.n = engine, int(levels), len(weights), int(weights[0].shape[0])
        self.rank, self.trace_override, self.negative = min(REPAIR_RANK, self.n) + 1, {}, NegativeBank()
        self._m10_lead = {}; _M10_IV.clear()   # M10: per-prediction index-view caches
        self.pending = self._tcs = self._tce = self._tc_QT = self.thin = self._wkey = self._tpend = None; self._views = {}
        self._tc_fold_ctx = None; self._m10_r3 = {}; self._hf_rows = False   # M10 T1: per-prediction constant memo
        self._dim, self._repair_ratios, self._dim_cast = _dimension_constants(self.n), {}, {}
        self._e_kept = self._e_kept_new = None   # E_PROP: kept-hub order of the newest admitted source
        self.e_X = None   # E_PROP: hub-coefficient chain of the previous layer (XH, XP, XR, XK, XG)
        self.e_ctx, self.e_dq = None, {}   # E_TU: engine context of the current layer; next-layer q4 corrections
        self.oplan = self.oprev = None   # OUT_THIN: (idx, inv, mask, W on COL x S, key, complement) of this transport; previous S mask
        self.nrow, self.nvrow, self.lv, self.rk, self.swL, self.q = [], [], {}, {}, {}, min(self.n, fmm.QUANTUM)
        fmm.HOST_LEVELS = self.levels
        fmm.reserve(self.levels, self.n, fmm.SYM_CHILDREN); fmm.reserve(self.levels - 1, self.n // 2, fmm.SPLIT_BATCH)
        if TCH not in (0, 1, 2, 3, 4): raise ValueError("TC history depth TCH must be 0 to 4")
        self.ngen = TCH + 1 if TCH else 2
        s, n, r = self.depth - 1, self.n, self.rank
        z, zz = lambda *shape: xp.empty(shape, dtype=xp.float32), lambda *shape: xp.zeros(shape, dtype=xp.float32)
        self.LEG2 = [zz(2 * s, n, n) for _ in range(self.ngen)]  # current to oldest H/P generations
        self._chain, self._qhist = [], None
        self.COMB2 = z(2 * s, n, n); self.CU, self.CP = self.COMB2[0::2], self.COMB2[1::2]
        self.LEFT, self.RIGHT, self.RECON, self.RREAD = zz(s, n, r), [z(s, r, n), z(s, r, n)], z(s, n, n), z(s, n, r)
        self.T1, self.T2, self.PH, self.MIX2, self.beta = z(s, n, n), z(s, n, n), z(s, n, n), z(n, n), zz(s, n)
        self.stars = GaussianStars(n, self._dim)
        self.SWQ = [xp.asarray(x, dtype=xp.float64) for x in _SWQ_LITERAL]
        self.sbG = self.sbY2 = self.sbH = None; self.sbd = 0   # age-switch metric tables
    def _denominator(self, name, dtype):
        key = (name, str(dtype))
        if key not in self._dim_cast: self._dim_cast[key] = self._dim[name].astype(dtype)
        return self._dim_cast[key]
    def _level(self, r):  # allocate each elder rank level once (coefficients Y/H, two right-basis buffers G, singular values)
        if r not in self.lv:
            cap, n = max(1, min(self.depth - 1, AGE_SWITCH_MAX_BIRTH + 1)), self.n
            Y2 = xp.zeros((2 * cap, n, r), dtype=xp.float32); G = [xp.empty((cap, r, n), dtype=xp.float32) for _ in range(self.ngen)]
            H = xp.empty((cap, self.rank, r), dtype=xp.float32); self.lv[r] = dict(Y2=Y2, YH=Y2[0::2], YP=Y2[1::2], H=H, G=G, s={})
        return self.lv[r]
    def _view(self, arr, *key, t=None):
        """arr[key] (then transposed by t), made once per (buffer, key): views of the persistent buffers repeat from layer to
        layer and every view is a round trip on the official runtime. key items: an int or a (start, stop[, step]) slice
        tuple; trailing full slices (None,) are dropped. Only for buffers that live for the whole prediction."""
        while key and key[-1] == (None,): key = key[:-1]
        if (hit := self._views.get(k := (id(arr), key, t))) is None:
            view = arr[tuple(slice(*e) if isinstance(e, tuple) else e for e in key)] if key else arr
            hit = self._views[k] = (arr, view if t is None else view.transpose(*t))
        return hit[1]
    def _elder_runs(self, sc): return _equal_groups(0, sc, [(self.rk[j], self.nrow[j]) for j in range(sc)])
    def _install(self, j, rank, basis, singular, col):  # install elder j's right basis at `rank`, lifted from the COL frame
        level = self.lv[rank]; basis, singular = [x if int(x.shape[0]) == rank else x[:rank] for x in (basis, singular)]
        xp.copyto(self._view(level["G"][0], j), basis if col is None else _scatter(basis, col[1], (1,), col[0]))
        level["s"][j], self.rk[j] = singular, rank
    def _recompress(self, j, r2, col):
        """Recompress elder j to rank r2 in its COL frame: the weighted right factor s G fixes the new basis; the same map T1
        updates Y (P/H) and H (repair) in their old order."""
        r, nr = self.rk[j], self.nrow[j]
        r2 = max(1, min(r2, r, self.n if col is None else int(col[0].shape[0])))
        A, B = self.lv[r], self._level(r2); s, G = A["s"][j], self._view(A["G"][0], j)
        if not bool(xp.all(s > 0)): raise DegenerateMoments('singular elder recompression')
        if col is not None: G = xp.take(G, col[0], axis=1)
        uu, ss, vh = xp.linalg.svd(xp.einsum("r,rj->rj", s, G), full_matrices=False, k=r2)   # exactly r2 columns / rows
        T1 = uu * (ss / s[:, None])
        fmm.matmul(A["Y2"][2 * j:2 * j + 2, :nr], T1, self.levels, out=B["Y2"][2 * j:2 * j + 2, :nr], b_shared=True)
        xp.matmul(A["H"][j], T1, out=B["H"][j]); self._install(j, r2, vh, ss, col)
    def _switch(self, j, start, col):
        """Age switch of source j: per-hub complete cubic metric (PHH, PPR, beta PPP as a two-row square-root block plus the
        repair-subspace Schur residual; cross-hub mode Grams omitted) with the total cubic trace row; the top-r SVD gives the
        right basis G; physical P/H coefficients from the SVD blocks where the hub block is conditioned, else by projection."""
        r, nr, nv = (self.n // 4 if j < EARLY else 3 * self.n // 16), self.nrow[j], self.nvrow[j]
        r = max(1, min(r, self.n if col is None else int(col[0].shape[0]),
                       2 * nr + self.rank + 1))
        level, f64, left, L = self._level(r), xp.float64, self.LEFT[j, :nr], self.LEG2[0]
        p, h, right = L[2 * j + 1, :nr], L[2 * j, :nr], self._view(self.RIGHT[0], j)
        if col is not None: p, h, right = [xp.take(a, col[0], axis=1) for a in (p, h, right)]
        if nv < nr: h = xp.copy(h); xp.copyto(h[nv:], 0.0)
        pd, hd, rd = [a.astype(f64) for a in (p, h, xp.matmul(left, right))]
        ld, rrd, bd = [a.astype(f64) for a in (left, right, self.beta[j, :nr])]
        a, b, c, d, e, f = [xp.einsum("ij,ij->i", x, y) for x, y in ((pd, pd), (hd, hd), (rd, rd), (pd, hd), (pd, rd), (hd, rd))]
        Y = xp.stack([a, b, c, d, e, f, bd * a, bd * d])  # the hub metric is a quadratic form of these row vectors
        values, vectors = xp.linalg.eigh(xp.einsum("xykl,kn,ln->nxy", self.SWQ[0], Y, Y))   # [[k00, k01], [k01, k11]]
        values = xp.maximum(values, 0.0); v0, v1 = values[:, :1], values[:, 1:]; positive = values > 1.4210854715202004e-14 * v1   # M10: views
        roots = xp.sqrt(values); inverse = xp.where(positive, 1.0 / xp.where(positive, roots, 1.0), 0.0)
        vr = vectors.transpose(0, 2, 1); offset = xp.einsum("nxy,ykl,kn,ln,nx->nx", vr, self.SWQ[1], Y, Y, inverse)   # (k02, k12)
        z = xp.einsum("nxj,nx->nxj", xp.matmul(vr, xp.stack([pd, hd], axis=1)), roots) + xp.einsum("nx,nj->nxj", offset, rd)
        schur = xp.maximum(a * a - xp.einsum("nx,nx->n", offset, offset), 0.0)
        ev, vec = xp.linalg.eigh(xp.matmul(ld.T, xp.einsum("n,nr->nr", schur, ld)))
        repair_rows = xp.matmul(xp.einsum("r,jr->rj", xp.sqrt(xp.maximum(ev, 0.0)), vec), rrd)
        T3 = xp.matmul(self.SWQ[2], Y); t0, t1, t2 = T3[0:1], T3[1:2], T3[2:3]  # total cubic trace row / sqrt 2: weights (b + 2 e + beta a, 2 d, a); M10: views
        trace = xp.matmul(t0, pd) + xp.matmul(t1, hd) + xp.matmul(t2, rd)   # (1, columns)
        stack = [xp.reshape(z, (2 * nr, int(p.shape[1]))), repair_rows, trace]
        u, s, vh = xp.linalg.svd(xp.concatenate(stack, axis=0).astype(p.dtype), full_matrices=False, k=r)   # exactly r columns / rows
        q = vh.T; yr = xp.matmul(right, q); proj = xp.reshape(u[: 2 * nr] * s, (nr, 2, r)).astype(f64)
        coefficients = xp.einsum("nxr,nx->nxr", proj - xp.einsum("nx,nr->nxr", offset, xp.matmul(ld, yr.astype(f64))), inverse)
        Yc = xp.matmul(vectors, coefficients).astype(p.dtype); yp, yh = Yc[:, 0], Yc[:, 1]   # M10: views, no unstack call
        good = ((v0 >= 1.52587890625e-5 * v1) & (v1 > 0.0)).reshape(nr)
        if count := int(xp.count_nonzero(~good)):  # ill-conditioned hubs: direct projection onto the basis
            order = xp.argsort(good); bad, keep, inv = order[:count], order[count:], xp.argsort(order)
            pb, hb = [xp.matmul(xp.take(x, bad, axis=0), q) for x in (p, h)]
            yp = xp.take(xp.concatenate([pb, xp.take(yp, keep, axis=0)], axis=0), inv, axis=0)
            yh = xp.take(xp.concatenate([hb, xp.take(yh, keep, axis=0)], axis=0), inv, axis=0)
        xp.copyto(level["H"][j], yr); xp.copyto(level["Y2"][2 * j + 1, :nr], yp); xp.copyto(level["Y2"][2 * j, :nr], yh)
        self.swL[j] = int(start); self._install(j, r, vh, s, col)
    def _transport(self, X, w, out):  # out[j] = X[j] @ w over the COL plan (thinned contraction); W transformed once per key
        if self.thin is not None: X, w = xp.take(X, self.thin[0], axis=2), self.thin[1]
        if self.oplan is not None:  # OUT_THIN: only the S columns, the rest exact zeros
            self._put_cols(out, fmm.matmul(X, self.oplan[3], self.levels, detach=False, b_shared=True, b_key=self.oplan[4])); return out
        return fmm.matmul(X, w, self.levels, out=out, b_shared=True, b_key=self._wkey)
    def _put_cols(self, tgt, vals):  # OUT_THIN: vals (the S columns, plan order) into tgt's last axis; exact zeros elsewhere
        idx, idd = self.oplan[0], self.oplan[5]
        if HYG_OPS:   # singleton views broadcast against tgt (no broadcast_to calls)
            hit = self._m10_lead.get(key := (id(idx), len(tgt.shape)))   # M10: once per (plan, target rank)
            if hit is None or hit[0] is not idx:
                lead = (None,) * (len(tgt.shape) - 1) + (slice(None),); hit = self._m10_lead[key] = (idx, idx[lead], idd[lead])
            xp.put_along_axis(tgt, hit[1], vals, axis=-1); xp.put_along_axis(tgt, hit[2], 0.0, axis=-1); return
        xp.put_along_axis(tgt, xp.broadcast_to(idx, tuple(int(x) for x in vals.shape)), vals, axis=-1)
        xp.put_along_axis(tgt, xp.broadcast_to(idd, tuple(int(x) for x in tgt.shape[:-1]) + (int(idd.shape[0]),)), 0.0, axis=-1)
    def _out_thin_plan(self, wcs, start):
        """OUT_THIN plan of the next layer's columns: S = the OUT_THIN_TAU top-gate plan of the next Gaussian gate (the
        deposit's GatePlans, same argsort, so S contains its THIN_OUT plan) plus OUT_THIN_PAD quanta; None = all columns."""
        gp, n, q = self.engine.gate_plans, self.n, self.q
        if gp.plan(THIN_OUT) is None: return None   # OUT keeps every column, so does S (lower threshold)
        mp = min(n, max(q, _pad(xp.count_nonzero(gp.gate > OUT_THIN_TAU), q) + OUT_THIN_PAD * q))
        if mp == n: return None
        idx = gp.order[n - mp:]; mask = (gp.inv >= n - mp).astype(xp.float32)
        return idx, gp.inv, mask, xp.take(wcs, idx, axis=1), ("bank-thinW-S", int(start), mp), gp.order[:n - mp]
    def _row_plan(self, idx):
        """SLICE_ROWS plan of the repeated-index rows: OUT rows with Gaussian gate in (SLICE_DEAD, 1 - SLICE_SAT), padded
        with the most uncertain remaining OUT rows; returns (ridx, rinv, positions of ridx inside idx, S = the dropped OUT rows)
        or None (= idx). L1 lean form: OUT = suffix of the gate argsort (gate_plans.order), so with SLICE_DEAD below THIN_OUT the
        kept rows are the contiguous segment order[s0 : s0 + m] (s0 = n - |OUT|) and the dropped ones its saturated tail; the
        same rows as the key argsort (padding = the least saturated), no argsort."""
        gp, n, q = self.engine.gate_plans, self.n, self.q; d = gp.gate
        m_out = n if idx is None else int(idx.shape[0]); s0 = n - m_out
        if gp.order is None: gp.order = xp.argsort(d); gp.inv = xp.argsort(gp.order); gp.idx = {}
        cnt = xp.count_nonzero(d < 1 - SLICE_SAT) - s0 if SLICE_DEAD <= THIN_OUT else None
        if cnt is None: raise ValueError("lean SLICE_ROWS plan needs SLICE_DEAD <= THIN_OUT")
        m = min(n, max(q, _pad(cnt, q)))
        if m >= m_out: return None
        ridx, S = gp.order[s0:s0 + m], gp.order[s0 + m:]
        g = gp.inv; rinv = xp.where((g >= s0) & (g < s0 + m), g + (n - m - s0), 0)
        if RESEARCH: RESEARCH_LOG.append(("rows", m, m_out))
        return ridx, rinv, (xp.arange(m) if SLICE_ROWS_TC == "block" else None), S
    def _r32(self, a, b, out, b_key=None):  # thin-repair product two levels shallower, single-item batches
        fmm.set_max_batch(1); out = fmm.matmul(a, b, self.levels - 2, out=out, b_key=b_key); fmm.set_max_batch(fmm.SYM_CHILDREN)
        return out
    def _switched_families(self, sc, idx, I=None, pre=False, ridx=False):  # slice families (C^T Y, G) of the switched elders, OUT/COL-cut
        if SB:
            if not sc: return []
            cidx = idx if ridx is False else ridx
            G = self.sbG[1 if pre else 0]; cut = I if pre else idx
            return [(self._sb_sy(0, sc, cidx)[None], G if cut is None else xp.take(G, cut, axis=2), True)]
        fams, V, C = [], self._view, self.COMB2; cidx = idx if ridx is False else ridx   # SLICE_ROWS: C^T rows on ridx
        for lo, hi, (rank, m) in self._elder_runs(sc):
            level, rows = self.lv[rank], ((2 * lo, 2 * hi, 2), (2 * lo + 1, 2 * hi, 2))   # rows of H and of P
            if (sy := _tcc_sy(self, lo, hi, rank, m, cidx)) is not None:
                gg, cut = V(level["G"][1 if pre else 0], (lo, hi)), I if pre else idx
                fams.append((sy, gg if cut is None else xp.take(gg, cut, axis=2), True)); continue
            if cidx is None: cu, cp = [V(C, e, (0, m), t=(0, 2, 1)) for e in rows]
            else: cu, cp = [xp.transpose(xp.take(V(C, e, (0, m)), cidx, axis=2), (0, 2, 1)) for e in rows]
            sy = fmm.matmul(cu, V(level["Y2"], rows[0], (0, m)), self.levels)
            sy = sy + fmm.matmul(cp, V(level["Y2"], rows[1], (0, m)), self.levels)
            gg, cut = V(level["G"][1 if pre else 0], (lo, hi)), I if pre else idx
            fams.append((sy, gg if cut is None else xp.take(gg, cut, axis=2), True))
        return fams
    def _ct(self, rows, m, idx):  # C^T of the coefficient rows (start, stop[, step]) and hub rows :m, OUT-cut
        V, C = self._view, self.COMB2
        return V(C, rows, (0, m), t=(0, 2, 1)) if idx is None else xp.transpose(xp.take(V(C, rows, (0, m)), idx, axis=2), (0, 2, 1))
    def _outside(self, col, I, hub, nr, w, cp, cut, gc=None):
        """Fresh CP rows outside COL keep their own product: cp^T (those rows) @ W (those rows; columns cut, scaled by the
        gate gc when given), or None when no kept row lies outside COL."""
        outside = col[2][: self.n - int(I.shape[0])]; pos = xp.take(hub, outside); sel = xp.nonzero(pos < nr)[0]
        if not int(sel.shape[0]): return None
        wt = xp.take(w, xp.take(outside, sel), axis=0); wt = wt if cut is None else xp.take(wt, cut, axis=1)
        if gc is not None: wt = xp.einsum("ij,j->ij", wt, gc)
        return fmm.matmul(xp.transpose(xp.take(cp, xp.take(pos, sel), axis=0)), wt, self.levels)
    def _elder_sy(self, lo, hi, rank, m, idx):  # C_H^T Y_H + C_P^T Y_P: one batched run when it fits one chunk, else two products
        if (sy := _tcc_sy(self, lo, hi, rank, m, idx)) is not None: return sy
        V, C, level, m1 = self._view, self.COMB2, self.lv[rank], max(self.nvrow[lo:hi])
        r0, r1 = (2 * lo, 2 * hi, 2), (2 * lo + 1, 2 * hi, 2)
        if idx is None: cu, cp = V(C, r0, (0, m1), t=(0, 2, 1)), V(C, r1, (0, m), t=(0, 2, 1))
        else: cu, cp = xp.transpose(xp.take(V(C, r0, (0, m1)), idx, axis=2), (0, 2, 1)), xp.transpose(xp.take(V(C, r1, (0, m)), idx, axis=2), (0, 2, 1))
        sy = fmm.matmul(cu, V(level["Y2"], r0, (0, m1)), self.levels)
        return sy + fmm.matmul(cp, V(level["Y2"], r1, (0, m)), self.levels)
    # ---- SB-T shared transported elder basis ------------------------------------------------------------------------------
    def consumer_lo(self, last_start):
        """Births excluded from the E_TU fresh term and the PK4 response fold: those the chassis schedule (SWITCH_AGE, EARLY,
        AGE_SWITCH_MAX_BIRTH) has switched in accumulates up to last_start; SB-T elders younger than that stay included
        (their legs are rebuilt every layer), so the consumers see the chassis's birth set and only the representation moves."""
        if not SB: return len(self.swL)
        k = sum(1 for j in range(min(AGE_SWITCH_MAX_BIRTH + 1, len(self.nrow))) if last_start >= j + SWITCH_AGE[0 if j < EARLY else 1])
        return min(len(self.swL), k)
    def _sb_switch(self, j, start, col):
        """Switch birth j at accumulate(start): least-squares coordinates of its gated legs (COL frame) and repair right factor
        on the shared basis; at the first switch the basis is this entrant's own top-d right singular subspace (tail rule)."""
        n, nr, nv, r = self.n, self.nrow[j], self.nvrow[j], self.rank
        L = self.LEG2[0]
        p, h, right = L[2 * j + 1, :nr], L[2 * j, :nr], self.RIGHT[0][j]
        if col is not None: p, h, right = [xp.take(a, col[0], axis=1) for a in (p, h, right)]
        if nv < nr: h = xp.copy(h); xp.copyto(h[nv:], 0.0)
        M = xp.concatenate([h[:nv], p], axis=0); ncol = int(M.shape[1])
        if self.sbG is None:
            kmax = max(1, min(ncol, int(M.shape[0]), 5 * n // 16))
            us, s, vh = xp.linalg.svd(M, full_matrices=False, k=kmax)
            _sb_finite(us); _sb_finite(s); _sb_finite(vh)
            tot = xp.einsum("ij,ij->", M, M); tail = tot - xp.cumsum(s * s)
            need = int(xp.count_nonzero(tail > _SB_TAU2 * tot)) + 1
            d = min(kmax, max(self.q, _pad(need, self.q))); sd = max(1, self.depth - 1)
            basis = vh[:d]
            if SB_E1 and SB_DENSE_REPAIR: Y = xp.multiply(us[:, :d], s[:d])   # SB_E1: M V_d = U_d S_d (SVD identity), no product
            else: Y = xp.matmul(M if SB_DENSE_REPAIR else xp.concatenate([M, right], axis=0), xp.transpose(basis))
            _sb_finite(Y)
            self.sbY2 = xp.zeros((2 * sd, n, d), dtype=xp.float32); self.sbH = None if SB_DENSE_REPAIR else xp.zeros((sd, r, d), dtype=xp.float32)
            self.sbG = [xp.zeros((1, d, n), dtype=xp.float32) for _ in range(self.ngen)]
            self.sbd = d
            xp.copyto(self.sbG[0][0], basis if col is None else _scatter(basis, col[1], (1,), col[0]))
        else:
            Qo, Ro = self._sb_qr(col)
            dg = xp.abs(xp.diag(Ro))
            if bool(xp.greater(xp.max(dg), xp.multiply(_SB_COND, xp.min(dg)))):
                self._sb_reorth(j); Qo, Ro = self._sb_qr(col)
            if SB_E1 and int(M.shape[0]) + (0 if SB_DENSE_REPAIR else int(right.shape[0])) > ncol:   # SB_E1: Y = M (Q R^-T)
                frame = _sb_finite(xp.linalg.solve(Ro, xp.transpose(Qo)))
                Y = xp.matmul(M if SB_DENSE_REPAIR else xp.concatenate([M, right], axis=0), xp.transpose(frame))
            else:
                c = xp.matmul(M if SB_DENSE_REPAIR else xp.concatenate([M, right], axis=0), Qo)
                _sb_finite(c)
                Y = xp.transpose(xp.linalg.solve(Ro, xp.transpose(c)))
        _sb_finite(Y)
        xp.copyto(self.sbY2[2 * j, :nv], Y[:nv]); xp.copyto(self.sbY2[2 * j + 1, :nr], Y[nv:nv + nr])
        if not SB_DENSE_REPAIR: xp.copyto(self.sbH[j], Y[nv + nr:])
        self.swL[j] = int(start); self.rk[j] = self.sbd
    def _sb_qr(self, col):
        G0 = self.sbG[0][0]; ncol = int(G0.shape[1]) if col is None else int(col[0].shape[0])
        if ncol < self.sbd: raise DegenerateMoments("shared-basis columns smaller than rank")
        Gc = G0 if col is None else xp.take(G0, col[0], axis=1)
        return _sb_validate_qr(*xp.linalg.qr(xp.transpose(Gc)), self.sbd)
    def _sb_reorth(self, sc):
        """Stage a finite coordinate change for every retained generation before committing."""
        G0 = self.sbG[0][0]; Qf, Rf = self._sb_qr(None); Rt = xp.transpose(Rf)
        generations = [(g, _sb_finite(xp.linalg.solve(Rt, self.sbG[g][0]))) for g in range(1, self.ngen - 1)]
        Y = _sb_finite(xp.matmul(self.sbY2[:2 * sc], Rt)) if sc else None
        Hh = _sb_finite(xp.matmul(self.sbH[:sc], Rt)) if sc and not SB_DENSE_REPAIR else None
        for g, value in generations: xp.copyto(self.sbG[g][0], value)
        xp.copyto(G0, xp.transpose(Qf))
        if Y is not None: xp.copyto(self.sbY2[:2 * sc], Y)
        if Hh is not None: xp.copyto(self.sbH[:sc], Hh)
    def _sb_transport(self, sc, w, Lb2, rb, cK, c_):
        """One transport of the shared basis for every elder; rebuild of the elders' legs Y Gs (S columns) and repair H Gs."""
        V, Ga, Gb = self._view, self.sbG[0], self.sbG[-1]
        GS = None
        if SB_E2 and self.oplan is not None:   # SB_E2: keep the S columns of the transported basis (= take(G0, S) bit for bit)
            X = Ga if self.thin is None else xp.take(Ga, self.thin[0], axis=2)
            GS = fmm.matmul(X, self.oplan[3], self.levels, b_shared=True, b_key=self.oplan[4])[0]
            self._put_cols(V(Gb, (0, 1), (None,), cK), GS)
        else: self._transport(Ga, w, V(Gb, (0, 1), (None,), cK))
        self.sbG.insert(0, self.sbG.pop())
        G0 = self.sbG[0][0]; gr = G0 if cK == (None,) else G0[:, cK[0]:cK[1]]
        bk = ("sb-GS", self._wkey[1], int(self.sbd)) if SB_E2 else None   # SB_E2: one B transform for every elder run of this layer
        for lo, hi, m in _equal_groups(0, sc, self.nrow):
            Y = V(self.sbY2, (2 * lo, 2 * hi), (0, m))
            if self.oplan is not None:
                res = fmm.matmul(Y, GS if GS is not None else xp.take(G0, self.oplan[0], axis=1), self.levels, detach=False, b_shared=True, b_key=bk)
                self._put_cols(V(Lb2, (2 * lo, 2 * hi), (0, m), cK), res)
            else: fmm.matmul(Y, gr, self.levels, out=V(Lb2, (2 * lo, 2 * hi), (0, m), cK), b_shared=True, b_key=None if bk is None else bk + (cK,))
        if SB_DENSE_REPAIR: return   # the elders' repair right factors are transported densely with the young ones
        r, d = self.rank, self.sbd
        prod = xp.matmul(xp.reshape(self.sbH[:sc], (sc * r, d)), gr)
        xp.copyto(rb[:sc, :, c_], xp.reshape(prod, (sc, r, int(gr.shape[1]))))
    def _sb_sy(self, lo, hi, cidx):
        """sum over elders lo..hi-1 of C_H^T Y_H + C_P^T Y_P, rows = the consumer columns (cidx cut): (rows, d)."""
        V, C, fams = self._view, self.COMB2, []
        for a, b, m in _equal_groups(lo, hi, self.nrow):
            rows = (2 * a, 2 * b)
            c = V(C, rows, (0, m), t=(0, 2, 1)) if cidx is None else xp.transpose(xp.take(V(C, rows, (0, m)), cidx, axis=2), (0, 2, 1))
            fams.append((c, V(self.sbY2, rows, (0, m))))
        return fmm.sum_families(fams, self.levels)
    def tc_stash_engine(self, v, d, e, a1, B): self._tce = dict(v=v, d=d, e=e, a1=a1, B=B)  # source-layer engine state (v = diag pc)
    def _tc_prepare(self, qt, ix, inv, st):
        """Prepare newest scalar TC and its Q31 factors before the fresh deposit.

        The four thin products in _rep_residual replace the four in the scalar
        _rep_apply. dq is their exact-real-arithmetic contraction reassociation.
        COMB2 and the physical trace/diagonal are never modified here.
        """
        se, n = self._tce, self.n
        kin, w, hub, d, c, a1 = st["k_in"], st["w"], st["hub"], se["d"], se["e"], se["a1"]
        I = xp.arange(n) if st["thin"] is None else st["thin"][0]
        J = xp.arange(n) if ix is None else ix
        nI, m = int(I.shape[0]), int(J.shape[0]); dI = xp.take(d, I)
        WJ = xp.take(w, J, axis=1); WIJ = xp.take(WJ, I, axis=0)
        rg = st["ra"][:kin]
        if st["sc"]: rg = xp.copy(rg); head = rg[:st["sc"]]; xp.multiply(head, d, out=head)
        A = xp.transpose(xp.reshape(xp.take(rg, I, axis=2), (kin * self.rank, nI)))
        B = xp.reshape(xp.transpose(xp.take(self.RREAD[:kin], J, axis=1), (0, 2, 1)), (kin * self.rank, m))
        gdc = _gate_div(dI)[:, None]; Q = (qt.T + xp.matmul(A, B)) / gdc   # M10: divisor column kept for the history reads
        Y = xp.einsum("i,ij->ij", dI, WIJ)
        cI, vI, aI = xp.take(c, I), xp.take(se["v"], I), xp.take(a1, I)
        # TAKE_COMPOSE: the hub-order rows are read only on I, so one take on the composed index hub[I] (same entries)
        htI = xp.take(xp.take(st["Lb2"][2 * kin], J, axis=-1), I if hub is None else xp.take(hub, I), axis=0)
        lf = xp.take(self.negative.last_left, I, axis=0); g = self.negative.kg[-1:]
        ft = xp.matmul(lf, xp.take(g, J, axis=-1))
        fgI = xp.take(st["fdiag"] if hub is None else xp.take(st["fdiag"], hub), I)
        sqc, cv = xp.sqrt(cI), cI * vI * dI + fgI - 0.5 * aI
        effc = xp.einsum("i,ij->ij", sqc, htI - ft) - xp.einsum("i,ij->ij", cv, WIJ)
        Zp = WIJ * effc - xp.outer(aI, xp.einsum("ij,ij->j", WJ, WJ) / self._denominator("2(n+4)", effc.dtype))
        svd = _rep_svd(se["B"], I, n // 256); Qt = Q - _rep_residual(svd, Y)
        dq = 12 * xp.einsum("ij,ij->j", Zp, Qt)
        mixed = _complete_bilinear(Qt, Y, Zp); PQ = WIJ * Qt
        cp = 3 * effc * Qt - 3 * xp.einsum("i,ij->ij", cv, PQ) + 6 * xp.einsum("i,ij->ij", dI, mixed)
        r3 = self._m10_r3.get(str(Qt.dtype))   # M10: -3 / (n + 4) formed once per prediction
        if r3 is None: r3 = self._m10_r3[str(Qt.dtype)] = _ratio(-3, self._dim["n+4"], Qt.dtype)
        s = r3 * xp.einsum("i,ij->j", aI, Qt)
        cpf = cp if st["thin"] is None else _scatter(cp, st["col"][1], (0,), I)
        cpf = cpf + xp.einsum("j,ij->ij", s, WJ)
        ch = 3 * xp.einsum("i,ij->ij", sqc, PQ)
        ctx = dict(dq=dq, A=A, B=B, svd=svd, I=I, J=J, dI=dI, sqc=sqc, cv=cv, aI=aI, gdc=gdc,
                   lf=lf, g=g, cpf=cpf, ch=ch)
        if FOLD_DIAGNOSTIC is not None:
            ctx["diagnostic"] = dict(Q=Q, Qt=Qt, Y=Y, Zp=Zp, effc=effc, WIJ=WIJ)
        return ctx

    def _tc_pending_folded(self):
        """Consume the cached newest scalar; leave the older scalar TC schedule intact."""
        ctx, self._tc_fold_ctx = self._tc_fold_ctx, None
        st = self._tcs; self._tcs = self._tce = None
        (_, ix, inv), self._tc_QT = self._tc_QT, None
        dq, J, B = ctx["dq"], ctx["J"], ctx["B"]
        if TCH:
            hc = getattr(self, "_hf_cache", None) or {}
            for h, qb in zip(self._chain, self._qhist or ()):
                zq = hc.get(h["st"]["k_in"])
                dq = dq + (12 * xp.einsum("ij,ij->j", zq[0], zq[1]) if zq is not None else self._tc_history(h, qb, J, B))
            self._hf_cache = None
            keep = {key: ctx[key] for key in ("svd", "I", "dI", "sqc", "cv", "aI", "A", "lf", "gdc")}
            keep["st"] = st
            self._qhist, self._chain = None, ([keep] + self._chain)[:TCH - 1]
        dq = dq if inv is None else _scatter(dq, inv, idx=ix); return dq if str(dq.dtype) == "float32" else dq.astype(xp.float32)   # M10

    def tc_terminal(self):
        """TERM_TC: the TC scalar dq of the terminal consumer, natural order (n,), or None. Same query as _deposit_pre (Horner form
        with the TC history chain, or the family sum) and the same _tc_prepare / tc_pending reads, on the transported terminal
        columns (plan order) with kappa4 weight above TERM_TC_TAU only; no slice block is formed."""
        st, P = self._tcs, self._tpend; self._tpend = None
        if st is None or P is None or self._tce is None: return None
        self.LEG2.insert(0, self.LEG2.pop())   # the terminal accumulate returns before its generation rotation: LEG2[0] = consumer legs
        try: return self._tc_terminal(st, P)
        finally: self.LEG2.append(self.LEG2.pop(0))
    def _tc_terminal(self, st, P):
        plan = self.term_plan; mt = self.n if plan is None else int(plan[0].shape[0]); idx = None if plan is None else xp.arange(mt); inv = None
        if TERM_TC_TAU:   # kappa4-weight crop of the transported terminal columns (positions in plan order)
            zp = self.engine.terminal_z if plan is None else xp.take(self.engine.terminal_z, plan[0]); zz = zp * zp
            if (sp := _plan(xp.abs((zz - 1) * xp.exp(zz * -0.5)), TERM_TC_TAU, mt, self.q)) is not None: idx, inv = sp[0], sp[1]
        k, sc, neg, V, C, LA2 = P["k"], P["sw"], self.negative, self._view, self.COMB2, P["LA2"]
        fresh, I = k - 1, (None if st["thin"] is None else st["thin"][0])
        if TCH and self._chain: qt, self._qhist = self._horner(P, idx, st, self._chain)
        else:
            self._qhist = None
            def pre(rows, m):
                c = V(C, rows, (0, m), t=(0, 2, 1)) if idx is None else xp.transpose(xp.take(V(C, rows, (0, m)), idx, axis=2), (0, 2, 1))
                return c, V(LA2, rows, (0, m)) if I is None else xp.take(V(LA2, rows, (0, m)), I, axis=2)
            fin = [pre((2 * lo + s, 2 * hi, 2), m) for s, counts in ((0, self.nvrow), (1, self.nrow)) for lo, hi, m in _equal_groups(sc, fresh, counts)]
            qt = fmm.sum_families(fin + self._switched_families(sc, idx, I, pre=True), self.levels)
            if (nb := neg.block(neg.tc_cache, idx, pre=True)) is not None: qt = qt - nb
        ctx = self._tc_prepare(qt, idx, inv, st)
        self._tc_QT = (qt, idx, inv)
        self._tc_fold_ctx = {key: ctx[key] for key in ("dq", "A", "B", "svd", "I", "J", "dI", "sqc", "cv", "aI", "lf", "gdc")}
        dq = self.tc_pending()
        return dq if plan is None else _scatter(dq, plan[1], idx=plan[0])
    def tc_pending(self):
        """Consume the deposit-owned TC query: scalar kappa4 correction dq (n,) at the consumer, or None. Zp = W o (effective
        fresh leg) on the source rows I, Q = old-source query + old repair; dq = 12 sum Zp Q minus the repeated-index image."""
        if self._tc_fold_ctx is not None: return self._tc_pending_folded()
        if not TCH: return self._tc_pending_original()
        st, se = self._tcs, self._tce; self._tcs = self._tce = None
        if st is None:
            self._chain = []; return None
        (qt, ix, inv), self._tc_QT = self._tc_QT, None
        n, kin, w, hub, d, c, a1 = self.n, st["k_in"], st["w"], st["hub"], se["d"], se["e"], se["a1"]; Lb2 = st["Lb2"]
        I = xp.arange(n) if st["thin"] is None else st["thin"][0]; J = xp.arange(n) if ix is None else ix
        nI, m = int(I.shape[0]), int(J.shape[0]); dI = xp.take(d, I); WJ = xp.take(w, J, axis=1); WIJ = xp.take(WJ, I, axis=0)
        rg = st["ra"][:kin]
        if st["sc"]: rg = xp.copy(rg); head = rg[: st["sc"]]; xp.multiply(head, d, out=head)
        A = xp.transpose(xp.reshape(xp.take(rg, I, axis=2), (kin * self.rank, nI)))
        B = xp.reshape(xp.transpose(xp.take(self.RREAD[:kin], J, axis=1), (0, 2, 1)), (kin * self.rank, m))
        # a row whose gate is exactly 0 (clipped corrected gate) carries a gated query 0 / 0: it contributes nothing
        Q = (xp.transpose(qt) + xp.matmul(A, B)) / _gate_div(dI)[:, None]; YIJ = xp.einsum("i,ij->ij", dI, WIJ)
        cI, vI, aI = xp.take(c, I), xp.take(se["v"], I), xp.take(a1, I); ht = xp.take(Lb2[2 * kin], J, axis=-1)
        if hub is not None: ht = xp.take(ht, hub, axis=0)
        lf = xp.take(self.negative.last_left, I, axis=0); ft = xp.matmul(lf, xp.take(self.negative.kg[-1:], J, axis=-1))
        fgI = xp.take(st["fdiag"] if hub is None else xp.take(st["fdiag"], hub), I)
        # effective fresh leg: sqrt(c) (H - F1) - (c v d + fdiag - a / 2) W, then the isotropic image of a
        sqc, cv = xp.sqrt(cI), cI * vI * dI + fgI - 0.5 * aI
        effc = xp.einsum("i,ij->ij", sqc, xp.take(ht, I, axis=0) - ft) - xp.einsum("i,ij->ij", cv, WIJ)
        Zp = WIJ * effc - xp.outer(aI, xp.einsum("ij,ij->j", WJ, WJ) / self._denominator("2(n+4)", effc.dtype))
        svd = _rep_svd(se["B"], I, n // 256); dq = 12 * xp.einsum("ij,ij->j", Zp, Q) - 12 * _rep_apply(svd, YIJ, Zp)
        if TCH:  # ages 2..TCH: the TC born at t - 2, t - 3, ... read at this consumer; this generation's state for later ones
            for h, qb in zip(self._chain, self._qhist or ()): dq = dq + self._tc_history(h, qb, J, B)
            self._qhist, self._chain = None, ([dict(st=st, svd=svd, I=I, dI=dI, sqc=sqc, cv=cv, aI=aI, A=A, lf=lf)] + self._chain)[: TCH - 1]
        dq = dq if inv is None else _scatter(dq, inv, idx=ix); return dq if str(dq.dtype) == "float32" else dq.astype(xp.float32)   # M10
    def _tc_pending_original(self):
        """Consume the deposit-owned TC query: scalar kappa4 correction dq (n,) at the consumer, or None. Zp = W o (effective
        fresh leg) on the source rows I, Q = old-source query + old repair; dq = 12 sum Zp Q minus the repeated-index image."""
        st, se = self._tcs, self._tce; self._tcs = self._tce = None
        if st is None: return None
        (qt, ix, inv), self._tc_QT = self._tc_QT, None
        n, kin, w, hub, d, c, a1 = self.n, st["k_in"], st["w"], st["hub"], se["d"], se["e"], se["a1"]; Lb2 = st["Lb2"]
        I = xp.arange(n) if st["thin"] is None else st["thin"][0]; J = xp.arange(n) if ix is None else ix
        nI, m = int(I.shape[0]), int(J.shape[0]); dI = xp.take(d, I); WJ = xp.take(w, J, axis=1); WIJ = xp.take(WJ, I, axis=0)
        rg = st["ra"][:kin]
        if st["sc"]: rg = xp.copy(rg); head = rg[: st["sc"]]; xp.multiply(head, d, out=head)
        A = xp.transpose(xp.reshape(xp.take(rg, I, axis=2), (kin * self.rank, nI)))
        B = xp.reshape(xp.transpose(xp.take(self.RREAD[:kin], J, axis=1), (0, 2, 1)), (kin * self.rank, m))
        Q = (xp.transpose(qt) + xp.matmul(A, B)) / dI[:, None]; YIJ = xp.einsum("i,ij->ij", dI, WIJ)
        cI, vI, aI = xp.take(c, I), xp.take(se["v"], I), xp.take(a1, I); ht = xp.take(Lb2[2 * kin], J, axis=-1)
        if hub is not None: ht = xp.take(ht, hub, axis=0)
        lf = xp.take(self.negative.last_left, I, axis=0); ft = xp.matmul(lf, xp.take(self.negative.last_g, J, axis=-1))
        fgI = xp.take(st["fdiag"] if hub is None else xp.take(st["fdiag"], hub), I)
        # effective fresh leg: sqrt(c) (H - F1) - (c v d + fdiag - a / 2) W, then the isotropic image of a
        effc = xp.einsum("i,ij->ij", xp.sqrt(cI), xp.take(ht, I, axis=0) - ft) - xp.einsum("i,ij->ij", cI * vI * dI + fgI - 0.5 * aI, WIJ)
        Zp = WIJ * effc - xp.outer(aI, xp.einsum("ij,ij->j", WJ, WJ) / self._denominator("2(n+4)", effc.dtype))
        dq = 12 * xp.einsum("ij,ij->j", Zp, Q) - 12 * _repeated_scalar(se["B"], I, YIJ, Zp, n // 16)
        dq = dq if inv is None else _scatter(dq, inv, idx=ix); return dq if str(dq.dtype) == "float32" else dq.astype(xp.float32)   # M10
    def _hist_fold_coef(self, h, qt2, J, Bt, theta, oidx=None):
        """HIST_FOLD coefficients of the history TC h at this consumer (OUT columns J): theta-scaled hub-family coefficient in the
        natural rows of its birth layer (n, m), edge-family coefficient in kept-hub order (nv, m), the rank-one F1 pair, and the
        (Zp, Qt) of its dq."""
        s2, n, I = h["st"], self.n, h["I"]; b, hub = s2["k_in"], s2["hub"]; nv = self.nvrow[b]
        gdc = h.get("gdc"); gdc = _gate_div(h["dI"])[:, None] if gdc is None else gdc   # M10: reuse the generation's column
        Q = (xp.transpose(qt2) + xp.matmul(h["A"], Bt[: b * self.rank])) / gdc
        legs = xp.take(self._view(self.LEG2[0], (2 * b, 2 * b + 2)), J, axis=-1)
        HIJ, PIJ = xp.unstack(xp.take(legs, I if hub is None else xp.take(hub, I), axis=1))
        gfull = self.negative.g_row(b); g = xp.take(gfull, J, axis=-1); ft = xp.matmul(h["lf"], g)
        rr = g[0] if oidx is J else (gfull[0] if oidx is None else xp.take(gfull, oidx, axis=-1)[0])   # line I: rank-one right factor on OUT (rows mode: J = row plan)
        sqc, cv, aI, dI = h["sqc"], h["cv"], h["aI"], h["dI"]
        effc = xp.einsum("i,ij->ij", sqc, HIJ - ft) - xp.einsum("i,ij->ij", cv, PIJ)
        Zp = PIJ * effc - xp.outer(aI, xp.einsum("ij,ij->j", legs[1], legs[1]) / self._denominator("2(n+4)", effc.dtype))
        Y = xp.einsum("i,ij->ij", dI, PIJ); Qt = Q - _rep_residual(h["svd"], Y)
        # HIST_FOLD lite: the kappa3-leg placement (distinct completion) is not folded for history TCs; theta = 3 x the base's
        # theta here: cp = 3 theta Qt (effc - cv P), ch = 3 theta sqc P Qt, s = -3 theta sum_i a_i Qt / (n + 4)
        cp = xp.einsum("ij,ij,j->ij", Qt, effc - xp.einsum("i,ij->ij", cv, PIJ), theta)
        if not (HF_LITE_ROWS and self._hf_rows):   # L1: on the row plan the lite history fold when HF_LITE_ROWS
            try: cp = cp + 2 * xp.einsum("i,ij,j->ij", dI, _complete_bilinear(Qt, Y, Zp), theta)   # LANE F: 6 theta d mixed (fresh form)
            except DegenerateMoments: pass   # LANE F: degenerate completion -> the lite fold (no fallback)
        r1 = self._m10_r3.get("r1")
        if r1 is None: r1 = self._m10_r3["r1"] = _ratio(-1, self._dim["n+4"], Qt.dtype)
        s = xp.einsum("i,ij,j->j", aI, Qt, theta) * r1
        ch = xp.einsum("i,ij,ij,j->ij", sqc, PIJ, Qt, theta)
        pnat = legs[1] if hub is None else xp.take(legs[1], hub, axis=0)        # the source's P leg at t, natural birth rows
        col = s2["col"]
        cpf = (cp if col is None else _scatter(cp, col[1], (0,), I)) + xp.einsum("j,ij->ij", s, pnat)
        nch = ch if col is None else _scatter(ch, col[1], (0,), I)
        chk = nch[:nv] if hub is None else xp.take(nch, (s2["m10_order"] if s2.get("m10_order") is not None else xp.argsort(hub))[:nv], axis=0)   # L1: M10 order == argsort(hub)
        return cpf, chk, xp.matmul(ch.T, h["lf"])[:, 0], rr, (Zp, Qt)
    def _tc_history(self, h, qt2, J, Bt):
        """TC source born at b (age t - b >= 2) read at consumer t, the pending formula: Q^(b)_t from the Horner deposit's
        generation-b query qt2 plus the old repair (A of generation b; consumer factor = the first b sources' rows of this
        consumer's B), Z from source b's own legs at t (P = the transported birth direction, H its completed square; F from its
        negative packet), the repeated-index subtraction with the SVD made of B_b[I, I] at consumer b + 1."""
        s2, n, I = h["st"], self.n, h["I"]; b, hub = s2["k_in"], s2["hub"]
        gdc = h.get("gdc"); gdc = _gate_div(h["dI"])[:, None] if gdc is None else gdc   # M10: reuse the generation's column
        Q = (xp.transpose(qt2) + xp.matmul(h["A"], Bt[: b * self.rank])) / gdc
        legs = xp.take(self._view(self.LEG2[0], (2 * b, 2 * b + 2)), J, axis=-1)   # source b at t: rows (H, P), kept-hub order
        # TAKE_COMPOSE: rows are read on I only (one take on the composed index hub[I]); the P column norms below sum over every
        # row, which the hub permutation does not change
        HIJ, PIJ = xp.unstack(xp.take(legs, I if hub is None else xp.take(hub, I), axis=1)); ft = xp.matmul(h["lf"], xp.take(self.negative.g_row(b), J, axis=-1))
        effc = xp.einsum("i,ij->ij", h["sqc"], HIJ - ft) - xp.einsum("i,ij->ij", h["cv"], PIJ)
        Zp = PIJ * effc - xp.outer(h["aI"], xp.einsum("ij,ij->j", legs[1], legs[1]) / self._denominator("2(n+4)", effc.dtype))
        return 12 * xp.einsum("ij,ij->j", Zp, Q) - 12 * _rep_apply(h["svd"], xp.einsum("i,ij->ij", h["dI"], PIJ), Zp)
    def _deposit_folded(self, P, idx, st, rp, qt, cidx, cinv):
        """Fuse the newest completed Q31 factors into the existing deposit.

        cpf is in full natural source-row order, including the isotropic term
        outside retained hubs. ch joins only the represented fresh H rows.
        Local coefficient sums do not modify the persistent physical bank.
        """
        n, k, V, C, w = self.n, P["k"], self._view, self.COMB2, P["w"]
        fresh, nv, nr = k - 1, self.nvrow[k - 1], self.nrow[k - 1]
        I = None if st["thin"] is None else st["thin"][0]
        hub, col = st["hub"], st["col"]
        if getattr(self, "_g_dcf", None) is not None:
            dg = self._g_dcf
            if cidx is not None: dg = xp.take(dg, cidx, axis=1)
            dg = xp.pad(dg, [[0, n - nr], [0, 0]])
            if hub is not None: dg = xp.take(dg, hub, axis=0)
            qt_tc = qt - (dg.T if I is None else xp.take(dg, I, axis=0).T)
        else:
            qt_tc = qt
        ctx = self._tc_prepare(qt_tc, cidx, cinv, st)
        mu, _, _, _, _, variance = self.engine._gate_cache
        theta = mu / (-3 * variance)   # M10: == -mu / (3 v) bitwise, one call fewer
        if cidx is not None: theta = xp.take(theta, cidx)
        base_ch_t = self._ct((2 * fresh, 2 * k, 2), nv, cidx)
        h_pre = V(P["LA2"], (2 * fresh, 2 * k, 2), (0, nv))
        if I is not None: h_pre = xp.take(h_pre, I, axis=2)
        natural_ch = ctx["ch"] if I is None else _scatter(ctx["ch"], col[1], (0,), I)
        if hub is None: kept_ch = natural_ch[:nv]
        else: kept_ch = xp.take(natural_ch, st["m10_order"][:nv], axis=0)   # M10: == argsort(hub)[:nv]
        ch_t = base_ch_t + xp.expand_dims(xp.einsum("ij,j->ji", kept_ch, theta), 0)
        assoc = rp is None   # assoc: the fresh H leg joins the block product through its transported form
        qa = qt if assoc else qt + fmm.sum_families([(ch_t, h_pre)], self.levels)
        cp0 = C[2 * fresh + 1, :nr]
        if cidx is not None: cp0 = xp.take(cp0, cidx, axis=1)
        cp0 = xp.pad(cp0, [[0, n - nr], [0, 0]])
        if hub is not None: cp0 = xp.take(cp0, hub, axis=0)
        cp = cp0 + xp.einsum("ij,j->ij", ctx["cpf"], theta)
        qa = qa + (cp.T if I is None else xp.take(cp, I, axis=0).T)
        row_positions = rp[2] if rp is not None and SLICE_ROWS_TC == "block" else None
        if row_positions is not None: qa = xp.take(qa, row_positions, axis=0)
        if I is not None and int(I.shape[0]) < n:   # tailcat: one product over the full COL order (outside rows first)
            tail_cp = xp.take(cp, col[2][:n - int(I.shape[0])], axis=0)
            if row_positions is not None: tail_cp = xp.take(tail_cp, row_positions, axis=1)
            wo = xp.take(w, col[2], axis=0)   # plan order = (outside, I)
            qcat, wcat = xp.concatenate([tail_cp.T, qa], axis=1), (wo if idx is None else xp.take(wo, idx, axis=1))
            if assoc: block = _tcc_assoc_block(self, st, qcat, wcat, ch_t, fresh, nv, idx)
            else: block = fmm.matmul(qcat, wcat, self.levels)
            I_tail = None
        elif assoc:
            wi = w if I is None else xp.take(w, I, axis=0)
            block = _tcc_assoc_block(self, st, qa, wi if idx is None else xp.take(wi, idx, axis=1), ch_t, fresh, nv, idx); I_tail = I
        else:
            wi = w if I is None else xp.take(w, I, axis=0)
            block = fmm.matmul(qa, wi if idx is None else xp.take(wi, idx, axis=1), self.levels); I_tail = I
        if I_tail is not None:
            outside = col[2][:n - int(I.shape[0])]
            if int(outside.shape[0]):
                tail_cp = xp.take(cp, outside, axis=0)
                if row_positions is not None: tail_cp = xp.take(tail_cp, row_positions, axis=1)
                wt = xp.take(w, outside, axis=0)
                if idx is not None: wt = xp.take(wt, idx, axis=1)
                tail = fmm.matmul(tail_cp.T, wt, self.levels); xp.add(block, tail, out=block)
        rank_left = theta * xp.matmul(ctx["ch"].T, ctx["lf"])[:, 0]
        if row_positions is not None: rank_left = xp.take(rank_left, row_positions)
        rank_right = ctx["g"][0] if idx is None else xp.take(ctx["g"][0], idx)
        block = block - xp.outer(rank_left, rank_right)
        self._tc_QT = (qt, cidx, cinv)
        # Keep only what scalar/history consumption needs; no n-by-n fold
        # coefficient or query is retained in the multi-generation TC chain.
        self._tc_fold_ctx = {key: ctx[key] for key in ("dq", "A", "B", "svd", "I", "J", "dI", "sqc", "cv", "aI", "lf", "gdc")}
        if FOLD_DIAGNOSTIC is not None:
            hf = st["Lb2"][2 * fresh]
            if hub is not None: hf = xp.take(hf, hub, axis=0)
            hi = xp.take(hf, ctx["I"], axis=0)
            FOLD_DIAGNOSTIC(dict(layer=self.engine.layer, qt=qt, base_ch_t=base_ch_t[0], h_pre=h_pre[0],
                                 cp0=cp0, w=w, I=I, J=ctx["J"], O=idx, theta=theta, cpf=ctx["cpf"], ch=ctx["ch"],
                                 H_I=hi, lf=ctx["lf"], g=ctx["g"], row_positions=row_positions, folded_block=block,
                                 tc=ctx["diagnostic"], dq=ctx["dq"]))
        if self._hf_rank is not None: block = block - self._hf_rank   # HIST_FOLD rank-one F1 parts
        return block - self.negative.block(self.negative.fresh_cache(), idx, ridx=False if rp is None else rp[0])

    def _deposit_pre(self, P, idx, inv, st, rp=None):
        """TC layers: the old-source free-slot query before its common COL lift (fresh H joins before the lift; fresh P rows
        inside COL join the lift, the rest keep their own product; fresh negative K uses full W). Qt = old PH + negative
        sources only (tc_pending adds the old repair once). Returns the slice block.
        SLICE_ROWS (rp = row plan): "rows" puts the whole query (C^T side, and so the TC dq) on the row plan; "block" keeps
        the query on OUT and restricts only the final slice product's rows."""
        n, k, sc, LA2, w, neg, V, C = self.n, P["k"], P["sw"], P["LA2"], P["w"], self.negative, self._view, self.COMB2
        fresh, nv, nr = k - 1, self.nvrow[k - 1], self.nrow[k - 1]
        I, col, hub = (None if st["thin"] is None else st["thin"][0]), st["col"], st["hub"]
        mode = None if rp is None else SLICE_ROWS_TC
        cidx, cinv = (rp[0], rp[1]) if mode == "rows" else (idx, inv)   # the query's C^T (repeated-index) rows
        def pre(rows, m):  # (C^T on the OUT columns, legs on the COL columns) of the rows (start, stop, 2) and hub rows :m
            c = V(C, rows, (0, m), t=(0, 2, 1)) if cidx is None else xp.transpose(xp.take(V(C, rows, (0, m)), cidx, axis=2), (0, 2, 1))
            return c, V(LA2, rows, (0, m)) if I is None else xp.take(V(LA2, rows, (0, m)), I, axis=2)
        self._hf_cache, self._hf_rank = None, None
        self._hf_rows = rp is not None   # L1: the row-plan deposit (HF_LITE_ROWS applies only there)
        if TCH and self._chain: qt, self._qhist = self._horner(P, cidx, st, self._chain, fold=(rp is None or SLICE_FOLD_ROWS) and TC_SLICE_FOLD, oidx=idx)
        else:
            fin = [pre((2 * lo + s, 2 * hi, 2), m) for s, counts in ((0, self.nvrow), (1, self.nrow)) for lo, hi, m in _equal_groups(sc, fresh, counts)]
            qt = fmm.sum_families(fin + self._switched_families(sc, cidx, I, pre=True), self.levels)
            if (nb := neg.block(neg.tc_cache, cidx, pre=True)) is not None: qt = qt - nb
        if TC_SLICE_FOLD: return self._deposit_folded(P, idx, st, rp, qt, cidx, cinv)
        qa = qt + fmm.sum_families([pre((2 * fresh, 2 * k, 2), nv)], self.levels)
        cp, hub = C[2 * fresh + 1, :nr], xp.arange(n) if hub is None else hub
        if cidx is not None: cp = xp.take(cp, cidx, axis=1)
        qa = qa + xp.transpose(xp.take(xp.pad(cp, [[0, n - nr], [0, 0]]), hub if I is None else xp.take(hub, I), axis=0))
        if mode == "block": qa, cp = xp.take(qa, rp[2], axis=0), xp.take(cp, rp[2], axis=1)   # final product rows only
        wi = w if I is None else xp.take(w, I, axis=0); block = fmm.matmul(qa, wi if idx is None else xp.take(wi, idx, axis=1), self.levels)
        if I is not None:  # fresh CP rows outside COL keep their own product
            outside = col[2][: n - int(I.shape[0])]; pos = xp.take(hub, outside); sel = xp.nonzero(pos < nr)[0]
            if int(sel.shape[0]):
                w_tail = xp.take(w, xp.take(outside, sel), axis=0); w_tail = w_tail if idx is None else xp.take(w_tail, idx, axis=1)
                tail = fmm.matmul(xp.transpose(xp.take(cp, xp.take(pos, sel), axis=0)), w_tail, self.levels); xp.add(block, tail, out=block)
        self._tc_QT = (qt, cidx, cinv)
        return block - neg.block(neg.fresh_cache(), idx, ridx=False if rp is None else rp[0])
    def _horner(self, P, idx, st, chain, fold=False, oidx=None):
        """TC-layer deposit started A = 1 + len(chain) generations back (Horner form of the family deposit). Every source joins
        once, at its entry generation: born inside the window -> its H leg one generation after birth and its P = identity on
        its kept rows at birth; older young sources -> both legs at t - A; elders -> C^T Y (once per run) times G at the
        generation of their last switch / recompression (or t - A); negative packets likewise. A Horner step is
        W_s[COL_{s-1}, COL_s] with the generation-s gate. The accumulated query at generation t - g (g >= 2) is the history
        query of the TC born at t - g (sources whose representation changed after t - g are absent from it; declared).
        Returns (qt at t - 1 = the family deposit's up to reassociation, [query at t - 2, t - 3, ...])."""
        n, t, neg, sc1, A, V, C = self.n, P["k"], self.negative, P["sw"], 1 + len(chain), self._view, self.COMB2
        gst = [st] + [h["st"] for h in chain]; sw = [x["sw"] for x in gst]; rk = [x["rk"] for x in gst]
        I = [None if x["thin"] is None else x["thin"][0] for x in gst]   # I[g - 1]: COL at generation t - g
        cols = lambda x, g: x if I[g - 1] is None else xp.take(x, I[g - 1], axis=2)
        state = lambda j, g: rk[g - 1][j] if j < sw[g - 1] else -1   # elder rank, or -1 (dense), at generation t - g
        ent = {j: (t - j - 1 if j >= t - A else next((g for g in range(1, A) if state(j, g) != state(j, g + 1)), A)) for j in range(t - 1)}
        fams = {g: [] for g in range(1, A + 1)}
        if (lo0 := sc1) < (hi0 := max(sc1, t - A)):   # young sources older than the window: both legs at t - A
            fams[A] += [(self._ct((2 * lo + s, 2 * hi, 2), m, idx), cols(V(self.LEG2[A], (2 * lo + s, 2 * hi, 2), (0, m)), A))
                        for s, counts in ((0, self.nvrow), (1, self.nrow)) for lo, hi, m in _equal_groups(lo0, hi0, counts)]
        hslot, hfp, fold = {}, {}, fold and HIST_FOLD
        for j in range(max(sc1, t - A), t - 1):  # born inside the window: H joins at generation j + 1
            g, nv = ent[j], self.nvrow[j]; hslot[j] = (g, len(fams[g]))
            fams[g].append((self._ct((2 * j, 2 * j + 1), nv, idx), cols(V(self.LEG2[g], (2 * j, 2 * j + 1), (0, nv)), g)))
        if fold:   # consumer gate theta, OUT columns and the consumer's repair read (as _tc_prepare builds them)
            J = xp.arange(n) if idx is None else idx; kin = st["k_in"]
            mu_, _, _, _, _, var_ = self.engine._gate_cache; theta = xp.take(-mu_ / var_, J)   # 3 x theta (see _hist_fold_coef)
            Bt = xp.reshape(xp.transpose(xp.take(self.RREAD[:kin], J, axis=1), (0, 2, 1)), (kin * self.rank, int(J.shape[0])))
            self._hf_cache, self._hf_rank = {}, None
        if sc1 and SB:
            for lo, hi, g in _equal_groups(0, sc1, [ent[j] for j in range(sc1)]):
                fams[g].append((self._sb_sy(lo, hi, idx)[None], cols(self.sbG[g], g), True))
        elif sc1: _tcc_elder_fams(self, sc1, idx, ent, fams, cols, V)   # elders: C^T Y per same-rank block, G at the entry generation
        acc, qh = None, [None] * (A - 1)
        for g in range(A, 0, -1):  # generation s = t - g
            if g < A:  # Horner step s - 1 -> s: P identity of the source born at s - 1 (+ its kept rows outside COL), W_s o gate_s
                b, sb, Ip, Ic = t - g - 1, gst[g], I[g], I[g - 1]; nr = self.nrow[b]
                hub = xp.arange(n) if sb["hub"] is None else sb["hub"]
                cp = V(C, 2 * b + 1, (0, nr)) if idx is None else xp.take(V(C, 2 * b + 1, (0, nr)), idx, axis=1)
                pb = acc + xp.transpose(xp.take(xp.pad(cp, [[0, n - nr], [0, 0]]), hub if Ip is None else xp.take(hub, Ip), axis=0))
                if b in hfp: pb = pb + xp.transpose(hfp.pop(b) if Ip is None else xp.take(hfp.pop(b), Ip, axis=0))   # HIST_FOLD hub part
                gate = gst[g - 1]["gate"]; gc = gate if Ic is None else xp.take(gate, Ic)
                w = sb["w"] if Ip is None else xp.take(sb["w"], Ip, axis=0); w = xp.einsum("ij,j->ij", w if Ic is None else xp.take(w, Ic, axis=1), gc)
                fams[g].insert(0, (xp.expand_dims(pb, 0), xp.expand_dims(w, 0)))   # (pb @ W) o gate = pb @ (W o gate)
                tail = None if Ip is None else self._outside(sb["col"], Ip, hub, nr, sb["w"], cp, Ic, gc)
                acc = fmm.sum_families(fams[g], self.levels)
                if tail is not None: xp.add(acc, tail, out=acc)
            elif fams[g]: acc = fmm.sum_families(fams[g], self.levels)
            else: acc = xp.zeros((n if idx is None else int(idx.shape[0]), n if I[g - 1] is None else int(I[g - 1].shape[0])), dtype=xp.float32)
            old = neg.tc_old if g == 1 else neg.tc_hist[g - 2]   # packets: all older ones at t - A, else the one born at s - 1
            if old is not None: acc = acc - neg.hist_block(old, 0 if g == A else int(old.shape[0]) // 2 - 1, idx)
            if g >= 2: qh[g - 2] = acc
            if fold and g >= 2 and (bt := t - g) in hslot and hslot[bt][0] == g - 1:   # HIST_FOLD for the TC born at t - g
                cpf, chk, rl, rr, zq = self._hist_fold_coef(chain[g - 2], acc, J, Bt, theta, oidx)
                hfp[bt] = cpf; self._hf_cache[bt] = zq
                gg, pos = hslot[bt]; ct, lg = fams[gg][pos]; fams[gg][pos] = (ct + xp.expand_dims(xp.transpose(chk), 0), lg)
                o = xp.outer(rl, rr); self._hf_rank = o if self._hf_rank is None else self._hf_rank + o
        return acc, qh
    def deposit(self, layer, mu, pc, gvw=None):  # G_Q31 wrapper: fold the T(u) (3,1) image into the coefficients for this deposit only
        self._g_dcf, self._g_sb_s = None, None
        undo = self._g_fold(self.pending, mu, pc, gvw) if (self.pending is not None) else None
        self._deposit_body(layer, mu, pc)
        self._g_dcf, self._g_sb_s = None, None
        if undo is not None:
            for dst, src in undo: xp.copyto(dst, src)
    def _g_fold(self, P, mu, pc, gvw=None):
        X, k = self.e_X, P["k"]; old, lo = k - 1, len(self.swL)
        if X is None or old <= lo or int(X[0].shape[0]) != 2 * old or int(X[1].shape[0]) != old: return None
        n, C, L, r = self.n, self.COMB2, self.LEG2[0], self.rank
        _, _, _, v = self.engine.live_gate(mu, pc); w = P["w"]
        if gvw is None: gvw = xp.einsum("aj,aj->j", w, w)
        r1 = self._m10_r3.get("r1")
        if r1 is None: r1 = self._m10_r3["r1"] = _ratio(-1, self._dim["n+4"], mu.dtype)
        s = (mu / v) * (gvw * r1)   # 3 theta gv / (n + 4), theta = -mu / (3 v)
        m, mv = max(self.nrow[lo:old]), max(self.nvrow[lo:old])
        xH, xP, xR = X[0][2 * lo:2 * old:2, :mv], X[0][2 * lo + 1:2 * old:2, :m], X[1][lo:old, :m]
        H, Pl, R = L[2 * lo:2 * old:2, :mv], L[2 * lo + 1:2 * old:2, :m], self.RECON[lo:old, :m]
        dcu = (xp.einsum("kh,khi->khi", xH, Pl[:, :mv]) + xp.einsum("kh,khi->khi", xP[:, :mv], H)) * s
        dcp = (xp.einsum("kh,khi->khi", xP, R) + xp.einsum("kh,khi->khi", xP * self.beta[lo:old, :m] + xR, Pl)) * s
        cu, cp, cs = C[2 * lo:2 * old:2, :mv], C[2 * lo + 1:2 * old:2, :m], C[2 * lo:2 * old, :m]
        undo = [(cs, xp.copy(cs)), (self.RREAD[lo:old], xp.copy(self.RREAD[lo:old]))]
        xp.add(cu, dcu, out=cu); xp.add(cp, dcp, out=cp); xp.add(cp[:, :mv], xp.einsum("kh,khi,i->khi", xH, H, s), out=cp[:, :mv])
        nr, kept = self.nrow[old], self._e_kept
        dl = X[4] if kept is None else xp.take(X[4], kept)
        cf = C[2 * old + 1, :nr]; undo.append((cf, xp.copy(cf)))
        rf = getattr(self, "_e_rad_fold", None)
        if rf is not None:
            vbg, cag, tv, sb = rf
            if kept is not None: vbg, cag = xp.take(vbg, kept), xp.take(cag, kept)
            self._g_dcf = dcf = (dl[:nr, None] * L[2 * old + 1, :nr] + cag[:nr, None] * tv[0] + vbg[:nr, None] * tv[1]) * s
            self._g_sb_s = sb * s
        else:
            self._g_dcf = dcf = xp.einsum("h,hi,i->hi", dl[:nr], L[2 * old + 1, :nr], s)
        xp.subtract(cf, dcf, out=cf)
        rread, rbf = P["side"]
        dr = xp.einsum("kh,khi,khr,i->kir", xP, Pl, self.LEFT[lo:old, :m], s)
        xp.add(self.RREAD[lo:old], dr, out=self.RREAD[lo:old])
        dr_side = xp.reshape(xp.pad(xp.transpose(dr, (1, 0, 2)), [[0, 0], [lo, k - old], [0, 0]]), (n, k * r))
        P["side"] = (rread + dr_side, rbf)
        return undo
    def _deposit_body(self, layer, mu, pc):  # deliver the pending (2,1) slice into repeated_bank[layer] once the target gate is known
        P, st, n = self.pending, self._tcs, self.n
        self.engine.l1_drop = None
        if P is None: return
        self.engine.live_gate(mu, pc)
        if self.engine.pk4 is not None: self.engine.pk4.fold_response(self)
        out = self.engine.gate_plans.plan(THIN_OUT)
        idx, inv = (None, None) if out is None else out[:2]
        rp = self._row_plan(idx) if SLICE_ROWS else None
        fill = None
        if rp is not None and int(rp[3].shape[0]):
            fill = _l1_fill(*self._drop_urow(P, idx, inv, rp[3], mu, pc))
        # Commit only after preflight. A failed guard uses these same once-folded factors.
        self.pending = None
        if fill is not None: return self._deposit_rows(layer, P, st, idx, inv, rp, fill, pc)
        if st is not None: block = self._deposit_pre(P, idx, inv, st)
        else:
            fams = P["families"]
            if idx is not None: fams = [(xp.take(a, idx, axis=1), xp.take(b, idx, axis=2)) for a, b in fams]
            fams = fams + self._switched_families(P["sw"], idx)
            block = fmm.sum_families(fams, self.levels, out=self.MIX2 if idx is None else None)
        rread, rbf = P["side"]  # rank-4 repair slice, on the OUT block only
        if idx is not None: rread, rbf = xp.take(rread, idx, axis=0), xp.take(rbf, idx, axis=1)
        xp.add(block, self._r32(rread, rbf, None), out=block)
        if st is None and (nb := self.negative.block(self.negative.cache, idx)) is not None: block = block - nb
        mix = block if idx is None else _scatter(block, inv, (0, 1), idx)
        if getattr(self, "_g_sb_s", None) is not None: mix = mix - xp.einsum("i,ij->ij", self._g_sb_s, pc)
        xp.fill_diagonal(mix, P["total"])
        # the incoming slice of this layer and its diagonal (the persistent MIX2 buffer is copied, the scattered block is not)
        self.engine.repeated_bank[layer], self.engine.third_bank[layer] = (xp.copy(mix) if idx is None and getattr(self, "_g_sb_s", None) is None else mix), P["total"]
        self.trace_override[layer] = P["trace"]
    def _drop_urow(self, P, idx, inv, omitted, mu, pc):
        """Two Gaussian-weight functionals of the current factored slice on omitted rows.

        All contractions are explicitly binary: public0.12.3 bills the full
        expression of a three-input einsum, irrespective of its backend path.
        Current E6/PK4 response folds, repair factors and negative packets are
        included; the separate newest/history TC fold is not reconstructed.
        """
        n, k, E = self.n, P["k"], self.engine
        _, z, d, _ = E.live_gate(mu, pc)
        phi = xp.exp(xp.einsum("i,i,->i", z, z, E.K.mhalf)) * 0.3989422804014327
        t = z * d + phi
        mask = None if idx is None else (inv >= n - int(idx.shape[0])).astype(d.dtype)
        wc = t * (1.0 - d) if mask is None else xp.einsum("i,i,i->i", t, 1.0 - d, mask)
        wr = xp.take(d - t * phi, omitted)
        mm = max(self.nrow[:k]); C, L = self.COMB2[:2 * k, :mm], self.LEG2[0][:2 * k, :mm]
        CS = xp.take(C, omitted, axis=2)
        a = xp.einsum("bh,bhn->n", xp.einsum("bhs,s->bh", CS, wr), L)
        c = xp.einsum("bhs,bh->s", CS, xp.einsum("bhn,n->bh", L, wc))
        rread, rbf = P["side"]; rs = xp.take(rread, omitted, axis=0)
        a = a + xp.matmul(xp.einsum("s,sk->k", wr, rs), rbf)
        c = c + xp.matmul(rs, xp.matmul(rbf, wc))
        if self.negative.cache is not None:
            an, kg = self.negative.cache; ans = xp.take(an, omitted, axis=1)
            a = a - xp.matmul(xp.einsum("ps,s->p", ans, wr), kg)
            c = c - xp.einsum("ps,p->s", ans, xp.matmul(kg, wc))
        if mask is not None: a = a * mask
        return a, _l1_put(n, omitted, c, c.dtype), _l1_put(n, omitted, wr, wr.dtype), wc
    def _deposit_rows(self, layer, P, st, idx, inv, rp, fill, pc):
        """SLICE_ROWS deposit: nonstar slice on (row plan) x OUT, scattered with separate row / column inverses."""
        n, (ridx, rinv, pos, omitted) = self.n, rp
        if st is not None: block = self._deposit_pre(P, idx, inv, st, rp)
        else:
            fams = [(xp.take(a, ridx, axis=1), b if idx is None else xp.take(b, idx, axis=2)) for a, b in P["families"]]
            block = fmm.sum_families(fams + self._switched_families(P["sw"], idx, ridx=ridx), self.levels)
        rread, rbf = P["side"]; rread, rbf = xp.take(rread, ridx, axis=0), rbf if idx is None else xp.take(rbf, idx, axis=1)
        xp.add(block, self._r32(rread, rbf, None), out=block)
        if st is None and (nb := self.negative.block(self.negative.cache, idx, ridx=ridx)) is not None: block = block - nb
        mix = _scatter2(block, rinv, inv) + fill
        if getattr(self, "_g_sb_s", None) is not None: mix = mix - xp.einsum("i,ij->ij", self._g_sb_s, pc)
        xp.fill_diagonal(mix, P["total"])
        self.engine.repeated_bank[layer], self.engine.third_bank[layer] = mix, P["total"]
        self.trace_override[layer] = P["trace"]
    def _e_pre(self, ectx, La2, ra, old):
        """E_TU / E_PROP, before gating: hub coefficients X2 = (H.u, P.u) interleaved and XR = R.u of the z_l legs, chained with
        the previous level (X = x + rho (X_prev - excluded diagonal on the P legs of the birth born at m - 1)); T(u)_aa read
        from the previous accumulate's slice coefficients (COMB2 = 2PH / H^2 + 2PR + beta P^2, T1 = P^2, negative cache)."""
        eu, n, sc, mm_old = ectx["u"], self.n, len(self.swL), max(self.nrow[:old])
        if sc < old:
            mv_y = max(self.nvrow[sc:old]); xh_y = xp.einsum("bij,j->bi", La2[2 * sc:2 * old:2, :mv_y], eu)
            if mv_y < mm_old: xh_y = xp.pad(xh_y, [[0, 0], [0, mm_old - mv_y]])
            xp_y = xp.einsum("bij,j->bi", La2[2 * sc + 1:2 * old:2, :mm_old], eu); x2_y = xp.reshape(xp.stack([xh_y, xp_y], axis=1), (2 * (old - sc), mm_old))
        else: x2_y = xp.empty((0, mm_old), dtype=eu.dtype)
        xr_y = xp.einsum("bir,br->bi", self.LEFT[sc:old, :mm_old], xp.einsum("brj,j->br", ra[sc:old], eu))
        if mm_old < n:
            x2_y = xp.pad(x2_y, [[0, 0], [0, n - mm_old]])
            xr_y = xp.pad(xr_y, [[0, 0], [0, n - mm_old]])
        x2 = xp.pad(x2_y, [[2 * sc, 0], [0, 0]]) if sc else x2_y
        xr = xp.pad(xr_y, [[sc, 0], [0, 0]]) if sc else xr_y
        neg, c0 = self.negative, self.negative.count - 2
        nk = c0 > 0 and neg.kg is not None and int(neg.kg.shape[0]) == 2 * c0
        ku = xp.matmul(neg.kg, eu) if nk else None
        prev, rho = self.e_X, ectx.get("rho")
        if E_PROP and prev is not None and rho is not None and int(prev[0].shape[0]) == 2 * (old - 1):
            dl = prev[3] if self._e_kept is None else xp.take(prev[3], self._e_kept)
            if int(dl.shape[0]) < n: dl = xp.pad(dl, [[0, n - int(dl.shape[0])]])
            x2 = x2 + rho * xp.pad(prev[0], [[0, 2], [0, 0]])
            xr = xr + rho * xp.concatenate([prev[1], xp.negative(dl)[None, :]], axis=0)
            if nk and prev[2] is not None and int(prev[2].shape[0]) == 2 * (c0 - 1):
                pk = prev[2]; h = c0 - 1
                ku = ku + rho * xp.concatenate([pk[:h], xp.zeros(1, dtype=ku.dtype), pk[h:], xp.zeros(1, dtype=ku.dtype)])
        lo_g = sc + int(sc <= AGE_SWITCH_MAX_BIRTH and old - sc == SWITCH_AGE[0 if sc < EARLY else 1] and old <= self.depth - 3)
        tdiag_g = xp.einsum("bi,bia->a", x2[2 * lo_g:2 * old, :mm_old], self.COMB2[2 * lo_g:2 * old, :mm_old]) + xp.einsum("bi,bia->a", xr[lo_g:old, :mm_old], self.T1[lo_g:old, :mm_old])
        tdiag = tdiag_g if lo_g == 0 else tdiag_g + xp.einsum("bi,bia->a", x2[:2 * lo_g, :mm_old], self.COMB2[:2 * lo_g, :mm_old]) + xp.einsum("bi,bia->a", xr[:lo_g, :mm_old], self.T1[:lo_g, :mm_old])
        if nk and neg.cache is not None and int(neg.cache[0].shape[0]) == 2 * c0: tdiag = tdiag - xp.einsum("p,pa->a", ku, neg.cache[0])
        g2 = ectx["g"] * ectx["g"]; dvec, dvec_g = g2 * tdiag, g2 * tdiag_g
        if E_PROP: self.e_X = (x2, xr, ku if nk else None, dvec, dvec_g)
        return dict(x2=x2, xr=xr, ku=ku if nk else None, c0=c0, g2=g2, dvec=dvec, dvec_g=dvec_g, ctx=ectx)
    def _e_read(self, e, old, w, gb, tplan, c_):
        """E_TU: dq4_j = 6 gv_j / (n + 4) sum_{a != b} W~_aj W~_bj [Tleg(X) - T_rad](u)_ab, read from this accumulate's slice
        coefficients of the next layer (COMB2, T1, negative cache)."""
        ctx, n, mm_old = e["ctx"], self.n, max(self.nrow[:old]); self._e_rad_fold = None
        read = xp.einsum("bi,bij->j", e["x2"][:, :mm_old], self.COMB2[:2 * old, :mm_old, c_]) + xp.einsum("bi,bij->j", e["xr"][:, :mm_old], self.T1[:old, :mm_old, c_])
        if e["ku"] is not None:
            a, ku, h = self.negative.cache[0], e["ku"], e["c0"]
            z1 = xp.zeros(1, dtype=ku.dtype)
            read = read - xp.einsum("p,pj->j", xp.concatenate([ku[:h], z1, ku[h:], z1]), a)
        ww, g2, dvec = w * w, e["g2"], e["dvec"]
        if E_RADIAL != "none":   # radial image of the bank history in the core: (v - pv) (x) pc u + pc u (x) (v - pv) + (v - pv).u pc
            vv, ca, sr, ua = ctx["rad"]; pv = ctx["pv"]; vb = vv - pv; sb = sr - xp.einsum("i,i->", pv, ua)
            vbg, cag = vb * gb, ca * gb
            tv = xp.matmul(xp.stack([vbg, cag]), w)
            rad_d = 2 * g2 * vb * ca + sb * ctx["ov"]
            dvec = dvec - rad_d
            vn = ctx["vnext"] if tplan is None else xp.take(ctx["vnext"], tplan[0])
            read = read - 2 * tv[0] * tv[1] - sb * vn
            if tplan is None and E_PROP:
                self.e_X = (self.e_X[0], self.e_X[1], self.e_X[2], self.e_X[3], e["dvec_g"] - rad_d)
                self._e_rad_fold = (vbg, cag, tv, sb)
        r1 = self._m10_r3.get("r1")
        if r1 is None: r1 = self._m10_r3["r1"] = _ratio(-1, self._dim["n+4"], read.dtype)
        corr = (xp.matmul(dvec, ww) - read) * r1   # E_GA: the core-diagonal read dga (q4 = 6 ga gv)
        if self.oplan is not None: corr = corr * self.oplan[2]
        if tplan is not None: corr = _scatter(corr, tplan[1], idx=tplan[0])
        return corr if str(corr.dtype) == "float32" else corr.astype(xp.float32)
    def accumulate(self, source, start, weights):
        """Admit the source born at `start` (hub-row thinning, repair SVD), transport the history to layer start + 1
        (gates, COL plan, elder switch/recompression), read out the kappa3 diagonal and trace; (2,1) slice factors -> pending."""
        n, r, V = self.n, self.rank, self._view; old, sc, gb = len(self.nrow), len(self.swL), self.engine.forecast_gates[start]
        tcs = self._tcs = dict(k_in=old, sc=sc) if TCP_START <= start + 1 <= self.depth - (1 if TERM_TC else 2) else None
        self.negative.tc_enabled = tcs is not None
        # fresh rows: keep gate > ROW_THIN (nr rows, padded); saturated rows (gate >= 1 - V_SAT) keep P only (rows >= nv)
        live, sat, mid = _hub_masks(source, gb); nr = min(n, _pad(xp.count_nonzero(live), self.q))
        if nr == 0: raise DegenerateMoments('empty retained cubic source')
        nv = min(nr, _pad(xp.count_nonzero(mid), self.q))
        kept = order = hubrank = None; source["_mid"] = mid
        if nr < n or nv < nr:
            order = source["_order"] = xp.argsort((2 * (1 - mid) - gb) - sat); kept = source["_kept"] = order[:nr]; hubrank = source["_rank"] = xp.argsort(order)   # M10
            for key in ("u", "r", "beta"): source[key] = xp.take(source[key], kept, axis=0)
            if nv < nr: tail = source["u"][nv:]; xp.multiply(tail, 0.0, out=tail)
        self.nvrow.append(nv); self.nrow.append(nr)
        self.negative.append(source["_negL"], source["_negG"], None if kept is None else (hubrank < nv).astype(xp.float32))
        if tcs is not None:  # TC: the F-diagonal curvature of the fresh source in kept-row order, zero padded
            g = source["_tc_fdiag"] if kept is None else xp.take(source["_tc_fdiag"], kept)
            if nv < nr: tail = g[nv:]; xp.multiply(tail, 0.0, out=tail)
            if nr < n: g = xp.pad(g, [[0, n - nr]])
            tcs.update(fdiag=g, hub=hubrank, m10_order=order)   # M10: the order whose argsort hubrank is
        col = self.engine.vertex_plans.plan(THIN_TAU); La2, Lb2, (ra, rb) = self.LEG2[0], self.LEG2[-1], self.RIGHT
        ectx, self.e_ctx = self.e_ctx, None
        epre = self._e_pre(ectx, La2, ra, old) if (E_TU and ectx is not None and old) else None
        self._e_kept = kept   # E_PROP: the source admitted now is the 'birth at m - 1' of the next accumulate
        if old:  # gate the history (gb scales the last axis; the star metric also its rows); recompress / switch elders
            if sc < old:
                mv_y, mm_y = max(self.nvrow[sc:old]), max(self.nrow[sc:old]); yh = La2[2 * sc:2 * old:2, :mv_y]; xp.multiply(yh, gb, out=yh); yp = La2[2 * sc + 1:2 * old:2, :mm_y]; xp.multiply(yp, gb, out=yp)
            y = ra[(0 if (SB and SB_DENSE_REPAIR) else sc):old]; xp.multiply(y, gb, out=y)
            if SB:
                if sc: xp.multiply(self.sbG[0], gb, out=self.sbG[0])
                due = ((old - sc == SB_AGE) if SB_AGE is not None else (sc <= AGE_SWITCH_MAX_BIRTH and old - sc == SWITCH_AGE[0 if sc < EARLY else 1])) and start <= self.depth - 3
                if due: self._sb_switch(sc, start, col); sc += 1
            else:
                for lo, hi, rank in _equal_groups(0, sc, self.rk):
                    G = V(self.lv[rank]["G"][0], (lo, hi)); xp.multiply(G, gb, out=G)
                for j in range(min(sc, EARLY)):
                    if start - self.swL[j] == RECOMP_AGE: self._recompress(j, n // 8, col)
                if sc <= AGE_SWITCH_MAX_BIRTH and old - sc == SWITCH_AGE[0 if sc < EARLY else 1] and start <= self.depth - 3:
                    self._switch(sc, start, col); sc += 1
        k = old + 1; xp.copyto(La2[2 * old, :nr], source["u"])
        u_, s_, vh_ = _svd_topk(source["r"], min(REPAIR_RANK, nr))  # birth repair: exactly k columns / rows
        rep = int(s_.shape[0])
        fitted_left = xp.multiply(u_, s_)
        correction, beta_delta = _repair_conservation(source["r"], fitted_left, vh_, kept, self._repair_ratios)
        xp.copyto(self.LEFT[old, :nr, :rep], fitted_left)
        xp.copyto(self.LEFT[old, :nr, rep:rep + 1], 1.0)
        if rep + 1 < r: xp.copyto(V(ra, old), 0.0)
        xp.copyto(ra[old, :rep], vh_)
        xp.copyto(ra[old, rep], correction)
        source["beta"] = source["beta"] + beta_delta
        xp.copyto(self.beta[old, :nr], source["beta"])
        target = start + 1; last, w, c_, cK, tplan = target == self.depth - 1, weights[target], slice(None), (None,), None
        if last:  # terminal: transport only to the columns whose q4 weight matters
            zt = self.engine.terminal_z; self.term_plan = tplan = _plan(xp.abs(zt) * xp.exp(0.5 - 0.5 * zt * zt), TERM_KEY, n, self.q)
            if TERM_U16:   # the TERM_KEY plan within U16
                if self.engine.diag_on: self.engine.diag["term_base_in"] = xp.ones(n, dtype=xp.float32) if tplan is None else (tplan[1] >= n - int(tplan[0].shape[0])).astype(xp.float32)
                self.term_plan = tplan = _sub_plan(tplan, xp.abs(zt) < TERM_U16_ALPHA, n, self.q)
            if TERM_U16: RESEARCH_LOG.append(("term", int(target), n if tplan is None else int(tplan[0].shape[0]), n))
            if tplan is not None: w, c_, cK = xp.take(w, tplan[0], axis=1), slice(0, int(tplan[0].shape[0])), (0, int(tplan[0].shape[0]))
        self.thin = None if col is None else (col[0], xp.take(w, col[0], axis=0)); self._wkey = ("bank-thinW", int(start))
        wcs, kcs = (w, n) if col is None else (self.thin[1], int(col[0].shape[0]))
        if OUT_THIN:  # research: S plan of the next layer's columns (None on the terminal transport, which has its own plan)
            if OUT_THIN_CHECK and self.oprev is not None:   # did this layer's COL plan stay inside the S of its transport?
                RESEARCH_LOG.append(("colmiss", int(start), kcs - int(xp.sum(self.oprev if col is None else xp.take(self.oprev, col[0]))), kcs))
            self.oplan = None if last else self._out_thin_plan(wcs, start); self.oprev = None if self.oplan is None else self.oplan[2]
            if RESEARCH: RESEARCH_LOG.append(("S", int(target), n if self.oplan is None else int(self.oplan[0].shape[0]), n))
        for lo, hi, m in _equal_groups(sc, k, self.nrow):  # joint H/P transport of the young sources (rows 2j H, 2j + 1 P)
            m1, fresh = max(self.nvrow[lo:hi]), int(hi == k); pe, nb = hi - fresh, 2 * (hi - lo) - fresh
            hp = V(La2, (2 * lo, 2 * lo + nb), (0, m1))
            if col is not None: hp = xp.take(hp, col[0], axis=2)
            outs = [(2 * (j - lo), 2 * (j - lo) + 2, V(Lb2, (2 * j, 2 * j + 2), (0, m1), cK)) for j in range(lo, pe)]
            if fresh: outs.append((nb - 1, nb, V(Lb2, (2 * pe, 2 * pe + 1), (0, m1), cK)))
            if len(outs) > 1: outs = [(0, nb, V(Lb2, (2 * lo, 2 * lo + nb), (0, m1), cK))]   # one contiguous output segment
            if self.oplan is None: fmm.matmul(hp, wcs, self.levels, out_segments=outs, b_shared=True, b_key=self._wkey)
            else:  # OUT_THIN: the S columns only, scattered (other columns exact zeros)
                res = fmm.matmul(hp, self.oplan[3], self.levels, detach=False, b_shared=True, b_key=self.oplan[4])
                for a0, a1, tgt in outs: self._put_cols(tgt, res if (a0, a1) == (0, nb) else res[a0:a1])
            if m1 < m and pe > lo: self._transport(V(La2, (2 * lo + 1, 2 * pe, 2), (m1, m)), w, V(Lb2, (2 * lo + 1, 2 * pe, 2), (m1, m), cK))
        xp.copyto(Lb2[2 * old + 1, :nr, c_], w if kept is None else xp.take(w, kept, axis=0))
        if self.oplan is not None:   # OUT_THIN: S columns only
            fp = Lb2[2 * old + 1, :nr]; xp.put_along_axis(fp, self.oplan[5][None, :] if HYG_OPS else xp.broadcast_to(self.oplan[5], (nr, int(self.oplan[5].shape[0]))), 0.0, axis=-1)
        if SB and sc: self._sb_transport(sc, w, Lb2, rb, cK, c_)
        elif sc:  # elders: transport G, rebuild P/H = Y @ G and the repair right factor
            segs = _equal_groups(0, sc, self.rk)
            for lo, hi, rank in segs:
                Ga, Gb = self.lv[rank]["G"][0], self.lv[rank]["G"][-1]; self._transport(V(Ga, (lo, hi)), w, V(Gb, (lo, hi), (None,), cK))
            for level in self.lv.values(): level["G"].insert(0, level["G"].pop())
            for lo, hi, (rank, m) in self._elder_runs(sc):
                level = self.lv[rank]
                if self.oplan is not None:  # OUT_THIN: rebuild on the S columns, scatter
                    tmp = [xp.empty((hi - lo, m, int(self.oplan[0].shape[0])), dtype=xp.float32) for _ in range(2)]
                    fmm.matmul_grouped(V(level["Y2"], (2 * lo, 2 * hi), (0, m)), xp.take(V(level["G"][0], (lo, hi)), self.oplan[0], axis=2), self.levels, tmp)
                    for a, tgt in zip(tmp, [V(Lb2, (2 * lo, 2 * hi, 2), (0, m), cK), V(Lb2, (2 * lo + 1, 2 * hi, 2), (0, m), cK)]):
                        self._put_cols(tgt, a)
                    continue
                fmm.matmul_grouped(V(level["Y2"], (2 * lo, 2 * hi), (0, m)), V(level["G"][0], (lo, hi), (None,), cK), self.levels,
                                   [V(Lb2, (2 * lo, 2 * hi, 2), (0, m), cK), V(Lb2, (2 * lo + 1, 2 * hi, 2), (0, m), cK)])
            for lo, hi, rank in segs:
                level = self.lv[rank]; xp.matmul(V(level["H"], (lo, hi)), V(level["G"][0], (lo, hi), (None,), cK), out=V(rb, (lo, hi), (None,), cK))
        r0 = 0 if (SB and SB_DENSE_REPAIR) else sc
        if tplan is not None: xp.matmul(ra[r0:k], w, out=rb[r0:k, :, c_])
        else:
            self._r32(xp.reshape(ra[r0:k], ((k - r0) * r, n)), w, xp.reshape(rb[r0:k], ((k - r0) * r, n)), b_key=("r32-W", int(start)))
        mm = max(self.nrow[:k])
        leftk, rbk, rr = self.LEFT[:k, :mm], rb[:k, :, c_], self.RECON[:k, :mm, c_]
        xp.matmul(leftk, rbk, out=rr)
        negdiag, negtrace = self.negative.advance(gb, w, self.thin)
        u, p, beta = Lb2[0:2 * k:2, :mm, c_], Lb2[1:2 * k:2, :mm, c_], self.beta[:k, :mm]
        cu, cp = self.COMB2[0:2 * k:2, :mm, c_], self.COMB2[1:2 * k:2, :mm, c_]
        p2 = xp.multiply(p, p, out=self.T1[:k, :mm, c_]); pu = xp.multiply(p, u, out=self.PH[:k, :mm, c_])
        t = xp.multiply(pu, u, out=self.T2[:k, :mm, c_])
        if not last:  # kappa3 diagonal 3 sum(P H H + P P R) + beta P^3; the repair part from the thin factors
            rreadk = self.RREAD[:k]; xp.matmul(p2.transpose(0, 2, 1), leftk, out=rreadk)
            s_diag = xp.einsum("kir,kri->i", rreadk, rbk); total = xp.einsum("kij->j", t) + s_diag
        else: xp.multiply(p2, rr, out=cu); xp.add(t, cu, out=t); total = xp.einsum("kij->j", t)
        xp.multiply(p2, p, out=cp); total = 3 * total + xp.einsum("ka,kai->i", beta, cp) - negdiag
        if tcs is not None: tcs.update(Lb2=Lb2, ra=ra, w=w if last else weights[target], thin=self.thin, col=col)   # terminal: plan columns
        if TCH and tcs is not None: tcs.update(sw=sc, rk=dict(self.rk), gate=gb)
        if last:  # terminal readout: diagonal only (no slice deposit)
            total = total if tplan is None else _scatter(total, tplan[1], idx=tplan[0])
            self.engine.repeated_bank[target] = self.engine.repeated_bank[target] + total
            if tcs is not None:   # TERM_TC: consumer coefficients C and the repair read B on the terminal columns, for tc_terminal
                rreadk = self.RREAD[:k, :int(p2.shape[2])]; xp.matmul(p2.transpose(0, 2, 1), leftk, out=rreadk)
                xp.multiply(u, u, out=cp); xp.multiply(p, rr, out=t); xp.multiply(t, 2.0, out=t); xp.add(cp, t, out=cp)
                xp.add(cp, xp.einsum("kij,ki->kij", p2, beta), out=cp); xp.add(pu, pu, out=cu)
                self._tpend = dict(k=k, sw=sc, LA2=La2, w=w)
                if epre is not None: self.e_dq[target] = self._e_read(epre, old, w, gb, tplan, c_)   # E_TU (terminal; TERM_TC formed cp/cu)
            return
        xp.multiply(u, u, out=cp)  # slice coefficients CP = U^2 + 2 P R + beta P^2, CU = 2 P H (in place, T2 scratch)
        xp.multiply(p, rr, out=t); xp.multiply(t, 2.0, out=t); xp.add(cp, t, out=cp)
        xp.add(cp, xp.einsum("kij,ki->kij", p2, beta), out=cp); xp.add(pu, pu, out=cu)
        if epre is not None: self.e_dq[target] = self._e_read(epre, old, w, gb, tplan, c_)   # E_TU (hidden)
        families = None if tcs is not None else [(V(self.COMB2, (2 * lo + c, 2 * hi, 2), (0, m), t=(0, 2, 1)), V(Lb2, (2 * lo + c, 2 * hi, 2), (0, m)))
                    for c, counts in ((0, self.nvrow), (1, self.nrow)) for lo, hi, m in _equal_groups(sc, k, counts)]   # M10: unread by TC deposits
        rread = xp.reshape(rreadk.transpose(1, 0, 2), (n, k * r))
        rbf = xp.reshape(rbk, (k * r, n))
        trace = xp.matmul(xp.sum(rread, axis=0), rbf)
        # coefficient / leg rows beyond a source's kept hubs are exact zeros: one product over every source (rows 2j H, 2j + 1 P)
        my = max(self.nrow[sc:k]); ck, lk = self.COMB2[:2 * k, :mm], Lb2[:2 * k, :mm]; csum = xp.sum(ck, axis=2)
        if OUT_THIN_ABL in ("trace", "rowshare") and (ab := self._out_thin_plan(wcs, start)) is not None:   # research ablation
            csum = xp.einsum("bki,i->bk", ck, ab[2])   # radial trace without the dead (non-S) rows' share
        draw = s_diag + xp.einsum("bki,bki->i", ck, lk)
        trace = trace + xp.einsum("bk,bkj->j", csum[2 * sc:, :my], Lb2[2 * sc:2 * k, :my])
        if SB and sc:
            tr = xp.einsum("ea,ear->r", csum[0:2 * sc], V(self.sbY2, (0, 2 * sc), (0, mm)))
            trace = trace + xp.matmul(tr, self.sbG[0][0])
        for lo, hi, rank in ([] if SB else _equal_groups(0, sc, self.rk)):
            tr = xp.einsum("ea,ear->er", csum[2 * lo : 2 * hi], V(self.lv[rank]["Y2"], (2 * lo, 2 * hi), (0, mm)))
            tr = xp.sum(xp.reshape(tr, (hi - lo, 2, rank)), axis=1)
            trace = trace + xp.einsum("br,brj->j", tr, V(self.lv[rank]["G"][0], (lo, hi)))
        if OUT_THIN_ABL == "diag":  # research ablation (no thinning): zero only the delivered kappa3 diagonal off the S plan
            if (ab := self._out_thin_plan(wcs, start)) is not None: total_out = total * ab[2]
        tr_mask = ab[2] if OUT_THIN_ABL == "trace" and ab is not None else None   # "trace": also zero the trace off S
        self.pending = dict(k=k, w=w, LA2=La2, families=families, side=(rread, rbf), total=total_out if OUT_THIN_ABL == "diag" and ab is not None else total, sw=sc,
                            trace=(trace - draw + total - negtrace + negdiag) if tr_mask is None else (trace - draw + total - negtrace + negdiag) * tr_mask)
        self.RIGHT = [rb, ra]; self.LEG2.insert(0, self.LEG2.pop())

# ---- Gaussian star history (Eppie-derived): a birth with third derivative f carries kappa4(x) = 4 sum_j f_j x_j (v_j . x)^3,
# v_j = row j of X = H / sqrt(e). At a consumer the scalar at direction w (column k of the transported legs) is
# 4 sum_j f_j S_jk X_jk^3 minus the isotropic core image of the same source and its diagonal powers (S = transported P leg).
# Constants are computed at birth from the operand evaluated later; the scalar joins q4 through the engine's tc_extra.

class GaussianStars:  # per-birth constants (bank row order) and the per-consumer scalar; reads bank legs, changes no state
    def __init__(self, n, dimension_constants=None):
        self.n, self.births = n, []
        self._dim = _dimension_constants(n) if dimension_constants is None else dimension_constants
    def birth(self, source, f_nat, nr, nv):
        """Record the fresh source after Bank.accumulate thinned it (u = kept-order H rows, row i = natural hub order[i]):
        one (6, n) array of row-order, zero-padded vectors isq, f, f cn, f de, powers, iso of the kept-hub star."""
        n, order, inv = self.n, source.get("_order"), source.get("_rank")
        V = xp.stack([source["e"], source["_mid"], f_nat, source["_dh"]])   # natural order; _dh = diagonal of H before thinning
        if order is not None: V = xp.take(V, source["_kept"], axis=1)
        e, mid, f, dh = xp.unstack(V); pos = e > 0; live = pos.astype(e.dtype) * mid
        if nv < nr: live = live * xp.asarray([1.0] * nv + [0.0] * (nr - nv), dtype=e.dtype)
        isq = live / xp.sqrt(xp.where(pos, e, 1.0)); f, de = f * live, dh * isq
        X0 = xp.einsum("ij,i->ij", source["u"], isq); sq0 = X0 * X0; cn, fde = xp.sum(sq0, axis=1), f * de; A = fde * cn
        S = xp.stack([A, xp.einsum("i,i,i->i", fde, de, de), isq, f, f * cn, fde])
        if nr < n: S = xp.pad(S, [[0, 0], [0, n - nr]])
        N, rows = xp.split(S, [2], axis=0); na, nb = xp.unstack(N if inv is None else xp.take(N, inv, axis=1))   # natural order
        c6 = xp.divide(_scalar(6.0), self._dim["n+4"])
        c3 = xp.divide(_scalar(3.0), self._dim["(n+2)(n+4)"])
        trace = xp.multiply(4.0, _scalar(xp.sum(A)))
        residual = 4 * nb - _scalar(xp.multiply(2.0, c6), nb.dtype) * (na + xp.matmul(fde, sq0)) + _scalar(xp.multiply(c3, trace), nb.dtype)
        offset = xp.divide(xp.multiply(3.0, _scalar(xp.sum(residual))), self._dim["(n-2)(n-1)"])
        powers = _ratio(self._dim["n+4"], self._dim["n-2"], residual.dtype) * residual - _scalar(offset, residual.dtype)
        iso_offset = _scalar(xp.multiply(c3, xp.subtract(trace, _scalar(xp.sum(powers)))), powers.dtype)
        PI = xp.stack([powers, _scalar(xp.negative(c6), powers.dtype) * powers - iso_offset])
        self.births.append(xp.concatenate([rows, PI if order is None else xp.take(PI, order, axis=1)], axis=0))
    def _part(self, H, S, lo, hi):
        """Sum over the sources lo..hi of the star scalar minus its core image and powers; H, S: their H and P legs."""
        n, (isq, f, fcn, fde, powers, iso) = self.n, xp.unstack(xp.stack(self.births[lo:hi]), axis=1)
        X = xp.einsum("mik,mi->mik", H, isq); X2, SX, S2 = X * X, S * X, S * S; cubic = xp.einsum("mi,mik->k", f, SX * X2)
        image = _ratio(12, self._dim["n+4"], X.dtype) * (xp.einsum("mi,mik->mk", fcn, SX) + xp.einsum("mi,mik->mk", fde, X2))
        image = image + xp.einsum("mi,mik->mk", iso, S2)
        return 4 * cubic - xp.einsum("mk,mk->k", image, xp.sum(S2, axis=1)) - xp.einsum("mi,mik->k", powers, S2 * S2)
    def consumer(self, bank, t):  # scalar dq4 (n,) at hidden consumer t from births of age STAR_AGES (legs in LEG[0]), or None
        lo, hi = max(0, t - STAR_AGES[1]), min(len(self.births), t - STAR_AGES[0] + 1)
        if not (STAR_START <= t <= bank.depth - 2) or hi <= lo: return None
        L = bank.LEG2[0]
        if STAR_ROWS:  # research: dq4 only on the columns with gate in (STAR_DEAD, 1 - STAR_SAT), padded; zeros elsewhere
            d = bank.engine.gate_plans.gate; req = (d > STAR_DEAD) * (d < 1 - STAR_SAT)
            kp = _key_plan(2 * req + xp.minimum(d, 1 - d), xp.count_nonzero(req), self.n, bank.q)
            if kp is not None:
                H, S = [xp.take(x, kp[0], axis=2) for x in (L[2 * lo:2 * hi:2], L[2 * lo + 1:2 * hi:2])]
                return _scatter(self._part(H, S, lo, hi), kp[1], idx=kp[0])
        return self._part(L[2 * lo:2 * hi:2], L[2 * lo + 1:2 * hi:2], lo, hi)
    def terminal(self, bank, t):
        """Scalar dq4 at the terminal consumer (births of age STAR_TERM); terminal legs sit in bank.LEG[1], columns 0..mt-1
        in TERM_KEY plan order (columns outside the plan were not transported and get 0)."""
        lo, hi = max(0, t - STAR_TERM[1]), min(len(self.births), t - STAR_TERM[0] + 1)
        if hi <= lo: return xp.zeros(self.n, dtype=xp.float32)
        plan = bank.term_plan; mt = self.n if plan is None else int(plan[0].shape[0])
        L = bank.LEG2[-1]; H, S = (L[2 * lo:2 * hi:2], L[2 * lo + 1:2 * hi:2]) if plan is None else (L[2 * lo:2 * hi:2, :, :mt], L[2 * lo + 1:2 * hi:2, :, :mt])
        part = self._part(H, S, lo, hi)
        return part if plan is None else _scatter(part, plan[1], idx=plan[0])


# ---- package_d5 helpers (baked from the lab arms a4_hubcrop [ponly] and tcf_calls [pair, seg, erank, tailcat, assoc]) ------
HUB_EPS = 1e-3   # hub crop: relative Frobenius accuracy of the dropped part of the born kappa3 tensor (see _hub_masks)


def _hub_masks(source, gb):
    """Born-hub keep masks (live, sat, mid). Chassis thresholds: live = gate > ROW_THIN, P-only (saturated) = gate >= 1 - V_SAT.
    On top, a saturated P-only hub is dropped when its own kappa3 weight 3 (|H|^2 + L^2) [mid] + 3 |R| + |beta| is among the
    smallest whose cumulative weight is <= HUB_EPS of the total over the kept hubs (triangle inequality: the dropped part of
    the born tensor is <= HUB_EPS of it in Frobenius norm). Every H row is kept; the rule only ever drops more."""
    live0 = gb > ROW_THIN; sat0 = (gb >= 1 - V_SAT).astype(xp.float32)
    live0f = live0.astype(xp.float32); mid0 = live0f * (1 - sat0)   # FUSE: the float live mask is made once (was twice)
    u, r, beta, negl = source["u"], source["r"], source["beta"], source["_negL"]
    hh = xp.einsum("ij,ij->i", u, u); rr = xp.einsum("ij,ij->i", r, r)
    w0 = (3 * (hh + negl[:, 0] * negl[:, 0]) * mid0 + 3 * xp.sqrt(rr) + xp.abs(beta)) * live0f
    s0 = xp.sort(w0); c0 = xp.cumsum(s0)
    thr0 = xp.max(s0 * (c0 <= HUB_EPS * c0[-1]))
    live = live0 & ((w0 > thr0) | (mid0 > 0)); mid = mid0; sat = live.astype(xp.float32) * (1 - mid)
    return live, sat, mid


def _tcc_mm_pinned(a, b, levels, B):
    """fmm.matmul(a, b, levels) for a non-shared, unkeyed 3-D batch a of 2B <= fmm._MAXB items, with the FMM depth the
    chassis would choose for a batch of B (the separate product): same plan path, same per-item arithmetic."""
    return fmm.matmul(a, b, levels, detach=True)


def _tcc_sy(self, lo, hi, rank, m, cidx, out=None):
    """C_H^T Y_H + C_P^T Y_P of elder sources lo..hi (rank, hub rows :m) as one batched run, or None when the merged batch
    would exceed one chunk. out: (hi - lo, rows, rank) destination of the sum."""
    B = hi - lo
    if 2 * B > fmm._MAXB: return None
    V, C, level, rows2 = self._view, self.COMB2, self.lv[rank], (2 * lo, 2 * hi)
    c2 = V(C, rows2, (0, m), t=(0, 2, 1)) if cidx is None else xp.transpose(xp.take(V(C, rows2, (0, m)), cidx, axis=2), (0, 2, 1))
    P = _tcc_mm_pinned(c2, V(level["Y2"], rows2, (0, m)), self.levels, B)
    return xp.add(P[0::2], P[1::2]) if out is None else xp.add(P[0::2], P[1::2], out=out)


def _tcc_assoc_block(self, st, q, wq, ch_t, fresh, nv, idx):
    """(qt + cp + ch_t h_pre) W = [qt + cp] W + ch_t (h_pre W): h_pre W is the fresh H leg the accumulate already transported
    (st Lb2 row 2 fresh, OUT_THIN S columns, S contains the OUT plan), so the deposit block is ONE leaf-space sum of two
    products (one combine / delivery) instead of two separate products."""
    ht = st["Lb2"][2 * fresh, :nv]
    if idx is not None: ht = xp.take(ht, idx, axis=1)
    return fmm.sum_families([(q[None], wq[None]), (ch_t, ht[None])], self.levels)


def _tcc_elder_fams(self, sc1, idx, ent, fams, cols, V):
    """Elder families of _horner with consecutive same-rank runs merged. Within each maximal block of consecutive elders
    sharing the rank, every run's C^T Y is written into one (block, rows, rank) buffer; then the (source, entry generation)
    groups of the block are appended in the chassis's order, adjacent groups with the same entry generation being one
    family (same items in the same order)."""
    runs = self._elder_runs(sc1); i = 0
    while i < len(runs):
        rank = runs[i][2][0]; j = i
        while j + 1 < len(runs) and runs[j + 1][2][0] == rank: j += 1
        blo, bhi = runs[i][0], runs[j][1]; G = self.lv[rank]["G"]
        if j == i:
            lo, hi, (_, m) = runs[i]; sy = self._elder_sy(lo, hi, rank, m, idx)
        else:   # one buffer for the block: pair runs add straight into their slice, other runs are copied in
            sy = xp.empty((bhi - blo, self.n if idx is None else int(idx.shape[0]), rank), dtype=xp.float32)
            for lo, hi, (_, m) in runs[i:j + 1]:
                if _tcc_sy(self, lo, hi, rank, m, idx, out=sy[lo - blo:hi - blo]) is None:
                    xp.copyto(sy[lo - blo:hi - blo], self._elder_sy(lo, hi, rank, m, idx))
        groups = []   # (a, c, g) in the chassis order: per run, its entry groups
        for lo, hi, _ in runs[i:j + 1]: groups += _equal_groups(lo, hi, ent)
        merged = []
        for a, c, g in groups:
            if merged and merged[-1][2] == g and merged[-1][1] == a: merged[-1][1] = c
            else: merged.append([a, c, g])
        for a, c, g in merged:
            fams[g].append((sy if (a, c) == (blo, bhi) else sy[a - blo:c - blo], cols(V(G[g], (a, c)), g), True))
        i = j + 1

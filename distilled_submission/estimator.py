"""Dense numpy rewrite of the ex_34 estimator (ARC white-box challenge, submission 337479). README.md next to this file
explains the model, the three ideas that carry it, what is kept and dropped, the memory census and every measured decision.

In short: for z_l = h_(l-1) W_l, h_l = relu(z_l), x ~ N(0, I_n), each hidden layer carries the pre-activation mean mu and
covariance pc, the third cumulant through its diagonal z3_i = kappa3(z_i), its (2,1) slice b_ij = kappa3(z_i, z_i, z_j) and
partial trace rt_j = sum_i b_ij, and the fourth cumulant through a trace core A = T + s I plus per-neuron corrections to
q4_i = kappa4(z_i). The ReLU is passed with Gaussian integrals corrected by the first Edgeworth terms; pairs use a 21-row
Mehler series. The input is the sphere (E[chi_n]/sqrt n restores the Gaussian radius) in the unit-prevariance gauge. The
third cumulants created by the ReLU ("sources", one per hidden layer) are stored over their birth-layer neurons ("hubs") as
three (n, n) legs u = C dv, X = extra, P = birth direction,
    B_s = sum_a [ e_a Sym(P_a, u_a, u_a) + (1/2) Sym6(P_a, u_a, X_a) + Sym(P_a, P_a, R_a) + beta_a P_a^(x3) ]
(ex_34's completed square with its spurious F (x) F part removed exactly; R a rank-2 repair plus a conservation column),
transported by Leg -> (Leg o dv) W and read at every later layer: the deposit (z3, b, rt), the trace correction TC with its
two-generation history, the directional response E6 and PK4. A covariance-star component Sym3(v (x) C) rides on the covariance.

Conventions: one function per mechanism, formulas in comments; plain `@` products only, each through `mm` (SVD and solve
through `svd_top`, `solve`) so the FLOPs are tallied; vectors float64, (n, n) legs and coefficient rows float32, float64
accumulators where many terms are summed; `col(v)` / `row(v)` scale rows / columns of a float32 matrix. Sections: 1 unary
law, 2 pair law, 3 covariance star, 4 engine (frame, layer loop, core, birth, terminal), 5 bank (transport, deposit, E6, TC),
6 PK4, 7 entry point.
"""
from __future__ import annotations

import math
import types

import numpy as np

F32, F64 = np.float32, np.float64

TCP_START = 4  # first consumer layer of the trace correction (TC); consumers run from here to the terminal layer
TRACE_HISTORY_DEPTH = 3  # the TCs born at the two previous consumers are re-read at every consumer (chain of 2)
REPAIR_RANK = 2  # SVD rank of the birth repair; one analytic conservation column is added (stored rank 3)
REPEATED_INDEX_RANK = 4  # n // 256: rank of the off-diagonal source slice inside the TC's repeated-index image
HUB_GATE_THRESHOLD, SATURATION_TOLERANCE = 0.01, 0.003  # a hub needs gate > 0.01; gate >= 0.997 keeps only its P leg (README)
TC_HUB_THRESHOLD = 0.04  # the TC sums over the newborn's hubs with gate above this (ex_34's COL rows)

FLOPS = {"matmul": 0.0, "svd": 0.0, "solve": 0.0}  # tally per prediction, read by the runner as Estimator.flops


def mm(a, b):  # a @ b, tallied at 2 (output entries) (contracted length); batched products count every batch
    out = a @ b
    FLOPS["matmul"] += 2.0 * out.size * a.shape[-1]
    return out


def svd_top(a, k):  # exact top-k singular triple (LAPACK), tallied at FlopScope's truncated-SVD tariff 4 m n k
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    FLOPS["svd"] += 4.0 * a.shape[0] * a.shape[1] * k
    return u[:, :k], s[:k], vh[:k]


def solve(a, b):  # float64 linear solve, tallied at (2/3) n^3 + 2 n^2 per right-hand side
    FLOPS["solve"] += 2.0 / 3.0 * a.shape[0] ** 3 + 2.0 * a.shape[0] * b.size
    return np.linalg.solve(np.asarray(a, F64), np.asarray(b, F64))


def col(v): return np.asarray(v, F32)[:, None]  # noqa: E704  M * col(v) scales row i of the float32 matrix M by v_i
def row(v): return np.asarray(v, F32)[None, :]  # noqa: E704  M * row(v) scales column j by v_j
_erf = np.vectorize(math.erf, otypes=[F64])
def normal_cdf(z): return 0.5 * (1.0 + _erf(np.asarray(z, F64) / math.sqrt(2.0)))  # noqa: E704
def normal_pdf(z): return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)  # noqa: E704


# ---- 1. Unary law: Gaussian ReLU integrals, Edgeworth correction, per-neuron statistics -------------------------------------------
def gaussian_relu_basis(sd, z, gate):
    """X ~ N(z sd, sd^2), gate = Phi(z): E relu^k (h, raw2..raw4) and d^k E relu/dmu^k (d, e, f, ds4..ds7) = (-1)^k He_(k-2)(z) phi/sd^(k-1)."""
    phi, zz, inv = normal_pdf(z), z * z, 1.0 / sd
    e = phi * inv
    return dict(
        h=sd * (z * gate + phi), raw2=sd**2 * ((1.0 + zz) * gate + z * phi), raw3=sd**3 * ((3.0 + zz) * z * gate + (2.0 + zz) * phi),
        raw4=sd**4 * ((3.0 + (6.0 + zz) * zz) * gate + (5.0 + zz) * z * phi),
        d=gate, e=e, f=-z * e * inv, ds4=(zz - 1.0) * e * inv**2, ds5=(3.0 - zz) * z * e * inv**3,
        ds6=(3.0 + (zz - 6.0) * zz) * e * inv**4, ds7=(-15.0 + (10.0 - zz) * zz) * z * e * inv**5,
    )


def edgeworth_moments(B, z3, q4):
    """E[g(X)] ~ E_G[g] + (z3/3!) d^3/dmu^3 E_G[g] + (q4/4!) d^4/dmu^4 E_G[g] for m..m4 and the vertices dv (clipped to [0, 1]), ev (>= 0), fv."""
    z36, q424 = z3 / 6.0, q4 / 24.0
    return dict(
        m=B["h"] + z36 * B["f"] + q424 * B["ds4"], m2=B["raw2"] + z3 * B["e"] / 3.0 + q4 * B["f"] / 12.0,
        m3=B["raw3"] + z3 * B["d"] + q4 * B["e"] / 4.0, m4=B["raw4"] + 4.0 * z3 * B["h"] + q4 * B["d"],
        dv=np.clip(B["d"] + z36 * B["ds4"] + q424 * B["ds5"], 0.0, 1.0), ev=np.maximum(B["e"] + z36 * B["ds5"] + q424 * B["ds6"], 0.0),
        fv=B["f"] + z36 * B["ds6"] + q424 * B["ds7"],
    )


def unary_statistics(B, C, gv, ga, z3, q4, stf_scale):
    """Cumulants of relu(X) and the per-neuron pair-law factors; families f, g, c (6, n) = mu-derivatives of E[(relu - m)^p], p = 1..3."""
    h, d, e, f = B["h"], B["d"], B["e"], B["f"]
    m, m2, m3, m4, dv, ev, fv = (C[k] for k in ("m", "m2", "m3", "m4", "dv", "ev", "fv"))
    mm_ = m * m
    D1 = [B[k] for k in ("h", "d", "e", "f", "ds4", "ds5", "ds6", "ds7")]  # derivatives of E relu
    D2 = [B["raw2"]] + [2.0 * r for r in D1[:5]]  # of E relu^2 (d/dmu E relu^p = p E relu^(p-1))
    D3 = [B["raw3"], 3.0 * B["raw2"]] + [6.0 * r for r in D1[:4]]  # of E relu^3
    fam_f = np.stack([D1[k] - (m if k == 0 else 0.0) for k in range(6)])
    fam_g = np.stack([D2[k] - 2.0 * m * D1[k] + (mm_ if k == 0 else 0.0) for k in range(6)])
    fam_c = np.stack([D3[k] - 3.0 * m * D2[k] + 3.0 * mm_ * D1[k] - (mm_ * m if k == 0 else 0.0) for k in range(6)])
    a1, a2 = fam_g[1], fam_g[2]  # first and second mu-derivatives of Var at the fixed centre
    u2, u3 = 2.0 * d - 2.0 * h * e - 2.0 * d * d, e - h * f - d * e  # full derivatives of the Gaussian-centred rows
    return dict(
        ov=m2 - mm_, k3=m3 - 3.0 * m * m2 + 2.0 * mm_ * m, k4=m4 - 4.0 * m * m3 + 12.0 * mm_ * m2 - 6.0 * mm_ * mm_ - 3.0 * m2 * m2,
        a1=a1, a2=a2, ga1=2.0 * h * (1.0 - d), ga2=2.0 * d - 2.0 * h * e,  # variance response at the Gaussian centre
        stf=stf_scale * d - stf_scale * m * e - 2.0 * d * d,  # own-covariance weight of the edge block (layer 0: 2 - 4/(n+2))
        dh=d / 2.0, gah=ga / 2.0, gvh=gv / 2.0, z36=z3 / 6.0, q424=q4 / 24.0, astar=0.5 * ev * gv,  # astar: the fresh core star's vector
        ddv=dv * dv, fv2=2.0 * fv, a1tc=2.0 * h - 2.0 * m * dv, dz=dv * dv * z3, m7=-3.0 * (a2 * d + a1 * e), f=fam_f, g=fam_g, c=fam_c,
        lc=np.stack([2.0 * e, 2.0 * e, e, 2.0 * d]), lr=np.stack([u3, u3, u2, u3]),  # edge-block weights (rows 2, 4, 6, 8)
    )


# ---- 2. Pair law: the 21 Mehler rows and the pair powers (next covariance, (2,1) slice `post`, (3,1) power, (2,2) row sums) ----
# Row q = (r, s, k): coefficient K_rs x^k / k! times E[F^(r+k)] E[G^(s+k)], left/right derivative orders r + k and s + k.
# Matrices 0..15 are distinct; rows 16..20 reuse the matrices 1, x, x^2/2, 1, x with z3_j/6 or q4_j/24 folded into the right vector.
MEHLER_MATRIX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 0, 1]
LEFT_ORDER = [0, 1, 2, 3, 2, 3, 2, 3, 1, 2, 3, 4, 1, 2, 4, 3, 0, 1, 2, 0, 1]
RIGHT_ORDER = [0, 1, 2, 3, 2, 3, 1, 2, 2, 3, 1, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5]
_SCRATCH = {}  # one reusable float64 (n, n) buffer per shape for the pair-power products


def mehler_matrices(x, qq4, bbh, q316):
    """1, x, x^2/2, x^3/6, Q22/4, x Q22/4, b/2, x b/2, b^T/2, x b^T/2, Q31/6, x Q31/6, Q31^T/6, x Q31^T/6, (x^2/2) b/2, (x^2/2) b^T/2."""
    x2 = x * x * 0.5
    return [np.ones_like(x), x, x2, x2 * x / 3.0, qq4, x * qq4, bbh, x * bbh, bbh.T, x * bbh.T, q316, x * q316, q316.T,
            x * q316.T, x2 * bbh, x2 * bbh.T]


def left_factors(fam, z36, q424):
    """ROW_FOLD: rows (3,0,k), (4,0,k) share the matrices of (0,0,k) and fold into its left vector; rows 17, 20, 18 reuse the folded 1, 1, 2."""
    L = [fam[o] for o in LEFT_ORDER]
    for k in range(3):
        L[k] = L[k] + z36 * fam[3 + k] + (q424 * fam[4 + k] if k < 2 else 0.0)
    L[17], L[20], L[18] = L[1], L[1], L[2]
    return L


def right_factors(fam, z36, q424):
    scale = [1.0] * 16 + [z36] * 3 + [q424] * 2
    return [fam[o] * c for o, c in zip(RIGHT_ORDER, scale)]


def pair_power(mats, left, right, rows=range(21)):
    """sum_q M_q o (left_q (x) right_q), accumulated in float64 through one scratch buffer (no per-row temporaries)."""
    out, tmp = np.zeros(mats[0].shape, F64), _SCRATCH.setdefault(mats[0].shape, np.empty(mats[0].shape, F64))
    for q in rows:
        np.multiply(left[q][:, None], right[q][None, :], out=tmp)
        np.multiply(mats[MEHLER_MATRIX[q]], tmp, out=tmp)
        out += tmp
    return out


# ---- 3. Covariance star: the kappa3 component Sym3(v (x) C) carried as one vector v ----------------------------------------------
def star_slice(v, C): return np.outer(np.diagonal(C), v) + 2.0 * v[:, None] * C  # noqa: E704  S_ij = Sym3(v (x) C)_iij = C_ii v_j + 2 v_i C_ij
def star_trace(v, C): return np.trace(C) * v + 2.0 * mm(C, v)  # noqa: E704  t_j = sum_i S_ij = tr(C) v_j + 2 (C v)_j


def star_vector(C, t):  # the v whose partial trace is t: (2 C + tr(C) I) v = t (ex_34's Chebyshev series equals this to eps32)
    C = np.asarray(C, F64)
    return solve(2.0 * C + np.trace(C) * np.eye(C.shape[0]), t)


def star_birth(a, M, Cov, pre_vector, pc, gate):
    """One star Sym3(v' (x) Cov(h)) with the partial trace of the fresh star Sym3(a (x) M) + the inherited Sym3(D v (x) D pc D)."""
    inherited, old_metric = gate * pre_vector, pc * np.outer(gate, gate)
    v = star_vector(Cov, star_trace(a, M) + star_trace(inherited, old_metric))
    return v, star_slice(v, Cov) - star_slice(inherited, old_metric)


# ---- 4. The engine: gauged frame, layer loop, unary + pair law, trace core, birth, terminal mean ---------------------------------
def prepare_frame(W_raw, prev_scale, mean, Cov):
    """Gauged frame: W = W_raw / prev_scale (rows), pc = W^T Cov W (W^T W at layer 0), columns scaled by 1/sqrt(diag pc); gate = Phi(z)."""
    W = W_raw if prev_scale is None else W_raw / col(prev_scale)
    premean = mm(mean, W)
    pc = mm(W.T, W) if Cov is None else mm(mm(W.T, Cov), W)
    scale = 1.0 / np.sqrt(np.diagonal(pc).astype(F64))
    fr = types.SimpleNamespace(W=W * row(scale), premean=premean * scale, pc=pc * col(scale) * row(scale), scale=scale)
    fr.v = np.diagonal(fr.pc).astype(F64)
    fr.sd, fr.z = np.sqrt(fr.v), fr.premean / np.sqrt(fr.v)
    fr.gate = normal_cdf(fr.z)
    return fr


class Engine:
    """One prediction: the layer loop, the trace core (T, s) and the previous layer's kappa4(h) carried across layers."""

    def __init__(self, weights):
        self.W_raw, self.n, self.L = weights, int(weights[0].shape[0]), len(weights)
        self.bank, self.pk4 = Bank(self.n, self.L), PK4(self.n, self.L)
        self.T, self.s_iso = np.zeros((self.n, self.n), F32), 0.0  # trace core A = T + s I (layer 0 uses the sphere's)

    def run(self):
        n, L, bank, pk4 = self.n, self.L, self.bank, self.pk4
        W_gauged = [None] * L
        frame = prepare_frame(self.W_raw[0], None, np.zeros(n), None)  # spherical input: pc = W^T W gauged
        W_gauged[0] = frame.W
        star, rows = np.zeros(n), []
        for l in range(L):
            W = W_gauged[l]
            pre_vector = mm(star, W)  # the covariance-star vector of the previous birth, transported into this layer
            if l == L - 1:
                rows.append(self.terminal_mean(W, mean, pre_vector))
                break
            # (2,1) slice b, kappa3 diagonal z3 and partial trace rt delivered by the bank, plus the covariance star's images
            b, z3, rt = bank.deposit(l, frame, pk4)
            b_star_free = b
            if l:
                b = b + star_slice(pre_vector, frame.pc)
                z3 = z3 + 3.0 * pre_vector * frame.v
                rt = rt + star_trace(pre_vector, frame.pc)
            # scalar kappa4 corrections: the trace correction and PK4 (single site + leg families)
            extra = bank.tc_pending()
            if l:
                gv = (W * W).sum(0, dtype=F64)
                pk = pk4.single_site(self.k4_prev, self.T, self.s_iso, W * W, gv) + pk4.read(bank)
                extra = pk if extra is None else extra + pk
            M = self.advance(l, frame, z3, b, rt, extra)
            mean = M["m"]
            rows.append(mean / frame.scale)
            self.k4_prev = M["k4"]
            source = self.birth(fr=frame, pre_vector=pre_vector, b=b, M=M)
            star = M["star"]
            # phase boundary: TC stash, next frame, directional context, transport of the history, PK4 birth record
            if TCP_START <= l + 1 <= L - 1:
                bank.tc_source = dict(v=frame.v, d=M["dv"], e=M["ev"], a1=M["a1tc"], B=b_star_free)
            if l + 1 < L - 1:
                nxt = prepare_frame(self.W_raw[l + 1], frame.scale, mean, M["cov"])
                W_gauged[l + 1], vnext = nxt.W, nxt.v
            else:  # the terminal layer is not gauged: only its prevariance and the terminal weights are needed
                Wt = self.W_raw[l + 1] / col(frame.scale)
                self.terminal_v = np.einsum("aj,aj->j", Wt, mm(M["cov"], Wt), dtype=F64)
                W_gauged[l + 1], vnext, nxt = Wt, self.terminal_v, None
            if l:
                bank.response_input = dict(u=M["ga1"], pv=pre_vector, rad=M["radial"], ov=M["ov"], g=M["dv"], vnext=vnext,
                                           rho=2.0 * M["kappa"] / (n + 4))
            bank.accumulate(source, l, W_gauged[l + 1], M["dv"])
            pk4.record_birth(l, source, mean, M["dv"], M["ev"], M["fv"], M["ov"])
            frame = nxt
        radius = math.sqrt(2.0) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2)) / math.sqrt(n)  # E[chi_n] / sqrt n
        return radius * np.stack(rows)

    def core_diagonal(self, l, gv, gm=None, W=None):
        """ga = diag(W^T A W) = diag(gm) + s gv (terminal: from W and T directly), plus the bank's directional response E_GA."""
        ga = (np.diagonal(gm).astype(F64) if gm is not None else np.einsum("aj,aj->j", W, mm(self.T, W), dtype=F64)) + self.s_iso * gv
        corr = self.bank.response_corrections.pop(l, None)
        return ga if corr is None else ga + corr

    def advance(self, l, fr, z3, b, rt, extra):
        """One hidden layer: core push, Edgeworth moments, unary statistics, Mehler rows, pair powers, next trace core."""
        n, W, pc = self.n, fr.W, fr.pc
        if rt is None:  # layer 0: no bank trace yet; the partial trace is the column sum of the (zero) slice
            rt = b.sum(0, dtype=F64)
        B = gaussian_relu_basis(fr.sd, fr.z, fr.gate)
        # core push gm = W^T A W, ga = diag gm + E_GA; the metric W^T W is carried as its diagonal gv (ex_34; dense gram x1.34).
        # Layer 0: the sphere's A = -I/(n+2) gives gm = -pc/(n+2) and the dense metric pc.
        if l == 0:
            gm, gram, gv = pc / F32(-(n + 2)), pc, fr.v
        else:
            gv = (W * W).sum(0, dtype=F64)
            gm, gram = mm(mm(W.T, self.T), W), np.diag(gv).astype(F32)
        ga = self.core_diagonal(l, gv, gm=gm)
        np.fill_diagonal(gm, ga)
        q4 = 6.0 * ga * gv + (0.0 if extra is None else extra)  # kappa4 diagonal of z
        C = edgeworth_moments(B, z3, q4)
        S = unary_statistics(B, C, gv, ga, z3, q4, 2.0 - 4.0 / (n + 2) if l == 0 else 2.0)
        # Mehler coefficient matrices. Q22/4 = (ga_i gv_j + gv_i ga_j)/4 + gm o gram; Q31/6 = (gv_i gm_ij + ga_i gram_ij)/2
        x = pc.copy()
        np.fill_diagonal(x, 0.0)
        Q22q = (np.outer(S["gah"], S["gvh"]) + np.outer(S["gvh"], S["gah"])).astype(F32) + gm * gram
        qq4, bbh = Q22q.copy(), (b * 0.5).astype(F32)
        np.fill_diagonal(qq4, 0.0)
        np.fill_diagonal(bbh, 0.0)
        q316 = col(S["gvh"]) * gm + col(S["gah"]) * gram
        mats = mehler_matrices(x, qq4, bbh, q316)
        z36, q424 = S["z36"], S["q424"]
        lf, lg, lc_ = (left_factors(S[k], z36, q424) for k in "fgc")
        rf, rg = right_factors(S["f"], z36, q424), right_factors(S["g"], z36, q424)
        cov, post, pv3 = pair_power(mats, lf, rf), pair_power(mats, lg, rf), pair_power(mats, lc_, rf)
        ov, k3, k4 = S["ov"], S["k3"], S["k4"]
        np.fill_diagonal(cov, ov)
        np.fill_diagonal(post, k3)
        # U_ROWSUM: the (2,2) pair power u = M - ov ov^T - 2 cov o cov (diagonal <- k4) is only read through its row sums
        u = pair_power(mats, lg, rg) - np.outer(ov, ov) - 2.0 * cov * cov
        np.fill_diagonal(u, 0.0)
        urow = u.sum(1) + k4
        cov32 = cov.astype(F32)
        radial = self.update_core(l, fr, B, S, mats, x, gm, gram, gv, ga, bbh, qq4, cov32, pv3, urow, rt)
        return dict(m=C["m"], dv=C["dv"], ev=C["ev"], fv=C["fv"], cov=cov32, post=post.astype(F32), stats=S, gm=gm, Q22q=Q22q,
                    self_coefficient=lg[1], ga1=S["ga1"], a1tc=S["a1tc"], ov=ov, k4=k4, **radial)

    def update_core(self, l, fr, B, S, mats, x, gm, gram, gv, ga, bbh, qq4, cov, pv3, urow, rt):
        """Next core from the mixed fourth trace t_ij = sum_k kappa4(h_i, h_j, h_k, h_k) = Z + Z^T (diagonal <- (2,2) row sums):
        T = t/(n+4), s = -tr(t)/(2(n+2)(n+4)); also the radial image (star fitted to rt) and kappa (E_PROP) for the bank's E6."""
        n, pc, d, v = self.n, fr.pc, fr.gate, fr.v
        a1, a2, ga1, ga2 = S["a1"], S["a2"], S["ga1"], S["ga2"]
        vv = star_vector(pc, rt)  # radial solve: the star vector whose partial trace is the cubic trace
        ca, sr = mm(pc, ga1), ga1 @ vv  # pc a and a . v of the radial image, a = ga1
        e, f = B["e"], B["f"]
        p, q2, bg, btg, qg = mm(x, ga1), mm(mats[2], ga2), mm(bbh, ga1), mm(bbh.T, ga2), mm(qq4, ga2)
        ca1, gma, gra = mm(pc, a1), mm(gm, a1), mm(gram, a1)
        a1d, rho = a1 * d, d * ga2 + e * ga1
        fp = f * p - 2.0 * a1d
        g_w = [vv * d, f * (q2 + bg + qg) + e * btg, fp + sr * e, -3.0 * f, fp, -f, fp, -3.0 * rho, e * p, d * ga1, S["m7"]]
        r_w = [ca * d, d, e, rho, e, rho, d, e, e, -2.0 * f, d]
        gmc = (a2 @ gv / 4.0) * d + e * gra + (0.0 if l == 0 else a2 * gv * d)
        grc = (a2 @ ga / 4.0) * d + e * gma
        cc = e * ca1 - v * (d * (a2 - 2.0 * d * d)) - v * (a1 * e) + (sr / 2.0) * d
        kappa = ((gmc + S["gvh"] * S["m7"]) @ d) / (d @ d)  # E_PROP
        half = pair_power(mats, g_w, r_w, rows=range(11))
        half += pc * np.outer(cc, d) + gm * np.outer(gmc, d) + x * np.outer(vv * e, ca * e) + gram * np.outer(grc, d)
        half += np.outer(0.5 * e * ga, d * gra) + np.outer(0.5 * e * gv, d * gma) + pv3 - (3.0 * S["ov"])[:, None] * cov
        if l < TCP_START:  # edge block: the merged (3,1)-type product; from TCP_START on the TC slice fold delivers this content
            left = sum(mats[q] * np.outer(S["lc"][k], S["lr"][k]) for k, q in enumerate((2, 4, 6, 8)))
            edge_left = left + pc * np.outer(S["dh"], S["stf"])
            half += (mm(edge_left.astype(F32), pc) - left * v[None, :]) * d[None, :]  # minus the k = j term of left @ pc
        trace = half + half.T
        np.fill_diagonal(trace, urow)
        self.s_iso = -np.trace(trace) / (2.0 * (n + 2) * (n + 4))
        self.T = (trace / (n + 4)).astype(F32)
        return dict(radial=(vv, ca, sr, ga1), kappa=kappa)

    def birth(self, fr, pre_vector, b, M):
        """Birth of this layer's source (hub rows a = neurons of this layer): legs u = pc diag(d), X = extra with hub diagonal
        X_aa = coef_a - 2 e_a d_a (the own (0,0,1) term reproduces post's first order), P = identity; repair R = post - images of the
        incoming slice and of the represented source; beta = kappa3(h) - dv^3 z3 - (3 e d^2 + 3 d X_aa); the covariance-star birth
        charges its slice change to R and beta. Hubs: d > 0.01; saturated hubs (d >= 0.997) keep only P, R, beta, their u, X rows
        being dropped after R and beta were formed from the full rows (ex_34; REVIEW S1)."""
        n, pc, S = self.n, fr.pc, M["stats"]
        d, e, f, coef = M["dv"], M["ev"], M["fv"], M["self_coefficient"]
        adm = d > HUB_GATE_THRESHOLD
        sq = adm & (d < 1.0 - SATURATION_TOLERANCE)
        # extra_ij = b_ij f_i d_j + b_ji e_i e_j + pc_ij^2 f_i e_j + (Q22/4)_ij 2 fv_i e_j: incoming slice, Mehler second order and Q22
        extra = b * np.outer(f, d) + b.T * np.outer(e, e) + (pc * pc) * np.outer(f, e) + M["Q22q"] * np.outer(S["fv2"], e)
        np.fill_diagonal(extra, coef - 2.0 * e * d)
        u, X = pc * d[None, :], extra
        Xd = np.diagonal(X).copy()
        # represented (2,1) slice at birth: hub a = i gives (2 e_i d_i + X_ii) u_ij + d_i X_ij, hub a = j gives e_j u_ji^2 + u_ji X_ji
        own = (2.0 * e * d + Xd)[:, None] * u + d[:, None] * X + (u * u).T * e[None, :] + (u * X).T
        R = M["post"].astype(F64) - b * np.outer(S["ddv"], d) - own
        beta = S["k3"] - d * S["dz"] - 3.0 * e * d * d - 3.0 * d * Xd
        u, X = u * sq[:, None], X * sq[:, None]
        star, star_delta = star_birth(S["astar"], M["gm"] * np.outer(d, d), M["cov"], pre_vector, pc, d)
        R = (R - star_delta) * adm[:, None]  # rows of neurons that are not hubs are dropped
        beta = (beta - np.diagonal(star_delta)) * adm
        np.fill_diagonal(R, 0.0)  # the repair's diagonal lives in beta
        M["star"] = star
        src = dict(u=u.astype(F32), X=X.astype(F32), R=R.astype(F32), beta=beta, e=e, coef=coef, adm=adm, sq=sq)
        del extra, u, X, own, R, star_delta  # float64 (n, n) temporaries of the birth
        return src

    def terminal_mean(self, W, mean, pre_vector):
        """h + z3 f/6 + q4 ds4/24 + (z3^2/72) ds6 in network coordinates, z3 = bank diagonal + 3 v_j C_jj, q4 = 6 ga gv + PK4 + TC."""
        bank, tv = self.bank, self.terminal_v
        premean = mm(mean, W)
        gv = (W * W).sum(0, dtype=F64)
        ga = self.core_diagonal(self.L - 1, gv, W=W)
        sd, z = np.sqrt(tv), premean / np.sqrt(tv)
        B = gaussian_relu_basis(sd, z, normal_cdf(z))
        fourth = 6.0 * ga * gv + self.pk4.single_site(self.k4_prev, self.T, self.s_iso, W * W, gv) + self.pk4.read(bank)
        tc = bank.tc_terminal()
        if tc is not None:
            fourth = fourth + tc
        third = bank.terminal_z3 + 3.0 * pre_vector * tv
        return B["h"] + third * B["f"] / 6.0 + fourth * B["ds4"] / 24.0 + (third * third / 72.0) * B["ds6"]


# ---- 5. The cubic source bank: transport, deposit, directional response E6, trace correction TC with its history ----------------
LEGS = ("u", "X", "P")  # per source and hub a: u_a = (C dv)_a (Gaussian leg), X_a = extra_a (coupling leg), P_a = birth direction


def completion_coefficients(quadratic, x):
    """Per column j: the minimum-norm symmetric cubic tensor with zero repeated-index entries and R(., e, e) = quadratic_j/|x_j|^2,
    e = x_j/|x_j|; returns (a, e, |x_j|, e^2), a the multiplier defining R (ex_34 B:783-823, transcribed)."""
    norm = np.sqrt((x * x).sum(0))
    safe = np.where(norm > 0, norm, 1.0)
    e = x / safe
    q = quadratic / (safe * safe)
    e2 = e * e
    diag = e2 * (6.0 * e2 - 4.0) + (1.0 - (e2 * e2).sum(0))
    if np.abs(diag).min() < 1e-10:
        raise ArithmeticError("degenerate distinct-completion diagonal")
    inv = 1.0 / diag
    first, e2i = inv * q, e2 * inv
    g00, g01, g11 = e2i.sum(0), (e2 * e2i).sum(0) - 0.5, (e2 * e2 * e2i).sum(0) - 0.5
    b0, b1 = (e * first).sum(0), (e * e2 * first).sum(0)
    det = g00 * g11 - g01 * g01
    if np.abs(det).min() < 1e-10:
        raise ArithmeticError("singular distinct-completion normal update")
    l0, l1 = (g11 * b0 - g01 * b1) / det, (g00 * b1 - g01 * b0) / det
    return first - inv * (e * l0 + e * e2 * l1), e, safe, e2


def complete_bilinear(quadratic, x, z):
    """The mixed action R(., x_j, z_j) of the completed tensor for every column j (vector operations only; ex_34 B:826-843)."""
    a, e, norm, e2 = completion_coefficients(quadratic, x)
    ez = e * z
    dot, third = ez.sum(0), (e2 * ez).sum(0)
    ae, az, ae2z = (a * e).sum(0), (a * z).sum(0), (a * e2 * z).sum(0)
    out = a * ((1.0 - 2.0 * e2) * dot - 2.0 * ez + 6.0 * e2 * ez - third) + e * ((dot - ez) * ae + (1.0 - e2) * az - 2.0 * ae2z)
    return out * norm


class RepeatedImage:
    """Repeated-index image of a source-layer (2,1) slice B (TC query triples with two equal indices) on directions Y (rows i at the
    source layer, columns j): B~ = off-diagonal B at rank REPEATED_INDEX_RANK, dd = diag B - 3 diag B~ (ex_34's rank; README)."""

    def __init__(self, B):
        off = np.asarray(B, F64).copy()
        diag = np.diagonal(off).copy()
        np.fill_diagonal(off, 0.0)
        u, s, vh = svd_top(off.astype(F32), REPEATED_INDEX_RANK)
        self.u, self.right = u.astype(F64), s[:, None] * vh.astype(F64)
        self.dd = diag - 3.0 * (self.u * self.right.T).sum(1)

    def residual(self, Y):
        y2 = Y * Y
        return (mm(self.right.T, mm(self.u.T, y2))  # k = l:        sum_k B~_ki Y_kj^2
                + 2.0 * Y * mm(self.u, mm(self.right, Y))  # i = k or i = l: 2 Y_ij sum_l B~_il Y_lj
                + self.dd[:, None] * y2)  # i = k = l:    dd_i Y_ij^2


class Bank:
    """The transported sources (README). legs[g][leg][s]: legs of source s, g layers back (legs[-1] is the output buffer);
    coef[leg][s]: C_u = 2 e P o u + P o X, C_X = P o u, C_P = e u o u + u o X + 2 P o R + beta P o P, so b_ij = sum_(s, a, leg)
    C_leg_ai Leg_aj; R = LEFT @ RIGHT (rank-2 repair + conservation column, the one low-rank piece); RREAD = (P o P)^T LEFT."""

    def __init__(self, n, L):
        self.n, self.L, S, self.k = n, L, L - 1, 0  # k: sources admitted so far
        z = lambda *shape: np.zeros(shape, F32)
        self.legs = [{leg: z(S, n, n) for leg in LEGS} for _ in range(TRACE_HISTORY_DEPTH + 1)]
        self.coef = {leg: z(S, n, n) for leg in LEGS}
        self.R, self.PP = z(S, n, n), z(S, n, n)
        self.LEFT, self.RIGHT, self.RREAD = z(S, n, REPAIR_RANK + 1), z(S, REPAIR_RANK + 1, n), z(S, n, REPAIR_RANK + 1)
        self.coef_saved, self.rread_saved = {leg: z(S, n, n) for leg in LEGS}, z(S, n, REPAIR_RANK + 1)  # undo copies (deposit)
        self.beta, self.e = np.zeros((S, n)), np.zeros((S, n))  # hub diagonal and birth curvature per source
        self.W_at, self.gate_at = {}, {}  # per layer: gauged weights into it, corrected gate of it (the Horner history steps)
        self.pending, self.terminal_z3 = None, np.zeros(n)
        self.tc_source = self.tc_state = self.tc_context = None  # newest TC: source-layer stash, transport record, prepared context
        self.tc_chain, self.history_queries, self.history_fold, self.response_corrections = [], None, {}, {}
        self.response_input = self.response_state = self.response_radial = self.directional_fold = self.directional_shared = None

    def accumulate(self, src, l, W, gate):
        """Admit the source born at l and transport every source to t = l + 1: old legs Leg -> (Leg o dv_l) W, the newborn ungated
        (its P is the identity, so its transport is W); then the coefficient rows, z3 and rt at t, and the directional read."""
        n, k_old, new, t = self.n, self.k, self.k, l + 1
        k = self.k = k_old + 1
        self.W_at[t], self.gate_at[l] = W, gate
        resp = self.prepare_directional(k_old) if (self.response_input is not None and k_old) else None
        cur, nxt, g = self.legs[0], self.legs[-1], row(gate)
        for leg in LEGS:
            cur[leg][:k_old] *= g
        self.RIGHT[:k_old] *= g
        cur["u"][new], cur["X"][new] = src["u"], src["X"]
        self.e[new] = src["e"]
        self.fit_repair(src, new)
        right_gated = self.RIGHT[:k].copy()  # the TC reads the source-layer repair factors after gating
        for s in range(k_old):
            for leg in LEGS:
                nxt[leg][s] = mm(cur[leg][s], W)
        nxt["u"][new], nxt["X"][new], nxt["P"][new] = mm(cur["u"][new], W), mm(cur["X"][new], W), W * col(src["adm"])
        z3, trace, represented = np.zeros(n), np.zeros(n), np.zeros(n)
        for s in range(k):
            u, X, P = nxt["u"][s], nxt["X"][s], nxt["P"][s]
            self.RIGHT[s] = mm(right_gated[s], W)
            R = self.R[s] = mm(self.LEFT[s], self.RIGHT[s])
            PP = np.multiply(P, P, out=self.PP[s])
            e = col(self.e[s])
            Pu = P * u
            self.coef["u"][s] = 2.0 * e * Pu + P * X
            self.coef["X"][s] = Pu
            self.coef["P"][s] = e * u * u + u * X + 2.0 * P * R + col(self.beta[s]) * PP
            self.RREAD[s] = mm(PP.T, self.LEFT[s])
            repair_diag = (self.RREAD[s] * self.RIGHT[s].T).sum(1, dtype=F64)
            # z3_j = sum_a 3 e P u^2 + 3 P u X + 3 P^2 R + beta P^3
            z3 += 3.0 * ((Pu * (e * u + X)).sum(0, dtype=F64) + repair_diag) + mm(self.beta[s], PP * P)
            trace += mm(self.RREAD[s].sum(0, dtype=F64), self.RIGHT[s])
            represented += repair_diag
            for leg in LEGS:
                trace += mm(self.coef[leg][s].sum(1, dtype=F64), nxt[leg][s])
                represented += (self.coef[leg][s] * nxt[leg][s]).sum(0, dtype=F64)
        trace = trace - represented + z3  # partial trace rt_j = sum_i b_ij with the diagonal replaced by z3
        if resp is not None:
            self.response_corrections[t] = self.read_directional(resp, k_old, W, gate)
        self.pending = dict(k=k, z3=z3, trace=trace)
        if TCP_START <= t <= self.L - 1:
            self.tc_state = dict(k_in=new, W=W, right_gated=right_gated)
        if t == self.L - 1:
            self.terminal_z3 += z3
        self.legs.insert(0, self.legs.pop())

    def fit_repair(self, src, s):
        """Rank-2 SVD of r plus a conservation column (hub indicator) (x) v restoring every column sum; the fit's diagonal moves into beta."""
        r, adm = src["R"], src["adm"]
        u, sv, vh = svd_top(r, REPAIR_RANK)
        left, right = (u * sv).astype(F64), vh.astype(F64)
        delta = -(left * right.T).sum(1)  # the fit's spurious diagonal
        colsum = r.sum(0, dtype=F64) - mm(left.sum(0), right)  # column sums of the dropped part
        m = adm.sum()
        v = np.where(adm, (colsum - delta) / (m - 1), colsum / m)  # on a hub's own column the diagonal entry is excluded
        self.LEFT[s, :, :REPAIR_RANK], self.LEFT[s, :, REPAIR_RANK] = left, adm
        self.RIGHT[s, :REPAIR_RANK], self.RIGHT[s, REPAIR_RANK] = right, v
        self.beta[s] = (src["beta"] + 3.0 * (delta - v)) * adm

    def deposit(self, t, fr, pk4):
        """Deliver (b, z3, rt) at consumer t: directional fold (this deposit only), PK4 fold, slice; the restore from copies taken
        before the directional fold leaves the PK4 fold only on the newest source's C_u, C_X rows (ex_34's asymmetry)."""
        if self.pending is None:
            return np.zeros((self.n, self.n), F32), np.zeros(self.n), None
        pend, self.pending = self.pending, None
        k, theta = pend["k"], -fr.premean / (3.0 * fr.v)  # theta: the (3,1)-to-(2,1) fold ratio at the Gaussian vertices
        undo = self.apply_directional_fold(k, fr, theta)
        pk4.fold(self, k, theta)
        if self.tc_state is not None:
            block = self.tc_deposit(t, k, theta)
        else:
            block = np.zeros((self.n, self.n), F64)
            for s in range(k):
                for leg in LEGS:
                    block += mm(self.coef[leg][s].T, self.legs[0][leg][s])
        for s in range(k):
            block += mm(self.RREAD[s], self.RIGHT[s])  # the repair's slice sum_a P_ai^2 R_aj
        if self.directional_shared is not None:
            block -= self.directional_shared[:, None] * fr.pc
        np.fill_diagonal(block, pend["z3"])
        for dst, saved in undo:
            dst[...] = saved
        self.directional_fold = self.directional_shared = None
        return block.astype(F32), pend["z3"], pend["trace"]

    # E6 directional response: T(u)_ab = sum_c kappa3(z)_abc u_c along u = ga1, beyond its radial image in the core
    def prepare_directional(self, old):
        """Pre-read at accumulate(l): x_leg = Leg . u, xr = R . u of the old sources, chained by rho = 2 kappa/(n+4) (E_PROP), and
        T(u)_aa from the current rows: dvec = dv^2 sum_(s, hub) [x_u C_u + x_X C_X + x_P C_P + xr P o P]."""
        ctx, n = self.response_input, self.n
        self.response_input = None
        u, rho = ctx["u"], ctx["rho"]
        x = {leg: np.zeros((old, n)) for leg in LEGS}
        xr = np.zeros((old, n))
        for s in range(old):
            for leg in LEGS:
                x[leg][s] = mm(self.legs[0][leg][s], u)
            xr[s] = mm(self.LEFT[s], mm(self.RIGHT[s], u))
        prev = self.response_state
        if prev is not None and prev["k"] == old - 1:
            for leg in LEGS:
                x[leg][: old - 1] += rho * prev["x"][leg]
            xr[: old - 1] += rho * prev["xr"]
            xr[old - 1] -= rho * prev["dvec"]  # the newest leg's P row carries the excluded previous diagonal read
        tdiag = np.zeros(n)
        for s in range(old):
            tdiag += mm(xr[s], self.PP[s]) + sum(mm(x[leg][s], self.coef[leg][s]) for leg in LEGS)
        g2 = ctx["g"] * ctx["g"]
        self.response_state = dict(k=old, x=x, xr=xr, dvec=g2 * tdiag, dvec_net=None)
        return dict(ctx=ctx, g2=g2)

    def read_directional(self, e, old, W, gb):
        """After the transport: dga_j = -(dvec @ W^2 - read_j)/(n+4) from the new rows, net of the radial image of the bank history
        ((v - pv) (x) pc u at first Mehler order); added to the next layer's ga (E_GA)."""
        ctx, X, n = e["ctx"], self.response_state, self.n
        read = np.zeros(n)
        for s in range(old):
            read += mm(X["xr"][s], self.PP[s]) + sum(mm(X["x"][leg][s], self.coef[leg][s]) for leg in LEGS)
        (vv, ca, sr, ua), pv = ctx["rad"], ctx["pv"]
        vb, sb = vv - pv, sr - pv @ ua
        vbg, cag = vb * gb, ca * gb
        tv = mm(np.stack([vbg, cag]), W)
        rad_d = 2.0 * e["g2"] * vb * ca + sb * ctx["ov"]
        X["dvec_net"] = X["dvec"] - rad_d
        read -= 2.0 * tv[0] * tv[1] + sb * ctx["vnext"]
        self.response_radial = (vbg, cag, tv, sb)
        return (mm(X["dvec_net"], W * W) - read) * (-1.0 / (n + 4))

    def apply_directional_fold(self, k, fr, theta):
        """Add theta x the (3,1) image of the directional correction, s = 3 theta gv/(n+4) times each old source's tensor contracted
        with u, to the coefficient rows and RREAD; the newest source's C_P row loses s (dl P + radial terms), which the TC query
        excludes. Returns the undo list [(view, copy)]."""
        S, old, n = self.response_state, k - 1, self.n
        if S is None or old < 1 or S["k"] != old:
            return []
        s = 3.0 * theta * (fr.W * fr.W).sum(0, dtype=F64) / (n + 4)
        s32, C, saved = row(s), self.coef, self.coef_saved
        for leg in LEGS:
            np.copyto(saved[leg][:k], C[leg][:k])
        np.copyto(self.rread_saved[:old], self.RREAD[:old])
        undo = [(C[leg][:old], saved[leg][:old]) for leg in LEGS] + [(self.RREAD[:old], self.rread_saved[:old]), (C["P"][old], saved["P"][old])]
        for a in range(old):
            xu, xX, xP = (col(S["x"][leg][a]) for leg in LEGS)
            xr = col(S["xr"][a])
            u, X, P, R = self.legs[0]["u"][a], self.legs[0]["X"][a], self.legs[0]["P"][a], self.R[a]
            e = col(self.e[a])
            C["u"][a] += (e * (xP * u + xu * P) + 0.5 * (xX * P + xP * X)) * s32
            C["X"][a] += 0.5 * (xu * P + xP * u) * s32
            C["P"][a] += (e * xu * u + 0.5 * (xX * u + xu * X) + xP * R + (xP * col(self.beta[a]) + xr) * P) * s32
            self.RREAD[a] += mm((xP * P * s32).T, self.LEFT[a])
        Pn = self.legs[0]["P"][old]
        dl = col(S["dvec_net"])
        if self.response_radial is not None:
            vbg, cag, tv, sb = self.response_radial
            dcf = (dl * Pn + col(cag) * row(tv[0]) + col(vbg) * row(tv[1])) * s32
            self.directional_shared = sb * s
        else:
            dcf = dl * Pn * s32
        self.directional_fold = dcf
        C["P"][old] -= dcf
        return undo

    # Trace correction TC (consumers t >= TCP_START): the newest source's quadratic response meets the kappa3 it inherited.
    def tc_deposit(self, t, k, theta):
        """TC-layer slice: the old sources' pre-lift query qt (with the history folds) lifted by W together with the newest source's
        rows plus theta x the TC fold, so every reader of b sees b + diag(theta) Q31 (the fold replaces the engine's edge block)."""
        st, new, th = self.tc_state, k - 1, theta[None, :]
        qt, self.history_queries = self.horner_query(t, 1 + len(self.tc_chain), theta, fold=True)
        qt_tc = qt if self.directional_fold is None else qt - self.directional_fold.T
        ctx = self.tc_context = self.prepare_tc(qt_tc, st)
        cur = self.legs[0]
        block = mm((qt + (self.coef["P"][new] + ctx["cpf"] * th).T).astype(F32), st["W"]).astype(F64)
        block += mm((self.coef["u"][new] + ctx["ch"] * th).T.astype(F32), cur["u"][new])
        return block + mm(self.coef["X"][new].T, cur["X"][new])

    def old_repair_read(self, right_gated, k):
        """Old-repair part of a TC query, sum_(s < k) RIGHT_gated[s]^T RREAD[s]^T (rows: source-layer neurons i, columns: consumer j)."""
        out = np.zeros((self.n, self.n), F64)
        for s in range(k):
            out += mm(right_gated[s].T, self.RREAD[s].T)
        return out

    def prepare_tc(self, qt, st):
        """Newest TC: dq_j = 12 sum_i Zp_ij Qt_ij over the newborn's hubs i; Q = (qt^T + old repair read)/dv_i (the old-source slice
        seen by hub i, un-gated; dv = 0 rows become 0), effc = e (u W) - cv W the newborn's quadratic-response leg (M = e u, the exact
        limit of ex_34's sqrt(e)(H - L G); cv = M_ii - a1/2 replaces the own-hub pairing), Zp = W o effc - a1 (x) |W_.j|^2/(2(n+4)),
        Qt = Q - repeated-index image. Fold of theta Q31: cp = 3 effc Qt - 3 cv W Qt + 6 dv x completion plus s W on the P leg,
        s = -3/(n+4) sum_i a1 Qt; ch = 3 e W Qt on the u leg."""
        se, n, kin, W = self.tc_source, self.n, st["k_in"], st["W"]
        d, e, v, a1 = se["d"], self.e[kin], se["v"], se["a1"]
        gdc = np.where(d > 0, d, np.inf)[:, None]
        Q = (qt.T + self.old_repair_read(st["right_gated"], kin)) / gdc
        Y = d[:, None] * W
        cv = e * v * d - 0.5 * a1
        effc = e[:, None] * self.legs[0]["u"][kin] - cv[:, None] * W
        adm = (d > TC_HUB_THRESHOLD)[:, None].astype(F64)  # the newborn's hubs that enter the TC
        Zp = (W * effc - np.outer(a1, (W * W).sum(0, dtype=F64) / (2.0 * (n + 4)))) * adm
        rep = RepeatedImage(se["B"])
        Qt = Q - rep.residual(Y)
        dq = 12.0 * (Zp * Qt).sum(0)
        PQ = W * Qt
        cp = (3.0 * effc * Qt - 3.0 * cv[:, None] * PQ + 6.0 * d[:, None] * complete_bilinear(Qt, Y, Zp)) * adm
        s = -3.0 / (n + 4) * (a1[:, None] * Qt).sum(0)
        return dict(dq=dq, right=st["right_gated"], rep=rep, d=d, e=e, cv=cv, a1=a1, gdc=gdc, k_in=kin, adm=adm,
                    cpf=cp + W * s[None, :], ch=3.0 * e[:, None] * PQ)

    def horner_query(self, t, A, theta, fold):
        """Old-source queries qh_g[j, i] = sum_(src <= t-g-1) sum_(a, leg) C_leg_aj Leg^(t-g)_ai (the slice with one index left at layer
        t-g; g = 1 the newest TC, g = 2, 3 the history TCs), evaluated by nesting: sources older than the window enter at generation
        A with their stored legs; at each later layer s+1 the source born at s enters (P = identity at s, u, X at s+1) and the query
        is lifted by W_(s+1) and gated by dv_(s+1). With `fold`, each history TC's response enters its newborn's rows (hfp, hck)."""
        n, hfp, hck, self.history_fold = self.n, {}, {}, {}
        qh, acc = [None] * (A - 1), np.zeros((n, n), F64)
        for j in range(t - A):
            for leg in LEGS:
                acc += mm(self.coef[leg][j].T, self.legs[A][leg][j])
        for g in range(A, 0, -1):
            s = t - g  # the query now carries legs at layer s
            if g < A:
                b = s - 1
                pb = acc + (self.coef["P"][b] + hfp.pop(b, 0.0)).T
                acc = mm(pb.astype(F32), self.W_at[s]).astype(F64) * self.gate_at[s][None, :]  # transport to layer s, then gate
                acc += mm((self.coef["u"][b] + hck.pop(b, 0.0)).T.astype(F32), self.legs[g]["u"][b])
                acc += mm(self.coef["X"][b].T, self.legs[g]["X"][b])
            if g >= 2:
                qh[g - 2] = acc
                if fold:
                    bt = t - g  # the TC born at consumer t-g+1 reads this query; its newborn is source bt
                    hfp[bt], hck[bt], self.history_fold[bt] = self.history_fold_coefficients(self.tc_chain[g - 2], acc, theta)
        return acc, qh

    def history_legs(self, h, qt2):
        """History TC with newborn b: its un-gated query Q, effective leg Zp and gated direction Y at this consumer (prepare_tc with
        W -> source b's P leg here, its stash at birth)."""
        b, n = h["k_in"], self.n
        Q = (qt2.T + self.old_repair_read(h["right"], b)) / h["gdc"]
        cur = self.legs[0]
        ub, Pb = cur["u"][b].astype(F64), cur["P"][b].astype(F64)
        effc = h["e"][:, None] * ub - h["cv"][:, None] * Pb
        Zp = (Pb * effc - np.outer(h["a1"], (Pb * Pb).sum(0) / (2.0 * (n + 4)))) * h["adm"]
        return Q, Zp, h["d"][:, None] * Pb, Pb, effc

    def history_fold_coefficients(self, h, qt2, theta):
        """Fold coefficients of a history TC (cp + s P on the P leg, ch on u; 3 theta included) and its (Zp, Qt)."""
        Q, Zp, Y, Pb, effc = self.history_legs(h, qt2)
        Qt = Q - h["rep"].residual(Y)
        th = 3.0 * theta[None, :]
        cp = (Qt * (effc - h["cv"][:, None] * Pb) * th + 2.0 * h["d"][:, None] * complete_bilinear(Qt, Y, Zp) * th) * h["adm"]
        s = -(1.0 / (self.n + 4)) * (h["a1"][:, None] * Qt * th).sum(0)
        return cp + Pb * s[None, :], h["e"][:, None] * Pb * Qt * th, (Zp, Qt)

    def tc_pending(self):
        """dq of the newest TC (None before TCP_START) plus the history TCs (from the fold's (Zp, Qt) or read directly); the chain advances."""
        ctx, self.tc_context = self.tc_context, None
        self.tc_state = self.tc_source = None
        if ctx is None:
            self.tc_chain = []
            return None
        dq = ctx["dq"]
        for h, qb in zip(self.tc_chain, self.history_queries or []):
            zq = self.history_fold.get(h["k_in"])
            if zq is None:
                Q, Zp, Y, _, _ = self.history_legs(h, qb)
                zq = (Zp, Q - h["rep"].residual(Y))
            dq = dq + 12.0 * (zq[0] * zq[1]).sum(0)
        self.history_fold, self.history_queries = {}, None
        keep = {key: ctx[key] for key in ("rep", "d", "e", "cv", "a1", "right", "gdc", "k_in", "adm")}
        self.tc_chain = [keep] + self.tc_chain[: TRACE_HISTORY_DEPTH - 2]
        return dq

    def tc_terminal(self):
        """Terminal TC: the query is built in history form on every terminal column; no slice is formed."""
        if self.tc_state is None:
            return None
        qt, self.history_queries = self.horner_query(self.L - 1, 1 + len(self.tc_chain), None, fold=False)
        self.tc_context = self.prepare_tc(qt, self.tc_state)
        return self.tc_pending()


# ---- 6. PK4: per-neuron fourth-cumulant corrections read elementwise from the legs ----------------------------------------------
class PK4:
    """Per source, g' = u + (X - coef P)/(2 e) (ex_34's dressed leg H/sqrt(e) without its self entry; 0 where e = 0 or no square),
    A0 = P g', A1 = P^2, A2 = g'^2: read_j = sum_a [4 c31 A0 A1 + 4 f3 A0 A2 - p_u A1^2 + 6 V A1 A2] + N_j sum_a [tr0 A0 + tr1 A1 + tr2 A2],
    N_j = sum_a A1: the star, u and (3,1) tree-level families minus their trace-core images."""

    def __init__(self, n, L):
        self.n, self.k, S = n, 0, L - 1
        self.live, self.inv2e, self.de = np.zeros((S, n)), np.zeros((S, n)), np.zeros((S, n))  # g' = live u + inv2e X - de P
        self.quartic, self.trace = np.zeros((S, 2, 2, n)), np.zeros((S, 3, n))  # [[4 c31, 4 f3], [-p_u, 6 V]] and the image rows

    def record_birth(self, j, src, m, d, e, f, ov):
        n, live = self.n, ((e > 0) & src["sq"]).astype(F64)  # hubs with a square and positive curvature
        inv2e = live / (2.0 * np.where(e > 0, e, 1.0))
        self.live[j], self.inv2e[j], self.de[j] = live, inv2e, src["coef"] * inv2e
        X = live[:, None] * src["u"] + inv2e[:, None] * src["X"]  # x' at birth (P = identity): its self entry is exactly zero
        np.fill_diagonal(X, 0.0)
        X2 = X * X
        cn = X2.sum(1)  # |x'_a|^2
        np.fill_diagonal(X2, cn)
        six_u = 12.0 * live * (d * (1.0 - d) - m * e)  # 6 V, V = A2 - 2 g^2
        four_c31 = 12.0 * live * (1.0 - d) * (ov - m * m)
        four_f3 = 4.0 * f * live
        tau6 = mm(six_u, X2)  # 6 x the u-family trace diagonal
        tr6 = tau6.sum()
        neg_p = tau6 / (n - 2) - tr6 / (2.0 * (n - 2) * (n - 1))  # -p_u
        self.quartic[j] = [[four_c31, four_f3], [neg_p, six_u]]
        self.trace[j, 0] = (four_c31 + four_f3 * cn) * (-3.0 / (n + 4))
        self.trace[j, 1] = six_u * cn * (-1.0 / (n + 4)) + neg_p * (-6.0 / (n + 4)) + tr6 / (2.0 * (n - 1) * (n + 4))
        self.trace[j, 2] = six_u * (-1.0 / (n + 4))
        self.k = j + 1

    def products(self, bank, s):
        """P, g' and the elementwise products P^2, g'^2, P g' of source s at the current consumer."""
        cur, P = bank.legs[0], bank.legs[0]["P"][s]
        gp = col(self.live[s]) * cur["u"][s] + col(self.inv2e[s]) * cur["X"][s] - col(self.de[s]) * P
        return P, gp, P * P, gp * gp, P * gp

    def read(self, bank):
        out = np.zeros(self.n)
        for s in range(self.k):
            P, gp, A1, A2, A0 = self.products(bank, s)
            q, tr = self.quartic[s], self.trace[s]
            out += mm(q[0, 0], A0 * A1) + mm(q[0, 1], A0 * A2) + mm(q[1, 0], A1 * A1) + mm(q[1, 1], A1 * A2)
            out += A1.sum(0, dtype=F64) * (mm(tr[0], A0) + mm(tr[1], A1) + mm(tr[2], A2))
        return out

    def single_site(self, k4_h, T, s_iso, W2, gv):
        """Exact kappa4(h_a) in place of the core's 6 A_aa: r = kappa4(h) - 6 (T_aa + s), p = (n+4)/(n-2) r - 3/((n-2)(n-1)) sum r,
        read sum_a p W^4 + |W_j|^2 (-6/(n+4) sum_a p W^2 + 3 sum p |W_j|^2/((n+2)(n+4)))."""
        n = self.n
        r = k4_h - 6.0 * (np.diagonal(T).astype(F64) + s_iso)
        p = r * ((n + 4) / (n - 2)) - r.sum() * (3.0 / ((n - 2) * (n - 1)))
        return mm(p, W2 * W2) + gv * (mm(p, W2) * (-6.0 / (n + 4)) + gv * (p.sum() * 3.0 / ((n + 2) * (n + 4))))

    def fold(self, bank, k, theta):
        """Persistent fold: theta (1/4) dF/dp on P and theta (1/4) dF/dg' on g' = live u + inv2e X - de P of the star + u quartic F."""
        n, th = self.n, row(theta)
        for s in range(k):
            P, gp, P2, g2, pg = self.products(bank, s)
            q, tr = self.quartic[s], self.trace[s]
            mixed_image = col(tr[0] - q[0, 0] * (-3.0 / (n + 4)))  # the A0 image row without its (3,1) part
            dir_image, gau_image = col(tr[1]), col(tr[2])
            star, pw, uc = col(q[0, 1]), col(q[1, 0]), col(q[1, 1])
            N = row(P2.sum(0, dtype=F64))
            image_read = row(mm(tr[0] - q[0, 0] * (-3.0 / (n + 4)), pg) + mm(tr[1], P2) + mm(tr[2], g2))
            grad_p = P * (pw * P2 + 0.5 * uc * g2 + 0.5 * (image_read + N * dir_image)) + gp * (0.25 * star * g2 + 0.25 * N * mixed_image)
            grad_g = (P * (0.5 * uc * pg + 0.25 * N * mixed_image) + gp * (0.75 * star * pg + 0.5 * N * gau_image)) * th
            bank.coef["u"][s] += col(self.live[s]) * grad_g
            bank.coef["X"][s] += col(self.inv2e[s]) * grad_g
            bank.coef["P"][s] += grad_p * th - col(self.de[s]) * grad_g


# ---- 7. Entry point (whestbench-style contract: Estimator().predict(mlp, budget) -> (depth, width)) -----------------------------
class Estimator:
    def __init__(self):
        self._fallback_reason = None

    def predict(self, mlp, budget=None):
        weights = [np.ascontiguousarray(np.asarray(w, F32)) for w in mlp.weights]
        for key in FLOPS:
            FLOPS[key] = 0.0
        result = Engine(weights).run()
        self.flops = dict(FLOPS)  # GF = sum / 1e9; multiplier = sum / 2^41
        return np.maximum(result, 0.0).astype(F32)  # ReLU means are non-negative

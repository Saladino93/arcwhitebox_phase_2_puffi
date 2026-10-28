"""PK4: native per-neuron fourth-cumulant (kappa4) diagonal reads from the legs the bank already carries (research arm,
2026-10-07, campaign 2026-10-07_pk4_1p6e9). Computation through flopscope.numpy only; no targets, no fits, no sampling.
Every switch lives in estimator.py (default OFF; with every switch OFF this module is never called).

Conventions (derivations and brute-force checks: ../ALGEBRA.md, ../research/algebra_gate.py). A birth at layer s owns, at a
consumer column j (the pre-activation z_t, t > s), the bank legs P_aj (= beta: the transported hub direction delta_a) and
H_aj (completed square: sqrt(e_a) (C d)_a + F_a, transported). X = H / sqrt(e) is the dressed Gaussian leg gamma INCLUDING
its self entry; gamma' = X - de * P removes the self entry (de_a = X0_aa at birth), so every family below has ZERO
diagonal entries K_aaaa (the diagonal is the single-site term's business) and exact tree-level power-cumulant vertices:

  S  star     K = sum_a f3_a Sym4(d_a, x'_a, x'_a, x'_a)                  read 4  sum_a f3_a P g'^3
  U  u-term   K = sum_a V_a  Sym6(d_a, d_a, x'_a, x'_a),  V = A2 - 2 g^2  read 6  sum_a V_a P^2 g'^2
  G  path     K = sum_{a!=b} e_a e_b C_ab Sym12(d_a, d_b, x'_a, x'_b)    read 12 sum_ab U~_a C_ab U~_b, U~ = e P g'  (age 1)
  A  A1-path  K = sum_{a!=c} A1_a e_c C_ac Sym12(d_a, d_a, d_c, x'_c)     read 12 sum_a A1_a P^2 (C_off U~)_a      (age 1)
  T  (3,1)    K = sum_a c31_a Sym4(x'_a, d_a, d_a, d_a)                  read 4  sum_a c31_a P^3 g'
  Q  (2,2)    K = 1/2 sum_{a!=b} A1_a A1_b C_ab Sym6(d_a, d_a, d_b, d_b)  read 3  sum_{a!=b} (A1 P^2)_a C_ab (A1 P^2)_b  (age 1)
  (A2 = 2g - 2 m e, A1 = 2 m (1 - g): centred-power Stein vertices of X_a^2; c31 = E d/dz X^3 - 3 Var g = 3 (1 - g)(E h^2 - 2 m^2);
   corrected vertices d, e, f, m, Var of the birth.  The G and Q reads share ONE cropped product C_off [U~ | A1 P^2].)

Trace-core image (CORESUB): the chassis core models kappa4(h) as Sym(A (x) I) from the partial trace tau = sum_c K_..cc.
For each family the image is the transported read of its own trace model WITH powers, A_K = (tau_K - diag p_K)/(n+4) + s'_K I,
6 A_K,aa + p_K,a = K_aaaa (= 0), i.e. the Gaussian-star / K4SRC convention of bank.GaussianStars:
  image_K(j) = 6 [ (P_j^T tau_K P_j - sum_a p_a P_aj^2) / (n+4) + s'_K N_j ] N_j + sum_a p_a P_aj^4,   N_j = sum_a P_aj^2.
Single site (SINGLE): the same powers fit for the TOTAL diagonal against the chassis core itself:
  r_a = kappa4(h_a) - 6 (T_aa + s),  dq_j = sum_a p_a W_aj^4 - 6 N_j [ sum_a p_a W_aj^2 / (n+4) - sum_a p_a N_j / (2(n+2)(n+4)) ].
Sign: every returned component is ADDED to the chassis q4 (pre-activation kappa4 diagonal, the engine's gauge)."""
import flopscope.numpy as xp
FOLD = True   # L3 knob (2026-10-17): False skips fold_response (the approximate star+u response closure in every deposit)
import bank
import fmm


def _scalar(value, dtype):
    return xp.asarray(value, dtype=dtype)


def powers(r, n, constants=None):
    """Joint (core, diagonal-powers) fit: p with 6 A'_aa + p_a = K_aaaa given the residual r = K_aaaa - 6 A_aa (A from tau)."""
    K = _fconst(n, r.dtype) if constants is None else constants
    return K["(n+4)/(n-2)"] * r - K["3/((n-2)(n-1))"] * xp.sum(r)


def single_site(k4h, adiag, w, ww, gv, n):
    """Single-site correction at the consumer read through w (rows = h_{t-1}, columns = z_t): exact sum_a W^4 kappa4(h_a)
    replacing the core model's own diagonal (r = k4h - 6 diag A, A = T + s I the chassis core). O(n^2)."""
    dt = w.dtype; K = _fconst(n, dt); p = powers(k4h - 6 * adiag, n, K); w4 = ww * ww
    q1, q2 = xp.matmul(p, w4), xp.matmul(p, ww)
    return q1 - 6 * gv * (q2 / K["n+4"] - xp.sum(p) * gv * K["1/(2(n+2)(n+4))"])


class Birth:
    """Constants of one birth (kept-hub order, hub rows 0..nr-1 of the bank legs). Built once, right after Bank.accumulate."""
    def __init__(self, s, n, q, source, m, d, e, f, pc, nr, nv, fams, coresub, path, htau, ov=None):
        self.s, self.nr = s, nr; dt = e.dtype; self.K = K = _fconst(n, dt)
        kept = source.get("_kept")
        nat = xp.stack([e, f, d, m, source["_mid"], source["_dh"], m if ov is None else ov])
        if kept is not None: nat = xp.take(nat, kept, axis=1)
        eK, fK, dK, mK, mid, dh, ovK = xp.unstack(nat)
        pos = eK > 0; live = pos.astype(dt) * mid
        if nv < nr: live = live * xp.asarray([1.0] * nv + [0.0] * (nr - nv), dtype=dt)
        isq = live / xp.sqrt(xp.where(pos, eK, 1.0)); de = dh * isq
        X0 = xp.einsum("ij,i->ij", source["u"], isq)                     # birth X rows (kept order) x natural columns
        kidx = xp.arange(n) if kept is None else kept
        X0p = xp.copy(X0); xp.put_along_axis(X0p, xp.reshape(kidx, (nr, 1)), 0.0, axis=1)   # x'_a: self entry removed
        cnp = xp.einsum("ij,ij->i", X0p, X0p)
        self.isq, self.de = isq, de
        self.f3, self.e = fK * live, eK * live
        self.V = (2 * dK - 2 * mK * eK - 2 * dK * dK) * live
        self.A1 = 2 * mK * (1 - dK) * live
        self.c31 = 3 * (1 - dK) * (ovK - mK * mK) * live                     # 3 (1 - g)(E h^2 - 2 m^2), E h^2 = Var + m^2
        if coresub and "t31" in fams: self.cT = K["12/(n+4)"] * self.c31
        if coresub and "star" in fams: self.cS = K["12/(n+4)"] * self.f3 * cnp
        if coresub and "u" in fams:   # tau_ii (natural) = (V @ X'^2)_i + V_i cn'_i ; tr tau = 2 sum V cn'
            self.cnpV = self.V * cnp
            hub = self.cnpV if kept is None else xp.take(xp.pad(self.cnpV, [[0, n - nr]]), source["_rank"])
            self.U_img = self._powers(xp.matmul(self.V, X0p * X0p) + hub, 2 * xp.sum(self.cnpV), n, kept, nr, K)
        self.path = None
        if path and ("path" in fams or "a1" in fams or "q22" in fams):   # age-1 path / A1-path / (2,2): hub support Uh in kept rows
            wt = self.e + xp.abs(self.A1)
            cnt = xp.count_nonzero(wt > htau * xp.max(wt))
            plan = bank._key_plan(wt, cnt, nr, q)
            Uh = xp.arange(nr) if plan is None else plan[0]; kU = xp.take(kidx, Uh)
            CU = xp.take(xp.take(pc, kU, axis=0), kU, axis=1); xp.fill_diagonal(CU, 0)
            eU, A1U = xp.take(self.e, Uh), xp.take(self.A1, Uh)
            P = dict(Uh=Uh, CU=CU, eU=eU, A1U=A1U, rho=xp.matmul(CU, A1U))
            if coresub and "q22" in fams:   # (2,2): tau = diag(A1 rho) on Uh, tr = sum A1 rho
                tq = A1U * P["rho"]; nat_q = xp.zeros(n, dtype=dt); xp.put_along_axis(nat_q, kU, tq, axis=0)
                P["Q_img"] = self._powers(nat_q, xp.sum(tq), n, kept, nr, K); P["Q_tau"] = tq
            if coresub == "exact" or coresub == "lite":
                XU = xp.take(X0p, Uh, axis=0); BU = xp.take(XU, kU, axis=1)         # x'_a(b), a, b in Uh (zero diagonal)
                KX = CU * BU                                                         # (C o X0') on the hub block
                tG = 4 * eU * xp.einsum("ab,ba,b->a", KX, BU, eU)                    # tau_ii of the path, i in Uh
                nat_t = xp.zeros(n, dtype=dt); xp.put_along_axis(nat_t, kU, tG, axis=0)
                P["G_img"] = self._powers(nat_t, xp.sum(tG), n, kept, nr, K)
                if coresub == "exact":
                    MU = fmm.gen(XU, XU.T)                                           # M' = x'_a . x'_b (one product per birth)
                    P["KXM"] = xp.concatenate([KX, CU * MU], axis=1)                 # [C o X0' | C o M'] (|Uh|, 2|Uh|)
            self.path = P

    @staticmethod
    def _powers(tau_nat, tr, n, kept, nr, constants=None):
        """(p in kept order, s') of a family with zero diagonal from its trace diagonal (natural order) and trace."""
        K = _fconst(n, tau_nat.dtype) if constants is None else constants
        c = K["1/(2(n+2)(n+4))"]
        p = powers(-6 * (tau_nat / K["n+4"] - tr * c), n, K)
        return (p if kept is None else xp.take(p, kept)), -(tr - xp.sum(p)) * c


def leg_terms(b, H, P, n, fams, coresub, age1, out=None):
    """Components (dict name -> (m,) vectors on the given columns) of birth b from its legs H, P (nr x m) at the consumer.
    Raw reads 'S','U','G','A' and images 'S_img','U_img','GA_img' (CORESUB), suffixed '@1' (age 1) or '@h' (age >= 2);
    the caller adds sum(raw) - sum(img) (see total)."""
    out = {} if out is None else out; K = b.K
    sfx = "@1" if age1 else "@h"
    def acc(name, v): out[name + sfx] = out[name + sfx] + v if name + sfx in out else v
    X = xp.einsum("ij,i->ij", H, b.isq); Gp = X - xp.einsum("ij,i->ij", P, b.de)          # gamma' (self entry removed)
    P2 = P * P; G2 = Gp * Gp
    if coresub: N = xp.sum(P2, axis=0)
    if "star" in fams:
        acc("S", 4 * xp.einsum("i,ij->j", b.f3, P * G2 * Gp))
        if coresub: acc("S_img", N * xp.einsum("i,ij->j", b.cS, P * Gp))
    if "u" in fams:
        acc("U", 6 * xp.einsum("i,ij->j", b.V, P2 * G2))
        if coresub:
            pU, sU = b.U_img
            tr = xp.einsum("i,ij->j", b.V, G2) + xp.einsum("i,ij->j", b.cnpV - pU, P2)
            acc("U_img", 6 * (tr / K["n+4"] + sU * N) * N + xp.einsum("i,ij->j", pU, P2 * P2))
    if "t31" in fams:
        acc("T", 4 * xp.einsum("i,ij->j", b.c31, P2 * P * Gp))
        if coresub: acc("T_img", N * xp.einsum("i,ij->j", b.cT, P * Gp))
    if age1 and b.path is not None and ("path" in fams or "a1" in fams or "q22" in fams):
        D = b.path; Uh = D["Uh"]
        Pu, Gu = xp.take(P, Uh, axis=0), xp.take(Gp, Uh, axis=0)
        eP = xp.einsum("i,ij->ij", D["eU"], Pu); Ut = eP * Gu; m = int(Ut.shape[1])
        if "q22" in fams:                                                                  # one product C_off [U~ | A1 P^2]
            AP = xp.einsum("i,ij->ij", D["A1U"], Pu * Pu); YY = fmm.gen(D["CU"], xp.concatenate([Ut, AP], axis=1))
            Y, YQ = YY[:, :m], YY[:, m:]
            acc("Q", 3 * xp.sum(AP * YQ, axis=0))
            if coresub:
                pQ, sQ = D["Q_img"]
                trq = xp.einsum("i,ij->j", D["Q_tau"], Pu * Pu)
                acc("Q_img", 6 * ((trq - xp.einsum("i,ij->j", pQ, P2)) / K["n+4"] + sQ * N) * N + xp.einsum("i,ij->j", pQ, P2 * P2))
        else: Y = fmm.gen(D["CU"], Ut)                                                     # C_off U~ (the one product)
        if "path" in fams: acc("G", 12 * xp.sum(Ut * Y, axis=0))
        if "a1" in fams: acc("A", 12 * xp.einsum("i,ij->j", D["A1U"], Pu * Pu * Y))
        if coresub and ("path" in fams or "a1" in fams):
            img = 0
            if "path" in fams:
                pG, sG = D["G_img"]
                img = img + 6 * (sG * N - xp.einsum("i,ij->j", pG, P2) / K["n+4"]) * N + xp.einsum("i,ij->j", pG, P2 * P2)
            tr = 0
            if "a1" in fams: tr = tr + 2 * xp.einsum("i,ij->j", D["rho"], Ut)
            if coresub == "exact":
                acc("x:GA_lite_img", img + 6 * tr / K["n+4"] * N)                           # diagnostic: the 'lite' image
                if "path" in fams and "a1" in fams: rhs = [4 * (xp.einsum("i,ij->ij", D["eU"], Gu) + xp.einsum("i,ij->ij", D["A1U"], Pu)), 2 * eP]
                elif "path" in fams: rhs = [4 * xp.einsum("i,ij->ij", D["eU"], Gu), 2 * eP]
                else: rhs = [4 * xp.einsum("i,ij->ij", D["A1U"], Pu), 0 * eP]
                Z = fmm.gen(D["KXM"], xp.concatenate(rhs, axis=0))                         # [C o X0' | C o M'] @ [...]
                tr = tr + xp.sum(eP * Z, axis=0)
            img = img + 6 * tr / K["n+4"] * N
            acc("GA_img", img)
    return out








# ---- fused production read (opsfrugal, 2026-10-07): identical math to Birth / leg_terms / single_site / total for the
# elementwise families (star, u, t31) with the core image, all births of a consumer's window in one set of calls. --------
# Per birth b (kept rows i, zero-padded to n) and consumer column j, with P = beta leg, g' = isq H - de P (self entry removed):
#   A0 = P g', A1 = P^2, A2 = g'^2 (one (3, m, rows, cols) buffer for the window's m births),
#   read_b(j) = sum_i [ 4 c31 A0 A1 + 4 f3 A0 A2 - pU A1 A1 + 6 V A1 A2 ]                    (T, S, U raw; U_img's powers)
#             + N_bj sum_i [ -(cS + cT) A0 + (-6/(n+4) (cnpV - pU) - 6 sU) A1 - 6/(n+4) V A2 ]   (images; N_bj = sum_i A1)
# (P^2 g'^2 = A1 A2 exactly as leg_terms forms P2 * G2). Coefficients live in per-birth rows of persistent buffers:
#   GC[s] = [isq, de], QC[s] = [[4 c31, 4 f3], [-pU, 6 V]] (k over A0, A1; l over A1, A2), NC[s] = the three image rows.
# Powers of the u-image (pk4.Birth._powers) in closed form, with tau6 = 6 tau (natural order, hub term included) and
# tr6 = sum tau6 (= 12 sum V cn' = 6 tr): pU = -tau6/(n-2) + tr6/(2(n-2)(n-1)), sU = -tr6/(12(n-1)(n+4)).
FCONST = (("1", 1, 1), ("-1", -1, 1), ("4", 4, 1), ("12", 12, 1), ("-3/(n+4)", -3, "n+4"), ("-1/(n+4)", -1, "n+4"), ("1/(n-2)", 1, "n-2"),
          ("-1/(2(n-2)(n-1))", -1, "2(n-2)(n-1)"), ("-6/(n+4)", -6, "n+4"), ("1/(2(n-1)(n+4))", 1, "2(n-1)(n+4)"),
          ("(n+4)/(n-2)", "n+4", "n-2"), ("-3/((n-2)(n-1))", -3, "(n-2)(n-1)"), ("3/((n+2)(n+4))", 3, "(n+2)(n+4)"), ("-6", -6, 1))

UNFUSED_CONST = (("12/(n+4)", 12, "n+4"), ("1/(2(n+2)(n+4))", 1, "2(n+2)(n+4)"),
                 ("3/((n-2)(n-1))", 3, "(n-2)(n-1)"))


def _fconst(n, dt, dimension_constants=None):
    """All structural arithmetic is public-xp float64, then cast exactly once."""
    val = bank._dimension_constants(n) if dimension_constants is None else dimension_constants
    names = tuple(val); rows = FCONST + UNFUSED_CONST
    pool = xp.concatenate([xp.ones(1, dtype=xp.float64), xp.stack([val[k] for k in names])])
    lookup = {name: i + 1 for i, name in enumerate(names)}
    num = xp.asarray([0.0 if isinstance(a, str) else a for _, a, _ in rows], dtype=xp.float64)
    for i, (_, a, _) in enumerate(rows):
        if isinstance(a, str): xp.copyto(num[i:i + 1], val[a])
    den = xp.take(pool, [lookup[b] if isinstance(b, str) else 0 for _, _, b in rows])
    v = xp.divide(num, den).astype(dt)
    out = {name: xp.squeeze(v[i:i + 1]) for i, (name, _, _) in enumerate(rows)}
    out["n+4"] = val["n+4"].astype(dt)
    return out   # 0-d arrays: no host scalar extraction


FUSED = True   # research switch: False = the per-birth / per-component reads (Birth, leg_terms, total) for every config


def fused_ok(cfg):   # the fused read covers exactly the production family set with the (lite / exact) core image
    return FUSED and bool(cfg["legs"]) and set(cfg["fams"]) == {"star", "u", "t31"} and bool(cfg["coresub"]) and not cfg.get("record")


class PK4:
    """Per-prediction PK4 state: birth constants and the consumer reads (hidden and terminal)."""
    def __init__(self, engine, n, depth, q, cfg):
        self.engine, self.n, self.depth, self.q, self.cfg, self.births = engine, n, depth, q, cfg, {}
        self.fused = fused_ok(cfg)
        if self.fused:
            dt = xp.float32; s = max(1, depth - 1); self.K = _fconst(n, dt, engine.bank._dim); self.nr = {}
            self.GC, self.QC, self.NC = xp.zeros((s, 2, n), dtype=dt), xp.zeros((s, 2, 2, n), dtype=dt), xp.zeros((s, 3, n), dtype=dt)
            self.ones = xp.ones(n, dtype=dt); self.A = None; self._legs4 = {}; self.sgn = xp.stack([self.K["1"], self.K["-1"]])
    # ---- fused path --------------------------------------------------------------------------------------------------
    def fold_response(self, bank_):
        """Approximate star+u response closure, including recursive TC feedback.

        Quarter-gradient of the represented residual quartic, times -mu/(3v),
        folded into the existing H/P deposit coefficients. This is not an
        exact Q31/all-reader operator. Switched LEG2 rows are stale: only the
        current unswitched sources participate. The cubic diagonal/trace and
        native scalar PK4 owners are not replaced by these coefficients.
        """
        if not FOLD: return   # L3 knob: FOLD False skips the star+u response fold
        if not self.fused:
            raise ValueError("PK4 response fold requires the fused S/U/T owner")
        lo, hi = len(bank_.swL), max(self.nr) + 1 if self.nr else 0
        if hi <= lo: return
        nr = max(self.nr[s] for s in range(lo, hi))
        legs = bank_.LEG2[0]
        H, P = legs[2 * lo:2 * hi:2, :nr], legs[2 * lo + 1:2 * hi:2, :nr]
        isq, de = self.GC[lo:hi, 0, :nr], self.GC[lo:hi, 1, :nr]
        G = isq[:, :, None] * H - de[:, :, None] * P
        Q, NC = self.QC[lo:hi, :, :, :nr], self.NC[lo:hi, :, :nr]
        # Remove t31 and exactly its NC0 core image; retain all star+u images.
        u = (NC[:, 0] - Q[:, 0, 0] * self.K["-3/(n+4)"])[:, :, None]
        v, w = NC[:, 1, :, None], NC[:, 2, :, None]
        b, c, d = Q[:, 0, 1, :, None], Q[:, 1, 0, :, None], Q[:, 1, 1, :, None]
        p2, g2, pg = P * P, G * G, P * G
        N = xp.sum(p2, axis=1)[:, None, :]
        L = xp.sum(u * pg + v * p2 + w * g2, axis=1)[:, None, :]
        d5, b25, b75, Nu25 = 0.5 * d, 0.25 * b, 0.75 * b, 0.25 * (N * u)
        A = P * (c * p2 + d5 * g2 + 0.5 * (L + N * v)) + G * (b25 * g2 + Nu25)
        mu, _, _, _, _, variance = self.engine._gate_cache
        theta = mu / (-3.0 * variance)
        B = (P * (d5 * pg + Nu25) + G * (b75 * pg + 0.5 * (N * w))) * theta
        cu, cp = bank_.COMB2[2 * lo:2 * hi:2, :nr], bank_.COMB2[2 * lo + 1:2 * hi:2, :nr]
        xp.add(cu, isq[:, :, None] * B, out=cu)
        xp.add(cp, A * theta - de[:, :, None] * B, out=cp)

    def _fbirth(self, s, source, m, d, e, f, nr, nv, ov):
        """Write birth s's coefficient rows (kept order, rows :nr) into GC / QC / NC."""
        K, kept = self.K, source.get("_kept")
        col = self.engine.vertex_plans.plan(bank.THIN_TAU)
        dh_src = source["_dh"] if col is None else xp.where(col[1] >= self.n - int(col[0].shape[0]), source["_dh"], 0.0)
        nat = xp.stack([e, f, d, m, source["_mid"], dh_src, m if ov is None else ov])
        if kept is not None: nat = xp.take(nat, kept, axis=1)
        eK, fK, dK, mK, mid, dh, ovK = (nat[i] for i in range(7))                      # basic row views
        pos = eK > 0; live = xp.where(pos, mid, 0.0)                                     # = pos.astype(dt) * mid (mid in {0, 1})
        if nv < nr: xp.copyto(live[nv:], 0.0)
        G, Q, N = self.GC[s, :, :nr], self.QC[s, :, :, :nr], self.NC[s, :, :nr]
        isq = xp.divide(live, xp.sqrt(xp.where(pos, eK, 1.0)), out=G[0])
        xp.multiply(dh, isq, out=G[1])                                                    # de (the read applies its sign)
        X = xp.einsum("ij,i->ij", source["u"], isq)                                      # birth X rows (kept) x natural columns
        if kept is None: xp.fill_diagonal(X, 0)
        else: xp.put_along_axis(X, kept[:, None], 0.0, axis=1)                           # x'_a: self entry removed
        sq = xp.multiply(X, X, out=X); cnp = xp.sum(sq, axis=1)
        if kept is None: xp.fill_diagonal(sq, cnp)                                       # self entries carry cn' (the hub term)
        else: xp.put_along_axis(sq, kept[:, None], cnp[:, None], axis=1)
        omd = 1 - dK
        V6 = xp.einsum("i,i,->i", live, dK * omd - mK * eK, K["12"], out=Q[1, 1])        # 6 V = 12 live (d (1 - d) - m e)
        xp.einsum("i,i,i,->i", live, omd, ovK - mK * mK, K["12"], out=Q[0, 0])          # 4 c31 = 12 live (1 - d)(Var - m^2)
        xp.einsum("i,i,->i", fK, live, K["4"], out=Q[0, 1])                              # 4 f3
        xp.einsum("li,li,->i", Q[0], xp.stack([self.ones[:nr], cnp]), K["-3/(n+4)"], out=N[0])   # -(cT + cS)
        xp.multiply(V6, K["-1/(n+4)"], out=N[2])                                         # -6 V / K["n+4"]
        tau6 = xp.matmul(V6, sq); tr6 = xp.sum(tau6)                                     # 6 x (u trace diagonal, natural), 6 tr
        tK = tau6 if kept is None else xp.take(tau6, kept)
        q10 = xp.multiply(tK, K["1/(n-2)"], out=Q[1, 0])
        xp.add(q10, tr6 * K["-1/(2(n-2)(n-1))"], out=q10)                                # -pU
        n1 = xp.add(xp.einsum("i,i,->i", V6, cnp, K["-1/(n+4)"]), xp.einsum("i,->i", q10, K["-6/(n+4)"]), out=N[1])
        xp.add(n1, tr6 * K["1/(2(n-1)(n+4))"], out=n1)                                   # -6/(n+4) (cnpV - pU) - 6 sU
        self.nr[s] = nr
    def _views4(self, L):   # (sources, 2, n, n) view of a LEG2 generation buffer: [s, 0] = H, [s, 1] = P (made once per buffer)
        v = self._legs4.get(id(L))
        if v is None or v[0] is not L: v = self._legs4[id(L)] = (L, xp.reshape(L, (int(L.shape[0]) // 2, 2) + tuple(int(x) for x in L.shape[1:])))
        return v[1]
    def fconsumer(self, bank_, t, terminal=False):
        """Sum over the window's births of every elementwise read minus its core image, (n,) natural order, or None."""
        c, n = self.cfg, self.n; alo, ahi = c["term_ages"] if terminal else c["ages"]
        lo, hi = max(0, t - ahi), min(t - alo + 1, max(self.nr) + 1 if self.nr else 0)
        if not terminal: lo = min(lo, len(bank_.swL))   # hidden: every birth not yet switched to an elder basis (switched LEG2 rows are stale)
        if hi <= lo: return None
        nrm, m = max(self.nr[s] for s in range(lo, hi)), hi - lo
        # exact_r2: H / P as strided row views of the LEG2 buffer (rows 2s / 2s + 1) instead of a (sources, 2, n, n) reshape of
        # the whole buffer (billed numel once per buffer); g' = isq H - de P as two scalings and one subtraction (same terms)
        L2 = bank_.LEG2[-1] if terminal else bank_.LEG2[0]
        if terminal:
            plan = bank_.term_plan; mc = n if plan is None else int(plan[0].shape[0]); J = None
            H, P = L2[2 * lo:2 * hi:2, :nrm, :mc], L2[2 * lo + 1:2 * hi:2, :nrm, :mc]
        else:
            plan, J = None, None; H, P = L2[2 * lo:2 * hi:2, :nrm], L2[2 * lo + 1:2 * hi:2, :nrm]
            mc = int(H.shape[2])
        # Local scratch died at function return (-768 MB peak RSS vs persistent self.A)
        A = xp.empty((3, m, nrm, mc), dtype=xp.float32)
        Gp = xp.einsum("mi,mij->mij", self.GC[lo:hi, 0, :nrm], H)   # g' = isq H - de P
        xp.subtract(Gp, xp.einsum("mi,mij->mij", self.GC[lo:hi, 1, :nrm], P), out=Gp)
        xp.multiply(P, Gp, out=A[0]); xp.multiply(P, P, out=A[1]); xp.multiply(Gp, Gp, out=A[2])
        # EINSUM_ORDER (2026-10-14): contract the two A slices first, then QC: same terms and bill, ~5x less server time than the
        #   default order (measured 101 -> 20 ms at 7 sources, 181 -> 33 ms at 14; relative difference <= 6e-6)
        v = xp.add(xp.einsum("mkli,kmij,lmij->j", self.QC[lo:hi, :, :, :nrm], A[0:2], A[1:3], optimize=[(1, 2), (0, 1)]),
                   xp.einsum("mki,kmij,mpj->j", self.NC[lo:hi, :, :nrm], A, A[1]))
        if J is not None: v = bank._scatter(v, J[1], idx=J[0])
        if plan is not None: v = bank._scatter(v, plan[1], idx=plan[0])
        return v
    def fsingle(self, k4h, core_t, core_s, ww, gv):
        """single_site with the metered structural constants: p = (n+4)/(n-2) r - 3/((n-2)(n-1)) sum r, r = k4h - 6 (T_aa + s);
        dq = sum_a p W^4 - 6 gv (sum_a p W^2 / K["n+4"] - sum p gv / (2 (n+2)(n+4)))."""
        K = self.K; r = k4h + (core_t.diagonal() + core_s) * K["-6"]
        p = r * K["(n+4)/(n-2)"] + xp.sum(r) * K["-3/((n-2)(n-1))"]
        q1, q2 = xp.einsum("i,ij,ij->j", p, ww, ww), xp.matmul(p, ww)
        return q1 + gv * (q2 * K["-6/(n+4)"] + gv * (xp.sum(p) * K["3/((n+2)(n+4))"]))
    def birth(self, s, source, m, d, e, f, pc, nr, nv, ov=None):
        c = self.cfg
        if not c["legs"]: return
        if self.fused: return self._fbirth(s, source, m, d, e, f, nr, nv, ov)
        self.births[s] = Birth(s, self.n, self.q, source, m, d, e, f, pc, nr, nv, c["fams"], c["coresub"],
                               path=(s + 1 <= self.depth - 2) or c["term"], htau=c["htau"], ov=ov)
        self.births.pop(s - max(c["ages"][1], c["term_ages"][1]) - 1, None)   # births older than every window are never read
    def _cols(self, gate, tau, size):   # uncertain-column plan: min(gate, 1 - gate) > tau, quantum padded (None = all)
        key = xp.minimum(gate, 1 - gate); return bank._key_plan(key, xp.count_nonzero(key > tau), size, self.q)
    def consumer(self, bank_, t, terminal=False):
        """dict of components (natural order, (n,) each) at consumer t. Elementwise families (star, u, t31) on the columns
        min(g, 1-g) > ctau; the age-1 product families (path, a1, q22) on the narrower min(g, 1-g) > ptau (hidden: the
        Gaussian gate of z_t; terminal: inside the bank's terminal plan, gate Phi(terminal z))."""
        c, n = self.cfg, self.n; lo, hi = c["term_ages"] if terminal else c["ages"]
        fe = tuple(x for x in c["fams"] if x in ("star", "u", "t31")); fp = tuple(x for x in c["fams"] if x in ("path", "a1", "q22"))
        if terminal:
            L, plan = bank_.LEG2[-1], bank_.term_plan; m = n if plan is None else int(plan[0].shape[0])
            g = self.engine.cdf(self.engine.terminal_z); g = g if plan is None else xp.take(g, plan[0])
            Je, Jp, outer = None, (self._cols(g, c["ptau"], m) if fp else None), (None if plan is None else plan[1])
            sel = lambda X, J: X[:, :m] if J is None else xp.take(X[:, :m], J[0], axis=1)
        else:
            L, g, outer = bank_.LEG2[0], self.engine.gate_plans.gate, None
            Je, Jp = (self._cols(g, c["ctau"], n) if fe else None), (self._cols(g, c["ptau"], n) if fp else None)
            sel = lambda X, J: X if J is None else xp.take(X, J[0], axis=1)
        ce, cp = {}, {}
        for s in range(max(0, t - hi), t - lo + 1):
            b = self.births.get(s)
            if b is None: continue
            H, P = L[2 * s, :b.nr], L[2 * s + 1, :b.nr]
            if fe: leg_terms(b, sel(H, Je), sel(P, Je), n, fe, c["coresub"], age1=(t - s == 1), out=ce)
            if fp and t - s == 1: leg_terms(b, sel(H, Jp), sel(P, Jp), n, fp, c["coresub"], age1=True, out=cp)
        out = {}
        for comp, J in ((ce, Je), (cp, Jp)):
            for k, v in comp.items():
                if J is not None: v = bank._scatter(v, J[1])
                if outer is not None: v = bank._scatter(v, outer)
                out[k] = out[k] + v if k in out else v
        return out


def total(parts):
    """sum of raw reads minus sum of images (dict of (n,) vectors) -> (n,) or None."""
    out = None
    for k, v in parts.items():
        if k == "single" or "@" not in k or k.startswith("x:"): continue
        sgn = -1 if "_img@" in k else 1
        out = sgn * v if out is None else out + sgn * v
    if "single" in parts: out = parts["single"] if out is None else out + parts["single"]
    return out

"""ARC K=3 SIMPLE in factored form: a numpy, float64 reference estimator.

Recurrence (research/digest_ex34/k3_original/dense_simple.py is the dense oracle; math/ARC_K3_BANKFREE_MATH.md
eqs. 1-7): per post-activation layer the state is (m, C, T, q) with m = E h, C = Cov(h), T = kappa3(h) kept as
symmetric CP factors T = Sym_avg(sum_r a_r (x) b_r (x) c_r) (Sym_avg = average over the 6 index permutations), and
q the radial fourth cumulant, Q_ijkl = (q/3)(d_ij d_kl + d_ik d_jl + d_il d_jk).

Row convention of the benchmark: z_l = h_(l-1) W_l, h_l = relu(z_l), h_(-1) = x ~ N(0, I).
  affine (eq. 1):  mbar = m W,  Cbar = W^T C W,  factors <- W^T factors (every leg),  M = W^T W,  q unchanged.
  nonlinear (eqs. 2-7), with Wick coefficients w_kp = E[d^k relu(Z)^p], Z ~ N(mbar_i, Cbar_ii):
    F_p = w_0p + Tbar_iii w_3p / 6 + Qbar_iiii w_4p / 24,  Qbar_iiii = q M_ii^2
    P_pq,ij = Cbar_ij w_1p,i w_1q,j + Tbar_iij w_2p,i w_1q,j / 2 + Tbar_jji w_1p,i w_2q,j / 2
              + (Cbar_ij^2 / 2 + Qbar_iijj / 4) w_2p,i w_2q,j,   Qbar_iijj = (q/3)(M_ii M_jj + 2 M_ij^2),  i != j
    m' = F_1;  C'_ii = F_2 - m'^2,  C'_ij = P_11;  T'_iii = F_3 - 3 m' F_2 + 2 m'^3;  T'_iij = P_21 - 2 m'_i P_11
    T'_ijk (distinct) = d_i d_j d_k Tbar_ijk + e_i d_j d_k Cbar_ij Cbar_ik + (two more hubs),  d = w_11, e = w_21
    q' = 3 / (n (n+2)) [ sum_i A_i + sum_(i != j) B_ij ],  A, B the kappa4 diagonal and (2,2) faces (eq. 7).
  Factored form of T': gate every factor leg by d in place (carry), append n tree columns (3 e_i delta_i, h_i, h_i)
  with h_ij = Cbar_ij d_j (zero diagonal), and n repair columns (delta_i, delta_i, r_i) with r_ij = 3 (T'_iij -
  Tprov_iij), r_ii = T'_iii - Tprov_iii, where the provisional faces of carry + tree are analytic in the faces read
  for the pair law: Tprov_iij = d_i^2 d_j Tbar_iij + e_j d_i^2 Cbar_ij^2, Tprov_iii = d_i^3 Tbar_iii (tree faces:
  only the hub j contributes to (i,i,j) because h_ii = 0). Rank grows by 2n per layer; faces are read once per layer.
  Two-leg storage (EXACT ARITHMETIC, the duplicate leg is removed, not approximated): every block has two equal legs
  (tree: b = c = h_i; repair: a = b = delta_i) and the same W^T and the same gate d act on every leg at every later
  layer, so the equality persists. The bank stores (x, y) per column: tree (x, y, y), repair (x, x, y). Faces of
  Sym_avg(sum_r a b c): T_iii = sum_r a_i b_i c_i, T_iij = (1/3) sum_r [a_i b_i c_j + a_i b_j c_i + a_j b_i c_i], i.e.
  tree block: T_iii = sum x y^2, T_iij = (1/3)[2 (x o y) y^T + (y o y) x^T];  repair: T_iii = sum x^2 y, T_iij = (1/3)[(x o x) y^T + 2 (x o y) x^T].
The last layer forms only m' = F_1 (needs T_iii only). Everything is float64 (float32 CP face reads cancel badly).

Memory census at n = 1024, L = 16 (float64):
  bank      (2, n, 2n(L-1)) = (2, 1024, 30720)  480 MiB, allocated once; column block l = layer l's tree + repair columns
  n x n     cov, cbar, off, metric, pair_core, p11, p21, p12, p22, k21/repair, k422, h, w   ~13 x 8 MiB = 104 MiB
  largest temporaries: one (n, n) half-block of the bank or an (n, n) product, 8 MiB each (affine chunks 16 MiB)
  expected peak RSS ~0.7 GiB.  No (n, n, k) or n^3 object is ever formed; all bank work is per block or per chunk.
FLOPs at n = 1024: transport 8 n^3 (l-1) + faces 8 n^3 (l-1) + congruences 6 n^3 per layer = 1896 n^3 = 2036 GF (0.93).
`mm` tallies matmul FLOPs as FlopScope would (m n (2k - 1)). Validation: compare_with_dense() against dense_simple.
"""
from __future__ import annotations

import math

import numpy as np

_ERFC = np.frompyfunc(math.erfc, 1, 1)
CHUNK = 2048  # bank columns per temporary: (1024, 2048) float64 = 16 MiB


class FlopTally:
    def __init__(self) -> None:
        self.flops = 0

    def mm(self, a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        m, k = a.shape
        n = b.shape[1]
        self.flops += m * n * (2 * k - 1)
        return np.matmul(a, b, out=out)

    @property
    def gf(self) -> float:
        return self.flops / 1e9


def relu_wick(mean: np.ndarray, var: np.ndarray, k: int, p: int = 1) -> np.ndarray:
    """E[d^k relu(Z)^p], Z ~ N(mean, var), the analytic law of dense_simple.relu_wick (vectorized)."""
    if np.any(~np.isfinite(var)) or np.any(var < 1e-10):
        raise ValueError("variance nonfinite or below the upstream clamp 1e-10")
    sigma = np.sqrt(var)
    alpha = mean / sigma
    phi = np.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * _ERFC(-alpha / math.sqrt(2)).astype(np.float64)
    if k >= p:
        if p > 1:
            return math.factorial(p) * relu_wick(mean, var, k - p + 1)
        if k == 1:
            return cdf
        h0, h1 = np.ones_like(alpha), alpha
        for order in range(1, k - 2):
            h0, h1 = h1, alpha * h1 - order * h0
        hermite = h0 if k == 2 else h1
        return (-1) ** (k - 2) * sigma ** (-(k - 1)) * hermite * phi
    moments = [cdf, phi]
    for j in range(2, p - k + 1):
        moments.append((-alpha) ** (j - 1) * phi + (j - 1) * moments[j - 2])
    power = p - k
    return math.factorial(p) / math.factorial(power) * sigma**power * sum(
        math.comb(power, j) * alpha ** (power - j) * moments[j] for j in range(power + 1)
    )


class K3State:
    """Post-activation state: mean, covariance, the CP factor bank (3, n, capacity) with `rank` used columns, q."""

    def __init__(self, n: int, depth: int) -> None:
        self.mean = np.zeros(n)
        self.cov = np.eye(n)
        self.n = n
        self.bank = np.zeros((2, n, 2 * n * max(1, depth - 1)))  # legs (x, y): tree (x, y, y), repair (x, x, y)
        self.rank = 0
        self.q = 0.0

    def blocks(self):
        """(tree slice, repair slice) of every stored block: tree = (x, y, y), repair = (x, x, y)."""
        n = self.n
        for lo in range(0, self.rank, 2 * n):
            yield slice(lo, lo + n), slice(lo + n, lo + 2 * n)

    def three_legs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expanded (a, b, c) factors, validation only (allocates three (n, rank) arrays)."""
        x, y = self.bank[0, :, : self.rank], self.bank[1, :, : self.rank]
        a, b, c = x.copy(), x.copy(), y.copy()
        for tree, _ in self.blocks():
            b[:, tree] = y[:, tree]
        return a, b, c


def transport_bank_(state: K3State, wt: np.ndarray, tally: FlopTally) -> None:
    """factors <- W^T factors, in place, one column chunk at a time (a column's image depends only on that column)."""
    for leg in range(2):
        for lo in range(0, state.rank, CHUNK):
            hi = min(lo + CHUNK, state.rank)
            state.bank[leg, :, lo:hi] = tally.mm(wt, state.bank[leg, :, lo:hi])


def repeated_faces(state: K3State, tally: FlopTally, want_slice: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """(T_iii, T_iij) of the two-leg bank (docstring), accumulated block by block; two (n, n)(n, n) products per half."""
    x, y = state.bank[0], state.bank[1]
    n = state.n
    t3 = np.zeros(n)
    t21 = np.zeros((n, n)) if want_slice else None
    for tree, rep in state.blocks():
        xy = x[:, tree] * y[:, tree]
        t3 += np.einsum("ij,ij->i", xy, y[:, tree])
        if want_slice:
            t21 += 2.0 * tally.mm(xy, y[:, tree].T)
            t21 += tally.mm(y[:, tree] * y[:, tree], x[:, tree].T)
        xy = x[:, rep] * y[:, rep]
        t3 += np.einsum("ij,ij->i", xy, x[:, rep])
        if want_slice:
            t21 += tally.mm(x[:, rep] * x[:, rep], y[:, rep].T)
            t21 += 2.0 * tally.mm(xy, x[:, rep].T)
    if want_slice:
        t21 /= 3.0
    return t3, t21


def layer_(state: K3State, w: np.ndarray, tally: FlopTally, mean_only: bool = False) -> np.ndarray:
    """Affine step then ReLU nonlinearity, updating `state` in place; returns the new mean. mean_only: last layer."""
    n = len(state.mean)
    wt = w.T
    mbar = state.mean @ w
    cbar = tally.mm(wt, tally.mm(state.cov, w))
    metric = tally.mm(wt, w)
    transport_bank_(state, wt, tally)
    var = np.diag(cbar).copy()
    coef = {(k, p): relu_wick(mbar, var, k, p) for p in range(1, 5) for k in range(5)}
    t3, t21 = repeated_faces(state, tally, want_slice=not mean_only)
    md = np.diag(metric)
    q4diag = state.q * md**2
    f = {p: coef[0, p] + t3 * coef[3, p] / 6 + q4diag * coef[4, p] / 24 for p in range(1, 5)}
    if mean_only:
        return f[1]
    q422 = state.q / 3 * (md[:, None] * md[None, :] + 2 * metric**2)
    off = cbar.copy()
    np.fill_diagonal(off, 0.0)
    pair_core = off**2 / 2 + q422 / 4

    def pair(p, r):
        out = (
            off * coef[1, p][:, None] * coef[1, r][None, :]
            + t21 * coef[2, p][:, None] * coef[1, r][None, :] / 2
            + t21.T * coef[1, p][:, None] * coef[2, r][None, :] / 2
            + pair_core * coef[2, p][:, None] * coef[2, r][None, :]
        )
        np.fill_diagonal(out, 0.0)
        return out

    p11, p21, p12, p22 = pair(1, 1), pair(2, 1), pair(1, 2), pair(2, 2)
    mean = f[1]
    cov = p11.copy()
    np.fill_diagonal(cov, f[2] - mean**2)
    d, e = coef[1, 1], coef[2, 1]
    # cubic: gate the carried legs in place, append the tree and repair blocks, provisional faces analytic
    rank = state.rank
    for leg in range(2):
        state.bank[leg, :, :rank] *= d[:, None]
    h = off * d[None, :]
    prov21 = (d * d)[:, None] * d[None, :] * t21 + e[None, :] * (d * d)[:, None] * off**2
    prov3 = d**3 * t3
    k3 = f[3] - 3 * mean * f[2] + 2 * mean**3
    k21 = p21 - 2 * mean[:, None] * p11
    repair = 3.0 * (k21 - prov21)
    np.fill_diagonal(repair, k3 - prov3)
    tree, rep = slice(rank, rank + n), slice(rank + n, rank + 2 * n)
    eye = np.eye(n)
    state.bank[0, :, tree] = 3.0 * np.diag(e)  # tree (x, y, y) = (3 e_i delta_i, h_i, h_i)
    state.bank[1, :, tree] = h.T
    state.bank[0, :, rep] = eye  # repair (x, x, y) = (delta_i, delta_i, r_i)
    state.bank[1, :, rep] = repair.T
    state.rank = rank + 2 * n
    # radial fourth (eq. 7)
    k422 = p22 - 2 * mean[:, None] * p12 - 2 * mean[None, :] * p21 + 4 * mean[:, None] * mean[None, :] * p11 - 2 * p11**2
    k4diag = f[4] - 4 * mean * f[3] - 3 * f[2] ** 2 + 12 * mean**2 * f[2] - 6 * mean**4
    np.fill_diagonal(k422, k4diag)
    state.q = float(3 * np.sum(k422) / (n * (n + 2)))
    state.mean, state.cov = mean, cov
    return mean


def predict_means(weights: list[np.ndarray], tally: FlopTally | None = None) -> np.ndarray:
    """(depth, n) post-activation means; the last layer is mean-only."""
    tally = tally or FlopTally()
    n, depth = weights[0].shape[0], len(weights)
    state = K3State(n, depth)
    rows = []
    for layer, w in enumerate(weights):
        rows.append(layer_(state, np.asarray(w, dtype=np.float64), tally, mean_only=layer == depth - 1))
    return np.stack(rows)


class Estimator:
    """Harness contract: Estimator().predict(mlp, budget) -> (depth, width) float32; mlp.weights are (n, n) arrays."""

    def __init__(self) -> None:
        self.tally = FlopTally()
        self._fallback_reason = None

    def setup(self, ctx=None) -> None:
        return None

    def predict(self, mlp, budget=None) -> np.ndarray:
        weights = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
        means = predict_means(weights, self.tally)
        self.gf = self.tally.gf
        self.multiplier = self.tally.flops / 2**41
        return np.maximum(means, 0.0).astype(np.float32)


def compare_with_dense(n: int = 12, depth: int = 4, seed: int = 0, dense_path: str | None = None) -> dict:
    """Validate against k3_original/dense_simple.py (column convention z = W x: pass W.T) on random weights."""
    import importlib.util
    import itertools
    import os

    path = dense_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "k3_original", "dense_simple.py")
    spec = importlib.util.spec_from_file_location("dense_simple", path)
    dense = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dense)
    rng = np.random.default_rng(seed)
    weights = [rng.standard_normal((n, n)) * math.sqrt(2.0 / n) for _ in range(depth)]
    tally = FlopTally()
    state = K3State(n, depth + 1)
    ref = dense.SimpleState.gaussian(np.zeros(n), np.eye(n))
    errors = {}
    for layer, w in enumerate(weights):
        layer_(state, w, tally)
        ref = dense.nonlinear(dense.linear(ref, w.T))
        a, b, c = state.three_legs()
        full = np.einsum("ir,jr,kr->ijk", a, b, c)
        full = sum(full.transpose(p) for p in itertools.permutations(range(3))) / 6.0
        errors[layer] = dict(
            mean=float(np.max(np.abs(state.mean - ref.mean))),
            cov=float(np.max(np.abs(state.cov - ref.covariance))),
            cubic=float(np.max(np.abs(full - ref.cubic.data))),
            q=float(abs(state.q - ref.fourth)),
            rank=int(state.rank),
        )
    return errors


if __name__ == "__main__":
    for n, depth in ((8, 3), (16, 5), (32, 4), (64, 3)):
        print(n, depth, compare_with_dense(n, depth))

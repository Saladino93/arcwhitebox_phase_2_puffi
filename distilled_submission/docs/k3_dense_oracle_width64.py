"""Bank-free NumPy reference for ARC K=3 SIMPLE, including pK and radial K4.

This is a semantic rewrite, not a cheap estimator: the current cubic occupies
8*n**3 bytes in float64 and its affine pushforward costs Theta(n**4).
No factor columns, source list, or layer/birth ages survive an update.
Equations specialize ARC's MIT-licensed factor_k3.py/kprop_harmonic.py at
93d091a4c26c042bfffa28f2e76a81bc0aba94bb. See LICENSE. AUGMENT is not implemented.
"""
from dataclasses import dataclass
from itertools import permutations
import math
import numpy as np

MAX_REFERENCE_WIDTH = 64


def check_width(n):
    """Reject benchmark-width allocations: this reference has dense n^4 work."""
    if n < 1 or n > MAX_REFERENCE_WIDTH:
        raise ValueError(f"semantic reference width must be 1..{MAX_REFERENCE_WIDTH}; got {n}")


def symmetrize(a):
    return sum(a.transpose(p) for p in permutations(range(a.ndim))) / math.factorial(a.ndim)


class DenseCubic:
    """Exactly one current symmetric cubic, with no history or factor cache."""

    def __init__(self, data):
        check_width(np.shape(data)[0])
        a = np.array(data, dtype=np.float64, copy=True)
        if a.ndim != 3 or len(set(a.shape)) != 1:
            raise ValueError("expected an n by n by n cubic")
        self.data = a

    @classmethod
    def zeros(cls, n):
        check_width(n)
        return cls(np.zeros((n, n, n)))

    @property
    def n(self):
        return self.data.shape[0]

    def copy(self):
        return DenseCubic(self.data)

    def add_source_(self, a, b, c):
        """Consume Sym(sum_r a_ir b_jr c_kr) immediately; retain no factors."""
        if a.shape != b.shape or a.shape != c.shape or a.shape[0] != self.n:
            raise ValueError("source factors must have the same n by rank shape")
        # Streaming rank columns bounds scratch to a cubic independently of rank.
        for r in range(a.shape[1]):
            raw = np.einsum("i,j,k->ijk", a[:, r], b[:, r], c[:, r])
            self.data += symmetrize(raw)
        return self

    def add_current_(self, other):
        self.data += other.data
        return self

    def affine(self, w):
        """Contract all three legs sequentially (three dense n^4 operations)."""
        check_width(w.shape[0])
        a = np.einsum("ai,ijk->ajk", w, self.data, optimize=True)
        a = np.einsum("bj,ajk->abk", w, a, optimize=True)
        return DenseCubic(np.einsum("ck,abk->abc", w, a, optimize=True))

    def scale_legs_(self, diagonal):
        self.data *= diagonal[:, None, None]
        self.data *= diagonal[None, :, None]
        self.data *= diagonal[None, None, :]
        return self

    def diagonal(self):
        i = np.arange(self.n)
        return self.data[i, i, i].copy()

    def repeated21(self):
        i = np.arange(self.n)
        return self.data[i, i, :].copy()

    def read(self, a, b, c):
        return np.einsum("ijk,i,j,k->", self.data, a, b, c, optimize=True)


@dataclass
class SimpleState:
    mean: np.ndarray
    covariance: np.ndarray
    cubic: DenseCubic
    # K4 = fourth/3 * (delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk).
    fourth: float = 0.0

    @classmethod
    def gaussian(cls, mean, covariance):
        mean = np.asarray(mean, dtype=np.float64)
        return cls(mean.copy(), np.array(covariance, dtype=np.float64, copy=True),
                   DenseCubic.zeros(len(mean)), 0.0)


@dataclass
class PreState:
    mean: np.ndarray
    covariance: np.ndarray
    cubic: DenseCubic
    fourth: float
    metric: np.ndarray


def linear(state, w, bias=None, set_metric=None):
    """Exact ARC affine step, including the independent radial fourth owner."""
    w = np.asarray(w, dtype=np.float64)
    check_width(w.shape[0])
    mean = w @ state.mean
    if bias is not None:
        mean = mean + np.asarray(bias)
    if set_metric is None:
        metric = w @ w.T
    else:
        metric = np.asarray(set_metric, dtype=np.float64)
        if metric.ndim == 0:
            metric = np.eye(w.shape[0]) * metric
        elif metric.ndim == 1:
            metric = np.diag(metric)
    return PreState(mean, w @ state.covariance @ w.T,
                    state.cubic.affine(w), state.fourth, metric)


def relu_wick(mean, variance, k, p=1):
    """E[derivative^k ReLU(Z)^p], same analytic law as ARC, no quadrature."""
    variance = np.asarray(variance)
    if not np.all(np.isfinite(variance)) or np.any(variance < 1e-10):
        raise ValueError("reference rejects nonfinite variance or the upstream clamp-active domain (variance < 1e-10)")
    sigma = np.sqrt(variance)
    alpha = mean / sigma
    phi = np.exp(-alpha * alpha / 2) / math.sqrt(2 * math.pi)
    cdf = np.array([.5*math.erfc(-float(x)/math.sqrt(2)) for x in alpha])
    if k >= p:
        if p > 1:
            return math.factorial(p) * relu_wick(mean, variance, k - p + 1)
        if k == 1:
            return cdf
        h0, h1 = np.ones_like(alpha), alpha
        for order in range(1, k-2):
            h0, h1 = h1, alpha*h1-order*h0
        hermite = h0 if k == 2 else h1
        return (-1)**(k-2) * sigma**(-(k-1)) * hermite * phi
    # Truncated Gaussian moments I_j = E[X^j 1(X > -alpha)].
    moments = [cdf, phi]
    for j in range(2, p-k+1):
        moments.append((-alpha)**(j-1) * phi + (j-1) * moments[j-2])
    power = p-k
    return math.factorial(p) / math.factorial(power) * sigma**power * sum(
        math.comb(power, j) * alpha**(power-j) * moments[j]
        for j in range(power+1))


def nonlinear(pre, wick=relu_wick):
    """Complete K=3 SIMPLE selected nonlinear terms, pK->K, and K4 projection.

    SIMPLE carries the fourth radial harmonic in addition to the full first
    three cumulants. BASE is a different approximation. This is not AUGMENT.
    """
    mu, covariance, cubic = pre.mean, pre.covariance, pre.cubic
    n = len(mu)
    var = np.diag(covariance)
    coef = {(k, p): wick(mu, var, k, p) for p in range(1, 5) for k in range(5)}
    t3 = cubic.diagonal()
    t21 = cubic.repeated21()
    md = np.diag(pre.metric)
    q4diag = pre.fourth * md**2
    q422 = pre.fourth / 3 * (md[:, None]*md[None, :] + 2*pre.metric**2)
    # Single-coordinate power cumulants are raw marginal moments F_p.
    f = {p: coef[0, p] + t3 * coef[3, p]/6 + q4diag*coef[4, p]/24
         for p in range(1, 5)}
    off = covariance.copy()
    np.fill_diagonal(off, 0)

    def pair(p, q):
        out = (off * coef[1, p][:, None] * coef[1, q][None, :]
               + t21 * coef[2, p][:, None] * coef[1, q][None, :]/2
               + t21.T * coef[1, p][:, None] * coef[2, q][None, :]/2
               + (off**2/2 + q422/4)*coef[2, p][:, None]*coef[2, q][None, :])
        np.fill_diagonal(out, 0)
        return out

    p11, p21, p12, p22 = pair(1, 1), pair(2, 1), pair(1, 2), pair(2, 2)
    mean = f[1]
    postcov = p11.copy()
    np.fill_diagonal(postcov, f[2]-mean**2)
    # 111: incoming cubic times three derivative legs + the light two-edge paths.
    post3 = cubic.copy().scale_legs_(coef[1, 1])
    d, e = coef[1, 1], coef[2, 1]
    # One tree source formed in O(n^3), consumed immediately, and discarded.
    # The identity hub leg collapses the rank sum before dense materialization.
    h = off*d[None, :]
    source = e[:, None, None]*h[:, :, None]*h[:, None, :]
    post3.data += source + source.transpose(1, 0, 2) + source.transpose(2, 1, 0)
    # pK conversion overwrites the repeated 21 and 3 faces exactly.
    k21 = p21 - 2*mean[:, None]*p11
    idx = np.arange(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                post3.data[i, i, j] = post3.data[i, j, i] = post3.data[j, i, i] = k21[i, j]
    post3.data[idx, idx, idx] = f[3] - 3*mean*f[2] + 2*mean**3
    # K4's radial projector reads precisely 4 and 22, never a hidden cubic bank.
    k422 = (p22 - 2*mean[:, None]*p12 - 2*mean[None, :]*p21
             + 4*mean[:, None]*mean[None, :]*p11 - 2*p11**2)
    k4diag = f[4] - 4*mean*f[3] - 3*f[2]**2 + 12*mean**2*f[2] - 6*mean**4
    np.fill_diagonal(k422, k4diag)
    fourth = float(3*np.sum(k422)/(n*(n+2)))
    return SimpleState(mean, postcov, post3, fourth)


def propagate(weights, state, biases=None, average_metric=None):
    """Apply ReLU after each supplied matrix. Caller chooses final affine read."""
    for layer, w in enumerate(weights):
        bias = None if biases is None else biases[layer]
        state = nonlinear(linear(state, w, bias=bias, set_metric=average_metric))
    return state

"""Total-degree polynomial basis built from Legendre polynomials.

For inputs in [-1, 1] the tensor-product Legendre basis
    phi_{a,b,c}(x) = P_a(x1) * P_b(x2) * P_c(x3),   a + b + c <= d
spans *exactly* the same function space as the ordinary monomial basis
{x1^a x2^b x3^c : a + b + c <= d}. It is simply a better-conditioned set of
coordinates for the same polynomial, so the fitted model is still a plain
polynomial of total degree d (it can be converted back to monomial
coefficients; see `to_monomial`).
"""
from itertools import product
import numpy as np
from numpy.polynomial import legendre as L


def exponents(n_features: int, degree: int):
    """All exponent tuples with sum <= degree, ordered by total degree."""
    out = [e for e in product(range(degree + 1), repeat=n_features) if sum(e) <= degree]
    out.sort(key=lambda e: (sum(e), [-v for v in e]))
    return np.array(out, dtype=int)


def _legendre_table(x: np.ndarray, degree: int) -> np.ndarray:
    """Return T[k] = P_k(x) for k = 0..degree (shape (degree+1, n))."""
    n = x.shape[0]
    T = np.empty((degree + 1, n))
    T[0] = 1.0
    if degree >= 1:
        T[1] = x
    for k in range(1, degree):  # Bonnet recursion
        T[k + 1] = ((2 * k + 1) * x * T[k] - k * T[k - 1]) / (k + 1)
    return T


class LegendrePolyFeatures:
    def __init__(self, degree: int, features=None):
        self.degree = degree
        self.features = features  # column indices to use (None = all)

    def fit(self, X):
        X = np.asarray(X, float)
        self.features_ = list(range(X.shape[1])) if self.features is None else list(self.features)
        self.exps_ = exponents(len(self.features_), self.degree)
        return self

    def transform(self, X):
        X = np.asarray(X, float)[:, self.features_]
        tables = [_legendre_table(X[:, j], self.degree) for j in range(X.shape[1])]
        Phi = np.ones((X.shape[0], len(self.exps_)))
        for j, T in enumerate(tables):
            Phi *= T[self.exps_[:, j]].T
        return Phi

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def n_terms(self):
        return len(self.exps_)


def to_monomial(exps: np.ndarray, coef: np.ndarray) -> dict:
    """Convert Legendre-basis coefficients to monomial coefficients {exp_tuple: c}."""
    d = exps.max() if len(exps) else 0
    # Legendre -> power series coefficient matrix: P_k(x) = sum_j M[k, j] x^j
    M = np.zeros((d + 1, d + 1))
    for k in range(d + 1):
        e = np.zeros(k + 1); e[k] = 1
        p = L.leg2poly(e)
        M[k, : len(p)] = p
    out = {}
    for e, c in zip(exps, coef):
        if c == 0:
            continue
        # expand product over features
        terms = {(): c}
        for k in e:
            new = {}
            for key, val in terms.items():
                for j in range(k + 1):
                    if M[k, j] != 0:
                        new[key + (j,)] = new.get(key + (j,), 0) + val * M[k, j]
            terms = new
        for key, val in terms.items():
            out[key] = out.get(key, 0) + val
    return out

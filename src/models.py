"""Polynomial regression estimators (all operate on a Legendre total-degree basis).

Every model here is a polynomial in the raw inputs of total degree <= `degree`;
they differ only in how the coefficients are estimated:

  ols      ordinary least squares
  ridge    L2-penalised least squares (intercept unpenalised)
  lasso    L1-penalised least squares (coordinate descent, sklearn)
  relaxed  Lasso used only to choose which terms to keep, then OLS refit on
           those terms (removes the Lasso shrinkage bias)
"""
import warnings
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

from polybasis import LegendrePolyFeatures

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class PolyRegressor:
    def __init__(self, degree, method="ols", alpha=0.0, features=None):
        self.degree = degree
        self.method = method
        self.alpha = alpha
        self.features = features

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.basis_ = LegendrePolyFeatures(self.degree, self.features).fit(X)
        A = self.basis_.transform(X)
        p = A.shape[1]
        self.support_ = np.arange(p)

        if self.method == "ols":
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
        elif self.method == "ridge":
            R = self.alpha * np.eye(p); R[0, 0] = 0.0
            coef = np.linalg.solve(A.T @ A + R, A.T @ y)
        elif self.method in ("lasso", "relaxed"):
            # column 0 is the constant P_0 = 1 -> let sklearn handle the intercept
            m = Lasso(alpha=self.alpha, max_iter=50000, tol=1e-5).fit(A[:, 1:], y)
            coef = np.r_[m.intercept_, m.coef_]
            if self.method == "relaxed":
                S = np.r_[0, 1 + np.flatnonzero(m.coef_)]
                c = np.linalg.lstsq(A[:, S], y, rcond=None)[0]
                coef = np.zeros(p); coef[S] = c
                self.support_ = S
            else:
                self.support_ = np.flatnonzero(coef)
        else:
            raise ValueError(self.method)
        self.coef_ = coef
        return self

    def predict(self, X):
        return self.basis_.transform(np.asarray(X, float)) @ self.coef_

    # ---- persistence -------------------------------------------------------
    def save(self, path):
        np.savez(path, degree=self.degree, method=self.method, alpha=self.alpha,
                 features=np.array(self.basis_.features_), coef=self.coef_,
                 exps=self.basis_.exps_)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = cls(int(z["degree"]), str(z["method"]), float(z["alpha"]), list(z["features"]))
        m.basis_ = LegendrePolyFeatures(m.degree, m.features)
        m.basis_.features_ = list(z["features"]); m.basis_.exps_ = z["exps"]
        m.coef_ = z["coef"]
        m.support_ = np.flatnonzero(m.coef_)
        return m

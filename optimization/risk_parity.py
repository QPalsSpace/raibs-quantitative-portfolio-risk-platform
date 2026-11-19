from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from risk.portfolio_risk import risk_contribution

def equal_risk_contribution(covariance: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    n = len(covariance)
    x0 = np.repeat(1.0 / n, n)
    S = covariance.to_numpy()

    def objective(w):
        sigma = np.sqrt(max(w @ S @ w, 1e-16))
        mrc = S @ w / sigma
        rc = w * mrc
        target = sigma / n
        return float(np.sum((rc - target) ** 2))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1.0}],
        options={"ftol":1e-12,"maxiter":2000},
    )
    if not result.success:
        raise RuntimeError(f"Risk-parity optimization failed: {result.message}")
    return pd.Series(result.x, index=covariance.index)

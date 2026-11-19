from __future__ import annotations
import numpy as np
import pandas as pd

def ewma_covariance(x: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    x = x.dropna(how="any")
    if len(x) < 2:
        raise ValueError("At least two observations are required.")
    S = np.cov(x.iloc[:min(20, len(x))].T, ddof=1)
    if S.ndim == 0:
        S = np.array([[float(S)]])
    for row in x.to_numpy()[20:]:
        v = row.reshape(-1, 1)
        S = lam * S + (1.0 - lam) * (v @ v.T)
    return pd.DataFrame(S, index=x.columns, columns=x.columns)

def covariance_to_correlation(cov: pd.DataFrame) -> pd.DataFrame:
    a = cov.to_numpy(dtype=float)
    d = np.sqrt(np.maximum(np.diag(a), 1e-15))
    corr = a / np.outer(d, d)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)

def ewma_correlation(standardized_returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    return covariance_to_correlation(ewma_covariance(standardized_returns, lam))

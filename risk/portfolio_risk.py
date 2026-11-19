from __future__ import annotations
import numpy as np
import pandas as pd

def portfolio_variance(weights: pd.Series, covariance: pd.DataFrame) -> float:
    w = weights.loc[covariance.index].to_numpy(dtype=float)
    return float(w @ covariance.to_numpy(dtype=float) @ w)

def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame, annualize: bool = False) -> float:
    sigma = np.sqrt(max(portfolio_variance(weights, covariance), 0.0))
    return float(sigma * np.sqrt(252.0) if annualize else sigma)

def marginal_risk_contribution(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    w = weights.loc[covariance.index]
    sigma = portfolio_volatility(w, covariance)
    if sigma <= 0:
        return pd.Series(0.0, index=w.index)
    return covariance @ w / sigma

def risk_contribution(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    return weights.loc[covariance.index] * marginal_risk_contribution(weights, covariance)

def correlation_from_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.maximum(np.diag(covariance.to_numpy()), 1e-15))
    c = covariance.to_numpy() / np.outer(d, d)
    return pd.DataFrame(c, index=covariance.index, columns=covariance.columns)

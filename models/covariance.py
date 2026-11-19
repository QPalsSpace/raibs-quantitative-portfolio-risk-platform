from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from .correlation import ewma_covariance, ewma_correlation

def rolling_covariance(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    x = returns.tail(window).dropna(how="any")
    return x.cov()

def ledoit_wolf_covariance(returns: pd.DataFrame, window: int = 250) -> pd.DataFrame:
    x = returns.tail(window).dropna(how="any")
    model = LedoitWolf().fit(x.to_numpy())
    return pd.DataFrame(model.covariance_, index=x.columns, columns=x.columns)

def garch_ewma_covariance(
    garch_sigmas: pd.Series,
    standardized_returns: pd.DataFrame,
    lam: float = 0.94,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = ewma_correlation(standardized_returns, lam)
    D = np.diag(garch_sigmas.loc[corr.index].to_numpy())
    cov = D @ corr.to_numpy() @ D
    cov = (cov + cov.T) / 2.0
    return (
        pd.DataFrame(cov, index=corr.index, columns=corr.columns),
        corr,
    )

def model_dispersion(covs: dict[str, pd.DataFrame], weights: pd.Series) -> dict[str, float]:
    vols = {}
    for name, cov in covs.items():
        w = weights.loc[cov.index].to_numpy()
        vols[name] = float(np.sqrt(max(w @ cov.to_numpy() @ w, 0.0)))
    vals = np.array(list(vols.values()))
    return {
        "min_vol": float(vals.min()),
        "max_vol": float(vals.max()),
        "dispersion": float(vals.max() - vals.min()),
        "cv": float(vals.std(ddof=0) / vals.mean()) if vals.mean() else 0.0,
        **{f"vol_{k}": v for k, v in vols.items()},
    }

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def _constraints():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

def minimum_variance(covariance: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    n = len(covariance)
    x0 = np.repeat(1.0 / n, n)
    S = covariance.to_numpy()
    result = minimize(
        lambda w: w @ S @ w,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[_constraints()],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    return pd.Series(result.x, index=covariance.index)

def maximum_sharpe(
    covariance: pd.DataFrame,
    annual_returns: pd.Series,
    rf_annual: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    n = len(covariance)
    x0 = np.repeat(1.0 / n, n)
    S = covariance.to_numpy() * 252.0
    mu = annual_returns.loc[covariance.index].to_numpy()
    rf = float(rf_annual)

    def objective(w):
        vol = np.sqrt(max(w @ S @ w, 1e-16))
        return -(w @ mu - rf) / vol

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[_constraints()],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Maximum-Sharpe optimization failed: {result.message}")
    return pd.Series(result.x, index=covariance.index)

def efficient_frontier(
    covariance: pd.DataFrame,
    annual_returns: pd.Series,
    points: int = 60,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """Generate the long-only mean-variance efficient frontier.

    Returns annualized volatility and expected annual return for each
    feasible target-return portfolio. Covariance is expected in daily units.
    """
    if points < 2:
        raise ValueError("points must be at least 2")

    assets = covariance.index
    mu = annual_returns.loc[assets].astype(float)
    annual_cov = covariance.loc[assets, assets].astype(float) * 252.0

    min_var_weights = minimum_variance(covariance, max_weight=max_weight)
    min_var_return = float(min_var_weights.dot(mu))
    max_return = float(mu.max())

    if max_return < min_var_return - 1e-10:
        raise RuntimeError("Efficient-frontier return range is infeasible.")

    targets = np.linspace(min_var_return, max_return, points)
    rows = []
    S = annual_cov.to_numpy()
    mu_array = mu.to_numpy()

    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, target=target: w @ mu_array - target},
        ]
        x0 = min_var_weights.loc[assets].to_numpy(dtype=float)
        result = minimize(
            lambda w: w @ S @ w,
            x0,
            method="SLSQP",
            bounds=[(0.0, max_weight)] * len(assets),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if result.success:
            variance = max(float(result.x @ S @ result.x), 0.0)
            rows.append({
                "expected_return": float(result.x @ mu_array),
                "volatility": float(np.sqrt(variance)),
            })

    frontier = pd.DataFrame(rows).drop_duplicates(subset=["expected_return"])
    if frontier.empty:
        raise RuntimeError("Efficient-frontier optimization produced no feasible points.")
    return frontier.sort_values("expected_return").reset_index(drop=True)


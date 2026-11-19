import numpy as np
import pandas as pd
from risk.portfolio_risk import portfolio_variance, portfolio_volatility, risk_contribution

def test_portfolio_variance():
    S = pd.DataFrame([[0.04,0.01],[0.01,0.09]], columns=["A","B"], index=["A","B"])
    w = pd.Series([0.6,0.4], index=["A","B"])
    expected = 0.6**2*0.04 + 2*0.6*0.4*0.01 + 0.4**2*0.09
    assert abs(portfolio_variance(w,S)-expected) < 1e-12

def test_risk_contributions_sum_to_volatility():
    S = pd.DataFrame([[0.04,0.01],[0.01,0.09]], columns=["A","B"], index=["A","B"])
    w = pd.Series([0.6,0.4], index=["A","B"])
    rc = risk_contribution(w,S)
    assert abs(rc.sum() - portfolio_volatility(w,S)) < 1e-12

def test_efficient_frontier_is_feasible_and_ordered():
    from optimization.mpt import efficient_frontier

    assets = ["A", "B", "C"]
    S = pd.DataFrame(
        [[0.0001, 0.00002, 0.00001],
         [0.00002, 0.000225, 0.00003],
         [0.00001, 0.00003, 0.0004]],
        index=assets,
        columns=assets,
    )
    mu = pd.Series({"A": 0.04, "B": 0.08, "C": 0.13})

    frontier = efficient_frontier(S, mu, points=12)

    assert len(frontier) >= 2
    assert frontier["expected_return"].is_monotonic_increasing
    assert (frontier["volatility"] > 0).all()

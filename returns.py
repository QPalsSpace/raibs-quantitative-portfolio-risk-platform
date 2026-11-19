from __future__ import annotations
import pandas as pd
import numpy as np

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="any")

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="any")

def annualized_mean(returns: pd.DataFrame, periods: int = 252) -> pd.Series:
    return returns.mean() * periods

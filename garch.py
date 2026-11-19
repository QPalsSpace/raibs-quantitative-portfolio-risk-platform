from __future__ import annotations
import pandas as pd
import numpy as np
from arch import arch_model

def fit_garch_models(
    returns: pd.DataFrame, p: int = 1, q: int = 1
) -> tuple[pd.Series, pd.DataFrame]:
    # arch is fitted to percentage returns; final sigma is converted back to decimal units.
    sigmas = {}
    standardized = {}
    for ticker in returns.columns:
        y = (returns[ticker].dropna() * 100.0).astype(float)
        model = arch_model(
            y, mean="Constant", vol="GARCH", p=p, q=q,
            dist="normal", rescale=False
        )
        result = model.fit(disp="off")
        variance_pct2 = float(result.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
        sigmas[ticker] = np.sqrt(max(variance_pct2, 0.0)) / 100.0
        z = result.resid / result.conditional_volatility
        standardized[ticker] = z.rename(ticker)
    z_df = pd.concat(standardized, axis=1).dropna(how="any")
    return pd.Series(sigmas, dtype=float), z_df

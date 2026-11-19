from __future__ import annotations
import pandas as pd
import yfinance as yf

def download_adjusted_close(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No market data returned by yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise RuntimeError("Adjusted Close/Close field was not returned.")
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index().dropna(how="all")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise RuntimeError(f"Missing requested tickers: {missing}")
    return prices[tickers].dropna(how="all")

def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    # Complete-case alignment is intentional for covariance estimation.
    return returns.dropna(how="any")

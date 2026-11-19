from __future__ import annotations
import json
import pandas as pd
from config.settings import settings
from data.adapters.yfinance_adapter import download_adjusted_close, prices_to_returns
from models.garch import fit_garch_models
from models.covariance import (
    rolling_covariance, ledoit_wolf_covariance, garch_ewma_covariance, model_dispersion
)
from risk.portfolio_risk import portfolio_volatility, risk_contribution
from optimization.mpt import minimum_variance, maximum_sharpe, efficient_frontier
from optimization.risk_parity import equal_risk_contribution

def load_portfolios() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_excel(settings.portfolio_file, sheet_name="Portfolio_Summary")
    holdings = pd.read_excel(settings.portfolio_file, sheet_name="Portfolio_Holdings")
    return summary, holdings

def run_once() -> dict:
    summary, holdings = load_portfolios()
    tickers = sorted(holdings["ticker"].unique().tolist())
    prices = download_adjusted_close(tickers, settings.history_period)
    returns = prices_to_returns(prices).tail(settings.max_obs)

    garch_sigma, z = fit_garch_models(returns, settings.garch_p, settings.garch_q)
    cov_garch, corr_garch = garch_ewma_covariance(garch_sigma, z, settings.ewma_lambda)
    cov_roll = rolling_covariance(returns, settings.rolling_window)
    cov_ewma = __import__("models.correlation", fromlist=["ewma_covariance"]).ewma_covariance(
        returns, settings.ewma_lambda
    )
    cov_lw = ledoit_wolf_covariance(returns, min(250, len(returns)))
    annual_mu = returns.mean() * 252.0

    portfolio_results = []
    for pid in summary["portfolio_id"]:
        h = holdings[holdings["portfolio_id"] == pid]
        w = h.set_index("ticker")["weight"].astype(float)
        covs = {
            "garch_ewma": cov_garch.loc[w.index, w.index],
            "rolling": cov_roll.loc[w.index, w.index],
            "ewma": cov_ewma.loc[w.index, w.index],
            "ledoit_wolf": cov_lw.loc[w.index, w.index],
        }
        disp = model_dispersion(covs, w)
        rc = risk_contribution(w, covs["garch_ewma"])
        portfolio_results.append({
            "portfolio_id": pid,
            "portfolio_name": h["portfolio_name"].iloc[0],
            "annualized_volatility": portfolio_volatility(w, covs["garch_ewma"], annualize=True),
            "daily_volatility": portfolio_volatility(w, covs["garch_ewma"]),
            "risk_contribution_sum": float(rc.sum()),
            "expected_return": float(w.dot(annual_mu.loc[w.index])),
            **disp,
        })

    opt_assets = summary.loc[summary["portfolio_id"].isin(["P28","P29","P31"])]
    opt_names = opt_assets["portfolio_id"].tolist()
    universe = sorted(set(holdings[holdings["portfolio_id"].isin(opt_names)]["ticker"]))
    S = cov_garch.loc[universe, universe]
    mv = minimum_variance(S)
    ms = maximum_sharpe(S, annual_mu, rf_annual=settings.rf_annual)
    rp = equal_risk_contribution(S)
    frontier = efficient_frontier(S, annual_mu.loc[universe], points=60)
    optimization = {
        "minimum_variance": mv.to_dict(),
        "maximum_sharpe": ms.to_dict(),
        "risk_parity": rp.to_dict(),
    }

    optimization_metrics = {}
    for name, weights in optimization.items():
        w = pd.Series(weights, dtype=float).reindex(universe).fillna(0.0)
        daily_variance = float(w.to_numpy() @ S.to_numpy() @ w.to_numpy())
        annual_vol = float((max(daily_variance, 0.0) * 252.0) ** 0.5)
        expected_return = float(w.dot(annual_mu.loc[universe]))
        optimization_metrics[name] = {
            "expected_return": expected_return,
            "annualized_volatility": annual_vol,
        }

    snapshot = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "assets": tickers,
        "observations": len(returns),
        "garch_sigma": garch_sigma.to_dict(),
        "expected_returns": annual_mu.to_dict(),
        "correlation": corr_garch.to_dict(),
        "portfolios": portfolio_results,
        "optimization": optimization,
        "optimization_metrics": optimization_metrics,
        "efficient_frontier": frontier.to_dict(orient="records"),
    }
    out = settings.output_dir / "latest_snapshot.json"
    out.write_text(json.dumps(snapshot, indent=2, default=float), encoding="utf-8")
    return snapshot

if __name__ == "__main__":
    result = run_once()
    print(f"Snapshot generated for {len(result['portfolios'])} portfolios and {len(result['assets'])} assets.")

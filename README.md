# RAIBS Dynamic Portfolio Risk & Optimization Engine

## 1. Objective

The engine estimates time-dependent individual volatility, dynamic correlation, covariance matrices, portfolio volatility and risk contributions, then provides MPT and risk-parity optimization.

Primary identity:

\[
\Sigma_t = D_t R_t D_t
\]

and:

\[
\sigma_{P,t} = \sqrt{w_t^\top \Sigma_t w_t}.
\]

The design follows a principle of sufficiency: every module must contribute directly to estimation, verification, attribution, optimization, API delivery or operational execution.

## 2. Chosen quantitative architecture

1. Adjusted daily prices from a provider adapter.
2. Simple returns in decimal units.
3. GARCH(1,1) per asset for conditional volatility.
4. Standardized GARCH residuals.
5. EWMA covariance/correlation of standardized residuals.
6. Dynamic covariance:
   \(\Sigma_t=D_tR_tD_t\).
7. Verification covariance models:
   rolling, EWMA and Ledoit-Wolf.
8. Portfolio variance and volatility.
9. Marginal and component risk contributions.
10. MPT minimum-variance and maximum-Sharpe optimization.
11. Equal-risk-contribution/risk-parity optimization.
12. JSON snapshot, FastAPI endpoint and Dash dashboard.
13. Long-only efficient-frontier analysis using the same historical return and covariance assumptions as MPT.
14. Continuous runner controlled by an environment variable.

GARCH is the primary econometric volatility model. The architecture does not call GARCH "machine learning"; ML challengers can be added later only if they materially improve forecasting.

## 2.1 Efficient-frontier analysis

The engine uses the same annualized historical mean-return vector already used by Maximum Sharpe:

\[\mu = 252\,\bar r\].

The annual covariance matrix is obtained from the daily primary covariance matrix as:

\[\Sigma_{annual}=252\Sigma_{daily}.\]

For a target expected return \(r^*\), the frontier solves:

\[\min_w\; w^\top\Sigma_{annual}w\]

subject to:

\[\sum_i w_i=1,\qquad w_i\ge0,\qquad w^\top\mu=r^*.\]

The snapshot now exposes `expected_returns`, `expected_return` for each workbook portfolio, `optimization_metrics`, and `efficient_frontier`. The Dash dashboard presents these through a discreet **EFFICIENCY ANALYSIS** button and modal rather than adding another permanent navigation tab.

This is an analytical risk-return view, not a claim that historical mean returns are reliable forecasts. The interpretation should therefore remain conditional on the return-estimation methodology, covariance model, constraints, turnover, liquidity and concentration controls.

## 3. Windows + Visual Studio Code: first deployment

### Recommended environment

Use **Python 3.11** for the first deployment. The application is ordinary Python modules; Jupyter is optional and is installed only because it is useful for research diagnostics.

### Method A — VS Code integrated PowerShell terminal

Open VS Code, open this project folder, then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m ipykernel install --user --name raibs-risk --display-name "Python (RAIBS Risk)"
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

Select the interpreter:

**Ctrl+Shift+P → Python: Select Interpreter → `.venv\Scripts\python.exe`**

For notebooks:

**Ctrl+Shift+P → Notebook: Select Notebook Kernel → `Python (RAIBS Risk)`**

### Method B — VS Code Command Prompt terminal

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m ipykernel install --user --name raibs-risk --display-name "Python (RAIBS Risk)"
```

### Method C — Anaconda/Miniconda

```powershell
conda create -n raibs-risk python=3.11 -y
conda activate raibs-risk
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name raibs-risk --display-name "Python (RAIBS Risk)"
```

For this project, do not mix a Conda environment and `.venv` accidentally. Pick one environment per project.

## 4. VS Code extensions

Install:

- Python
- Pylance
- Jupyter
- Python Debugger
- GitLens (optional)

The project does **not** require a special kernel to run `.py` modules. A kernel is required only for Jupyter notebooks.

## 5. First execution

From the project root:

```powershell
python run_once.py
```

This downloads the configured historical sample, fits the GARCH models, constructs the dynamic covariance matrix and writes:

```text
outputs/latest_snapshot.json
```

Then launch the dashboard:

```powershell
python app/dashboard.py
```

For the API:

```powershell
uvicorn api.main:app --reload
```

Continuous operation:

```powershell
python runner.py
```

The refresh interval is controlled by:

```text
RAIBS_REFRESH_SECONDS
```

Do not interpret a 10-second loop against daily Yahoo data as true intraday risk. For production daily risk, schedule a post-close run. True intraday risk requires an intraday-capable data feed and appropriate market-calendar handling.

## 6. Validation

Run:

```powershell
python -m compileall .
```

and, if pytest is installed:

```powershell
python -m pytest -q
```

Check that:

- portfolio weights sum to 1;
- covariance matrices are symmetric;
- portfolio variance is non-negative up to numerical tolerance;
- risk contributions sum to portfolio volatility;
- optimization weights sum to 1 and respect bounds;
- snapshot timestamp advances on each successful run.

## 7. Unit conventions

Prices are in provider currency.

Returns are decimal:

\[
r_t = \frac{P_t}{P_{t-1}}-1.
\]

GARCH is fitted to percentage returns for numerical convention:

\[
100r_t.
\]

The forecast is converted back to decimal volatility before covariance construction.

Daily covariance has units of decimal-return squared. Annualized volatility is:

\[
\sigma_{\text{annual}}=\sigma_{\text{daily}}\sqrt{252}.
\]

Never annualize twice.

## 8. Operational architecture

```text
Provider
   ↓
Data Adapter
   ↓
Prices → Returns
   ↓
GARCH Volatility ─────────┐
   ↓                      │
Standardized Residuals    │
   ↓                      │
EWMA Correlation ─────────┤
                          ↓
                    Σ_t = D_t R_t D_t
                          ↓
       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
Portfolio Risk      Verification       Optimization
       ↓                  ↓                  ↓
       └──────────────────┼──────────────────┘
                          ↓
                    JSON Snapshot
                     ↙          ↘
                  Dash          API
```

## 9. Provider abstraction

`data/adapters/yfinance_adapter.py` is deliberately isolated. A later institutional provider should implement the same conceptual interface without rewriting the risk engine.

## 10. Portfolio test set

The workbook contains 31 cases. P01–P30 reproduce the supplied compositions. P31 is added as a Risk-Parity Candidate to provide a dedicated test case for equal-risk-contribution optimization.

The workbook is therefore an input specification, not a claim that P28/P29/P31 are already optimized portfolios. The optimization engine computes the optimized weights from the current covariance/return estimates.

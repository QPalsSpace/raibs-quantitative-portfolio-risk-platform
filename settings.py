from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    history_period: str = os.getenv("RAIBS_HISTORY_PERIOD", "5y")
    max_obs: int = int(os.getenv("RAIBS_MAX_OBS", "1250"))
    garch_p: int = int(os.getenv("RAIBS_GARCH_P", "1"))
    garch_q: int = int(os.getenv("RAIBS_GARCH_Q", "1"))
    ewma_lambda: float = float(os.getenv("RAIBS_EWMA_LAMBDA", "0.94"))
    rolling_window: int = int(os.getenv("RAIBS_ROLLING_WINDOW", "60"))
    rf_annual: float = float(os.getenv("RAIBS_RF_ANNUAL", "0.00"))
    refresh_seconds: int = int(os.getenv("RAIBS_REFRESH_SECONDS", "3600"))
    portfolio_file: Path = ROOT / os.getenv(
        "RAIBS_PORTFOLIO_FILE", "portfolios/RAIBS_PORTFOLIO_RISK.xlsx"
    )
    output_dir: Path = ROOT / "outputs"
    cache_dir: Path = ROOT / "data" / "cache"

settings = Settings()
settings.output_dir.mkdir(exist_ok=True)
settings.cache_dir.mkdir(parents=True, exist_ok=True)

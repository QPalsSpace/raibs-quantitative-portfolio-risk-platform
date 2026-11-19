import time
from pipeline import run_once
from config.settings import settings

if __name__ == "__main__":
    print(f"RAIBS risk engine runner started; refresh={settings.refresh_seconds}s")
    while True:
        try:
            run_once()
            print("Risk snapshot updated.")
        except Exception as exc:
            print(f"Risk cycle failed: {type(exc).__name__}: {exc}")
        time.sleep(settings.refresh_seconds)

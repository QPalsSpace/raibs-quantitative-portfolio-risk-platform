from fastapi import FastAPI, HTTPException
import json
from pathlib import Path

app = FastAPI(title="RAIBS Dynamic Portfolio Risk API", version="1.0.0")
SNAPSHOT = Path(__file__).resolve().parents[1] / "outputs" / "latest_snapshot.json"

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/risk")
def risk():
    if not SNAPSHOT.exists():
        raise HTTPException(404, "No risk snapshot available. Run python run_once.py first.")
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))

@app.get("/risk/{portfolio_id}")
def portfolio_risk(portfolio_id: str):
    if not SNAPSHOT.exists():
        raise HTTPException(404, "No risk snapshot available.")
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for p in data["portfolios"]:
        if p["portfolio_id"] == portfolio_id:
            return p
    raise HTTPException(404, f"Portfolio {portfolio_id} not found.")

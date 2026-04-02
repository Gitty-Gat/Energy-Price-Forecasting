from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI

app = FastAPI(title="Energy Forecast API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp_utc": datetime.now(timezone.utc).isoformat()}


@app.post("/forecast/run")
def trigger_forecast() -> dict[str, str]:
    return {"status": "accepted", "detail": "Wire to pipeline orchestrator in next slice."}

"""Tiny experiment tracker + model registry (JSON on disk). Also logs to MLflow when USE_MLFLOW=1."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts"
RUNS, REG = ART / "runs.json", ART / "registry.json"


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def log_run(experiment: str, model: str, params: dict, metrics: dict) -> str:
    ART.mkdir(exist_ok=True)
    rid = uuid.uuid4().hex[:8]
    runs = _read(RUNS)
    runs.append({"run_id": rid, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": experiment, "model": model,
                 "params": params, "metrics": {k: round(float(v), 4) for k, v in metrics.items()}})
    RUNS.write_text(json.dumps(runs, indent=2))
    if os.getenv("USE_MLFLOW") == "1":
        try:
            import mlflow
            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=model):
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        except Exception:
            pass
    return rid


def load_runs() -> list:
    return _read(RUNS)


def load_registry() -> list:
    return _read(REG)


def register(run_id: str, name: str, stage: str = "Staging") -> None:
    reg = [r for r in load_registry() if not (r["name"] == name and r["stage"] == stage == "Production")]
    for r in reg:
        if r["name"] == name and stage == "Production" and r["stage"] == "Production":
            r["stage"] = "Archived"
    reg.append({"name": name, "run_id": run_id, "stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    REG.write_text(json.dumps(reg, indent=2))

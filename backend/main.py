from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Business Entity Resolution API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


OUTPUT_DIR = Path("output")


@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "business-entity-resolution",
    }


@app.get("/api/output-status")
def output_status():

    matching = (
        OUTPUT_DIR
        / "matching_results.tsv"
    )

    candidates = (
        OUTPUT_DIR
        / "candidate_pairs.tsv"
    )

    return {
        "matching_results": matching.exists(),
        "candidate_pairs": candidates.exists(),
    }


@app.get("/api/metrics")
def metrics():

    metrics_file = (
        OUTPUT_DIR
        / "metrics.json"
    )

    if not metrics_file.exists():

        return {
            "status": "not_available"
        }

    import json

    with open(
        metrics_file,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)
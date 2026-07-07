# HackerRank Orchestrate Solution

This folder contains a deterministic Python baseline for the multi-modal evidence review task.

## Run Predictions

```bash
python code/main.py --input dataset/claims.csv --output output.csv --dataset-dir dataset
```

The script writes the exact required schema to `output.csv`.

## Run Evaluation

```bash
python code/evaluation/main.py
```

Evaluation uses `dataset/sample_claims.csv`, compares `text_only` and `final` strategies, writes `code/evaluation/sample_predictions_final.csv`, and generates `code/evaluation/evaluation_report.md`.

## Approach

The default strategy is dependency-free and deterministic. It extracts claimed issue type and object part from the chat transcript, validates referenced image files, applies prompt-injection and user-history risk flags, and emits concise evidence decisions. It is intended as a reproducible floor that can later be upgraded by adding a vision-model observation layer before final scoring.

## Run The Web Demo

```bash
python code/app.py
```

Open `http://localhost:8000` to try a claim in the browser.

API endpoints:

- `GET /health`
- `GET /schema`
- `POST /api/predict`
- `POST /api/run-test`

## Deploy

The repository includes a `Dockerfile` and `render.yaml`. On Render, create a new Blueprint from this repo or deploy it as a Docker web service. The app reads the platform-provided `PORT` environment variable automatically.

# MovieLens LLM Recommender API

A small FastAPI service that accepts a MovieLens 100K dataset zip in standard GroupLens layout and runs an end-to-end binary recommendation pipeline.

## What it expects

Upload a zip containing these files:
- `u1.base`
- `u1.test`
- `u.user`
- `u.item`

A normal `ml-100k.zip` from GroupLens works.

## What it does

1. Loads `u1.base` / `u1.test`
2. Binarizes ratings using `rating >= positive_threshold`
3. Builds user and movie profiles
4. Builds pairwise examples for test user-item pairs
5. Scores each pair with either:
   - `backend=mock` for smoke testing with no model download
   - `backend=hf` for local Hugging Face inference
6. Saves predictions and computes:
   - Accuracy / Precision / Recall / F1 / ROC-AUC
   - HitRate@K / NDCG@K

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:
- `http://127.0.0.1:8000/docs`

## Minimal smoke test

Use the official GroupLens zip or your own zip with the same internal files.

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -F "dataset=@ml-100k.zip" \
  -F "backend=mock" \
  -F "max_rows=500" \
  -F 'top_k_json=[5,10,20]'
```

## Hugging Face mode

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -F "dataset=@ml-100k.zip" \
  -F "backend=hf" \
  -F "model_name=Qwen/Qwen2.5-1.5B-Instruct" \
  -F "max_rows=200"
```

This is slower and depends on local RAM/GPU.

## Downloading results

After a run:
- `GET /runs` lists run folders
- `GET /download?run_dir=<absolute_run_dir>` downloads a zip of that run

## Notes

- `backend=mock` is included so the API can be smoke tested immediately on a fresh laptop.
- `backend=hf` uses a free local Hugging Face model; no API key is needed.
- Ranking metrics are computed over the candidate items present in the test split for each user.
- This is a practical demo API, not a production recommender stack.

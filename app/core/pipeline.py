from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import pandas as pd

from app.core.data import prepare_data, save_builder_outputs
from app.core.eval import evaluate_predictions
from app.core.model import get_recommender
from app.schemas import RunConfig, MetricsSummary, RunResponse


def run_pipeline(dataset_path: str, work_root: str, config: RunConfig) -> RunResponse:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(work_root) / f"run_{timestamp}"
    builder_dir = run_dir / "builder_outputs"
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    run_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(
        dataset_path=dataset_path,
        work_dir=run_dir,
        positive_threshold=config.positive_threshold,
        max_history=config.max_history,
    )
    output_files: List[str] = []
    if config.save_outputs:
        output_files.extend(save_builder_outputs(data, builder_dir))

    recommender = get_recommender(
        backend=config.backend,
        model_name=config.model_name,
        use_4bit_if_available=config.use_4bit_if_available,
    )

    test_examples = data["test_examples"].copy()
    if config.max_rows is not None:
        test_examples = test_examples.head(config.max_rows).copy()

    records = []
    start = time.time()
    for _, row in test_examples.iterrows():
        score, pred, raw = recommender.score_row(row)
        rec = row.to_dict()
        rec["score"] = float(score)
        rec["y_pred"] = int(pred)
        rec["llm_raw_output"] = raw
        records.append(rec)

    pred_df = pd.DataFrame(records)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "test_predictions_full.csv"
    pred_df.to_csv(pred_path, index=False)
    output_files.append(str(pred_path))

    summary, per_user_path = evaluate_predictions(pred_df, config.top_k_list, eval_dir)
    output_files.append(str(eval_dir / "evaluation_summary.json"))
    output_files.append(per_user_path)

    metadata = {
        "elapsed_seconds": round(time.time() - start, 3),
        "backend": config.backend,
        "model_name": config.model_name,
        "dataset_path": dataset_path,
        "run_dir": str(run_dir),
    }
    meta_path = run_dir / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    output_files.append(str(meta_path))

    return RunResponse(
        status="ok",
        message="Pipeline completed successfully.",
        run_dir=str(run_dir),
        config=config,
        metrics=MetricsSummary(**summary),
        output_files=output_files,
    )

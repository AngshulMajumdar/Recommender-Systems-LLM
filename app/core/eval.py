from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def dcg_at_k(relevances: List[int], k: int) -> float:
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    return float(np.sum([rel / math.log2(idx + 2) for idx, rel in enumerate(relevances)]))


def ndcg_at_k(sorted_labels: List[int], k: int) -> float:
    dcg = dcg_at_k(sorted_labels, k)
    ideal = sorted(sorted_labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else float(dcg / idcg)


def hit_rate_at_k(sorted_labels: List[int], k: int) -> float:
    return 1.0 if np.sum(sorted_labels[:k]) > 0 else 0.0


def evaluate_predictions(df: pd.DataFrame, top_k_list: List[int], output_dir: str | Path) -> Tuple[Dict, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["label"] = df["label"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df["score"] = df["score"].astype(float)
    df["score_for_rank"] = df["score"] + 1e-9 * df["item_id"].astype(float)

    y_true = df["label"].values
    y_pred = df["y_pred"].values
    y_score = df["score"].values

    summary = {
        "num_test_rows": int(len(df)),
        "num_users_in_test": int(df["user_id"].nunique()),
        "positive_rate_test": float(df["label"].mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1 and len(np.unique(y_score)) > 1:
        try:
            summary["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            summary["roc_auc"] = None
    else:
        summary["roc_auc"] = None

    ranking_metrics: Dict[str, float] = {}
    per_user_rows = []
    for k in top_k_list:
        hr_list = []
        ndcg_list = []
        for user_id, grp in df.groupby("user_id"):
            grp_sorted = grp.sort_values(["score_for_rank"], ascending=False)
            labels = grp_sorted["label"].tolist()
            hr_list.append(hit_rate_at_k(labels, k))
            ndcg_list.append(ndcg_at_k(labels, k))
        ranking_metrics[f"HitRate@{k}"] = float(np.mean(hr_list))
        ranking_metrics[f"NDCG@{k}"] = float(np.mean(ndcg_list))

    for user_id, grp in df.groupby("user_id"):
        grp_sorted = grp.sort_values(["score_for_rank"], ascending=False)
        labels = grp_sorted["label"].tolist()
        row = {
            "user_id": int(user_id),
            "num_test_items": int(len(grp_sorted)),
            "num_relevant_test_items": int(np.sum(labels)),
        }
        for k in top_k_list:
            row[f"HitRate@{k}"] = hit_rate_at_k(labels, k)
            row[f"NDCG@{k}"] = ndcg_at_k(labels, k)
        per_user_rows.append(row)
    per_user_df = pd.DataFrame(per_user_rows)
    per_user_path = output_dir / "evaluation_per_user.csv"
    per_user_df.to_csv(per_user_path, index=False)

    summary["ranking_metrics"] = ranking_metrics
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary, str(per_user_path)

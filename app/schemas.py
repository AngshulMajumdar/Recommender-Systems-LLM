from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    model_name: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct")
    positive_threshold: int = Field(default=4, ge=1, le=5)
    max_history: int = Field(default=10, ge=1, le=50)
    max_rows: Optional[int] = Field(default=None, ge=1)
    top_k_list: List[int] = Field(default_factory=lambda: [5, 10, 20])
    backend: str = Field(default="mock", description="mock or hf")
    use_4bit_if_available: bool = True
    save_outputs: bool = True


class MetricsSummary(BaseModel):
    num_test_rows: int
    num_users_in_test: int
    positive_rate_test: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    ranking_metrics: Dict[str, float]


class RunResponse(BaseModel):
    status: str
    message: str
    run_dir: str
    config: RunConfig
    metrics: MetricsSummary
    output_files: List[str]

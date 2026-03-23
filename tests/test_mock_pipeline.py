from pathlib import Path

from app.core.pipeline import run_pipeline
from app.schemas import RunConfig


def test_run_pipeline_mock():
    dataset_path = Path(__file__).resolve().parents[2] / "sample_data" / "ml-100k-mini.zip"
    if not dataset_path.exists():
        return
    config = RunConfig(backend="mock", max_rows=2)
    response = run_pipeline(str(dataset_path), str(Path(__file__).resolve().parents[2] / "test_runs"), config)
    assert response.status == "ok"
    assert response.metrics.num_test_rows == 2

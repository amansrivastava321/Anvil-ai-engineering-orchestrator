"""Tests for app.services.learning_service — LearningService performance tracking."""
import pytest
from app.services.learning_service import LearningService


@pytest.fixture
def svc(tmp_path):
    return LearningService(data_dir=str(tmp_path / "perf"))


def test_record_execution_writes_file(tmp_path, svc):
    svc.record_execution({
        "run_id": "run-abc",
        "task_type": "debug",
        "model_used": "qwen2.5-coder:7b",
        "status": "completed",
    })
    files = list((tmp_path / "perf" / "executions").glob("*.json"))
    assert len(files) == 1


def test_score_execution_writes_feedback(tmp_path, svc):
    svc.score_execution("run-xyz", 0.9)
    files = list((tmp_path / "perf" / "feedback").glob("*.json"))
    assert len(files) == 1


def test_score_execution_clamps_to_0_1(tmp_path, svc):
    import json
    svc.score_execution("run-1", 2.5)
    path = list((tmp_path / "perf" / "feedback").glob("*.json"))[0]
    data = json.loads(path.read_text())
    assert data["score"] == 1.0


def test_recommend_model_returns_none_when_no_data(svc):
    assert svc.recommend_model("debug") is None


def test_recommend_model_picks_best_success_rate(tmp_path, svc):
    import json
    executions_dir = tmp_path / "perf" / "executions"
    # model_a: 1/2 success
    for i, status in enumerate(["completed", "failed"]):
        (executions_dir / f"model_a_{i}.json").write_text(json.dumps({
            "task_type": "debug", "model_used": "model_a", "status": status
        }))
    # model_b: 2/2 success
    for i in range(2):
        (executions_dir / f"model_b_{i}.json").write_text(json.dumps({
            "task_type": "debug", "model_used": "model_b", "status": "completed"
        }))
    assert svc.recommend_model("debug") == "model_b"


def test_recommend_workflow_returns_none_when_no_data(svc):
    assert svc.recommend_workflow("debug") is None


def test_recommend_workflow_picks_best(tmp_path, svc):
    import json
    executions_dir = tmp_path / "perf" / "executions"
    for i, status in enumerate(["completed", "failed"]):
        (executions_dir / f"wf_a_{i}.json").write_text(json.dumps({
            "task_type": "audit", "workflow_type": "audit", "status": status
        }))
    for i in range(3):
        (executions_dir / f"wf_b_{i}.json").write_text(json.dumps({
            "task_type": "audit", "workflow_type": "report", "status": "completed"
        }))
    assert svc.recommend_workflow("audit") == "report"


def test_get_stats_counts_files(tmp_path, svc):
    svc.record_execution({"run_id": "r1", "task_type": "x", "model_used": "m", "status": "completed"})
    svc.score_execution("r1", 0.8)
    stats = svc.get_stats()
    assert stats["executions_recorded"] == 1
    assert stats["feedback_given"] == 1


def test_recommend_model_ignores_other_task_types(tmp_path, svc):
    import json
    executions_dir = tmp_path / "perf" / "executions"
    (executions_dir / "x.json").write_text(json.dumps({
        "task_type": "other", "model_used": "model_x", "status": "completed"
    }))
    assert svc.recommend_model("debug") is None

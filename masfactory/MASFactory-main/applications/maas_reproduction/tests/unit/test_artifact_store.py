import pytest

from applications.maas_reproduction.maas_reproduction.adapters.artifact_store import ArtifactStore


def test_artifact_store_allows_token_usage_keys(tmp_path):
    path = ArtifactStore(tmp_path).write_sample_result({
        "problem_index": 0,
        "prompt_tokens": 12,
        "completion_tokens": 4,
    })
    assert path.exists()


def test_artifact_store_rejects_expected_answer_recursively(tmp_path):
    with pytest.raises(ValueError, match="expected_answer"):
        ArtifactStore(tmp_path).write_sample_result({
            "problem_index": 0,
            "metadata": {"expected_answer": "secret"},
        })

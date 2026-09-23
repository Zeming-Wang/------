import pytest
import torch

from applications.maas_reproduction.maas_reproduction.runtime.source_controller_bundle import (
    SourceControllerBundleError,
    load_source_controller_bundle,
    validate_query_embedding_fixtures,
)


GSM8K_CATALOG = (
    "Generate",
    "GenerateCoT",
    "MultiGenerateCoT",
    "ScEnsemble",
    "Programmer",
    "SelfRefine",
    "EarlyStop",
)


def _payload(**overrides):
    payload = {
        "artifact_type": "maas_source_controller_bundle",
        "format_version": 1,
        "dataset": "GSM8K",
        "controller_state_dict": {"weight": torch.tensor([1.0])},
        "operator_catalog": GSM8K_CATALOG,
        "operator_embeddings": torch.zeros(7, 384, dtype=torch.float32),
        "embedding_spec": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
        },
        "controller_spec": {"input_dim": 384, "hidden_dim": 32, "num_layers": 4},
        "metadata": {"source": "fixture"},
    }
    payload.update(overrides)
    return payload


def test_source_bundle_preserves_the_source_catalog_embedding_alignment(tmp_path):
    embeddings = torch.arange(7 * 384, dtype=torch.float32).reshape(7, 384)
    path = tmp_path / "source_controller_bundle.pt"
    torch.save(_payload(operator_embeddings=embeddings), path)

    bundle = load_source_controller_bundle(
        path,
        expected_dataset="GSM8K",
        expected_catalog=GSM8K_CATALOG,
        expected_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    assert bundle.operator_catalog == GSM8K_CATALOG
    assert torch.equal(bundle.operator_embeddings, embeddings)


def test_source_bundle_rejects_embeddings_that_break_catalog_alignment(tmp_path):
    path = tmp_path / "bad_shape.pt"
    torch.save(_payload(operator_embeddings=torch.zeros(6, 384)), path)

    with pytest.raises(SourceControllerBundleError, match="shape"):
        load_source_controller_bundle(
            path,
            expected_dataset="GSM8K",
            expected_catalog=GSM8K_CATALOG,
            expected_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )


def test_query_fixture_check_detects_encoder_drift(tmp_path):
    path = tmp_path / "query_fixtures.pt"
    fixtures = tuple(
        {"text": f"question {index}", "embedding": torch.ones(384)}
        for index in range(5)
    )
    torch.save(_payload(metadata={"query_embedding_fixtures": fixtures}), path)
    bundle = load_source_controller_bundle(
        path,
        expected_dataset="GSM8K",
        expected_catalog=GSM8K_CATALOG,
        expected_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    class DriftedProvider:
        def encode(self, _text):
            return torch.zeros(384)

    with pytest.raises(SourceControllerBundleError, match="query embedding fixture"):
        validate_query_embedding_fixtures(bundle, DriftedProvider())


@pytest.mark.parametrize(
    ("payload_override", "expected_dataset", "expected_catalog", "message"),
    [
        ({"dataset": "MATH"}, "GSM8K", GSM8K_CATALOG, "dataset"),
        ({"operator_catalog": tuple(reversed(GSM8K_CATALOG))}, "GSM8K", GSM8K_CATALOG, "catalog"),
        (
            {"embedding_spec": {"model_name": "different-model", "dimension": 384}},
            "GSM8K",
            GSM8K_CATALOG,
            "embedding model",
        ),
    ],
)
def test_source_bundle_rejects_runtime_contract_mismatch(
    tmp_path, payload_override, expected_dataset, expected_catalog, message
):
    path = tmp_path / "contract_mismatch.pt"
    torch.save(_payload(**payload_override), path)

    with pytest.raises(SourceControllerBundleError, match=message):
        load_source_controller_bundle(
            path,
            expected_dataset=expected_dataset,
            expected_catalog=expected_catalog,
            expected_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )


def test_source_bundle_explicitly_rejects_humaneval_first_release(tmp_path):
    humaneval_catalog = (*GSM8K_CATALOG[:4], "Test", *GSM8K_CATALOG[5:])
    path = tmp_path / "humaneval.pt"
    torch.save(
        _payload(
            dataset="HumanEval",
            operator_catalog=humaneval_catalog,
            operator_embeddings=torch.zeros(7, 384),
        ),
        path,
    )

    with pytest.raises(SourceControllerBundleError, match="only GSM8K and MATH"):
        load_source_controller_bundle(
            path,
            expected_dataset="HumanEval",
            expected_catalog=humaneval_catalog,
            expected_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )

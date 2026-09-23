"""Read-only loader for Controller artifacts exported by source MaAS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class SourceControllerBundleError(ValueError):
    """Raised when a source Controller bundle cannot be used for testing."""


@dataclass(frozen=True)
class SourceControllerBundle:
    dataset: str
    controller_state_dict: Mapping[str, Any]
    operator_catalog: tuple[str, ...]
    operator_embeddings: Any
    embedding_spec: Mapping[str, Any]
    controller_spec: Mapping[str, int]
    metadata: Mapping[str, Any]


def validate_query_embedding_fixtures(
    bundle: SourceControllerBundle,
    embedding_provider: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """Fail fast when the live query encoder has drifted from the source one."""
    import torch

    fixtures = bundle.metadata.get("query_embedding_fixtures", ())
    if not fixtures:
        return
    if not isinstance(fixtures, (tuple, list)) or not 5 <= len(fixtures) <= 20:
        raise SourceControllerBundleError(
            "query embedding fixtures must contain between 5 and 20 entries"
        )
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, Mapping) or not isinstance(fixture.get("text"), str):
            raise SourceControllerBundleError(f"query embedding fixture {index} is malformed")
        expected = fixture.get("embedding")
        if not isinstance(expected, torch.Tensor) or tuple(expected.shape) != (384,):
            raise SourceControllerBundleError(f"query embedding fixture {index} is malformed")
        actual = torch.as_tensor(embedding_provider.encode(fixture["text"])).detach().cpu().float()
        try:
            torch.testing.assert_close(actual, expected.detach().cpu().float(), rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise SourceControllerBundleError(
                f"query embedding fixture {index} does not match the source encoder"
            ) from exc


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch releases before ``weights_only``.
        return torch.load(path, map_location="cpu")


def load_source_controller_bundle(
    path: str | Path,
    *,
    expected_dataset: str,
    expected_catalog: Sequence[str],
    expected_embedding_model: str,
) -> SourceControllerBundle:
    """Load and validate a source artifact without constructing runtime state."""
    import torch

    payload = _torch_load(Path(path))
    if not isinstance(payload, Mapping):
        raise SourceControllerBundleError("source bundle must contain a mapping")
    if payload.get("artifact_type") != "maas_source_controller_bundle":
        raise SourceControllerBundleError("unsupported source bundle artifact type")
    if payload.get("format_version") != 1:
        raise SourceControllerBundleError("unsupported source bundle format version")

    dataset = payload.get("dataset")
    if dataset != expected_dataset:
        raise SourceControllerBundleError(
            f"source bundle dataset {dataset!r} does not match {expected_dataset!r}"
        )
    if dataset not in {"GSM8K", "MATH"}:
        raise SourceControllerBundleError(
            "source Controller bundles currently support only GSM8K and MATH"
        )
    catalog = tuple(payload.get("operator_catalog", ()))
    if catalog != tuple(expected_catalog):
        raise SourceControllerBundleError("source bundle operator catalog does not match dataset contract")

    embedding_spec = payload.get("embedding_spec")
    if not isinstance(embedding_spec, Mapping):
        raise SourceControllerBundleError("source bundle embedding_spec must be a mapping")
    if embedding_spec.get("model_name") != expected_embedding_model:
        raise SourceControllerBundleError("source bundle embedding model does not match query embedder")

    controller_spec = payload.get("controller_spec")
    if not isinstance(controller_spec, Mapping):
        raise SourceControllerBundleError("source bundle controller_spec must be a mapping")
    required_controller_spec = {"input_dim": 384, "hidden_dim": 32, "num_layers": 4}
    if dict(controller_spec) != required_controller_spec:
        raise SourceControllerBundleError("source bundle controller spec is incompatible")
    if embedding_spec.get("dimension") != controller_spec["input_dim"]:
        raise SourceControllerBundleError("source bundle embedding dimension is incompatible")

    state_dict = payload.get("controller_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise SourceControllerBundleError("source bundle controller_state_dict must be non-empty")
    if any(not isinstance(key, str) or not key for key in state_dict):
        raise SourceControllerBundleError("source bundle controller_state_dict has an invalid key")

    embeddings = payload.get("operator_embeddings")
    required_shape = (len(catalog), controller_spec["input_dim"])
    if not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != required_shape:
        raise SourceControllerBundleError(
            f"source bundle operator_embeddings shape must be {required_shape}"
        )
    if embeddings.dtype != torch.float32:
        raise SourceControllerBundleError("source bundle operator_embeddings dtype must be float32")
    if not bool(torch.isfinite(embeddings).all()):
        raise SourceControllerBundleError("source bundle operator_embeddings must be finite")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise SourceControllerBundleError("source bundle metadata must be a mapping")

    return SourceControllerBundle(
        dataset=dataset,
        controller_state_dict=state_dict,
        operator_catalog=catalog,
        operator_embeddings=embeddings.detach().cpu().contiguous(),
        embedding_spec=dict(embedding_spec),
        controller_spec=dict(controller_spec),
        metadata=dict(metadata),
    )


__all__ = [
    "SourceControllerBundle",
    "SourceControllerBundleError",
    "load_source_controller_bundle",
    "validate_query_embedding_fixtures",
]

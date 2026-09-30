"""Export source MaAS Controller state and operator embeddings for test-only use."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPERATOR_CATALOGS = {
    "GSM8K": (
        "Generate",
        "GenerateCoT",
        "MultiGenerateCoT",
        "ScEnsemble",
        "Programmer",
        "SelfRefine",
        "EarlyStop",
    ),
    "MATH": (
        "Generate",
        "GenerateCoT",
        "MultiGenerateCoT",
        "ScEnsemble",
        "Programmer",
        "SelfRefine",
        "EarlyStop",
    ),
    "HumanEval": (
        "Generate",
        "GenerateCoT",
        "MultiGenerateCoT",
        "ScEnsemble",
        "Test",
        "SelfRefine",
        "EarlyStop",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_state_dict(path: Path):
    import torch

    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("controller .pth must contain a non-empty state_dict")
    return state_dict


def _validate_controller_state_dict(state_dict) -> None:
    expected = {}
    for layer in range(4):
        expected[f"layers.{layer}.operator_encoder.weight"] = (32, 384 if layer == 0 else 768)
        expected[f"layers.{layer}.operator_encoder.bias"] = (32,)
        expected[f"layers.{layer}.query_encoder.weight"] = (32, 384)
        expected[f"layers.{layer}.query_encoder.bias"] = (32,)
    actual_keys = set(state_dict)
    if actual_keys != set(expected):
        missing = sorted(set(expected) - actual_keys)
        unexpected = sorted(actual_keys - set(expected))
        raise ValueError(f"incompatible Controller state_dict; missing={missing}, unexpected={unexpected}")
    for key, shape in expected.items():
        if tuple(getattr(state_dict[key], "shape", ())) != shape:
            raise ValueError(f"incompatible Controller tensor shape for {key}: expected {shape}")


def _operator_descriptions(operator_json_path: Path, catalog: Sequence[str]) -> tuple[str, ...]:
    # This is the exact formatting from source GraphUtils._load_operator_description.
    # Keeping it local prevents a one-shot artifact exporter from importing the
    # full legacy MaAS runtime and all of its unrelated version-pinned plugins.
    operator_data = json.loads(operator_json_path.read_text(encoding="utf-8"))
    if not isinstance(operator_data, dict):
        raise ValueError("operator.json must contain an object")
    descriptions = []
    for index, name in enumerate(catalog):
        matched = operator_data.get(name, {})
        description = matched.get("description", "No description available")
        interface = matched.get("interface", "No interface specified")
        descriptions.append(f"{index}. {name}: {description}, with interface {interface}.")
    return tuple(descriptions)


def _query_texts(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("query fixture file must contain a non-empty JSON list of strings")
    if not 5 <= len(value) <= 20:
        raise ValueError("query fixture file must contain between 5 and 20 questions")
    return tuple(value)


def _controller_parity_fixtures(
    *, state_dict, operator_embeddings, catalog, query_texts, embedding_fn,
    controller_factory=None,
):
    """Capture real source Controller layer distributions with forward hooks."""
    import torch

    if not query_texts:
        return ()
    if controller_factory is None:
        from maas.ext.maas.models.controller import MultiLayerController
        controller_factory = lambda: MultiLayerController(device=torch.device("cpu"))
    controller = controller_factory()
    controller.load_state_dict(state_dict, strict=True)
    controller.eval()
    fixtures = []
    original_rng = torch.random.get_rng_state()
    try:
        for fixture_index, text in enumerate(query_texts):
            captured = []
            hooks = []
            for layer_index, layer in enumerate(controller.layers):
                hooks.append(layer.register_forward_hook(
                    lambda _module, inputs, output, index=layer_index: captured.append(
                        (index, inputs, output)
                    )
                ))
            seed = 1729 + fixture_index
            torch.manual_seed(seed)
            try:
                layer_log_probs, selected_layers = controller.forward(
                    text, operator_embeddings, catalog
                )
            finally:
                for hook in hooks:
                    hook.remove()
            layer_rows = []
            previous_indices: list[int] = []
            for layer_index, inputs, output in sorted(captured, key=lambda item: item[0]):
                log_probs, probs = output
                selected_names = tuple(selected_layers[layer_index])
                selected_indices = tuple(catalog.index(name) for name in selected_names)
                layer_rows.append({
                    "layer_index": layer_index,
                    "previous_operator_indices": tuple(previous_indices),
                    "selected_indices": selected_indices,
                    "log_probs": log_probs.detach().cpu().float().squeeze(0).contiguous(),
                    "probs": probs.detach().cpu().float().squeeze(0).contiguous(),
                })
                previous_indices = list(selected_indices)
            aggregate = layer_log_probs[0]
            for value in layer_log_probs[1:]:
                aggregate = aggregate + value
            fixtures.append({
                "text": text,
                "seed": seed,
                "query_embedding": torch.as_tensor(embedding_fn(text)).detach().cpu().float().contiguous(),
                "layers": tuple(layer_rows),
                "aggregate_log_prob": aggregate.detach().cpu().float().reshape(()),
            })
    finally:
        torch.random.set_rng_state(original_rng)
    return tuple(fixtures)


def export_source_controller_bundle(
    *,
    dataset: str,
    controller_path: str | Path,
    operator_json_path: str | Path,
    output_path: str | Path,
    query_fixture_path: str | Path | None = None,
    embedding_fn: Callable[[str], object] | None = None,
    controller_factory: Callable[[], object] | None = None,
) -> Path:
    """Create the complete, test-only artifact consumed by the reproduction."""
    import torch

    if dataset not in OPERATOR_CATALOGS:
        raise ValueError("unsupported source bundle dataset")
    controller_path = Path(controller_path).resolve()
    operator_json_path = Path(operator_json_path).resolve()
    output_path = Path(output_path).resolve()
    query_fixture_path = Path(query_fixture_path).resolve() if query_fixture_path else None
    if not controller_path.is_file():
        raise FileNotFoundError(controller_path)
    if not operator_json_path.is_file():
        raise FileNotFoundError(operator_json_path)

    state_dict = _load_state_dict(controller_path)
    _validate_controller_state_dict(state_dict)
    catalog = OPERATOR_CATALOGS[dataset]
    descriptions = _operator_descriptions(operator_json_path, catalog)
    if embedding_fn is None:
        from sentence_transformers import SentenceTransformer

        # Equivalent to source get_sentence_embedding, while avoiding import of
        # the unrelated legacy MaAS application stack.
        source_model = SentenceTransformer(EMBEDDING_MODEL)
        embedding_fn = lambda text: torch.tensor(source_model.encode(text))
    operator_embeddings = torch.stack(
        [torch.as_tensor(embedding_fn(text)) for text in descriptions]
    ).detach().cpu().to(dtype=torch.float32).contiguous()
    if tuple(operator_embeddings.shape) != (len(catalog), 384):
        raise ValueError("source embedding function must return 384-dimensional vectors")
    if not bool(torch.isfinite(operator_embeddings).all()):
        raise ValueError("source operator embeddings must be finite")

    query_texts = _query_texts(query_fixture_path)
    if dataset == "HumanEval" and not query_texts:
        raise ValueError("HumanEval source bundles require 5-20 query fixtures")
    fixtures = tuple(
        {
            "text": text,
            "embedding": torch.as_tensor(embedding_fn(text)).detach().cpu().to(dtype=torch.float32).contiguous(),
        }
        for text in query_texts
    )
    if any(tuple(item["embedding"].shape) != (384,) for item in fixtures):
        raise ValueError("source query embedding fixtures must be 384-dimensional")

    payload = {
        "artifact_type": "maas_source_controller_bundle",
        "format_version": 2 if dataset == "HumanEval" else 1,
        "dataset": dataset,
        "controller_state_dict": {
            key: value.detach().cpu().clone() for key, value in state_dict.items()
        },
        "operator_catalog": tuple(catalog),
        "operator_embeddings": operator_embeddings,
        "embedding_spec": {"model_name": EMBEDDING_MODEL, "dimension": 384},
        "controller_spec": {"input_dim": 384, "hidden_dim": 32, "num_layers": 4},
        "metadata": {
            "controller_path": str(controller_path),
            "controller_sha256": _sha256(controller_path),
            "operator_json_path": str(operator_json_path),
            "operator_json_sha256": _sha256(operator_json_path),
            "operator_descriptions": descriptions,
            "query_embedding_fixtures": fixtures,
            "controller_parity_fixtures": _controller_parity_fixtures(
                state_dict=state_dict,
                operator_embeddings=operator_embeddings,
                catalog=catalog,
                query_texts=query_texts,
                embedding_fn=embedding_fn,
                controller_factory=controller_factory,
            ),
            "workflow_contract": (
                "maas-humaneval-source-v1" if dataset == "HumanEval" else "maas-source-v1"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(OPERATOR_CATALOGS))
    parser.add_argument("--controller-pth", required=True, type=Path)
    parser.add_argument("--operator-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--query-fixture-file",
        type=Path,
        default=None,
        help="optional JSON list containing 5-20 fixed questions for encoder parity checks",
    )
    args = parser.parse_args(argv)
    output = export_source_controller_bundle(
        dataset=args.dataset,
        controller_path=args.controller_pth,
        operator_json_path=args.operator_json,
        output_path=args.output,
        query_fixture_path=args.query_fixture_file,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EMBEDDING_MODEL", "OPERATOR_CATALOGS", "export_source_controller_bundle", "main"]

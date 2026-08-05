import argparse
import json
from datetime import datetime
from typing import Any
from tqdm import tqdm

from ddo.consultation import run_consultation
from ddo.paths import OUTPUT_DIR
from ddo.runtime.app_runtime import app_runtime
from ddo.runtime.bootstrap import bootstrap_app_runtime
from ddo.runtime.config_types import RuntimeConfig
from ddo.services.utils_data import load_dataset


def _init_results(cfg: RuntimeConfig) -> dict[str, Any]:
    return {
        "metrics": {
            "Acc_wo_iq": None,
            "Acc": None,
            "Avg_n": None,
        },
        "records": [],
        "token_usage": {
            "total": {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "by_component": {},
        },
        "settings": {
            "dataset_name": cfg.dataset_name,
            "device": cfg.device,
            "normal_agent_model": cfg.api_cfg.for_normal_agents.model_name,
            "confidence_agent_model": cfg.api_cfg.for_confidence_agent.model_name,
            "best_settings": cfg.best_settings,
            "confidence_estimation": cfg.basic_cfg.get("confidence_estimation", {}),
            "api_reliability": cfg.basic_cfg.get("api_reliability", {}),
        },
    }


def _get_disease_label_rank(
    diagnostic_confidence: dict[str, float] | None,
    disease_label: str,
) -> int:
    if not diagnostic_confidence:
        return 999

    rank = 0
    previous_confidence: float | None = None
    for disease, confidence in sorted(
        diagnostic_confidence.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        if previous_confidence is None or confidence < previous_confidence:
            rank += 1
            previous_confidence = confidence

        if disease == disease_label:
            return rank

    return 999


def _calculate_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    correct_wo_iq_count = 0
    correct_count = 0
    turn_count = 0

    for record in records:
        disease_label = record["disease_label"]
        initial_confidence = record["initial_diagnostic_confidence"]
        final_confidence = record.get("final_diagnostic_confidence")

        if not final_confidence:
            if record["interactions"]:
                final_confidence = record["interactions"][-1]["diagnostic_confidence"]
            else:
                final_confidence = initial_confidence

        is_initial_correct = _get_disease_label_rank(initial_confidence, disease_label) == 1
        is_final_correct = _get_disease_label_rank(final_confidence, disease_label) == 1

        correct_wo_iq_count += int(is_initial_correct)
        correct_count += int(is_final_correct)
        turn_count += len(record["interactions"])

    records_count = len(records)
    return {
        "Acc_wo_iq": round(correct_wo_iq_count / records_count, 3) if records_count > 0 else 0,
        "Acc": round(correct_count / records_count, 3) if records_count > 0 else 0,
        "Avg_n": round(turn_count / records_count, 1) if records_count > 0 else 0,
    }


def _build_results_path(cfg: RuntimeConfig) -> str:
    results_dir = OUTPUT_DIR / "evaluation" / cfg.dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg.api_cfg.for_confidence_agent.model_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = results_dir / f"results_{model_name}_{timestamp}.json"

    return str(results_path)


def _save_results(results: dict[str, Any], results_path: str) -> str:
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results_path


def _update_token_usage(results: dict[str, Any]) -> None:
    if app_runtime.token_usage is None:
        return

    results["token_usage"] = app_runtime.token_usage.snapshot()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the DDO MASFactory consultation evaluation.",
    )
    parser.parse_args(argv)

    cfg = bootstrap_app_runtime()

    from ddo.graphs.g_consultation import g_consultation

    g_consultation.build()

    dataset = load_dataset(cfg.dataset_name, "test")

    results = _init_results(cfg)
    results_path = _build_results_path(cfg)

    process_bar = tqdm(total=len(dataset), desc="Processing cases")
    for case in dataset:
        record = run_consultation(
            g_consultation=g_consultation,
            case=case,
        )
        results["records"].append(record)
        results["metrics"] = _calculate_metrics(results["records"])
        _update_token_usage(results)
        _save_results(results, results_path)
        process_bar.update(1)
    process_bar.close()

    _update_token_usage(results)
    _save_results(results, results_path)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

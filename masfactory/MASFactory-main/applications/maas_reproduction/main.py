"""CLI entry point for the MaAS MASFactory reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maas_reproduction.config import SUPPORTED_DATASETS, SUPPORTED_MODES
from maas_reproduction.nodes.config_node import config_forward
from maas_reproduction.runtime import build_runtime_attributes
from maas_reproduction.workflow import build_maas_reproduction_graph


def run(input_data: dict[str, object], specific_indices: list[int] | None = None) -> dict[str, object]:
    config = config_forward(input_data, {}) #将CLI参数转化为settings
    runtime_attributes = build_runtime_attributes(
        config["settings"],
        specific_indices=specific_indices,
    ) #初始化controller、optimizer、workflow、数据集等对象
    graph = build_maas_reproduction_graph() #创建一个rootgraph
    graph.build()
    output, _attributes = graph.invoke(input_data, attributes=runtime_attributes)
    return output


def main() -> None:
    args = _parse_args()
    application_root = Path(__file__).resolve().parent
    output = run(
        {
            "application_root": application_root,
            "dataset": args.dataset,
            "mode": args.mode,
            "sample": args.sample,
            "round_number": args.round_number,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "is_textgrad": args.is_textgrad,
            "opt_model_name": args.opt_model_name,
            "exec_model_name": args.exec_model_name,
        },
        specific_indices=args.indices,
    )
    print(json.dumps(output, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MaAS MASFactory reproduction.")
    parser.add_argument("--dataset", choices=sorted(SUPPORTED_DATASETS), required=True)
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), default="Graph")
    parser.add_argument("--sample", type=int, default=4)
    parser.add_argument("--round-number", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--opt-model-name", default="gpt-4o-mini")
    parser.add_argument("--exec-model-name", default="gpt-4o-mini")
    parser.add_argument("--is-textgrad", action="store_true")
    parser.add_argument("--indices", type=int, nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Command-line entry point for the native MaAS reproduction."""
from __future__ import annotations
import argparse
from pathlib import Path
from .maas_reproduction.runtime.settings import load_settings
from .maas_reproduction.runtime.bootstrap import build_runtime

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "test", "smoke"), default="test")
    parser.add_argument("--dataset", choices=("GSM8K", "MATH", "HumanEval"), default=None)
    parser.add_argument("--split", choices=("train", "test"), default=None)
    parser.add_argument("--sample", type=int, default=None, help="number of dataset samples to run")
    parser.add_argument("--config-root", type=Path, default=Path(__file__).parent / "assets" / "config")
    parser.add_argument("--fake-model", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config_root, mode_override=args.mode, dataset_override=args.dataset,
                             split_override=args.split, sample_override=args.sample)
    runtime = build_runtime(settings, fake=args.fake_model or args.mode == "smoke")
    if args.resume is not None:
        if settings.mode != "train":
            raise ValueError("--resume is only valid in train mode")
        payload = runtime.checkpoint_manager.load(args.resume, controller=runtime.policy_controller,
                                                  optimizer=runtime.optimizer,
                                                  expected_operator_catalog=runtime.operator_catalog)
        runtime.dataset_runner.cursor = int(payload["cursor"])
        runtime.dataset_runner.epoch = int(payload["epoch"])
    results = runtime.dataset_runner.run(limit=settings.sample)
    for result in results: print(result)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["main"]

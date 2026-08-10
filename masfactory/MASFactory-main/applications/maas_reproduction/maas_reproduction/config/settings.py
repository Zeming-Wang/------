"""Runtime configuration and path resolution for the migrated MaAS application.

This module is intentionally limited to configuration concerns. It does not instantiate
LLM providers, define the MaAS search space, or execute optimization logic. Keeping those
responsibilities outside this module gives the later MASFactory Graph and runtime nodes a
single, explicit configuration contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .experiments import EXPERIMENT_CONFIGS

DatasetName = Literal["GSM8K", "MATH", "HumanEval"]
ExecutionMode = Literal["Graph", "Test"]
DatasetSplit = Literal["train", "test"]
QuestionType = Literal["math", "code", "qa"]

SUPPORTED_DATASETS: frozenset[str] = frozenset({"GSM8K", "MATH", "HumanEval"})
SUPPORTED_MODES: frozenset[str] = frozenset({"Graph", "Test"})
SUPPORTED_SPLITS: frozenset[str] = frozenset({"train", "test"})
SUPPORTED_QUESTION_TYPES: frozenset[str] = frozenset({"math", "code", "qa"})


def _require_non_empty(value: str, field_name: str) -> str:
    """Validate a required string configuration value."""
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _require_positive_int(value: int, field_name: str) -> int:
    """Validate a positive integer configuration value."""
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _require_non_negative_float(value: float, field_name: str) -> float:
    """Validate a non-negative floating-point configuration value."""
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value

#负责路径管理 数据集 optimized architecture checkpoints runs
@dataclass(frozen=True)
class MaASPaths:
    """Filesystem locations used by the migrated MaAS runtime.

    Args:
        application_root: Root directory of the MASFactory MaAS application.
        data_root: Dataset directory containing ``*_train.jsonl`` and ``*_test.jsonl``.
        optimized_root: Directory containing optimized architecture templates.
        checkpoint_root: Directory for controller checkpoints.
        runs_root: Directory for runtime results and logs.
    """

    application_root: Path
    data_root: Path
    optimized_root: Path
    checkpoint_root: Path
    runs_root: Path

    def __post_init__(self) -> None:
        for field_name in (
            "application_root",
            "data_root",
            "optimized_root",
            "checkpoint_root",
            "runs_root",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                raise TypeError(f"{field_name} must be a pathlib.Path")

    @classmethod
    def from_application_root(cls, application_root: Path) -> "MaASPaths":
        """Build default application-relative paths without creating directories."""
        root = application_root.expanduser().resolve()
        return cls(
            application_root=root,
            data_root=root / "assets" / "data",
            optimized_root=root / "assets" / "optimized",
            checkpoint_root=root / "assets" / "optimized",
            runs_root=root / "runs",
        )

    @classmethod
    def from_environment(cls, application_root: Path) -> "MaASPaths":
        """Resolve path overrides from environment variables.

        Supported variables:
            ``MAAS_DATA_ROOT``
            ``MAAS_OPTIMIZED_ROOT``
            ``MAAS_CHECKPOINT_ROOT``
            ``MAAS_RUNS_ROOT``
        """
        defaults = cls.from_application_root(application_root)
        return cls(
            application_root=defaults.application_root,
            data_root=_resolve_path_override("MAAS_DATA_ROOT", defaults.data_root),
            optimized_root=_resolve_path_override(
                "MAAS_OPTIMIZED_ROOT",
                defaults.optimized_root,
            ),
            checkpoint_root=_resolve_path_override(
                "MAAS_CHECKPOINT_ROOT",
                defaults.checkpoint_root,
            ),
            runs_root=_resolve_path_override("MAAS_RUNS_ROOT", defaults.runs_root),
        )

    def dataset_file(self, dataset: DatasetName, split: DatasetSplit) -> Path:
        """Return the JSONL path for a dataset split."""
        _validate_dataset(dataset)
        _validate_split(split)
        return self.data_root / f"{dataset.lower()}_{split}.jsonl"

    def optimized_dataset_root(self, dataset: DatasetName) -> Path:
        """Return the optimized architecture root for a dataset."""
        _validate_dataset(dataset)
        return self.optimized_root / dataset

    def optimized_split_root(self, dataset: DatasetName, split: DatasetSplit) -> Path:
        """Return the optimized architecture root for a dataset split."""
        _validate_split(split)
        return self.optimized_dataset_root(dataset) / split

    def run_root(self, dataset: DatasetName, mode: ExecutionMode) -> Path:
        """Return the runtime output root for a dataset and execution mode."""
        _validate_dataset(dataset)
        _validate_mode(mode)
        return self.runs_root / dataset / mode

    def round_root(
        self,
        dataset: DatasetName,
        mode: ExecutionMode,
        round_number: int,
    ) -> Path:
        """Return the runtime directory for one optimization round."""
        _require_positive_int(round_number, "round_number")
        return self.run_root(dataset, mode) / f"round_{round_number}"

    def controller_checkpoint(
        self,
        dataset: DatasetName,
        round_number: int,
        sample: int,
    ) -> Path:
        """Return the MaAS controller checkpoint path.

        MaAS stores controller checkpoints under the optimized train round directory.
        Test mode loads the checkpoint produced by Graph mode for the same round/sample.
        """
        _require_positive_int(round_number, "round_number")
        _require_positive_int(sample, "sample")
        return (
            self.checkpoint_root
            / dataset
            / "train"
            / f"round_{round_number}"
            / f"{dataset}_controller_sample{sample}.pth"
        )

# 负责优化参数 对应相关maas参数
@dataclass(frozen=True)
class OptimizerSettings:
    """Parameters controlling MaAS optimization and evaluation."""

    sample: int
    round_number: int
    batch_size: int
    learning_rate: float
    is_textgrad: bool
    opt_model_name: str
    exec_model_name: str

    def __post_init__(self) -> None:
        _require_positive_int(self.sample, "sample")
        _require_positive_int(self.round_number, "round_number")
        _require_positive_int(self.batch_size, "batch_size")
        _require_non_negative_float(self.learning_rate, "learning_rate")
        _require_non_empty(self.opt_model_name, "opt_model_name")
        _require_non_empty(self.exec_model_name, "exec_model_name")


#graph node之间共享的总配置 将相关的内容集中起来 构造maas optimizer时会提供完整参数
@dataclass(frozen=True)
class MaASRuntimeSettings:
    """Complete configuration passed from the MASFactory workflow to MaAS runtime nodes."""

    dataset: DatasetName
    mode: ExecutionMode
    paths: MaASPaths
    optimizer: OptimizerSettings
    question_type: QuestionType
    operators: tuple[str, ...]
    opt_llm_config: object
    exec_llm_config: object

    def __post_init__(self) -> None:
        _validate_dataset(self.dataset)
        _validate_mode(self.mode)
        _validate_question_type(self.question_type)
        if not isinstance(self.paths, MaASPaths):
            raise TypeError("paths must be an instance of MaASPaths")
        if not isinstance(self.optimizer, OptimizerSettings):
            raise TypeError("optimizer must be an instance of OptimizerSettings")
        if not self.operators:
            raise ValueError("operators must not be empty")
        if self.opt_llm_config is None:
            raise ValueError("opt_llm_config must not be None")
        if self.exec_llm_config is None:
            raise ValueError("exec_llm_config must not be None")

    @classmethod
    def from_experiment(
        cls,
        dataset: DatasetName,
        mode: ExecutionMode,
        paths: MaASPaths,
        optimizer: OptimizerSettings,
        opt_llm_config: object,
        exec_llm_config: object,
    ) -> "MaASRuntimeSettings":
        """Build runtime settings from the fixed MaAS experiment search space."""
        _validate_dataset(dataset)
        experiment = EXPERIMENT_CONFIGS[dataset]
        return cls(
            dataset=dataset,
            mode=mode,
            paths=paths,
            optimizer=optimizer,
            question_type=experiment.question_type,
            operators=experiment.operators,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
        )

    @property
    def dataset_file(self) -> Path:
        """Return the configured input dataset file."""
        split: DatasetSplit = "test" if self.mode == "Test" else "train"
        return self.paths.dataset_file(self.dataset, split)

    @property
    def architecture_root(self) -> Path:
        """Return the optimized architecture directory for the active mode."""
        split: DatasetSplit = "test" if self.mode == "Test" else "train"
        return self.paths.optimized_split_root(self.dataset, split)

    @property
    def run_directory(self) -> Path:
        """Return the runtime output directory for the active round."""
        return self.paths.round_root(
            self.dataset,
            self.mode,
            self.optimizer.round_number,
        )


def _resolve_path_override(environment_name: str, default: Path) -> Path:
    """Resolve one optional absolute or relative path override."""
    raw_value = os.getenv(environment_name)
    if raw_value is None or not raw_value.strip():
        return default
    return Path(raw_value).expanduser().resolve()


def _validate_dataset(dataset: str) -> None:
    """Validate a dataset name against the currently implemented MaAS datasets."""
    if dataset not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {supported}")


def _validate_mode(mode: str) -> None:
    """Validate the optimizer execution mode."""
    if mode not in SUPPORTED_MODES:
        supported = ", ".join(sorted(SUPPORTED_MODES))
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {supported}")


def _validate_split(split: str) -> None:
    """Validate a dataset split."""
    if split not in SUPPORTED_SPLITS:
        supported = ", ".join(sorted(SUPPORTED_SPLITS))
        raise ValueError(f"Unsupported split '{split}'. Supported splits: {supported}")


def _validate_question_type(question_type: str) -> None:
    """Validate a MaAS question type."""
    if question_type not in SUPPORTED_QUESTION_TYPES:
        supported = ", ".join(sorted(SUPPORTED_QUESTION_TYPES))
        raise ValueError(
            f"Unsupported question type '{question_type}'. Supported question types: {supported}"
        )

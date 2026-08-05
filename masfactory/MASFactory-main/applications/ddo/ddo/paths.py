from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DDO_ROOT = PROJECT_ROOT / "ddo"

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
POLICY_DIR = PROJECT_ROOT / "policy"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def get_policy_run_dir(dataset_name: str):
    """Return the local policy directory, accepting the typo used by the source repo."""
    candidates = [
        POLICY_DIR / dataset_name / "custom_rl_training",
        POLICY_DIR / dataset_name / "curstom_rl_training",
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]

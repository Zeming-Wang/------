"""Show what a MaAS run actually did, epoch by epoch.

Read-only.  Answers three questions that the raw artefacts do not answer at a
glance: did every epoch complete, how did each epoch differ, and did the policy
weights actually move between epochs.

    python scripts/inspect_run.py                 # newest run
    python scripts/inspect_run.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "output"


def newest_run(root: Path) -> Path:
    runs = sorted(root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit(f"no run_* directory under {root}")
    return runs[0]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(run: Path) -> list[dict]:
    path = run / "logs" / "run.log"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def epoch_table(events: list[dict]) -> None:
    """Per-epoch sample outcomes, straight out of run.log."""
    per_epoch: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        if event.get("event") == "sample_completed":
            per_epoch[event.get("epoch")].append(event)
    if not per_epoch:
        print("  (no sample_completed events)")
        return
    print(f"  {'epoch':>5} {'samples':>8} {'success':>8} {'fail':>6} {'avg cost':>10}  statuses")
    for epoch in sorted(per_epoch):
        rows = per_epoch[epoch]
        ok = sum(1 for r in rows if r.get("status") == "success")
        costs = [r.get("cost_delta") for r in rows if isinstance(r.get("cost_delta"), (int, float))]
        statuses = sorted({str(r.get("status")) for r in rows})
        avg = f"{sum(costs) / len(costs):.6f}" if costs else "-"
        print(f"  {epoch:>5} {len(rows):>8} {ok:>8} {len(rows) - ok:>6} {avg:>10}  {','.join(statuses)}")


def checkpoint_table(run: Path) -> None:
    """Cursor/metrics per checkpoint, plus how far the policy moved between them."""
    files = sorted((run / "checkpoints").glob("checkpoint_epoch_*.pt"))
    latest = run / "checkpoints" / "latest.pt"
    if not files and not latest.exists():
        print("  (no checkpoints)")
        return
    try:
        import torch
    except ImportError:
        print("  (torch unavailable; skipping checkpoint inspection)")
        return

    previous = None
    print(f"  {'file':<28} {'epoch':>5} {'cursor':>6} {'updates':>8} {'samples':>8} {'|dW| vs prev':>13}")
    for path in files + ([latest] if latest.exists() else []):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metrics = payload.get("metadata", {}).get("runner_metrics", {})
        controller = payload.get("controller", {})
        drift = "-"
        if previous is not None and controller:
            total = 0.0
            for key, value in controller.items():
                old = previous.get(key)
                if torch.is_tensor(value) and torch.is_tensor(old) and value.shape == old.shape:
                    total += float((value - old).abs().sum())
            drift = f"{total:.6f}"
        print(f"  {path.name:<28} {payload.get('epoch', '-'):>5} {payload.get('cursor', '-'):>6} "
              f"{metrics.get('update_count', '-'):>8} {metrics.get('sample_count', '-'):>8} {drift:>13}")
        previous = controller or previous

    if len(files) >= 2:
        first = torch.load(files[0], map_location="cpu", weights_only=False).get("controller", {})
        last = torch.load(files[-1], map_location="cpu", weights_only=False).get("controller", {})
        total = 0.0
        for key, value in first.items():
            other = last.get(key)
            if torch.is_tensor(value) and torch.is_tensor(other) and value.shape == other.shape:
                total += float((value - other).abs().sum())
        verdict = "policy unchanged -- no optimizer step ever ran" if total == 0.0 \
            else f"policy moved by {total:.6f} in total absolute weight"
        print(f"\n  {files[0].name} -> {files[-1].name}: {verdict}")


def main(argv: list[str]) -> int:
    run = Path(argv[1]).resolve() if len(argv) > 1 else newest_run(OUTPUT_ROOT)
    print(f"run: {run}\n")

    print("[config]")
    for key, value in read_json(run / "config.json").items():
        print(f"  {key:18}: {value}")

    print("\n[metrics.json]  (written only when a run reaches the end)")
    for key, value in read_json(run / "metrics.json").items():
        print(f"  {key:18}: {value}")

    print("\n[per epoch, from logs/run.log]")
    epoch_table(load_events(run))

    print("\n[checkpoints]")
    checkpoint_table(run)

    print("\nNote: sample_i.json is overwritten by each epoch, so results/ only holds the last epoch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

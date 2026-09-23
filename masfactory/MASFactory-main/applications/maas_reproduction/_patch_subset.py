"""One-shot patch: add --subset so an epoch can complete on a small run."""
from pathlib import Path


def patch(path, pairs):
    p = Path(path)
    with open(p, "r", encoding="utf-8", newline="") as handle:
        text = handle.read()
    assert "\r\n" not in text, f"{path} has CRLF"
    for old, new in pairs:
        count = text.count(old)
        assert count == 1, f"{path}: expected 1 occurrence, found {count} for:\n{old[:140]}"
        text = text.replace(old, new, 1)
    with open(p, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    print("patched", path)


patch("main.py", [(
    '    parser.add_argument("--epochs", type=int, default=None, help="override the configured number of epochs")\n',
    '    parser.add_argument("--epochs", type=int, default=None, help="override the configured number of epochs")\n'
    '    parser.add_argument("--subset", type=int, default=None,\n'
    '                        help="use only the first N dataset samples, so a small run can still complete an epoch")\n',
), (
    "    if args.fresh and args.resume is not None:\n"
    '        raise ValueError("--fresh and --resume are mutually exclusive")\n',
    "    if args.fresh and args.resume is not None:\n"
    '        raise ValueError("--fresh and --resume are mutually exclusive")\n'
    "    if args.subset is not None and args.subset <= 0:\n"
    '        raise ValueError("--subset must be positive")\n',
), (
    "    runtime = build_runtime(settings, fake=args.fake_model or args.mode == \"smoke\",\n"
    "                            output_root_override=output_override)\n",
    "    runtime = build_runtime(settings, fake=args.fake_model or args.mode == \"smoke\",\n"
    "                            output_root_override=output_override, subset=args.subset)\n",
), (
    "    for result in runtime.dataset_runner.iter_run(limit=settings.sample):\n"
    "        print(result)\n",
    "    # ``--subset`` already bounds the run by the size of the subset, so the\n"
    "    # per-invocation budget only applies when it was asked for explicitly.\n"
    "    limit = settings.sample\n"
    "    if args.subset is not None and args.sample is None:\n"
    "        limit = None\n"
    "    for result in runtime.dataset_runner.iter_run(limit=limit):\n"
    "        print(result)\n",
)])

patch("maas_reproduction/runtime/bootstrap.py", [(
    "def build_runtime(settings: RuntimeSettings, *, fake: bool = False, manager: Any = None,\n"
    "                  output_root_override: Path | None = None) -> RuntimeContext:\n",
    "def build_runtime(settings: RuntimeSettings, *, fake: bool = False, manager: Any = None,\n"
    "                  output_root_override: Path | None = None,\n"
    "                  subset: int | None = None) -> RuntimeContext:\n",
), (
    "    runner = DatasetRunner(graph=root, dataset=dataset, epochs=settings.epochs,\n",
    "    if subset is not None:\n"
    "        if isinstance(subset, bool) or not isinstance(subset, int) or subset <= 0:\n"
    '            raise ValueError("subset must be a positive int")\n'
    "        dataset = dataset[:subset]\n"
    "    runner = DatasetRunner(graph=root, dataset=dataset, epochs=settings.epochs,\n",
), (
    '                           run_config={"dataset": settings.dataset, "split": settings.split,\n',
    '                           run_config={"dataset": settings.dataset, "split": settings.split,\n'
    '                                       "subset": subset,\n',
)])

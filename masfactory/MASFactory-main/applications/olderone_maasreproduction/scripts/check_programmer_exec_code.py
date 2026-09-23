"""Check optimized Programmer.exec_code outside the LLM workflow."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
import sys


async def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    maas_root = os.getenv("METAGPT_PROJECT_ROOT")
    if maas_root:
        sys.path.insert(0, maas_root)
    sys.path.insert(0, str(project_root))

    results = []
    for module_name in (
        "assets.optimized.GSM8K.train.template.operator",
        "assets.optimized.GSM8K.test.template.operator",
        "assets.optimized.MATH.train.template.operator",
        "assets.optimized.MATH.test.template.operator",
    ):
        module = importlib.import_module(module_name)
        programmer = module.Programmer(llm=None)
        status, output = await programmer.exec_code(
            "def solve():\n"
            "    return 72\n",
            timeout=60,
        )
        results.append({"module": module_name, "status": status, "output": output})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

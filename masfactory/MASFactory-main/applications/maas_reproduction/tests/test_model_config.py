from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class ModelConfigTest(unittest.TestCase):
    def test_resolve_model_configs_prefers_metagpt_project_root(self) -> None:
        from maas_reproduction.config.model_config import resolve_model_configs

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "maas" / "configs"
            config_root.mkdir(parents=True)
            (root / "maas" / "__init__.py").write_text("", encoding="utf-8")
            (config_root / "__init__.py").write_text("", encoding="utf-8")
            (config_root / "models_config.py").write_text(
                "class ModelsConfig:\n"
                "    @classmethod\n"
                "    def default(cls):\n"
                "        return {'opt': {'model': 'opt'}, 'exec': {'model': 'exec'}}\n",
                encoding="utf-8",
            )
            _clear_maas_modules()
            try:
                with patch.dict("os.environ", {"METAGPT_PROJECT_ROOT": str(root)}):
                    opt_config, exec_config = resolve_model_configs("opt", "exec")
            finally:
                _clear_maas_modules()
                if str(root) in sys.path:
                    sys.path.remove(str(root))

        self.assertEqual(opt_config, {"model": "opt"})
        self.assertEqual(exec_config, {"model": "exec"})


def _clear_maas_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "maas" or module_name.startswith("maas."):
            del sys.modules[module_name]


if __name__ == "__main__":
    unittest.main()

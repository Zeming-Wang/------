from pathlib import Path
import unittest


class OptimizedAssetsTest(unittest.TestCase):
    def test_workflows_accumulate_logprob_as_tensor(self) -> None:
        assets_root = Path(__file__).resolve().parents[1] / "assets" / "optimized"
        graph_files = sorted(assets_root.glob("*/*/graph.py"))

        self.assertTrue(graph_files)
        for graph_file in graph_files:
            source = graph_file.read_text(encoding="utf-8")
            self.assertIn("sum_log_prob = torch.tensor(0.0, device=self.device)", source)
            self.assertNotIn("log_probs_layers[layer_idx].item()", source)


if __name__ == "__main__":
    unittest.main()

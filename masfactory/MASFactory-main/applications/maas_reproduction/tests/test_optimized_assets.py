from pathlib import Path
import unittest


class OptimizedAssetsTest(unittest.TestCase):
    def test_gsm8k_cot_prompt_escapes_literal_answer_braces(self) -> None:
        prompt_file = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "optimized"
            / "GSM8K"
            / "train"
            / "template"
            / "op_prompt.py"
        )
        source = prompt_file.read_text(encoding="utf-8")

        self.assertIn(r"\boxed{{-2}}, \boxed{{2}}", source)

    def test_gsm8k_cot_prompt_formats_without_replacement_errors(self) -> None:
        from assets.optimized.GSM8K.train.template import op_prompt

        formatted = op_prompt.GENERATE_COT_PROMPT.format(
            input="test problem",
            instruction="solve carefully",
        )

        self.assertIn(r"\boxed{-2}", formatted)
        self.assertIn(r"\boxed{144\pi}", formatted)

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

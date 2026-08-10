import unittest

from maas_reproduction.nodes.result_node import result_forward


class ResultNodeTest(unittest.TestCase):
    def test_returns_training_loop_summary(self) -> None:
        result = result_forward(
            {
                "average_score": 0.75,
                "round": 2,
                "checkpoint_path": "checkpoint.pth",
                "result_path": "runs",
                "runtime_metadata": {"dataset": "GSM8K"},
            },
            {},
        )

        self.assertEqual(result["average_score"], 0.75)
        self.assertEqual(result["round"], 2)
        self.assertEqual(result["checkpoint_path"], "checkpoint.pth")
        self.assertEqual(result["result_path"], "runs")
        self.assertEqual(result["runtime_metadata"], {"dataset": "GSM8K"})


if __name__ == "__main__":
    unittest.main()

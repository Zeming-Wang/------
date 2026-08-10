import unittest

from maas_reproduction.graphs.training_loop import (
    CONTROLLER_TO_ARCHITECTURE_KEYS,
    EVALUATOR_TO_LOSS_KEYS,
    LOSS_TO_CONTROLLER_KEYS,
    TRAINING_LOOP_PUSH_KEYS,
)


class TrainingLoopGraphTest(unittest.TestCase):
    def test_declares_plan_edge_keys(self) -> None:
        self.assertEqual(
            set(CONTROLLER_TO_ARCHITECTURE_KEYS),
            {"problem", "entry_point", "expected_answer", "problem_index"},
        )
        self.assertEqual(
            set(EVALUATOR_TO_LOSS_KEYS),
            {"score", "cost", "logprob", "problem_index"},
        )
        self.assertTrue(all(key.startswith("result_") for key in LOSS_TO_CONTROLLER_KEYS))
        self.assertEqual(
            set(TRAINING_LOOP_PUSH_KEYS),
            {"average_score", "round", "checkpoint_path", "result_path", "runtime_metadata"},
        )


if __name__ == "__main__":
    unittest.main()

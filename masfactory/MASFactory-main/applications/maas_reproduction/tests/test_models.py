import importlib
import sys
import types
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


def install_sentence_transformers_stub() -> None:
    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name

        def parameters(self):
            return []

        def encode(self, sentence: str):
            return [0.0] * 384

    module.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = module


class ModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if torch is None:
            raise unittest.SkipTest("torch is required for MaAS controller tests")
        install_sentence_transformers_stub()
        cls.utils = importlib.import_module("maas_reproduction.models.utils")
        cls.controller = importlib.import_module("maas_reproduction.models.controller")

    def test_sample_operators_selects_until_threshold(self) -> None:
        probs = torch.tensor([1.0, 0.0, 0.0])

        selected = self.utils.sample_operators(probs, threshold=0.3)

        self.assertEqual(selected.tolist(), [0])

    def test_operator_selector_returns_operator_distribution(self) -> None:
        selector = self.controller.OperatorSelector(
            input_dim=4,
            hidden_dim=2,
            device=torch.device("cpu"),
            is_first_layer=True,
        )
        query_embedding = torch.ones(4)
        operator_embeddings = torch.eye(3, 4)

        log_probs, probs = selector(query_embedding, operator_embeddings)

        self.assertEqual(log_probs.shape, (1, 3))
        self.assertEqual(probs.shape, (1, 3))
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.tensor([1.0])))

    def test_controller_replaces_first_layer_earlystop_with_generate(self) -> None:
        module = self.controller
        original_encoder = module.sentence_encoder
        original_sampler = module.sample_operators
        module.sentence_encoder = lambda query: torch.ones(4)
        module.sample_operators = lambda probs, threshold=0.3: torch.tensor([1])
        try:
            controller = module.MultiLayerController(
                input_dim=4,
                hidden_dim=2,
                num_layers=4,
                device=torch.device("cpu"),
            )
            operator_embeddings = torch.eye(2, 4)

            log_probs_layers, selected_names_layers = controller(
                "question",
                operator_embeddings,
                ["Generate", "EarlyStop"],
            )
        finally:
            module.sentence_encoder = original_encoder
            module.sample_operators = original_sampler

        self.assertEqual(selected_names_layers, [["Generate"]])
        self.assertEqual(len(log_probs_layers), 1)
        self.assertLess(log_probs_layers[0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()

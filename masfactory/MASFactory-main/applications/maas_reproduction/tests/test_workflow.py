import unittest

from maas_reproduction.workflow import build_maas_reproduction_graph


class WorkflowTest(unittest.TestCase):
    def test_builds_root_graph(self) -> None:
        graph = build_maas_reproduction_graph()
        graph.build()

        self.assertTrue(graph.check_built())


if __name__ == "__main__":
    unittest.main()

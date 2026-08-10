import unittest
from unittest.mock import Mock, patch


class MainTest(unittest.TestCase):
    def test_run_builds_runtime_attributes_before_invoking_graph(self) -> None:
        from applications.maas_reproduction import main

        graph = Mock()
        graph.invoke.return_value = ({"average_score": 1.0}, {"attrs": True})
        settings = object()
        runtime_attrs = {"settings": settings, "problems": []}

        with patch.object(main, "config_forward", return_value={"settings": settings}) as config_forward:
            with patch.object(main, "build_runtime_attributes", return_value=runtime_attrs) as build_attrs:
                with patch.object(main, "build_maas_reproduction_graph", return_value=graph):
                    output = main.run({"dataset": "GSM8K", "application_root": "app"}, specific_indices=[0])

        graph.build.assert_called_once_with()
        config_forward.assert_called_once_with({"dataset": "GSM8K", "application_root": "app"}, {})
        build_attrs.assert_called_once_with(settings, specific_indices=[0])
        graph.invoke.assert_called_once_with(
            {"dataset": "GSM8K", "application_root": "app"},
            attributes=runtime_attrs,
        )
        self.assertEqual(output, {"average_score": 1.0})


if __name__ == "__main__":
    unittest.main()

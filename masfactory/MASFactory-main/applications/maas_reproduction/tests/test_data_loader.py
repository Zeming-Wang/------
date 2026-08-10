import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from maas_reproduction.runtime.data_loader import load_jsonl_data, load_problems


class DataLoaderTest(unittest.TestCase):
    def test_load_jsonl_data_reads_all_records(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "sample.jsonl"
            records = [{"question": "q1"}, {"question": "q2"}]
            data_file.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(load_jsonl_data(data_file), records)

    def test_load_jsonl_data_filters_existing_indices(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "sample.jsonl"
            records = [{"question": "q1"}, {"question": "q2"}]
            data_file.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(load_jsonl_data(data_file, specific_indices=[1, 5]), [{"question": "q2"}])

    def test_load_problems_uses_settings_dataset_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "sample.jsonl"
            data_file.write_text(json.dumps({"question": "q1"}) + "\n", encoding="utf-8")

            class Settings:
                dataset_file = data_file

            self.assertEqual(load_problems(Settings()), [{"question": "q1"}])


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest

from maas_reproduction.runtime.async_runner import run_async_once


async def _return_value() -> str:
    return "ok"


class AsyncRunnerTest(unittest.TestCase):
    def test_runs_coroutine_from_sync_context(self) -> None:
        self.assertEqual(run_async_once(_return_value()), "ok")

    def test_rejects_running_event_loop(self) -> None:
        async def call_inside_loop() -> None:
            coro = _return_value()
            try:
                with self.assertRaises(RuntimeError):
                    run_async_once(coro)
            finally:
                coro.close()

        asyncio.run(call_inside_loop())


if __name__ == "__main__":
    unittest.main()

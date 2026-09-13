from ..workflow import BenchmarkCompletionGraph
class HumanEvalCompletionGraph(BenchmarkCompletionGraph):
    def __init__(self, name: str = "humaneval_completion", **kwargs): super().__init__(name, dataset="HumanEval", **kwargs)
__all__ = ["HumanEvalCompletionGraph"]

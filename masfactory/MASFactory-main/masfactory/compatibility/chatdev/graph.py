from masfactory.compatibility.common.output_root import WorkflowOutputRootGraph


class ChatDevRootGraph(WorkflowOutputRootGraph):
    _outputs_key = "chatdev_outputs"
    _preferred_suffixes = ("DONE", "FINAL", "END", "HAS_CODE", "NO_CODE")

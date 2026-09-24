"""Dify / graphon official workflow node type strings (DSL `data.type` values)."""

from __future__ import annotations

# graphon BuiltinNodeTypes (v0.3.1) — primary workflow node vocabulary.
GRAPHON_BUILTIN_NODE_KINDS: frozenset[str] = frozenset(
    {
        "start",
        "end",
        "answer",
        "llm",
        "knowledge-retrieval",
        "if-else",
        "code",
        "template-transform",
        "question-classifier",
        "http-request",
        "tool",
        "datasource",
        "variable-aggregator",
        "variable-assigner",  # legacy name for variable-aggregator
        "loop",
        "loop-start",
        "loop-end",
        "iteration",
        "iteration-start",
        "parameter-extractor",
        "assigner",
        "document-extractor",
        "list-operator",
        "agent",
        "human-input",
    }
)

# Dify workflow layer extensions (api/core/workflow/nodes + triggers).
DIFY_EXTENDED_NODE_KINDS: frozenset[str] = frozenset(
    {
        "knowledge-index",
        "trigger-webhook",
        "trigger-schedule",
        "trigger-plugin",
    }
)

# Canvas-only / annotation nodes (not executed in Dify runtime).
DIFY_UI_NODE_KINDS: frozenset[str] = frozenset(
    {
        "custom-note",
        "note",
    }
)

OFFICIAL_DIFY_NODE_KINDS: frozenset[str] = (
    GRAPHON_BUILTIN_NODE_KINDS | DIFY_EXTENDED_NODE_KINDS | DIFY_UI_NODE_KINDS
)

# Nodes that only exist inside loop/iteration containers in the main graph export.
CONTAINER_MARKER_KINDS: frozenset[str] = frozenset(
    {
        "loop-start",
        "loop-end",
        "iteration-start",
    }
)

DIFY_ENTRY_NODE_KINDS: frozenset[str] = frozenset(
    {
        "start",
        "datasource",
        "trigger-webhook",
        "trigger-schedule",
        "trigger-plugin",
    }
)

DIFY_END_NODE_KINDS: frozenset[str] = frozenset({"end", "answer"})

from __future__ import annotations

from workflows.common.tools import excerpt, parse_bool, write_text


def finalize_outline_iteration(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    outline_markdown = attrs.get("outline_markdown", "")
    source_map_markdown = attrs.get("source_map_markdown", "")
    outline_path = write_text(attrs["outline_path"], outline_markdown)
    source_map_path = write_text(attrs["source_map_path"], source_map_markdown)
    human_ready = parse_bool(attrs.get("outline_human_decision"))
    ready = human_ready

    feedback = str(attrs.get("outline_human_feedback", "")).strip()
    outline_excerpt = excerpt(outline_markdown, limit=10000)
    source_map_excerpt = excerpt(source_map_markdown, limit=6000)

    report = "\n".join(
        [
            "# Outline Review",
            "",
            f"- Human ready: {human_ready}",
            f"- Final ready: {ready}",
            "- Gate policy: human review is authoritative",
            "",
            "## Human Feedback",
            feedback,
        ]
    )
    review_path = write_text(attrs["outline_review_path"], report)
    return {
        "outline_ready": ready,
        "outline_iteration_feedback": feedback,
        "outline_review_path": review_path,
        "outline_markdown": outline_markdown,
        "source_map_markdown": source_map_markdown,
        "outline_path": outline_path,
        "source_map_path": source_map_path,
        "outline_excerpt": outline_excerpt,
        "source_map_excerpt": source_map_excerpt,
    }


def should_terminate_outline(_: dict[str, object], attrs: dict[str, object]) -> bool:
    return parse_bool(attrs.get("outline_ready"))


__all__ = [
    "finalize_outline_iteration",
    "should_terminate_outline",
]

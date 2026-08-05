from __future__ import annotations

import json
import re
from pathlib import Path

from tools import html_to_pdf_with_playwright, merge_pdfs, pdf_to_pngs, pdf_to_text
from workflows.common.tools import (
    ensure_dir,
    excerpt,
    get_stage_batch_size,
    normalize_outline_slide_sections,
    parse_bool,
    parse_outline_slide_sections,
    render_html_asset_validation,
    render_slide_batch_markdown,
    validate_html_local_image_refs,
    write_text,
)


def _candidate_page_refs(asset_dir: str | Path) -> list[str]:
    asset_root = Path(str(asset_dir)).expanduser().resolve()
    refs: list[str] = []
    for paper_dir in sorted(path for path in asset_root.glob("paper*") if path.is_dir()):
        for page_file in sorted((paper_dir / "pages").glob("page*.png")):
            refs.append(f"../assets/{paper_dir.name}/pages/{page_file.name}")
    return refs


def _count_html_slides(html_content: str) -> int:
    return len(re.findall(r'(?is)<section\b[^>]*class=["\'][^"\']*\bslide\b', html_content or ""))


def _rewrite_missing_asset_refs(html_content: str, html_path: str | Path, asset_dir: str | Path) -> str:
    _, missing_refs = validate_html_local_image_refs(html_content, html_path)
    if not missing_refs:
        return html_content

    candidate_refs = _candidate_page_refs(asset_dir)
    if not candidate_refs:
        return html_content

    replacement_index = 0
    rewritten = html_content
    for item in missing_refs:
        src = item["src"]
        if "/paper" in src:
            continue
        if not src.startswith("../assets/"):
            continue
        if replacement_index >= len(candidate_refs):
            break
        rewritten = rewritten.replace(src, candidate_refs[replacement_index], 1)
        replacement_index += 1
    return rewritten


def persist_html_outputs(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    html_content = str(attrs.get("html_content", ""))
    structure_notes_markdown = attrs.get("structure_notes_markdown", "")
    html_path = Path(str(attrs["html_path"])).expanduser().resolve()
    html_content = _rewrite_missing_asset_refs(html_content, html_path, attrs.get("asset_dir", ""))
    html_path_str = write_text(html_path, html_content)
    structure_notes_path = write_text(attrs["structure_notes_path"], structure_notes_markdown)
    local_refs, missing_refs = validate_html_local_image_refs(html_content, html_path_str)
    asset_validation_markdown = render_html_asset_validation(local_refs, missing_refs)
    return {
        "html_content": html_content,
        "html_path": html_path_str,
        "structure_notes_path": structure_notes_path,
        "html_excerpt": excerpt(html_content, limit=14000),
        "structure_notes_excerpt": excerpt(structure_notes_markdown, limit=7000),
        "html_asset_validation_markdown": asset_validation_markdown,
        "html_missing_asset_refs": "\n".join(item["src"] for item in missing_refs),
        "html_has_missing_assets": bool(missing_refs),
    }


def _export_single_html_document(
    *,
    html_content: str,
    structure_notes_markdown: str,
    asset_dir: str | Path,
    html_path: str | Path,
    structure_notes_path: str | Path,
    pdf_path: str | Path,
    preview_dir: str | Path,
    export_report_path: str | Path,
    strategy_label: str,
    page_budget: object | None = None,
) -> dict[str, object]:
    persisted = persist_html_outputs(
        {},
        {
            "html_content": html_content,
            "structure_notes_markdown": structure_notes_markdown,
            "asset_dir": asset_dir,
            "html_path": str(html_path),
            "structure_notes_path": str(structure_notes_path),
        },
    )

    html_path_str = str(persisted["html_path"])
    pdf_path = Path(str(pdf_path))
    preview_dir = ensure_dir(preview_dir)
    export_report_path = Path(str(export_report_path))
    playwright_strategy = "slide_screenshots" if strategy_label == "slide_screenshots_to_pdf" else "print_pdf"

    success = False
    error_text = ""
    pdf_text = ""
    preview_paths: list[Path] = []
    html_body = Path(html_path_str).read_text(encoding="utf-8", errors="ignore")
    expected_slide_count = 0
    try:
        expected_slide_count = max(0, int(page_budget or 0))
    except (TypeError, ValueError):
        expected_slide_count = 0
    actual_slide_count = _count_html_slides(html_body)
    local_refs, missing_refs = validate_html_local_image_refs(html_body, html_path_str)
    asset_validation_markdown = render_html_asset_validation(local_refs, missing_refs)

    try:
        html_to_pdf_with_playwright(
            html_path_str,
            pdf_path,
            strategy=playwright_strategy,
            package_dir=ensure_dir(preview_dir.parent / "playwright_env"),
            viewport=(1600, 900),
        )
        preview_paths = pdf_to_pngs(pdf_path, preview_dir / "slide", dpi=144)
        text_path = str(pdf_path).replace(".pdf", ".txt")
        pdf_to_text(pdf_path, output_txt=text_path)
        pdf_text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
        success = not missing_refs
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"

    if missing_refs:
        missing_text = "; ".join(item["src"] for item in missing_refs)
        error_text = f"{error_text} | missing local image refs: {missing_text}".strip(" |")
    if expected_slide_count > 0 and actual_slide_count != expected_slide_count:
        mismatch = f"slide count mismatch: expected {expected_slide_count}, got {actual_slide_count}"
        error_text = f"{error_text} | {mismatch}".strip(" |")
        success = False

    preview_manifest = "\n".join(str(path) for path in preview_paths) if preview_paths else "(no previews generated)"
    export_report = "\n".join(
        [
            "# Export Report",
            "",
            f"- HTML: {html_path_str}",
            f"- PDF: {pdf_path}",
            f"- Preview dir: {preview_dir}",
            f"- Strategy label: {strategy_label}",
            f"- Playwright strategy: {playwright_strategy}",
            f"- Expected slide count: {expected_slide_count or '(not set)'}",
            f"- Actual slide count: {actual_slide_count}",
            f"- Success: {success}",
            f"- Error: {error_text or '(none)'}",
            "",
            "## HTML Asset Validation",
            asset_validation_markdown,
            "",
            "## Preview Files",
            preview_manifest,
            "",
            "## PDF Text Excerpt",
            excerpt(pdf_text, limit=7000) if pdf_text else "(not available)",
        ]
    )
    write_text(export_report_path, export_report)
    rendered_pdf_summary = "\n".join(
        [
            "# Rendered PDF Summary",
            "",
            f"- PDF path: {pdf_path}",
            f"- Preview dir: {preview_dir}",
            f"- Export success: {success}",
            f"- Strategy: {strategy_label}",
            f"- Expected slide count: {expected_slide_count or '(not set)'}",
            f"- Actual slide count: {actual_slide_count}",
            "",
            "## HTML Asset Validation",
            asset_validation_markdown,
            "",
            "## Preview Files",
            preview_manifest,
            "",
            "## PDF Text Excerpt",
            excerpt(pdf_text, limit=5000) if pdf_text else "(not available)",
        ]
    )
    return {
        "html_content": str(persisted["html_content"]),
        "html_path": html_path_str,
        "html_excerpt": str(persisted["html_excerpt"]),
        "structure_notes_markdown": structure_notes_markdown,
        "structure_notes_path": str(persisted["structure_notes_path"]),
        "pdf_export_success": success,
        "html_actual_slide_count": actual_slide_count,
        "html_expected_slide_count": expected_slide_count,
        "rendered_pdf_path": str(pdf_path),
        "rendered_pdf_preview_dir": str(preview_dir),
        "rendered_pdf_summary_markdown": rendered_pdf_summary,
        "export_report_path": str(export_report_path),
        "html_asset_validation_markdown": asset_validation_markdown,
        "html_missing_asset_refs": "\n".join(item["src"] for item in missing_refs),
        "html_has_missing_assets": bool(missing_refs),
    }


def export_html_and_pdf(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    return _export_single_html_document(
        html_content=str(attrs.get("html_content", "")),
        structure_notes_markdown=str(attrs.get("structure_notes_markdown", "")),
        asset_dir=str(attrs.get("asset_dir", "")),
        html_path=str(attrs["html_path"]),
        structure_notes_path=str(attrs["structure_notes_path"]),
        pdf_path=str(attrs["rendered_pdf_path"]),
        preview_dir=str(attrs["rendered_pdf_preview_dir"]),
        export_report_path=str(attrs["export_report_path"]),
        strategy_label=str(attrs.get("pdf_export_strategy") or "slide_screenshots_to_pdf"),
        page_budget=attrs.get("page_budget"),
    )


def _extract_slide_sections(html_text: str) -> tuple[str, list[str], str]:
    pattern = re.compile(r"(?is)<section\b[^>]*class=[\"'][^\"']*\bslide\b[^\"']*[\"'][^>]*>.*?</section>")
    matches = list(pattern.finditer(html_text))
    if not matches:
        return html_text, [], ""
    prefix = html_text[: matches[0].start()]
    suffix = html_text[matches[-1].end() :]
    sections = [match.group(0).strip() for match in matches]
    return prefix, sections, suffix


def _merge_html_decks(html_paths: list[str | Path], output_html_path: str | Path) -> str:
    resolved_paths = [Path(str(path)).expanduser().resolve() for path in html_paths]
    if not resolved_paths:
        return write_text(output_html_path, "")

    first_html = resolved_paths[0].read_text(encoding="utf-8", errors="ignore")
    prefix, first_sections, suffix = _extract_slide_sections(first_html)
    merged_sections = list(first_sections)

    for path in resolved_paths[1:]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        _, sections, _ = _extract_slide_sections(text)
        merged_sections.extend(sections)

    if merged_sections:
        merged_html = prefix.rstrip() + "\n\n" + "\n\n".join(merged_sections) + "\n\n" + suffix.lstrip()
    else:
        merged_html = first_html
    return write_text(output_html_path, merged_html)


def prepare_html_batch_plan(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    sections = normalize_outline_slide_sections(
        parse_outline_slide_sections(attrs.get("outline_markdown", "")),
        page_budget=attrs.get("page_budget"),
    )
    batch_size = get_stage_batch_size(default=10)
    batch_count = max(1, (len(sections) + batch_size - 1) // batch_size) if sections else 1
    return {
        "html_batch_sections_json": json.dumps(sections, ensure_ascii=False),
        "html_batch_size": batch_size,
        "html_batch_count": batch_count,
        "html_batch_index": 0,
        "html_batch_outputs_json": "[]",
    }


def prepare_html_batch_iteration(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    sections = json.loads(str(attrs.get("html_batch_sections_json") or "[]"))
    batch_size = int(attrs.get("html_batch_size") or get_stage_batch_size(default=10))
    batch_count = int(attrs.get("html_batch_count") or 1)
    batch_index = int(attrs.get("html_batch_index") or 0)

    if sections:
        start = batch_index * batch_size
        end = min(start + batch_size, len(sections))
        current_sections = sections[start:end]
        current_batch_markdown = render_slide_batch_markdown(
            current_sections,
            batch_index=batch_index + 1,
            batch_count=batch_count,
        )
        batch_label = (
            f"batch {batch_index + 1}/{batch_count} | slides "
            f"{current_sections[0]['number']}-{current_sections[-1]['number']}"
            if current_sections
            else f"batch {batch_index + 1}/{batch_count}"
        )
    else:
        current_batch_markdown = str(attrs.get("outline_markdown", ""))
        batch_label = "batch 1/1 | full deck"

    return {
        "html_current_batch_markdown": current_batch_markdown,
        "html_batch_label": batch_label,
        "html_batch_html_path": str(
            ensure_dir(Path(str(attrs.get("html_path", "") or ".")).expanduser().resolve().parent / "batches" / f"batch_{batch_index + 1:02d}")
            / "slides.html"
        ),
        "html_batch_structure_notes_path": str(
            ensure_dir(Path(str(attrs.get("html_path", "") or ".")).expanduser().resolve().parent / "batches" / f"batch_{batch_index + 1:02d}")
            / "structure_notes.md"
        ),
        "html_batch_pdf_path": str(
            ensure_dir(Path(str(attrs.get("html_path", "") or ".")).expanduser().resolve().parent / "batches" / f"batch_{batch_index + 1:02d}")
            / "slides.pdf"
        ),
        "html_batch_preview_dir": str(
            ensure_dir(Path(str(attrs.get("html_path", "") or ".")).expanduser().resolve().parent / "batches" / f"batch_{batch_index + 1:02d}" / "preview")
        ),
        "html_batch_export_report_path": str(
            ensure_dir(Path(str(attrs.get("html_path", "") or ".")).expanduser().resolve().parent / "batches" / f"batch_{batch_index + 1:02d}")
            / "export_report.md"
        ),
    }


def export_html_batch_pdf(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    export_result = _export_single_html_document(
        html_content=str(attrs.get("html_content", "")),
        structure_notes_markdown=str(attrs.get("structure_notes_markdown", "")),
        asset_dir=str(attrs.get("asset_dir", "")),
        html_path=str(attrs.get("html_batch_html_path", "")),
        structure_notes_path=str(attrs.get("html_batch_structure_notes_path", "")),
        pdf_path=str(attrs.get("html_batch_pdf_path", "")),
        preview_dir=str(attrs.get("html_batch_preview_dir", "")),
        export_report_path=str(attrs.get("html_batch_export_report_path", "")),
        strategy_label=str(attrs.get("pdf_export_strategy") or "slide_screenshots_to_pdf"),
    )
    return {
        "html_batch_html_content": export_result["html_content"],
        "html_batch_html_path": export_result["html_path"],
        "html_batch_structure_notes_markdown": export_result["structure_notes_markdown"],
        "html_batch_structure_notes_path": export_result["structure_notes_path"],
        "html_batch_pdf_export_success": export_result["pdf_export_success"],
        "html_batch_pdf_path": export_result["rendered_pdf_path"],
        "html_batch_preview_dir": export_result["rendered_pdf_preview_dir"],
        "html_batch_rendered_pdf_summary_markdown": export_result["rendered_pdf_summary_markdown"],
        "html_batch_export_report_path": export_result["export_report_path"],
        "html_batch_asset_validation_markdown": export_result["html_asset_validation_markdown"],
        "html_batch_missing_asset_refs": export_result["html_missing_asset_refs"],
        "html_batch_has_missing_assets": export_result["html_has_missing_assets"],
    }


def finalize_html_batch_iteration(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    batch_index = int(attrs.get("html_batch_index") or 0)
    batch_count = int(attrs.get("html_batch_count") or 1)
    next_index = batch_index + 1
    outputs = json.loads(str(attrs.get("html_batch_outputs_json") or "[]"))
    outputs.append(
        {
            "batch_index": batch_index + 1,
            "html_path": str(attrs.get("html_batch_html_path", "")),
            "structure_notes_path": str(attrs.get("html_batch_structure_notes_path", "")),
            "pdf_path": str(attrs.get("html_batch_pdf_path", "")),
            "preview_dir": str(attrs.get("html_batch_preview_dir", "")),
            "export_report_path": str(attrs.get("html_batch_export_report_path", "")),
            "pdf_export_success": parse_bool(attrs.get("html_batch_pdf_export_success")),
            "has_missing_assets": parse_bool(attrs.get("html_batch_has_missing_assets")),
            "asset_validation_markdown": str(attrs.get("html_batch_asset_validation_markdown", "")),
            "rendered_pdf_summary_markdown": str(attrs.get("html_batch_rendered_pdf_summary_markdown", "")),
        }
    )
    return {
        "html_batch_index": next_index,
        "html_batch_iteration_ready": next_index >= batch_count,
        "html_batch_sections_json": attrs.get("html_batch_sections_json", "[]"),
        "html_batch_size": int(attrs.get("html_batch_size") or get_stage_batch_size(default=10)),
        "html_batch_count": batch_count,
        "html_batch_outputs_json": json.dumps(outputs, ensure_ascii=False),
    }


def assemble_html_batch_outputs(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    batch_outputs = json.loads(str(attrs.get("html_batch_outputs_json") or "[]"))
    batch_outputs = sorted(batch_outputs, key=lambda item: int(item.get("batch_index") or 0))

    html_paths = [item["html_path"] for item in batch_outputs if str(item.get("html_path", "")).strip()]
    notes_paths = [item["structure_notes_path"] for item in batch_outputs if str(item.get("structure_notes_path", "")).strip()]
    pdf_paths = [item["pdf_path"] for item in batch_outputs if str(item.get("pdf_path", "")).strip()]

    final_html_path = str(attrs.get("html_path", ""))
    final_notes_path = str(attrs.get("structure_notes_path", ""))
    final_pdf_path = Path(str(attrs.get("rendered_pdf_path", ""))).expanduser().resolve()
    final_preview_dir = ensure_dir(attrs.get("rendered_pdf_preview_dir", final_pdf_path.parent / "preview"))
    final_export_report_path = Path(str(attrs.get("export_report_path", final_pdf_path.parent / "export_report.md"))).expanduser().resolve()

    merged_html_path = _merge_html_decks(html_paths, final_html_path) if html_paths else write_text(final_html_path, "")
    merged_notes = []
    for index, path in enumerate(notes_paths, start=1):
        text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        merged_notes.append(f"# Batch {index}\n\n{text}")
    merged_notes_path = write_text(final_notes_path, "\n\n".join(merged_notes).strip())

    merge_success = False
    merge_error = ""
    preview_paths: list[Path] = []
    pdf_text = ""
    try:
        if pdf_paths:
            merge_pdfs(pdf_paths, final_pdf_path)
            preview_paths = pdf_to_pngs(final_pdf_path, final_preview_dir / "slide", dpi=144)
            text_path = str(final_pdf_path).replace(".pdf", ".txt")
            pdf_to_text(final_pdf_path, output_txt=text_path)
            pdf_text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
            merge_success = True
    except Exception as exc:
        merge_error = f"{type(exc).__name__}: {exc}"

    merged_html_text = Path(merged_html_path).read_text(encoding="utf-8", errors="ignore")
    local_refs, missing_refs = validate_html_local_image_refs(merged_html_text, merged_html_path)
    html_asset_validation_markdown = render_html_asset_validation(local_refs, missing_refs)
    preview_manifest = "\n".join(str(path) for path in preview_paths) if preview_paths else "(no previews generated)"

    batch_manifest_lines = []
    for item in batch_outputs:
        batch_manifest_lines.extend(
            [
                f"- Batch {item.get('batch_index')}: PDF export success = {item.get('pdf_export_success')}, missing assets = {item.get('has_missing_assets')}",
                f"  - html: {Path(str(item.get('html_path', ''))).name}",
                f"  - pdf: {Path(str(item.get('pdf_path', ''))).name}",
            ]
        )
    export_report = "\n".join(
        [
            "# Batch Assembly Report",
            "",
            f"- Final HTML: {merged_html_path}",
            f"- Final PDF: {final_pdf_path}",
            f"- Preview dir: {final_preview_dir}",
            f"- Batch count: {len(batch_outputs)}",
            f"- Merge success: {merge_success}",
            f"- Merge error: {merge_error or '(none)'}",
            "",
            "## Batch Outputs",
            "\n".join(batch_manifest_lines) or "(none)",
            "",
            "## HTML Asset Validation",
            html_asset_validation_markdown,
            "",
            "## Preview Files",
            preview_manifest,
            "",
            "## PDF Text Excerpt",
            excerpt(pdf_text, limit=5000) if pdf_text else "(not available)",
        ]
    )
    write_text(final_export_report_path, export_report)

    rendered_pdf_summary = "\n".join(
        [
            "# Rendered PDF Summary",
            "",
            f"- PDF path: {final_pdf_path}",
            f"- Preview dir: {final_preview_dir}",
            f"- Batch count: {len(batch_outputs)}",
            f"- Merge success: {merge_success}",
            "",
            "## HTML Asset Validation",
            html_asset_validation_markdown,
            "",
            "## Preview Files",
            preview_manifest,
            "",
            "## PDF Text Excerpt",
            excerpt(pdf_text, limit=5000) if pdf_text else "(not available)",
        ]
    )
    return {
        "html_content": merged_html_text,
        "structure_notes_markdown": Path(merged_notes_path).read_text(encoding="utf-8", errors="ignore"),
        "html_path": merged_html_path,
        "structure_notes_path": merged_notes_path,
        "html_excerpt": excerpt(merged_html_text, limit=14000),
        "structure_notes_excerpt": excerpt(Path(merged_notes_path).read_text(encoding="utf-8", errors="ignore"), limit=7000),
        "pdf_export_success": merge_success and not missing_refs and all(parse_bool(item.get("pdf_export_success")) for item in batch_outputs),
        "rendered_pdf_path": str(final_pdf_path),
        "rendered_pdf_preview_dir": str(final_preview_dir),
        "rendered_pdf_summary_markdown": rendered_pdf_summary,
        "export_report_path": str(final_export_report_path),
        "html_asset_validation_markdown": html_asset_validation_markdown,
        "html_missing_asset_refs": "\n".join(item["src"] for item in missing_refs),
        "html_has_missing_assets": bool(missing_refs),
    }


def finalize_html_review(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    export_success = parse_bool(attrs.get("pdf_export_success"))
    human_ready = parse_bool(attrs.get("html_human_decision"))
    has_missing_assets = parse_bool(attrs.get("html_has_missing_assets"))
    ready = export_success and human_ready and not has_missing_assets

    feedback = ""
    if not ready:
        feedback = "\n\n".join(
            part
            for part in [
                "PDF export summary:\n" + str(attrs.get("rendered_pdf_summary_markdown", "")).strip(),
                "Asset validation:\n" + str(attrs.get("html_asset_validation_markdown", "")).strip(),
                "Human review:\n" + str(attrs.get("html_human_feedback", "")).strip(),
            ]
            if part.strip()
        ).strip()

    report = "\n".join(
        [
            "# HTML + PDF Review",
            "",
            f"- PDF export success: {export_success}",
            f"- Human ready: {human_ready}",
            f"- Missing local asset refs: {has_missing_assets}",
            f"- Final ready: {ready}",
            "- Gate policy: PDF export must succeed; human review is authoritative; local asset refs must resolve.",
            "",
            "## Rendered PDF Summary",
            str(attrs.get("rendered_pdf_summary_markdown", "")),
            "",
            "## Asset Validation",
            str(attrs.get("html_asset_validation_markdown", "")),
            "",
            "## Human Feedback",
            str(attrs.get("html_human_feedback", "")),
        ]
    )
    review_path = write_text(attrs["html_review_path"], report)
    return {
        "html_ready": ready,
        "html_iteration_feedback": feedback,
        "html_review_path": review_path,
        "html_content": attrs.get("html_content", ""),
        "structure_notes_markdown": attrs.get("structure_notes_markdown", ""),
        "html_path": attrs.get("html_path", ""),
        "structure_notes_path": attrs.get("structure_notes_path", ""),
        "html_excerpt": attrs.get("html_excerpt", ""),
        "structure_notes_excerpt": attrs.get("structure_notes_excerpt", ""),
        "html_asset_validation_markdown": attrs.get("html_asset_validation_markdown", ""),
        "html_missing_asset_refs": attrs.get("html_missing_asset_refs", ""),
        "html_has_missing_assets": has_missing_assets,
        "rendered_pdf_path": attrs.get("rendered_pdf_path", ""),
        "rendered_pdf_preview_dir": attrs.get("rendered_pdf_preview_dir", ""),
        "rendered_pdf_summary_markdown": attrs.get("rendered_pdf_summary_markdown", ""),
        "export_report_path": attrs.get("export_report_path", ""),
    }


def should_terminate_html(_: dict[str, object], attrs: dict[str, object]) -> bool:
    return parse_bool(attrs.get("html_ready"))


def should_terminate_html_batch_loop(_: dict[str, object], attrs: dict[str, object]) -> bool:
    return parse_bool(attrs.get("html_batch_iteration_ready"))


__all__ = [
    "assemble_html_batch_outputs",
    "export_html_batch_pdf",
    "export_html_and_pdf",
    "finalize_html_batch_iteration",
    "finalize_html_review",
    "prepare_html_batch_iteration",
    "prepare_html_batch_plan",
    "persist_html_outputs",
    "should_terminate_html",
    "should_terminate_html_batch_loop",
]

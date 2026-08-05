from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any

from tools import pdf_extract_images, pdf_info, pdf_to_pngs, pdf_to_text


def excerpt(text: Any, *, limit: int = 6000) -> str:
    if text is None:
        return ""
    raw = str(text)
    return raw if len(raw) <= limit else raw[:limit] + "\n...(truncated)..."


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_text(path: str | Path, content: Any) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("" if content is None else str(content), encoding="utf-8")
    return str(target)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "ready",
        "approve",
        "approved",
        "ok",
        "pass",
        "passed",
        "go",
    }


def parse_paper_paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(Path(item).expanduser().resolve()) for item in value if str(item).strip()]
    parts = re.split(r"[\n,;]+", str(value))
    return [str(Path(part.strip()).expanduser().resolve()) for part in parts if part.strip()]


def get_stage_batch_size(default: int = 5) -> int:
    raw = str(os.getenv("OHNOPPT_STAGE_BATCH_SIZE", default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(1, value)


def parse_outline_slide_sections(outline_markdown: Any) -> list[dict[str, Any]]:
    text = str(outline_markdown or "").strip()
    if not text:
        return []

    slide_plan_index = text.find("# Slide Plan")
    if slide_plan_index >= 0:
        text = text[slide_plan_index:]

    pattern = re.compile(
        r"(?ms)^## Slide\s+(\d+)\s*[—-]\s*(.+?)\n(.*?)(?=^## Slide\s+\d+\s*[—-]|\Z)"
    )
    sections: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        number = int(match.group(1))
        title = match.group(2).strip()
        body = match.group(3).strip()
        conclusion_match = re.search(r"\*\*Main Conclusion:\*\*\s*(.+)", body)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
        section_markdown = f"## Slide {number} — {title}\n{body}".strip()
        sections.append(
            {
                "number": number,
                "title": title,
                "conclusion": conclusion,
                "section_markdown": section_markdown,
            }
        )
    return sections


def normalize_outline_slide_sections(
    sections: list[dict[str, Any]],
    *,
    page_budget: int | None = None,
) -> list[dict[str, Any]]:
    unique_sections: list[dict[str, Any]] = []
    seen_numbers: set[int] = set()

    for item in sections:
        number = int(item.get("number") or 0)
        if number <= 0 or number in seen_numbers:
            continue
        seen_numbers.add(number)
        unique_sections.append(item)

    unique_sections.sort(key=lambda item: int(item.get("number") or 0))

    if page_budget is not None:
        try:
            budget = max(1, int(page_budget))
        except (TypeError, ValueError):
            budget = 0
        if budget > 0:
            unique_sections = unique_sections[:budget]

    return unique_sections


def render_slide_batch_markdown(
    sections: list[dict[str, Any]],
    *,
    batch_index: int,
    batch_count: int,
) -> str:
    if not sections:
        return ""
    start_slide = sections[0]["number"]
    end_slide = sections[-1]["number"]
    lines = [
        f"# Current Slide Batch ({batch_index}/{batch_count})",
        "",
        f"- This batch should cover Slide {start_slide} to Slide {end_slide}.",
        "- Only add or revise slides in this batch unless a global consistency adjustment is truly necessary.",
        "",
    ]
    for item in sections:
        lines.append(item["section_markdown"])
        lines.append("")
    return "\n".join(lines).strip()


def render_previous_slides_summary(sections: list[dict[str, Any]]) -> str:
    if not sections:
        return "# Previous Slides Summary\n\n(no previous slides)\n"

    lines = ["# Previous Slides Summary", ""]
    for item in sections:
        summary_line = f"- Slide {item['number']}: {item.get('title', '')}"
        conclusion = str(item.get("conclusion", "")).strip()
        if conclusion:
            summary_line += f" | {conclusion}"
        lines.append(summary_line)
    lines.append("")
    return "\n".join(lines)


def _display_name(path: str | Path) -> str:
    return Path(str(path)).name


_IMG_SRC_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)


def validate_html_local_image_refs(html_content: Any, html_path: str | Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    html_path = Path(html_path).expanduser().resolve()
    refs: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for raw_src in _IMG_SRC_RE.findall(str(html_content or "")):
        src = unescape(raw_src).strip()
        if not src or src.startswith(("http://", "https://", "data:", "file://", "//")):
            continue
        resolved = Path(src).expanduser()
        if not resolved.is_absolute():
            resolved = (html_path.parent / resolved).resolve()
        exists = resolved.exists()
        record = {
            "src": src,
            "resolved_path": str(resolved),
            "exists": "yes" if exists else "no",
        }
        refs.append(record)
        if not exists:
            missing.append(record)

    return refs, missing


def render_html_asset_validation(refs: list[dict[str, str]], missing: list[dict[str, str]]) -> str:
    lines = [
        "# HTML Asset Validation",
        "",
        f"- Total local image refs: {len(refs)}",
        f"- Missing local image refs: {len(missing)}",
        "",
    ]

    if refs:
        lines.extend(["## Local Image Refs", ""])
        for item in refs:
            lines.append(f"- `{item['src']}` -> `{item['resolved_path']}` ({item['exists']})")
        lines.append("")

    if missing:
        lines.extend(["## Missing Refs", ""])
        for item in missing:
            lines.append(f"- `{item['src']}` -> `{item['resolved_path']}`")
        lines.append("")

    return "\n".join(lines).strip()


def init_pipeline_run(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    run_name = str(attrs.get("run_name") or f"ppt_pipeline_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_root = ensure_dir(attrs.get("output_dir") or (Path.cwd() / "runs_text"))
    run_dir = ensure_dir(output_root / run_name)

    asset_dir = ensure_dir(run_dir / "assets")
    source_dir = ensure_dir(run_dir / "00_sources")
    outline_dir = ensure_dir(run_dir / "01_outline")
    html_dir = ensure_dir(run_dir / "02_html")
    pdf_dir = ensure_dir(run_dir / "03_pdf")
    pptx_dir = ensure_dir(run_dir / "04_pptx")

    paper_paths = parse_paper_paths(attrs.get("paper_paths"))
    python_bin = str(Path(attrs.get("python_bin") or "").expanduser().resolve()) if attrs.get("python_bin") else ""

    manifest = {
        "run_name": run_name,
        "paper_paths": paper_paths,
        "output_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_variant": "text_only",
    }
    manifest_path = write_text(run_dir / "run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "asset_dir": str(asset_dir),
        "source_dir": str(source_dir),
        "outline_dir": str(outline_dir),
        "html_dir": str(html_dir),
        "pdf_dir": str(pdf_dir),
        "pptx_dir": str(pptx_dir),
        "paper_paths": paper_paths,
        "paper_paths_text": "\n".join(paper_paths),
        "paper_count": len(paper_paths),
        "python_bin": python_bin,
        "asset_manifest_path": str(source_dir / "asset_manifest.md"),
        "source_asset_index_path": str(source_dir / "source_asset_index.md"),
        "outline_path": str(outline_dir / "outline.md"),
        "source_map_path": str(outline_dir / "source_map.md"),
        "outline_review_path": str(outline_dir / "outline_review.md"),
        "html_path": str(html_dir / "slides.html"),
        "structure_notes_path": str(html_dir / "structure_notes.md"),
        "html_review_path": str(html_dir / "html_review.md"),
        "rendered_pdf_path": str(pdf_dir / "slides.pdf"),
        "rendered_pdf_preview_dir": str(pdf_dir / "preview"),
        "export_report_path": str(pdf_dir / "export_report.md"),
        "recreated_script_path": str(pptx_dir / "recreate_pdf_to_pptx.py"),
        "recreated_pptx_path": str(pptx_dir / "recreated_from_pdf.pptx"),
        "recreated_validation_pdf_path": str(pptx_dir / "recreated_from_pdf.pdf"),
        "recreated_preview_dir": str(pptx_dir / "preview"),
        "recreation_notes_path": str(pptx_dir / "recreation_notes.md"),
        "pptx_review_path": str(pptx_dir / "pptx_review.md"),
        "run_manifest_path": manifest_path,
        "pdf_export_strategy": attrs.get("preferred_export_strategy") or "slide_screenshots_to_pdf",
        "pdf_needs_html_fix": False,
        "outline_iteration_feedback": "",
        "html_iteration_feedback": "",
        "pptx_iteration_feedback": "",
    }


def prepare_source_materials(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    source_dir = ensure_dir(attrs["source_dir"])
    asset_dir = ensure_dir(attrs["asset_dir"])
    papers = parse_paper_paths(attrs.get("paper_paths"))
    items: list[dict[str, Any]] = []
    asset_items: list[dict[str, Any]] = []
    context_parts: list[str] = []

    for index, paper in enumerate(papers, start=1):
        paper_path = Path(paper)
        paper_stem = paper_path.stem
        paper_slug = f"paper{index:02d}"
        paper_source_dir = ensure_dir(source_dir / paper_slug)
        txt_path = paper_source_dir / "text" / f"{paper_stem}.txt"
        preview_files: list[Path] = []
        info: dict[str, Any] = {}
        text = ""
        page_files: list[Path] = []
        embedded_files: list[Path] = []
        page_dir = ensure_dir(asset_dir / paper_slug / "pages")
        embedded_dir = ensure_dir(asset_dir / paper_slug / "embedded_images")

        if paper_path.suffix.lower() == ".pdf":
            info = pdf_info(paper_path)
            pdf_to_text(paper_path, output_txt=txt_path, layout=False)
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            page_assets = pdf_to_pngs(paper_path, page_dir / "page", dpi=144)
            for page_number, generated_path in enumerate(page_assets, start=1):
                normalized_path = page_dir / f"page{page_number:02d}.png"
                if generated_path != normalized_path:
                    shutil.move(str(generated_path), str(normalized_path))
                page_files.append(normalized_path)
                if page_number == 1:
                    preview_files.append(normalized_path)
                asset_items.append(
                    {
                        "paper_index": index,
                        "asset_id": f"{paper_slug.upper()}_PAGE_{page_number:02d}",
                        "page_number": page_number,
                        "path": str(normalized_path),
                        "html_relative_path": f"../assets/{paper_slug}/pages/{normalized_path.name}",
                        "kind": "paper_page_preview",
                    }
                )
            try:
                raw_embedded_files = pdf_extract_images(paper_path, embedded_dir / "image")
                for embedded_index, embedded_file in enumerate(raw_embedded_files, start=1):
                    suffix = embedded_file.suffix.lower() or ".bin"
                    normalized_path = embedded_dir / f"image{embedded_index:02d}{suffix}"
                    if embedded_file != normalized_path:
                        shutil.move(str(embedded_file), str(normalized_path))
                    embedded_files.append(normalized_path)
                    asset_items.append(
                        {
                            "paper_index": index,
                            "asset_id": f"{paper_slug.upper()}_EMBED_{embedded_index:02d}",
                            "page_number": "",
                            "path": str(normalized_path),
                            "html_relative_path": f"../assets/{paper_slug}/embedded_images/{normalized_path.name}",
                            "kind": "embedded_image",
                        }
                    )
            except Exception:
                embedded_files = []
        else:
            text = paper_path.read_text(encoding="utf-8", errors="ignore")
            txt_path.write_text(text, encoding="utf-8")

        item = {
            "index": index,
            "slug": paper_slug,
            "path": str(paper_path),
            "text_path": str(txt_path),
            "preview_paths": [str(path) for path in preview_files],
            "page_preview_count": len(page_files),
            "embedded_image_count": len(embedded_files),
            "page_dir": str(page_dir),
            "embedded_dir": str(embedded_dir),
            "title": info.get("Title", paper_stem),
            "pages": info.get("Pages", ""),
        }
        items.append(item)

        context_parts.append(
            "\n".join(
                [
                    f"# Paper {index}",
                    f"Path: {paper_path}",
                    f"Title: {item['title']}",
                    f"Pages: {item['pages']}",
                    "Excerpt:",
                    excerpt(text, limit=12000),
                ]
            )
        )

    manifest_lines = ["# Source Manifest", ""]
    for item in items:
        manifest_lines.extend(
            [
                f"## Paper {item['index']}",
                f"- Slug: `{item['slug']}`",
                f"- Path: {item['path']}",
                f"- Text: {item['text_path']}",
                f"- Preview: {item['preview_paths'][0] if item['preview_paths'] else '(not generated)'}",
                f"- Title: {item['title']}",
                f"- Pages: {item['pages'] or '(unknown)'}",
                f"- Page previews dir: {item['page_dir']}",
                f"- Embedded images dir: {item['embedded_dir']}",
                f"- Page preview count: {item['page_preview_count']}",
                f"- Embedded image count: {item['embedded_image_count']}",
                "",
            ]
        )

    source_manifest = "\n".join(manifest_lines).strip() + "\n"
    source_context = "\n\n".join(context_parts).strip()
    asset_lines = [
        "# Asset Manifest",
        "",
        "Use `html_relative_path` when writing HTML or python-pptx code that needs to embed a local asset.",
        "",
    ]
    if asset_items:
        for item in asset_items:
            asset_lines.extend(
                [
                    f"## {item['asset_id']}",
                    f"- Type: {item['kind']}",
                    f"- Paper index: {item['paper_index']}",
                    f"- Page number: {item['page_number'] or '(n/a)'}",
                    f"- Absolute path: {item['path']}",
                    f"- HTML relative path: {item['html_relative_path']}",
                    "",
                ]
            )
    else:
        asset_lines.extend(["(no reusable local assets generated)", ""])
    asset_manifest = "\n".join(asset_lines).strip() + "\n"

    asset_index_lines = [
        "# Source Asset Index",
        "",
        "These are the stable local assets that downstream agents can reference.",
        "Prefer embedded images for figure insertion. Use page previews for layout/context or when no figure-level asset exists.",
        "",
    ]
    for item in items:
        asset_index_lines.extend(
            [
                f"## {item['slug'].upper()}",
                f"- PDF path: `{item['path']}`",
                f"- First-page preview: `{item['preview_paths'][0] if item['preview_paths'] else '(not available)'}`",
                f"- Page preview directory: `{item['page_dir']}`",
                f"- Embedded image directory: `{item['embedded_dir']}`",
                "",
            ]
        )
        for asset in [entry for entry in asset_items if entry["paper_index"] == item["index"]]:
            asset_index_lines.append(
                f"- `{asset['asset_id']}` -> `{asset['html_relative_path']}` ({asset['kind']})"
            )
        asset_index_lines.append("")

    source_manifest_path = write_text(source_dir / "source_manifest.md", source_manifest)
    asset_manifest_path = write_text(attrs["asset_manifest_path"], asset_manifest)
    source_asset_index_path = write_text(attrs["source_asset_index_path"], "\n".join(asset_index_lines).strip() + "\n")
    source_context_path = write_text(source_dir / "source_context.md", source_context)

    verbose_lines = [
        "# Source Reference Package",
        "",
        "This is the text-only source package. Use the local extracted assets below instead of inventing filenames.",
        "",
        "## Source Manifest",
        excerpt(source_manifest.strip(), limit=2500),
        "",
    ]

    for item in items:
        page_files = [entry for entry in asset_items if entry["paper_index"] == item["index"] and entry["kind"] == "paper_page_preview"]
        embedded_files = [entry for entry in asset_items if entry["paper_index"] == item["index"] and entry["kind"] == "embedded_image"]
        paper_slug = item["slug"]
        verbose_lines.extend(
            [
                f"## {paper_slug.upper()} Assets",
                f"- Page preview directory name: `{_display_name(item['page_dir'])}`",
                f"- Page preview HTML relative pattern: `../assets/{paper_slug}/pages/pageNN.png`",
                f"- Embedded image directory name: `{_display_name(item['embedded_dir'])}`",
                f"- Embedded image HTML relative pattern: `../assets/{paper_slug}/embedded_images/imageNN.ext`",
                "- Do not shorten these to `../assets/pageNN.png`.",
                "",
                "### Sample Page Previews",
            ]
        )
        if page_files:
            verbose_lines.extend(
                f"- abs: `{entry['path']}` | html: `{entry['html_relative_path']}`"
                for entry in page_files[:3]
            )
        else:
            verbose_lines.append("- (none)")
        verbose_lines.extend(["", "### Sample Embedded Images"])
        if embedded_files:
            verbose_lines.extend(
                f"- abs: `{entry['path']}` | html: `{entry['html_relative_path']}`"
                for entry in embedded_files[:6]
            )
            if len(embedded_files) > 6:
                verbose_lines.append(f"- ... {len(embedded_files) - 6} more files in the same directory")
        else:
            verbose_lines.append("- (none)")
        verbose_lines.append("")

    verbose_lines.extend(
        [
            "## Source Context Excerpt",
            excerpt(source_context, limit=4000),
            "",
            "## On-Disk Reference Files",
            f"- Asset manifest: `{_display_name(asset_manifest_path)}`",
            f"- Source asset index: `{_display_name(source_asset_index_path)}`",
            f"- Source context: `{_display_name(source_context_path)}`",
        ]
    )

    compact_lines = [
        "# Outline Source Reference Package",
        "",
        "Use this compact package to build the outline. Focus on narrative, method, experiments, limitations, and discussion value.",
        "",
    ]
    for item in items:
        paper_slug = item["slug"]
        compact_lines.extend(
            [
                f"## {paper_slug.upper()}",
                f"- Title: {item['title']}",
                f"- Pages: {item['pages'] or '(unknown)'}",
                f"- Page preview HTML pattern: `../assets/{paper_slug}/pages/pageNN.png`",
                f"- Embedded image HTML pattern: `../assets/{paper_slug}/embedded_images/imageNN.ext`",
                "",
            ]
        )
    compact_lines.extend(
        [
            "## Source Context Excerpt",
            excerpt(source_context, limit=5000),
            "",
            "## Reference Files",
            f"- Asset manifest: `{_display_name(asset_manifest_path)}`",
            f"- Source asset index: `{_display_name(source_asset_index_path)}`",
            f"- Source context: `{_display_name(source_context_path)}`",
        ]
    )

    source_reference_markdown = "\n".join(verbose_lines).strip() + "\n"
    compact_source_reference_markdown = "\n".join(compact_lines).strip() + "\n"
    use_compact_outline_source = os.getenv("OHNOPPT_COMPACT_OUTLINE_SOURCE", "1").strip().lower() in {"1", "true", "yes", "on"}
    outline_source_reference_markdown = compact_source_reference_markdown if use_compact_outline_source else source_reference_markdown

    return {
        "asset_dir": str(asset_dir),
        "asset_manifest_path": asset_manifest_path,
        "source_asset_index_path": source_asset_index_path,
        "source_manifest_path": source_manifest_path,
        "source_context_path": source_context_path,
        "source_reference_markdown": source_reference_markdown,
        "outline_source_reference_markdown": outline_source_reference_markdown,
        "outline_source_mode": "compact" if use_compact_outline_source else "verbose",
    }


def finalize_pipeline(_: dict[str, object], attrs: dict[str, object]) -> dict[str, object]:
    result = {
        "run_dir": attrs.get("run_dir", ""),
        "outline_path": attrs.get("outline_path", ""),
        "html_path": attrs.get("html_path", ""),
        "rendered_pdf_path": attrs.get("rendered_pdf_path", ""),
        "recreated_pptx_path": attrs.get("recreated_pptx_path", ""),
        "recreated_validation_pdf_path": attrs.get("recreated_validation_pdf_path", ""),
        "outline_review_path": attrs.get("outline_review_path", ""),
        "html_review_path": attrs.get("html_review_path", ""),
        "pptx_review_path": attrs.get("pptx_review_path", ""),
        "ppt_pipeline_success": parse_bool(attrs.get("pptx_ready")),
    }

    manifest_path = Path(str(attrs["run_manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_outputs"] = result
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return result


__all__ = [
    "ensure_dir",
    "excerpt",
    "finalize_pipeline",
    "get_stage_batch_size",
    "init_pipeline_run",
    "parse_bool",
    "parse_outline_slide_sections",
    "prepare_source_materials",
    "render_previous_slides_summary",
    "render_html_asset_validation",
    "render_slide_batch_markdown",
    "validate_html_local_image_refs",
    "write_text",
]

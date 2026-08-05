from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL_CACHE_ROOT = REPO_ROOT / "data" / "tool_cache"


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, capture_output=True)


def find_pdf_files(pdf_dir: str | Path) -> list[Path]:
    root = Path(pdf_dir).expanduser().resolve()
    return sorted(path for path in root.rglob("*.pdf") if path.is_file())


def pdf_info(pdf_path: str | Path) -> dict[str, str]:
    target = Path(pdf_path).expanduser().resolve()
    completed = run_command(["pdfinfo", str(target)])
    info: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def count_pdf_pages(pdf_path: str | Path) -> int:
    info = pdf_info(pdf_path)
    raw = info.get("Pages", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def extract_pdf_text(pdf_path: str | Path, output_txt: str | Path) -> str:
    source = Path(pdf_path).expanduser().resolve()
    target = Path(output_txt).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    run_command(["pdftotext", "-layout", str(source), str(target)])
    return target.read_text(encoding="utf-8", errors="ignore")


def pdf_first_page_preview(pdf_path: str | Path, output_prefix: str | Path) -> str:
    source = Path(pdf_path).expanduser().resolve()
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "pdftoppm",
            "-png",
            "-f",
            "1",
            "-l",
            "1",
            str(source),
            str(prefix),
        ]
    )
    generated_paths = sorted(prefix.parent.glob(f"{prefix.name}-*.png"))
    if not generated_paths:
        raise FileNotFoundError(f"No preview image generated for {source}")
    return str(generated_paths[0])


def dump_json(path: str | Path, payload: object) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def _ensure_positive_page(page: int, *, name: str) -> int:
    if int(page) < 1:
        raise ValueError(f"{name} must be >= 1, got {page}")
    return int(page)


def _normalize_page_range(pdf_path: str | Path, first_page: int, last_page: int | None) -> tuple[int, int]:
    page_count = count_pdf_pages(pdf_path)
    first = _ensure_positive_page(first_page, name="first_page")
    if page_count <= 0:
        raise ValueError(f"Could not determine page count for PDF: {pdf_path}")
    if first > page_count:
        raise ValueError(f"first_page {first} exceeds page count {page_count} for {pdf_path}")

    if last_page is None or int(last_page) == 0:
        last = page_count
    else:
        last = _ensure_positive_page(int(last_page), name="last_page")
    if last < first:
        raise ValueError(f"last_page {last} must be >= first_page {first}")
    return first, min(last, page_count)


def _ensure_output_dir(output_dir: str | Path | None, pdf_path: str | Path, kind: str) -> Path:
    if output_dir:
        target = Path(output_dir).expanduser().resolve()
    else:
        source = Path(pdf_path).expanduser().resolve()
        target = DEFAULT_TOOL_CACHE_ROOT / kind / source.stem
    target.mkdir(parents=True, exist_ok=True)
    return target


def _sort_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda item: item.name)


def _extract_rendered_page_number(path: Path) -> int | None:
    match = re.search(r"-(\d+)\.png$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def make_asset_descriptor(path: str | Path, modality: str, label: str, *, paper_id: str = "") -> dict[str, str]:
    asset_path = Path(path).expanduser().resolve()
    descriptor = {
        "path": str(asset_path),
        "label": label,
        "modality": modality.lower(),
        "source_kind": "path",
    }
    if paper_id:
        descriptor["paper_id"] = paper_id
    return descriptor


def pdf_extract_text_pages(
    pdf_path: str,
    first_page: int = 1,
    last_page: int | None = None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """Extract text from a PDF page range.

    Args:
        pdf_path: Absolute or relative path to the source PDF.
        first_page: First page number to extract, 1-based.
        last_page: Last page number to extract, inclusive. Use 0 or omit for "to the end".
        max_chars: Maximum number of characters returned in the tool result.

    Returns:
        A dict containing the normalized page range, extracted text, and truncation metadata.
    """

    source = Path(pdf_path).expanduser().resolve()
    first, last = _normalize_page_range(source, first_page, last_page)
    completed = run_command(
        [
            "pdftotext",
            "-layout",
            "-f",
            str(first),
            "-l",
            str(last),
            str(source),
            "-",
        ]
    )
    text = completed.stdout
    clipped = text[: max(0, int(max_chars))]
    return {
        "pdf_path": str(source),
        "first_page": first,
        "last_page": last,
        "text": clipped,
        "char_count": len(text),
        "truncated": len(clipped) < len(text),
    }


def pdf_render_pages_to_images(
    pdf_path: str,
    first_page: int = 1,
    last_page: int | None = None,
    dpi: int = 144,
    output_dir: str = "",
) -> dict[str, Any]:
    """Render PDF pages into PNG images.

    Args:
        pdf_path: Absolute or relative path to the source PDF.
        first_page: First page number to render, 1-based.
        last_page: Last page number to render, inclusive. Use 0 or omit for "to the end".
        dpi: Render resolution. Higher values improve detail but cost more storage.
        output_dir: Optional directory for generated images. Defaults to the nowwhat tool cache.

    Returns:
        A dict with generated image paths for each rendered page.
    """

    source = Path(pdf_path).expanduser().resolve()
    first, last = _normalize_page_range(source, first_page, last_page)
    target_dir = _ensure_output_dir(output_dir, source, "rendered_pages")
    prefix = target_dir / f"{source.stem}_pages_f{first}_l{last}_d{int(dpi)}"
    for stale_path in target_dir.glob(f"{prefix.name}-*.png"):
        stale_path.unlink(missing_ok=True)
    run_command(
        [
            "pdftocairo",
            "-png",
            "-r",
            str(int(dpi)),
            "-f",
            str(first),
            "-l",
            str(last),
            str(source),
            str(prefix),
        ]
    )
    rendered_paths = []
    for image_path in target_dir.glob(f"{prefix.name}-*.png"):
        page_number = _extract_rendered_page_number(image_path)
        if page_number is None:
            continue
        actual_page = first + page_number - 1
        if first <= actual_page <= last:
            rendered_paths.append((actual_page, image_path))
    rendered_paths.sort(key=lambda item: item[0])
    pages = []
    for actual_page, image_path in rendered_paths:
        pages.append({"page": actual_page, "image_path": str(image_path)})
    return {
        "pdf_path": str(source),
        "first_page": first,
        "last_page": last,
        "dpi": int(dpi),
        "output_dir": str(target_dir),
        "pages": pages,
    }


def pdf_extract_embedded_images(
    pdf_path: str,
    first_page: int = 1,
    last_page: int | None = None,
    output_dir: str = "",
) -> dict[str, Any]:
    """Extract embedded raster images from a PDF page range.

    Args:
        pdf_path: Absolute or relative path to the source PDF.
        first_page: First page number to inspect, 1-based.
        last_page: Last page number to inspect, inclusive. Use 0 or omit for "to the end".
        output_dir: Optional directory for extracted images. Defaults to the nowwhat tool cache.

    Returns:
        A dict with the extracted file paths. Formats depend on the source PDF content.
    """

    source = Path(pdf_path).expanduser().resolve()
    first, last = _normalize_page_range(source, first_page, last_page)
    target_dir = _ensure_output_dir(output_dir, source, "embedded_images")
    prefix = target_dir / f"{source.stem}_embedded"
    run_command(
        [
            "pdfimages",
            "-f",
            str(first),
            "-l",
            str(last),
            "-all",
            str(source),
            str(prefix),
        ]
    )
    extracted_paths = _sort_paths([path for path in target_dir.glob(f"{prefix.name}-*") if path.is_file()])
    return {
        "pdf_path": str(source),
        "first_page": first,
        "last_page": last,
        "output_dir": str(target_dir),
        "images": [str(path) for path in extracted_paths],
        "image_count": len(extracted_paths),
    }

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(f"Required binary not found in PATH: {name}")
    return path


def run_command(
    args: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=str(_as_path(cwd)) if cwd else None,
        env=env,
        check=check,
        text=True,
        capture_output=True,
    )


def pdf_info(pdf_path: str | Path, *, pdfinfo_bin: str = "pdfinfo") -> dict[str, str]:
    require_binary(pdfinfo_bin)
    pdf_path = _as_path(pdf_path)
    result = run_command([pdfinfo_bin, str(pdf_path)])
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def pdf_to_text(
    pdf_path: str | Path,
    *,
    output_txt: str | Path | None = None,
    layout: bool = False,
    pdftotext_bin: str = "pdftotext",
) -> str | Path:
    require_binary(pdftotext_bin)
    pdf_path = _as_path(pdf_path)
    flags: list[str] = [pdftotext_bin]
    if layout:
        flags.append("-layout")
    if output_txt is None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        run_command([*flags, str(pdf_path), str(tmp_path)])
        text = tmp_path.read_text(encoding="utf-8", errors="ignore")
        tmp_path.unlink(missing_ok=True)
        return text

    output_txt = _as_path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    run_command([*flags, str(pdf_path), str(output_txt)])
    return output_txt


def pdf_to_pngs(
    pdf_path: str | Path,
    output_prefix: str | Path,
    *,
    dpi: int | None = None,
    first_page: int | None = None,
    last_page: int | None = None,
    single_file: bool = False,
    pdftoppm_bin: str = "pdftoppm",
) -> list[Path]:
    require_binary(pdftoppm_bin)
    pdf_path = _as_path(pdf_path)
    output_prefix = _as_path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    args: list[str] = [pdftoppm_bin, "-png"]
    if dpi is not None:
        args.extend(["-r", str(dpi)])
    if first_page is not None:
        args.extend(["-f", str(first_page)])
    if last_page is not None:
        args.extend(["-l", str(last_page)])
    if single_file:
        args.append("-singlefile")
    args.extend([str(pdf_path), str(output_prefix)])
    run_command(args)

    if single_file:
        return [output_prefix.with_suffix(".png")]
    return sorted(
        output_prefix.parent.glob(f"{output_prefix.name}-*.png"),
        key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
    )


def pdf_extract_images(
    pdf_path: str | Path,
    output_prefix: str | Path,
    *,
    pdfimages_bin: str = "pdfimages",
) -> list[Path]:
    require_binary(pdfimages_bin)
    pdf_path = _as_path(pdf_path)
    output_prefix = _as_path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    run_command([pdfimages_bin, "-all", str(pdf_path), str(output_prefix)])
    return sorted(path for path in output_prefix.parent.glob(f"{output_prefix.name}-*") if path.is_file())


def crop_image(
    image_path: str | Path,
    output_path: str | Path,
    box: tuple[int, int, int, int],
) -> Path:
    image_path = _as_path(image_path)
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(image_path).crop(box).save(output_path)
    return output_path


def images_to_pdf(image_paths: Iterable[str | Path], output_pdf: str | Path) -> Path:
    paths = [_as_path(path) for path in image_paths]
    if not paths:
        raise ValueError("No images were provided.")
    output_pdf = _as_path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    images = [Image.open(path).convert("RGB") for path in paths]
    images[0].save(output_pdf, save_all=True, append_images=images[1:])
    return output_pdf


def merge_pdfs(pdf_paths: Iterable[str | Path], output_pdf: str | Path, *, pdfunite_bin: str = "pdfunite") -> Path:
    require_binary(pdfunite_bin)
    paths = [_as_path(path) for path in pdf_paths if str(path).strip()]
    if not paths:
        raise ValueError("No PDF files were provided.")
    output_pdf = _as_path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    run_command([pdfunite_bin, *[str(path) for path in paths], str(output_pdf)])
    return output_pdf


def convert_office_to_pdf(
    input_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    soffice_bin: str = "soffice",
) -> Path:
    require_binary(soffice_bin)
    input_path = _as_path(input_path)
    output_dir = _as_path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command([soffice_bin, "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(input_path)])
    return output_dir / f"{input_path.stem}.pdf"


def compile_latex_with_tectonic(
    tex_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    tectonic_bin: str = "tectonic",
) -> Path:
    require_binary(tectonic_bin)
    tex_path = _as_path(tex_path)
    output_dir = _as_path(output_dir) if output_dir else tex_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command([tectonic_bin, "--outdir", str(output_dir), str(tex_path)], cwd=tex_path.parent)
    return output_dir / f"{tex_path.stem}.pdf"


def ensure_playwright_env(
    *,
    package_dir: str | Path,
    playwright_version: str = "1.58.2",
    browser: str = "chromium",
) -> Path:
    require_binary("npm")
    require_binary("npx")
    package_dir = _as_path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    package_json = package_dir / "package.json"
    if not package_json.exists():
        run_command(["npm", "init", "-y"], cwd=package_dir)

    playwright_dir = package_dir / "node_modules" / "playwright"
    if not playwright_dir.exists():
        run_command(["npm", "install", f"playwright@{playwright_version}"], cwd=package_dir)

    browser_marker = package_dir / f".{browser}_installed"
    if not browser_marker.exists():
        run_command(["npx", "playwright", "install", browser], cwd=package_dir)
        browser_marker.write_text(playwright_version, encoding="utf-8")

    return package_dir


def html_to_pdf_with_playwright(
    html_path: str | Path,
    output_pdf: str | Path,
    *,
    viewport: tuple[int, int] = (1280, 720),
    strategy: str = "slide_screenshots",
    package_dir: str | Path | None = None,
    playwright_version: str = "1.58.2",
) -> Path:
    require_binary("node")
    html_path = _as_path(html_path)
    output_pdf = _as_path(output_pdf)
    package_dir = _as_path(package_dir) if package_dir else output_pdf.parent / "playwright_env"
    package_dir = ensure_playwright_env(package_dir=package_dir, playwright_version=playwright_version)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ohnoppt_html_export_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        png_dir = temp_dir / "slides"
        png_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".mjs",
            prefix="export_",
            dir=package_dir,
            delete=False,
        ) as tmp_script:
            js_path = Path(tmp_script.name)

        script = f"""
import fs from "node:fs";
import path from "node:path";
import {{ chromium }} from "playwright";

const htmlPath = {json.dumps(str(html_path))};
const outputPdf = {json.dumps(str(output_pdf))};
const pngDir = {json.dumps(str(png_dir))};
const strategy = {json.dumps(strategy)};

const browser = await chromium.launch({{ headless: true }});
try {{
  const page = await browser.newPage({{
    viewport: {{ width: {viewport[0]}, height: {viewport[1]} }},
    deviceScaleFactor: 1,
  }});

  await page.goto(`file://${{htmlPath}}`, {{ waitUntil: "load" }});
  await page.waitForLoadState("networkidle");
  await page.emulateMedia({{ media: "screen" }});

  if (strategy === "slide_screenshots") {{
    const slides = page.locator(".slide");
    const count = await slides.count();
    if (count === 0) {{
      await page.pdf({{
        path: outputPdf,
        printBackground: true,
        preferCSSPageSize: true,
        margin: {{ top: "0", right: "0", bottom: "0", left: "0" }},
      }});
    }} else {{
      for (let i = 0; i < count; i += 1) {{
        const slide = slides.nth(i);
        await slide.screenshot({{
          path: path.join(pngDir, `slide-${{String(i + 1).padStart(2, "0")}}.png`),
        }});
      }}
    }}
  }} else {{
    await page.pdf({{
      path: outputPdf,
      printBackground: true,
      preferCSSPageSize: true,
      margin: {{ top: "0", right: "0", bottom: "0", left: "0" }},
    }});
  }}
}} finally {{
  await browser.close();
}}
"""
        try:
            js_path.write_text(script, encoding="utf-8")
            run_command(["node", str(js_path)], cwd=package_dir)
        finally:
            js_path.unlink(missing_ok=True)

        if strategy == "slide_screenshots":
            pngs = sorted(png_dir.glob("slide-*.png"))
            if pngs:
                return images_to_pdf(pngs, output_pdf)

    if not output_pdf.exists():
        raise RuntimeError("Playwright export did not produce the requested PDF.")
    return output_pdf

# HTML -> PDF Export Task Prompt Template

Export an HTML slide deck into a stable PDF.

## Inputs

- HTML file: `{{html_path}}`
- Asset directory: `{{assets_dir}}`
- Output PDF: `{{pdf_output_path}}`
- Preview directory: `{{preview_dir}}`

## Task Requirements

1. First decide on the export strategy:
   - If the print CSS is stable, exporting directly from the browser to PDF is allowed.
   - If print mode breaks the layout, switch to "capture each slide and assemble the PDF."
2. After export, generate at least key-page PNG previews.
3. Check:
   - Whether the page count is correct
   - Whether each page is complete
   - Whether there are blank pages, pages missing images, or pages that only contain titles
   - Whether fonts finished loading
   - Whether animation initial states or lazy-loaded elements failed to appear
4. If you find problems, do not stop at a bad result. Continue refining the export strategy.

## Outputs

- `slides.pdf`
- `preview/slide-*.png`
- `export_report.md`
  - Which export strategy was used
  - viewport / page size
  - Which pages were validated
  - What known issues still remain

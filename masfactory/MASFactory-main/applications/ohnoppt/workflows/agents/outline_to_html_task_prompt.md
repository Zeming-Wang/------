# Outline -> HTML Slides Task Prompt Template

Implement an HTML slide deck from the given outline.

## Inputs

- Outline file: `{{outline_path}}`
- Reference images / chart assets: `{{assets_dir}}`
- Style brief: `{{style_brief}}`
- Output path: `{{html_output_path}}`

## Task Requirements

1. Each outline slide must map to one HTML slide section.
2. Every slide must retain:
   - a title
   - a main conclusion
   - supporting content
   - the required figures / tables / cards
3. Design a unified CSS visual system.
4. Prepare for export and reconstruction:
   - Use stable classes such as `.slide`, `.panel`, `.metric-card`, and `.figure-card`
   - Do not rely on complex JavaScript animations
   - Do not absolutely position everything into an unmaintainable layout
   - If you use figures from the source paper, keep them as independent image nodes instead of baking them into a giant background
   - Only reference files that truly exist in the Asset Manifest; if no suitable image exists, switch to HTML components
   - Asset paths must resolve directly from the directory containing `slides.html`, for example `../assets/...`
5. After the deck is complete, include a structure note file describing:
   - which components each slide uses
   - which elements should later be reconstructed carefully in `python-pptx`
   - which elements are acceptable to keep as standalone images

## Outputs

- `slides.html`
- `assets/`, if needed
- `structure_notes.md`

## Quality Requirements

- Slides should fit a 16:9 presentation format
- Fonts and card sizes must remain readable after printing or screenshot export
- Do not overload key slides
- Minimize reliance on special print-CSS hacks; prioritize a stable slide structure under screen rendering

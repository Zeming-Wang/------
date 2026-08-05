# PDF -> python-pptx Recreation Task Prompt Template

Reconstruct an editable `pptx` in `python-pptx` from the rendered result of the target PDF.

## Inputs

- Target PDF: `{{target_pdf_path}}`
- Per-page target PDF previews: `{{preview_dir}}`
- Reusable image assets: `{{assets_dir}}`
- Optional source files, if available: `{{source_code_paths}}`
- Output directory: `{{output_dir}}`

## Task Requirements

1. First analyze the layout and visual primitives of each page in the target PDF.
2. Extract a set of reusable `python-pptx` helper functions.
3. Rebuild 2 to 3 representative slides first to validate the style and layout strategy, then expand to the full deck.
4. Rebuild every slide using those helper functions.
5. You must export:
   - the generation script
   - `pptx`
   - a validation PDF
6. You must perform at least one sampling check that includes:
   - the cover slide
   - one method slide
   - one result slide
   - one summary slide
7. If overflow, clipping, or tables breaking out of cards appears, keep fixing the deck instead of stopping at the first draft.

## Reconstruction Priorities

1. Page structure
2. Title hierarchy
3. Card and block system
4. Image-to-text proportion
5. Color and emphasis relationships
6. Detail shadows and whitespace

## Outputs

- `recreate_*.py`
- `*.pptx`
- `*.pdf`
- `recreation_notes.md`
  - Which elements were rebuilt with shapes/text
  - Which elements directly reuse original image assets
  - Which capability limits still remain
  - Which slides used style simplifications for stability

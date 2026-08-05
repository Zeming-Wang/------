# Requirements -> Outline Task Prompt Template

Generate an outline that can be used directly to build presentation slides from the following requirements and reference materials.

## Inputs

- Requirement description: `{{requirements_text}}`
- Reference material paths: `{{source_paths}}`
- Target audience: `{{audience}}`
- Slide budget: `{{page_budget}}`
- Presentation objective: `{{presentation_goal}}`

## Task Requirements

1. First extract the real question this presentation needs to answer.
2. Design a clear narrative spine.
3. Output an outline organized by slides.
4. For each slide, explicitly state:
   - the main conclusion
   - supporting points
   - required figures or images
   - the most suitable slide type:
     - `statement`
     - `explainer`
     - `method`
     - `evidence`
     - `comparison`
     - `summary`
5. If the reference material is a paper, cover at least:
   - problem background
   - method core
   - experimental setup
   - main results
   - validity explanation / ablations / error analysis
   - limitations and personal judgment
6. Add a `source_map` to help downstream agents trace the basis for each slide.

## Output Format

- Title
- Audience
- Narrative
- Talk Objective
- Slide Plan
  - Slide 1 ...
  - Slide 2 ...
  - ...
- Source Map
- Assets Needed
- Risks

## Additional Requirements

- The outline must serve downstream HTML slide production, so it must have slide-level granularity.
- Every slide must be implementable as an individual slide; do not provide vague sections.
- If a slide contains too much information, split it proactively instead of pushing the implementation burden downstream.

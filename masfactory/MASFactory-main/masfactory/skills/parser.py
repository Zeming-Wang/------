from pathlib import Path
from typing import Any

from .errors import SkillParseError


def parse_skill_markdown(path: str | Path) -> tuple[dict[str, Any], str, str]:
    """Parse a SKILL.md file into frontmatter, body text, and raw markdown.

    Args:
        path: Path to the `SKILL.md` file.

    Returns:
        A tuple of `(frontmatter, body, raw_markdown)`.

    Raises:
        SkillParseError: If the file cannot be read, the frontmatter is malformed,
            the frontmatter is not a mapping, or the markdown body is empty.
    """
    skill_md_path = Path(path).expanduser().resolve()
    try:
        raw = skill_md_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillParseError(f"Failed to read skill file '{skill_md_path}': {exc}") from exc

    frontmatter: dict[str, Any] = {}
    body = raw

    if raw.startswith("---\n"):
        closing_index = raw.find("\n---\n", 4)
        if closing_index == -1:
            raise SkillParseError(f"Invalid frontmatter in '{skill_md_path}': missing closing '---'")
        frontmatter_text = raw[4:closing_index]
        body = raw[closing_index + 5 :]
        try:
            import yaml  # noqa: WPS433
        except Exception as exc:
            raise SkillParseError("PyYAML is required to parse skill frontmatter.") from exc
        try:
            parsed = yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
        except Exception as exc:
            raise SkillParseError(f"Invalid YAML frontmatter in '{skill_md_path}': {exc}") from exc
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise SkillParseError(f"Invalid frontmatter in '{skill_md_path}': expected a mapping.")
        frontmatter = {str(k): v for k, v in parsed.items()}

    body = body.strip()
    if not body:
        raise SkillParseError(f"Skill file '{skill_md_path}' must contain non-empty markdown instructions.")

    return frontmatter, body, raw

class SkillError(Exception):
    """Base class for all public errors raised by MASFactory Skills APIs."""


class SkillNotFoundError(SkillError):
    """Raised when the target skill directory or its required SKILL.md file does not exist."""


class InvalidSkillPackageError(SkillError):
    """Raised when a skill path exists but does not conform to the expected package layout."""


class SkillParseError(SkillError):
    """Raised when SKILL.md cannot be read or parsed into a valid skill definition."""

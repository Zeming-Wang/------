"""
Patch MASFactory Agent to remove prompt wrapping and JSON format requirements.

MASFactory wraps prompts in "MESSAGE TO YOU:" and adds JSON output instructions.
This conflicts with MultiAgentESC prompts that expect plain text format like
"Response: [strategy] text".
"""

import re
import json as _json

from masfactory.components.agents.agent import Agent
from masfactory.core.message.base import StatefulFormatter


class PlainTextFormatter(StatefulFormatter):
    """Plain text formatter: clean prompts in, plain text out."""

    def __init__(self):
        super().__init__()
        self._is_input_formatter = True
        self._is_output_formatter = True
        self._agent_introducer = ""

    def dump(self, message):
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            if len(message) == 1:
                val = next(iter(message.values()))
                if isinstance(val, str):
                    return val
            lines = []
            for key, val in message.items():
                if val is None:
                    continue
                rendered = val if isinstance(val, str) else str(val)
                lines.append(f"{key}:\n{rendered}")
            return "\n\n".join(lines)
        return str(message)

    def format(self, message):
        raw = message
        if isinstance(raw, dict):
            return self._normalize_dict(raw)
        text = str(raw)
        text = re.sub(r"<think.*?</think\s*>", "", text, flags=re.DOTALL).strip()
        output_key = list(self._field_keys.keys())[0] if self._field_keys else "output"
        for cand in re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL):
            cand = cand.strip()
            try:
                parsed = _json.loads(cand)
                return self._normalize_dict(parsed)
            except Exception:
                pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = _json.loads(text[start:end + 1])
                return self._normalize_dict(parsed)
            except Exception:
                pass
        return {output_key: text.strip()}

    def _normalize_dict(self, d):
        output_key = list(self._field_keys.keys())[0] if self._field_keys else "output"
        result = {output_key: ""}
        if output_key in d:
            result[output_key] = d[output_key]
        else:
            for v in d.values():
                if isinstance(v, str):
                    result[output_key] = v
                    break
        return result


_original_input_prompt = Agent._input_prompt
_original_output_keys_prompt = Agent._output_keys_prompt


def _patched_input_prompt(self):
    formatted_input = self._prompt_template_format(self._prompt_template)
    return {"input": formatted_input}


def _patched_output_keys_prompt(self):
    return {}


Agent._input_prompt = _patched_input_prompt
Agent._output_keys_prompt = _patched_output_keys_prompt

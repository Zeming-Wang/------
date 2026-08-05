from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


class LLMJsonError(RuntimeError):
    pass


class NowWhatLLM:
    def __init__(self, *, model_name: str, api_key: str, base_url: str | None = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def _extract_text(self, response: object) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()

        output = getattr(response, "output", None) or []
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for block in content:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    parts.append(block_text)
        return "\n".join(part for part in parts if part).strip()

    def _extract_json_block(self, text: str) -> Any:
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last != -1 and last > first:
            return json.loads(cleaned[first : last + 1])

        first = cleaned.find("[")
        last = cleaned.rfind("]")
        if first != -1 and last != -1 and last > first:
            return json.loads(cleaned[first : last + 1])

        raise LLMJsonError("Model response did not contain valid JSON.")

    def _chat_text(self, *, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_output_tokens,
        )
        return self._extract_text(response)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        pdf_path: str | Path | None = None,
        max_output_tokens: int = 2400,
    ) -> Any:
        prompt = user_prompt
        if pdf_path:
            prompt = f"{user_prompt}\n\nReference PDF path: {Path(pdf_path).name}"
        raw_text = self._chat_text(
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_output_tokens=max_output_tokens,
        )
        return self._extract_json_block(raw_text)

    def generate_markdown(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 3200,
    ) -> str:
        return self._chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )

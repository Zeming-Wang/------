from __future__ import annotations

SKIP_LANGFLOW_NODE_KINDS = frozenset({"note", "noteNode", "genericNode"})

CHAT_INPUT_KINDS = frozenset({"ChatInput", "TextInput"})
CHAT_OUTPUT_KINDS = frozenset({"ChatOutput", "TextOutput"})
PROMPT_KINDS = frozenset({"Prompt", "PromptComponent"})
LOOP_KINDS = frozenset({"LoopComponent", "Loop"})
LLM_KINDS = frozenset(
    {
        "LanguageModelComponent",
        "ChatOpenAI",
        "OpenAIModel",
        "AnthropicModel",
        "OllamaModel",
        "GroqModel",
        "AzureOpenAIModel",
    }
)

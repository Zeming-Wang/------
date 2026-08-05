"""
Debate component: multi-agent discussion to select best response.
"""

from masfactory import Agent, OpenAIModel, RootGraph
from ..patch_masfactory import PlainTextFormatter


def run_debate(model: OpenAIModel, context, emo_and_reason, cau_and_reason, int_and_reason, responses):
    responses_template = "\n\n".join(responses)
    admin_prompt = f"""### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

Based on the provided information and dialogue context, please select the most appropriate response from the following options and explain why.

### Response
{responses_template}

Your answer must include the following elements:
Response: the most appropriate response and the strategy used in this response.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Response: [strategy] [response]
Reasoning: [reasoning]"""

    debate_history = []
    for i in range(len(responses)):
        if debate_history:
            history_text = "\n\n".join(debate_history)
            current_prompt = admin_prompt + "\n\n" + history_text
        else:
            current_prompt = admin_prompt

        g = RootGraph(f"debate_turn_{i}")
        node = g.create_node(Agent, f"debater_{i}",
            model=model,
            instructions=(
                "You are a psychologist who is good at listening to others' opinions and reflecting on your own thoughts. "
                "You are currently participating in a group discussion about which response is the most appropriate, "
                f"and you are inclined to support the response \"{responses[i]}\". "
                "However, during the discussion, you need to carefully consider others' perspectives "
                "and reflect on your own viewpoint, ultimately reaching a reliable answer."
            ),
            prompt_template="{debate_prompt}",
            model_settings={"max_tokens": 400},
            formatters=PlainTextFormatter(),
        )
        g.edge_from_entry(node, {"debate_prompt": ""})
        g.edge_to_exit(node, {"opinion": ""})
        g.build()
        out, _ = g.invoke({"debate_prompt": current_prompt})
        debate_history.append(out["opinion"])

    return debate_history

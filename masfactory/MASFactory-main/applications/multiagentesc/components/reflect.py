"""
Reflect component: agents reflect on debate and finalize their stance.
"""

from masfactory import Agent, OpenAIModel, RootGraph
from ..patch_masfactory import PlainTextFormatter


def run_reflect(model: OpenAIModel, context, emo_and_reason, cau_and_reason, int_and_reason,
                debate_history, responses):
    discussion_content = "\n\n".join(debate_history)
    admin_prompt = f"""### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

Based on the provided information and the context of the dialogue, a group discussion is taking place to determine which response is the most appropriate.

### Discussion content
{discussion_content}

You should carefully analyze the various different viewpoints above, reflect on your own thoughts, and ultimately arrive at a convincing result. Your thought can be changed if you believe the viewpoints of others are more reasonable.

Your answer must include the following elements:
Response: the most appropriate response and the strategy used in this response.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Response: [strategy] [response]
Reasoning: [reasoning]"""

    reflection_history = []
    for i in range(len(responses)):
        if reflection_history:
            history_text = "\n\n".join(reflection_history)
            current_prompt = admin_prompt + "\n\n" + history_text
        else:
            current_prompt = admin_prompt

        g = RootGraph(f"reflect_turn_{i}")
        node = g.create_node(Agent, f"reflector_{i}",
            model=model,
            instructions=(
                "You are a psychologist who is good at listening to others' opinions and reflecting on your own thoughts. "
                "You are currently participating in a group discussion about which response is the most appropriate, "
                f"and you are inclined to support the response \"{responses[i]}\". "
                "However, during the discussion, you need to carefully consider others' perspectives "
                "and reflect on your own viewpoint, ultimately reaching a reliable answer. "
                "Your thought can be changed if you believe the viewpoints of others are more reasonable."
            ),
            prompt_template="{reflect_prompt}",
            model_settings={"max_tokens": 400},
            formatters=PlainTextFormatter(),
        )
        g.edge_from_entry(node, {"reflect_prompt": ""})
        g.edge_to_exit(node, {"reflection": ""})
        g.build()
        out, _ = g.invoke({"reflect_prompt": current_prompt})
        reflection_history.append(out["reflection"])

    return reflection_history

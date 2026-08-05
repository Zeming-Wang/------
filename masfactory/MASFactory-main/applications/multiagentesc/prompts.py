"""
Prompt templates for MultiAgentESC.
"""

prompts = {
    "behavior_control": '''### Instruction
You are a psychological counseling expert. You will be provided with an incomplete conversation between an Assistant and a User.
Please analyze whether this conversation reflects the user's current emotional state, the reason the user is seeking emotional support, and how the user plans to cope with the event.
If all three points are reflected, please reply "YES," otherwise reply "NO."

### Conversation
{context}

Your answer must include two parts:
1. "YES" or "NO"
2. If "YES", briefly explain how the conversation reflects these elements; if "NO", explain which elements are missing.

Your answer must follow this format:
1. [YES or NO]
2. [explaination]
''',

    "zero_shot": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Your task is to play a role as 'Assistant' and generate a response based on the given dialogue context.

### Dialogue context
{context}

Your answer must be fewer than 30 words and must follow this format:
Response: [response]
''',

    "get_emotion": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Please infer the emotional state expressed in the user's last utterance.

### Dialogue context
{context}

Your answer must include the following elements:
Emotion: the emotion user expressed in their last utterance.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Emotion: [emotion]
Reasoning: [reasoning]
''',

    "get_cause": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Another agent analyzes the conversation and infers the emotional state expressed by the user in their last utterance.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

Please infer the specific event that led to the user's emotional state based on the dialogue context. Your answer must include the following elements:
Event: the specific event that led to the user's emotional state.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Event: [event]
Reasoning: [reasoning]
''',

    "get_intention": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Other agents have analyzed the conversation, infering the emotional state expressed by the user in their last utterance and the specific event that led to the user's emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

Please reasonably infer the user's intention based on the dialogue context, with the goal of addressing the event that lead to their emotional state. Your answer must include the following elements:
Intention: user's intention which aims to address the event that lead to their emotional state.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Intention: [intention]
Reasoning: [reasoning]
''',

    "select_strategy": '''### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

Based on the provided information and dialogue context, please select a strategy for the 'Assistant' to generate an appropriate response, and explain why. Your strategy should differnet from others as much as possible.
The following are examples of different strategies, all presented in the format of <post\n[strategy] response>.

### Examples
{examples}

Your answer must include the following elements:
Strategy: Strategy for generating an response. The strategy must appear in the examples. Please choose different strategy from others as much as possible.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Strategy: [strategy]
Reasoning: [reasoning]
''',

    "response_with_strategy": '''You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}


Please generate a response from the Assistant's perspective using the {strategy} strategy.
The following are examples of this strategy, all presented in the format of <post\n[strategy] response>.

### Examples
{examples}

Your answer must be fewer than 30 words and must follow this format:
Response: [strategy] [response]
''',

    "debate": '''### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

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
Reasoning: [reasoning]''',

    "reflect": '''### You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

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
Reasoning: [reasoning]''',

    "judge": '''You will be provided with a dialogue context between an 'Assistant' and a 'User'.

### Dialogue context
{context}

The following are responses generated by the therapist using different strategies, all presented in the format of <[strategy] response>. Please select the most appropriate response and explain why.

### Examples
{template_responses}

Your answer must include the following elements:
Response: the most appropriate response and the strategy used in this response.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Response: [strategy] [response]
Reasoning: [reasoning]''',

    "self_reflection": '''You will be provided with a dialogue context between an 'Assistant' and a 'User'.

### Dialogue context
{context}

The following is a responses generated by the therapist using {pred_strategy} strategy, presented in the format of <[strategy] response>. Please analyze whether this response is consistent with the ongoing conversation, whether it aligns with the strategy, and whether it effectively helps alleviate the user's emotional stress.

### Response
[{pred_strategy}] {response}

If the respones meets the above requirements, please return it as is; if not, please modify the response and provide a more refined version. Refined version must less than 30 words.

Your answer must include the following elements:
Response: original response or refined response and the strategy used in this response.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Response: [strategy] [origianl/refined response]
Reasoning: [reasoning]''',
}


def get_prompt(prompt_name):
    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found.")
    return prompts[prompt_name]

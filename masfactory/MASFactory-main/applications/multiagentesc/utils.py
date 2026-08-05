"""
Utility functions for MultiAgentESC on MASFactory.
"""

import re
import json
import heapq
import numpy as np
from collections import Counter
from sentence_transformers import util
from masfactory import VectorRetriever, SentenceTransformerEmbedder
from masfactory.adapters.context.types import ContextQuery


def clean_strategy(strategies):
    cleaned_strategy = set()
    for s in strategies:
        if s.lower() == "question":
            cleaned_strategy.add("Question")
        elif s.lower() == "restatement or paraphrasing":
            cleaned_strategy.add("Restatement or Paraphrasing")
        elif s.lower() == "reflection of feelings":
            cleaned_strategy.add("Reflection of feelings")
        elif s.lower() == "self-disclosure":
            cleaned_strategy.add("Self-disclosure")
        elif s.lower() == "affirmation and reassurance":
            cleaned_strategy.add("Affirmation and Reassurance")
        elif s.lower() == "providing suggestions":
            cleaned_strategy.add("Providing Suggestions")
        elif s.lower() == "information":
            cleaned_strategy.add("Information")
        elif s.lower() == "others":
            cleaned_strategy.add("Others")
    return list(cleaned_strategy)


def vote(results):
    count = {}
    strat2response = {}
    for result in results:
        try:
            response, _ = result.split("\nReasoning")
            strategy, response = re.findall(r'Response:\s*\[([a-zA-Z ]+)\](.*)', response)[0]
            strategy, response = strategy.strip(), response.strip()
            if strategy not in count:
                count[strategy] = 0
            count[strategy] += 1
            if strategy not in strat2response:
                strat2response[strategy] = response
        except:
            continue
    if len(count) == 0:
        return ["None"], ["None"]
    max_count = max(count.values())
    max_strat = [key for key, value in count.items() if value == max_count]
    responses = [strat2response[strat] for strat in max_strat]
    return max_strat, responses


def json2natural(history):
    natural_language = ""
    for u in history:
        content = u["content"].strip()
        role = u["role"].capitalize() if "role" in u.keys() else u["speaker"].capitalize()
        if role == "Supporter":
            role = "Assistant"
        if role == "Seeker":
            role = "User"
        natural_language += f"{role}: {content} "
    return natural_language.strip()


def build_dataset(dataset_path, start_id=100):
    with open(dataset_path, "r", encoding="UTF-8") as f:
        dataset = json.load(f)
    record_seeker_supporter = {}
    record_seeker = {}
    for id in range(start_id, len(dataset)):
        dialog = dataset[id]["dialog"]
        for turn in range(len(dialog) - 1):
            if dialog[turn]["speaker"] == "seeker" and dialog[turn + 1]["speaker"] == "supporter":
                doc_id = f"s{id}_t{turn}"
                record_seeker_supporter[doc_id] = {
                    "seeker": dialog[turn]["content"],
                    "supporter": dialog[turn + 1]["content"],
                    "strategy": dialog[turn + 1]["annotation"]["strategy"],
                }
                record_seeker[doc_id] = dialog[turn]["content"]
    return record_seeker_supporter, record_seeker


def build_retriever(record_seeker, embedding_fn):
    retriever = VectorRetriever(
        documents=record_seeker,
        embedding_function=embedding_fn,
        similarity_threshold=-1,
        context_label="MultiAgentESC",
    )
    return retriever


def get_top_k(record_seeker_supporter, retriever, post, top_k=10):
    blocks = retriever.get_blocks(ContextQuery(query_text=post), top_k=top_k)
    top_pairs = []
    for block in blocks:
        doc_id = block.metadata["doc_id"]
        if doc_id in record_seeker_supporter:
            pair = record_seeker_supporter[doc_id]
            top_pairs.append((pair["seeker"], f"[{pair['strategy']}] {pair['supporter']}"))
    return top_pairs


def parse_single_response(response):
    try:
        response = re.findall(r'Response:\s*(.*)', response)[0].strip()
    except:
        response = response.strip() if response and response.strip() else "None"
    return response


def _strip_reasoning(text):
    if "\nReasoning" in text:
        text = text.split("\nReasoning")[0].strip()
    if "Reasoning:" in text:
        text = text.split("Reasoning:")[0].strip()
    return text


def parse_strategy_response(response, strategy=""):
    try:
        response = re.findall(r'Response:\s*\[[a-zA-Z ]+\](.*)', response)[0].strip()
    except:
        if strategy and strategy in response:
            response = response.split(strategy, 1)[1].strip()
            if response.startswith("]"):
                response = response[1:].strip()
        else:
            response = "None"
    return _strip_reasoning(response)


def filter_examples_by_strategy(pairs, target_strategy):
    examples = ""
    for pair in pairs:
        strat = pair[1].split("]", 1)[0].strip("[").strip()
        if strat == target_strategy:
            examples += f"{pair[0]}\n{pair[1]}\n\n"
    return examples.strip()

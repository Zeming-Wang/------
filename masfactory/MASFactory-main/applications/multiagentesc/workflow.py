"""
Main workflow for MultiAgentESC on MASFactory.
"""

import re
from .patch_masfactory import PlainTextFormatter
from masfactory import Agent, OpenAIModel, RootGraph
from .prompts import get_prompt
from .utils import (
    clean_strategy, vote, json2natural, get_top_k,
    parse_single_response, parse_strategy_response,
    filter_examples_by_strategy,
)
from .components.strategy_discussion import create_strategy_discussion_graph, run_strategy_discussion
from .components.debate import run_debate
from .components.reflect import run_reflect


def _create_graph_with_agent(model, name, prompt_template, instructions,
                              in_keys, out_key, model_settings=None):
    kwargs = dict(
        cls=Agent,
        name="agent",
        model=model,
        instructions=instructions,
        prompt_template=prompt_template,
        formatters=PlainTextFormatter(),
    )
    if model_settings:
        kwargs["model_settings"] = model_settings
    g = RootGraph(name)
    g.create_node(**kwargs)
    g.edge_from_entry(g._nodes["agent"], in_keys)
    g.edge_to_exit(g._nodes["agent"], {out_key: ""})
    g.build()
    return g


def _make_is_complex_wrapper(is_complex_graph):
    def is_complex(context):
        out, _ = is_complex_graph.invoke({"context": context})
        flag_str = out.get("flag", "")
        return "yes" in flag_str.lower()
    return is_complex


def _make_judge_wrapper(judge_graph):
    def judge(context, strategies, responses):
        template = []
        for strategy, response in zip(strategies, responses):
            template.append(f"[{strategy}] {response}")
        template_responses = "\n\n".join(template)
        out, _ = judge_graph.invoke({
            "context": context,
            "template_responses": template_responses,
        })
        try:
            strategy, response = re.findall(
                r'Response:\s*\[([a-zA-Z ]+)\](.*)', out["judgement"]
            )[0]
        except:
            strategy, response = "None", "None"
        return strategy.strip(), response.strip()
    return judge


def _make_self_reflection_wrapper(reflection_graph):
    def self_reflection(context, pred_strategy, response):
        out, _ = reflection_graph.invoke({
            "context": context,
            "pred_strategy": pred_strategy,
            "response": response,
        })
        try:
            strategy, response = re.findall(
                r'Response:\s*\[([a-zA-Z ]+)\](.*)', out["reflection"]
            )[0]
        except:
            raw = out.get("reflection", "")
            try:
                after_resp = re.findall(r'Response:\s*(.*)', raw)[0].strip()
                m = re.match(r'\[([a-zA-Z ]+)\]\s*(.*)', after_resp)
                if m:
                    strategy, response = m.group(1), m.group(2)
                else:
                    strategy, response = pred_strategy, after_resp
            except:
                strategy, response = pred_strategy, response
        if isinstance(response, str) and "Reasoning:" in response:
            response = response.split("Reasoning:")[0].strip()
        return strategy.strip(), response.strip()
    return self_reflection


def create_workflow(model):
    settings_100 = {"max_tokens": 100}
    settings_400 = {"max_tokens": 400}

    is_complex_graph = _create_graph_with_agent(
        model, "is_complex",
        get_prompt("behavior_control"),
        "You are a psychological counseling expert.",
        {"context": ""},
        "flag",
        model_settings=settings_100,
    )
    zero_shot_graph = _create_graph_with_agent(
        model, "zero_shot",
        get_prompt("zero_shot"),
        "You are a psychological counseling expert.",
        {"context": ""},
        "response",
        model_settings=settings_100,
    )
    emotion_graph = _create_graph_with_agent(
        model, "emotion",
        get_prompt("get_emotion"),
        "You are a psychological counseling expert.",
        {"context": ""},
        "emotion_result",
        model_settings=settings_400,
    )
    cause_graph = _create_graph_with_agent(
        model, "cause",
        get_prompt("get_cause"),
        "You are a psychological counseling expert.",
        {"context": "", "emo_and_reason": ""},
        "cause_result",
        model_settings=settings_400,
    )
    intention_graph = _create_graph_with_agent(
        model, "intention",
        get_prompt("get_intention"),
        "You are a psychological counseling expert.",
        {"context": "", "emo_and_reason": "", "cau_and_reason": ""},
        "intention_result",
        model_settings=settings_400,
    )
    response_gen_graph = _create_graph_with_agent(
        model, "response_gen",
        get_prompt("response_with_strategy"),
        "You are a psychological counseling expert.",
        {"context": "", "emo_and_reason": "", "cau_and_reason": "",
         "int_and_reason": "", "strategy": "", "examples": ""},
        "response",
        model_settings=settings_100,
    )
    judge_graph = _create_graph_with_agent(
        model, "judge",
        get_prompt("judge"),
        "You are a psychological counseling expert.",
        {"context": "", "template_responses": ""},
        "judgement",
        model_settings=settings_400,
    )
    reflection_graph = _create_graph_with_agent(
        model, "self_reflection",
        get_prompt("self_reflection"),
        "You are a psychological counseling expert.",
        {"context": "", "pred_strategy": "", "response": ""},
        "reflection",
        model_settings=settings_400,
    )
    strategy_discussion_graph = create_strategy_discussion_graph(model)

    graphs = {
        "is_complex": is_complex_graph,
        "zero_shot": zero_shot_graph,
        "emotion": emotion_graph,
        "cause": cause_graph,
        "intention": intention_graph,
        "response_gen": response_gen_graph,
        "judge": judge_graph,
        "self_reflection": reflection_graph,
        "strategy_discussion": strategy_discussion_graph,
    }
    utils = {
        "json2natural": json2natural,
        "get_top_k": get_top_k,
        "clean_strategy": clean_strategy,
        "parse_single_response": parse_single_response,
        "parse_strategy_response": parse_strategy_response,
        "filter_examples_by_strategy": filter_examples_by_strategy,
        "run_strategy_discussion": run_strategy_discussion,
        "run_debate": run_debate,
        "run_reflect": run_reflect,
        "vote": vote,
        "judge": _make_judge_wrapper(judge_graph),
        "self_reflection": _make_self_reflection_wrapper(reflection_graph),
        "is_complex": _make_is_complex_wrapper(is_complex_graph),
    }
    return graphs, utils


def process(message, attributes, model, graphs, utils):
    dialog = message.get("dialog", [])
    count = int(attributes.get("count", 0))
    history = list(attributes.get("history", []))
    ret = list(attributes.get("ret", []))
    record_seeker_supporter = attributes.get("record_seeker_supporter", {})
    retriever = attributes.get("retriever")

    while count < len(dialog):
        save = {}

        if count != 0 and dialog[count]["speaker"] == "supporter":
            is_single = (
                (count < len(dialog) - 1 and dialog[count + 1]["speaker"] != "supporter") or
                (count == len(dialog) - 1)
            )

            if is_single:
                save["strategy"] = dialog[count]["annotation"]["strategy"]
                save["reference"] = dialog[count]["content"].strip()
            else:
                save["strategy"] = (
                    f"{dialog[count]['annotation']['strategy']} and "
                    f"{dialog[count + 1]['annotation']['strategy']}"
                )
                save["reference"] = (
                    dialog[count]["content"].strip() + " " + dialog[count + 1]["content"].strip()
                )

            context = utils["json2natural"](history)
            save["context"] = context
            post = history[-1]["content"] if history else ""

            history.append({
                "content": dialog[count]["content"].strip(),
                "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"
            })
            if not is_single:
                history.append({
                    "content": dialog[count + 1]["content"].strip(),
                    "role": "user" if dialog[count + 1]["speaker"] == "seeker" else "assistant"
                })
                count += 2
            else:
                count += 1

            if count <= 5 or not utils["is_complex"](context):
                out_zs, _ = graphs["zero_shot"].invoke({"context": context})
                save["response"] = utils["parse_single_response"](out_zs.get("response", ""))
                save["pred_strategy"] = "None"
            else:
                out_emo, _ = graphs["emotion"].invoke({"context": context})
                emo_result = out_emo.get("emotion_result", "")
                try:
                    if "</think" in emo_result:
                        emo_result = emo_result.split("</think")[1].strip()
                    emotion = re.findall(r"Emotion:(.*)", emo_result.split("\n")[0])[0].strip()
                except:
                    emotion = "Negative"
                emo_and_reason = emo_result

                out_cause, _ = graphs["cause"].invoke({
                    "context": context,
                    "emo_and_reason": emo_and_reason,
                })
                cau_result = out_cause.get("cause_result", "")
                try:
                    if "</think" in cau_result:
                        cau_result = cau_result.split("</think")[1].strip()
                    cause = re.findall(r"Event:(.*)", cau_result.split("\n")[0])[0].strip()
                except:
                    cause = "Not mention"
                cau_and_reason = cau_result

                out_int, _ = graphs["intention"].invoke({
                    "context": context,
                    "emo_and_reason": emo_and_reason,
                    "cau_and_reason": cau_and_reason,
                })
                int_result = out_int.get("intention_result", "")
                try:
                    if "</think" in int_result:
                        int_result = int_result.split("</think")[1].strip()
                    intention = re.findall(r"Intention:(.*)", int_result.split("\n")[0])[0].strip()
                except:
                    intention = "Not mention"
                int_and_reason = int_result

                pairs = utils["get_top_k"](record_seeker_supporter, retriever, post, top_k=10)
                examples_str = "\n\n".join([f"{p[0]}\n{p[1]}" for p in pairs])

                pred_strategies = utils["run_strategy_discussion"](
                    graphs["strategy_discussion"],
                    context=context,
                    emo_and_reason=emo_and_reason,
                    cau_and_reason=cau_and_reason,
                    int_and_reason=int_and_reason,
                    examples=examples_str,
                )
                pred_strategies = utils["clean_strategy"](pred_strategies)

                save["emotion"] = emotion
                save["cause"] = cause
                save["intention"] = intention

                if len(pred_strategies) == 0:
                    out_zs, _ = graphs["zero_shot"].invoke({"context": context})
                    save["response"] = utils["parse_single_response"](out_zs.get("response", ""))
                    save["pred_strategy"] = "None"
                else:
                    if len(pred_strategies) == 1:
                        examples_filtered = utils["filter_examples_by_strategy"](
                            pairs, pred_strategies[0]
                        )
                        out_resp, _ = graphs["response_gen"].invoke({
                            "context": context,
                            "emo_and_reason": emo_and_reason,
                            "cau_and_reason": cau_and_reason,
                            "int_and_reason": int_and_reason,
                            "strategy": pred_strategies[0],
                            "examples": examples_filtered,
                        })
                        response = utils["parse_strategy_response"](out_resp.get("response", ""), pred_strategies[0])
                        save["ori_response"] = response
                        save["pred_strategy"] = pred_strategies[0]
                    else:
                        responses_list = []
                        for strat in pred_strategies:
                            examples_filtered = utils["filter_examples_by_strategy"](
                                pairs, strat
                            )
                            out_resp, _ = graphs["response_gen"].invoke({
                                "context": context,
                                "emo_and_reason": emo_and_reason,
                                "cau_and_reason": cau_and_reason,
                                "int_and_reason": int_and_reason,
                                "strategy": strat,
                                "examples": examples_filtered,
                            })
                            resp = utils["parse_strategy_response"](out_resp.get("response", ""), strat)
                            responses_list.append(f"[{strat}] {resp}")

                        debate_history = utils["run_debate"](
                            model, context, emo_and_reason, cau_and_reason,
                            int_and_reason, responses_list
                        )
                        reflection_result = utils["run_reflect"](
                            model, context, emo_and_reason, cau_and_reason,
                            int_and_reason, debate_history, responses_list
                        )
                        strats, final_responses = utils["vote"](reflection_result)
                        strats = utils["clean_strategy"](strats)
                        if not strats:
                            strats, final_responses = ["None"], ["None"]

                        if len(strats) == 1 and strats[0] != "None":
                            pred_strategy_str = strats[0].strip()
                            response = final_responses[0].strip()
                            save["ori_response"] = response
                            save["pred_strategy"] = pred_strategy_str
                        else:
                            pred_strategy_str, response = utils["judge"](
                                context, strats, final_responses
                            )
                            cleaned = utils["clean_strategy"]([pred_strategy_str])
                            if cleaned:
                                pred_strategy_str = cleaned[0]
                            save["ori_response"] = response
                            save["pred_strategy"] = pred_strategy_str

                    pred_str = save["pred_strategy"]
                    orig_resp = save.get("ori_response", "")
                    if not orig_resp or orig_resp == "None":
                        out_zs, _ = graphs["zero_shot"].invoke({"context": context})
                        save["response"] = utils["parse_single_response"](out_zs.get("response", ""))
                        save["pred_strategy"] = "None"
                    else:
                        _, response_out = utils["self_reflection"](
                            context, pred_str, orig_resp
                        )
                        if response_out.startswith("Refined response:"):
                            response_out = response_out[len("Refined response:"):].strip()
                        if (not response_out or response_out == "None") and orig_resp:
                            response_out = orig_resp
                        save["response"] = response_out

            ret.append(save)
        else:
            history.append({
                "content": dialog[count]["content"].strip(),
                "role": "user" if dialog[count]["speaker"] == "seeker" else "assistant"
            })
            count += 1

    return {"history": history, "ret": ret, "count": count}

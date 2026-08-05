import argparse
import json
import os
from pathlib import Path


def find_env_file(start_path):
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        env_path = candidate / ".env"
        if env_path.exists():
            return env_path
    return None


# Load .env file automatically
def load_env_file(dotenv_path):
    if dotenv_path and os.path.exists(dotenv_path):
        with open(dotenv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val

# Prefer the application-local .env, but remain compatible with the current repo layout.
load_env_file(find_env_file(__file__))

from utils.openai_model import OpenAIModel
from eval_functions.eval_criterion import evaluate_aut, evaluate_scientific, evaluate_wkct
import logging
from automation_csv import calculate_mean_std, write_results_to_csv

TASK_PATHS = {
    "AUT": "Results/AUT/Output",
    "Scientific": "Results/Scientific/Output",
    "Instances": "Results/Instances/Output",
    "Similarities": "Results/Similarities/Output",
}

EVAL_PATHS = {
    "AUT": "Results/AUT/Eval_Result",
    "Scientific": "Results/Scientific/Eval_Result",
    "Instances": "Results/Instances/Eval_Result",
    "Similarities": "Results/Similarities/Eval_Result",
}

def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def build_default_input_file_path(args):
    task_folder = TASK_PATHS[args.task]
    return os.path.join(Path(__file__).parent, '..', task_folder, f"{args.input_file.split('_')[1]}_agent", f"{args.input_file}.json")


def build_default_output_file_path(args):
    output_folder = EVAL_PATHS[args.task]
    return os.path.join(
        Path(__file__).parent,
        '..',
        output_folder,
        f"{args.input_file.split('_')[1]}_agent",
        f"evaluation_{args.input_file}_{args.type}_{args.version}.json",
    )


def normalize_output_path(output_file_path):
    output_file_path = os.path.abspath(output_file_path)
    if os.name == 'nt' and len(output_file_path) >= 250 and not output_file_path.startswith('\\\\?\\'):
        output_file_path = '\\\\?\\' + output_file_path
    return output_file_path


def load_responses(args):
    input_file_path = args.input_json_path if args.input_json_path else build_default_input_file_path(args)
    ensure_folder_exists(os.path.dirname(input_file_path))
    with open(input_file_path, "r") as file:
        return json.load(file)


def build_model(args):
    api_key = os.getenv("OPENAI_API_KEY")
    requested_model = "gpt-4-0125-preview" if args.version == "4" else "gpt-3.5-turbo-0125"
    actual_model = os.getenv("MODEL_NAME", requested_model)
    print(f"Requested judge model: {requested_model}, actual model from env: {actual_model}")
    cache_file_name = args.cache_file or f"cache_{args.version}.pickle"
    return OpenAIModel(cache_file_name, requested_model, api_key)


def evaluate_responses(args, responses):
    model = build_model(args)
    total_results = []
    sampling_criteria = ["originality", "elaboration"]
    evaluation_criteria = ["fluency", "flexibility"]
    selected_criteria = evaluation_criteria + sampling_criteria 

    if args.task == "AUT":
        for response_obj in responses:
            item = response_obj['item']
            uses = response_obj.get('uses', [])
            item_results = {"item": item}
            if not uses:  # Check if 'uses' is empty
                for criterion in selected_criteria:
                    responses = [{"response": "No uses provided", "score": 0}]
                    item_results[criterion] = responses
                    log_score = {f"average_{criterion}": 0}
                    item_results[criterion].append(log_score)
            else:
                for criterion in evaluation_criteria:
                    result = evaluate_aut(model, response_obj, criterion, args.type, args.sample)
                    item_results[criterion] = [result]
                    model.save_cache()
                for criterion in sampling_criteria:
                    total = []
                    for use in uses:
                        result = evaluate_aut(model, {"item": item, "uses": [use]}, criterion, args.type, 1)
                        total.append(result)
                        print(f"Item: {item}, Use: {use}, {criterion.capitalize()} Score: {result['average_score']}")
                    item_results[criterion] = total
                    model.save_cache()
                for criterion in evaluation_criteria:
                    avg_score = sum(res['average_score'] for res in item_results[criterion]) / len(item_results[criterion])
                    log_score = {f"average_{criterion}": avg_score}
                    item_results[criterion].append(log_score)
                for criterion in sampling_criteria:
                    avg_score = sum(res['average_score'] for res in item_results[criterion]) / len(item_results[criterion])
                    log_score = {f"average_{criterion}": avg_score}
                    item_results[criterion].append(log_score)
            total_results.append(item_results)      

    elif args.task == "Scientific":
        print("Scientific Task")
        for response_obj in responses:
            question = response_obj['question']
            answer = response_obj.get('answer',[])
            question_results = {"question": question}
            if not answer:  # Check if 'answer' is empty
                for criterion in selected_criteria:
                    responses = [{"answer": "No responses provided", "score": 0}]
                    question_results[criterion] = responses
                    log_score = {f"average_{criterion}": 0}
                    question_results[criterion].append(log_score)
            else:
                for criterion in evaluation_criteria:
                    result = evaluate_scientific(model, response_obj, criterion, args.type, args.sample)
                    question_results[criterion] = [result]
                    model.save_cache()
                for criterion in sampling_criteria:
                    total = []
                    for ans in answer:
                        result = evaluate_scientific(model, {"question": question, "answer": [ans]}, criterion, args.type, 1)
                        total.append(result)
                        print(f"Question: {question}, Answer: {ans}, {criterion.capitalize()} Score: {result['average_score']}")
                    question_results[criterion] = total
                    model.save_cache()
                for criterion in evaluation_criteria:
                    avg_score = sum(res['average_score'] for res in question_results[criterion]) / len(question_results[criterion])
                    log_score = {f"average_{criterion}": avg_score}
                    question_results[criterion].append(log_score)
                for criterion in sampling_criteria:
                    avg_score = sum(res['average_score'] for res in question_results[criterion]) / len(question_results[criterion])
                    log_score = {f"average_{criterion}": avg_score}
                    question_results[criterion].append(log_score)
            total_results.append(question_results)

    elif args.task == "Instances" or args.task == "Similarities":
        print("WKCT Task")
        for response_obj in responses:
            question = response_obj['question']
            answer = response_obj.get('answer',[])
            question_results = {"question": question}
            if not answer:  # Check if 'answer' is empty
                for criterion in selected_criteria:
                    responses = [{"answer": "No responses provided", "score": 0}]
                    question_results[criterion] = responses
                    log_score = {f"average_{criterion}": 0}
                    question_results[criterion].append(log_score)
            else:
                for criterion in evaluation_criteria:
                    result = evaluate_wkct(model, response_obj, criterion, args.type, args.sample)
                    question_results[criterion] = [result]
                    model.save_cache()
                for criterion in sampling_criteria:
                    total = []
                    for ans in answer:
                        result = evaluate_wkct(model, {"question": question, "answer": [ans]}, criterion, args.type, 1)
                        total.append(result)
                        print(f"Question: {question}, Answer: {ans}, {criterion.capitalize()} Score: {result['average_score']}")
                    question_results[criterion] = total
                    model.save_cache()
                for criterion in evaluation_criteria:
                    avg_score = sum(res['average_score'] for res in question_results[criterion]) / len(question_results[criterion])
                    log_score = {f"average_{criterion}": avg_score}
                    question_results[criterion].append(log_score)
                for criterion in sampling_criteria:
                    avg_score = sum(res['average_score'] for res in question_results[criterion]) / len(question_results[criterion])
                    log_score = {f"average_{criterion}": avg_score}
                    question_results[criterion].append(log_score)

            total_results.append(question_results)

    return total_results


def write_evaluation_json(total_results, output_file_path):
    output_file_path = normalize_output_path(output_file_path)
    ensure_folder_exists(os.path.dirname(output_file_path))
    with open(output_file_path, "w") as outfile:
        json.dump(total_results, outfile, indent=4)
    print(f"Results saved to {output_file_path}")


def write_leaderboard_from_results(args, total_results):
    mean_std_results = calculate_mean_std(total_results)
    output_csv_path = os.path.join(Path(__file__).parent, '..', 'Results', 'LeaderBoard', f'LeaderBoard-{args.task}.csv')
    ensure_folder_exists(os.path.dirname(output_csv_path))
    write_results_to_csv(args.input_file, mean_std_results, output_csv_path, args.version)


def auto_grade(args):
    print("AUTO GRADE STARTED, Input_file: ", args.input_file)
    if args.input_file:
        print(f"{args.input_file.split('_')[1]}_agent")
    elif not getattr(args, "input_json_path", None):
        raise ValueError("Either --input_file or --input-json-path must be provided.")
    responses = load_responses(args)
    total_results = evaluate_responses(args, responses)
    output_json_path = getattr(args, "output_json_path", None)
    output_file_path = output_json_path if output_json_path else build_default_output_file_path(args)
    write_evaluation_json(total_results, output_file_path)
    if getattr(args, "no_leaderboard", False) or args.output == 'n':
        print('Output will not be saved in Leader Board!')
    else:
        write_leaderboard_from_results(args, total_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # PARSERS
    parser = argparse.ArgumentParser(description="Evaluate responses based on specified criteria using OpenAI's API.")
    parser.add_argument("-v", "--version", default="3", choices=["3", "4"], help="Version of the OpenAI model to use.")
    parser.add_argument("-i", "--input_file", default=None, help="Name of the input file located in the Results directory.")
    parser.add_argument("-t", "--type", default="sampling", choices=["default", "sampling"], help="Variant of the evaluation.")
    parser.add_argument("-s", "--sample", default=3, type=int, help="Number of times to sample the evaluation.")
    parser.add_argument("-d", "--task", default="AUT", choices = ["AUT", "Scientific", "Instances", "Similarities"], help="Task for the evaluation. Default is AUT.")
    parser.add_argument("-o", "--output", default="n", choices=["y", "n"], help="Output into LeaderBoard or not")
    parser.add_argument("--input-json-path", default=None, help="Direct path to an input JSON file.")
    parser.add_argument("--output-json-path", default=None, help="Direct path to write evaluation JSON.")
    parser.add_argument("--no-leaderboard", action="store_true", help="Skip writing the leaderboard CSV.")
    parser.add_argument("--cache-file", default=None, help="Override the evaluation cache pickle filename.")
    args = parser.parse_args()
    auto_grade(args)


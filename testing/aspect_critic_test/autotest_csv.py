import requests
import re
import json
import time
import os
import csv
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore, RougeScore
from ragas.metrics import AspectCritic
from langchain.llms import Ollama
from ragas.llms import LangchainLLMWrapper
from ragas.exceptions import RagasOutputParserException

from datetime import datetime


API_URL = "http://localhost:8000/ask"
METADATA_URL = "http://localhost:8000/metadata"
CSV_FILE = "/Users/lelange/Uni/askTHWS/testing/aspect_critic_test/fragenkatalog_askTHWS.csv"


def query_api(question):
    """
    Queries the API and handles both successful and error responses.
    Returns the JSON response and the HTTP status code.
    """
    try:
        response = requests.post(API_URL, json={"query": question}, timeout=10000)
        if response.status_code == 200:
            return response.json(), response.status_code
        else:
            try:
                error_json = response.json()
            except requests.exceptions.JSONDecodeError:
                error_json = {"detail": response.text}
            return error_json, response.status_code
    except requests.exceptions.RequestException as e:
        return {"detail": f"Failed to connect to API: {e}"}, 503


def get_metadata():
    """Gets metadata from the API."""
    try:
        response = requests.get(METADATA_URL)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch metadata: {e}")
        return {}


def load_test_cases(csv_path: str):
    """Loads question/gold-answer pairs from a CSV file."""
    cases = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row['Question']
            gold = row['Answer']
            cases.append((question, gold))
    return cases


def run_tests():
    """Main testing routine using CSV and AspectCritic."""
    metadata = get_metadata()
    if not metadata:
        print("Aborting tests due to failed metadata fetch.")
        return

    all_cases = load_test_cases(CSV_FILE)
    cases = all_cases[6:16]
    if not cases:
        print("No test cases found.")
        return

    os.makedirs("test_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_csv = os.path.join("test_results", f"results_{timestamp}.csv")

    # Initialize local Ollama LLM client
    ollama_client = Ollama(
        base_url="http://127.0.0.1:11434",
        model="gemma3:4b"
    )
    wrapped_llm = LangchainLLMWrapper(ollama_client)

    correctness_evaluator = AspectCritic(
        name="correctness",
        definition="Ist die generierte Antwort korrekt im Vergleich zur Gold-Antwort?",
        llm=wrapped_llm
    )
    bleu_evaluator = BleuScore()
    f1_evaluator = RougeScore(rouge_type="rougeL", mode="fmeasure")

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question', 'bot_answer', 'gold_answer', 'genauigkeit', 'F1', 'BLEU'])
        for i, (question, gold_answer) in enumerate(cases):
            print(f"Testing {i+1}/{len(cases)}: {question}")
            start_time = time.time()
            res, status_code = query_api(question)
            duration = time.time() - start_time

            if status_code == 200:
                # Extract raw answer (could be dict/list or markdown JSON code block)
                if isinstance(res, list) and len(res) > 0 and 'content' in res[0]:
                    raw = res[0]['content']
                else:
                    ans_field = res.get('answer')
                    if isinstance(ans_field, dict):
                        raw = ans_field.get('answer', '')
                    elif isinstance(ans_field, str):
                        raw = ans_field
                    else:
                        raw = ''

                # Clean citations
                raw = re.sub(r"\s*<(\d+)>", "", raw).strip()

                # If wrapped in a JSON code block, extract inner JSON
                md_match = re.search(r"```json\s*(.*?)```", raw, re.S)
                if md_match:
                    json_part = md_match.group(1)
                    try:
                        parsed = json.loads(json_part)
                        if isinstance(parsed, list) and parsed:
                            item = parsed[0]
                            # prefer 'text' field, fallback to 'content'
                            answer = item.get('text') or item.get('content') or ''
                        else:
                            answer = raw
                    except json.JSONDecodeError:
                        answer = raw
                else:
                    answer = raw

                answer = answer.strip()
            else:
                answer = ''

            # Build a sample for evaluation
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                reference=gold_answer
            )
            # Correctness (binary)
            try:
                correctness_score = asyncio.run(correctness_evaluator.single_turn_ascore(sample))
            except RagasOutputParserException as e:
                print(f"⚠️ Warning: Correctness evaluation failed for question {i+1}: {e}")
                correctness_score = None

            # BLEU score
            try:
                bleu_score = asyncio.run(bleu_evaluator.single_turn_ascore(sample))
            except Exception as e:
                print(f"⚠️ Warning: BLEU evaluation failed for question {i+1}: {e}")
                bleu_score = None

            # F1 score (Rouge-L F1)
            try:
                f1_score = asyncio.run(f1_evaluator.single_turn_ascore(sample))
            except Exception as e:
                print(f"⚠️ Warning: F1 evaluation failed for question {i+1}: {e}")
                f1_score = None

            writer.writerow([
                question,
                answer,
                gold_answer,
                correctness_score,
                f1_score,
                bleu_score
            ])

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    run_tests()

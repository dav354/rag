#!/usr/bin/env python3
import csv
import os
import asyncio
from datetime import datetime

# Konfiguration: Pfade hartcodiert
INPUT_CSV_PATH = '/Users/lelange/Uni/askTHWS/testing/llm_eval_pipeline/answered/answered_20250713_27B.csv'
OUTPUT_DIR = '/Users/lelange/Uni/askTHWS/testing/aspect_critic_test/eval_results'

from langchain.llms import Ollama
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic, BleuScore, RougeScore
from ragas.dataset_schema import SingleTurnSample
from ragas.exceptions import RagasOutputParserException

def evaluate_sample(sample, correctness_evaluator, bleu_evaluator, rouge_evaluator):
    try:
        correctness = asyncio.run(correctness_evaluator.single_turn_ascore(sample))
    except Exception as e:
        print(f"⚠️ Correctness evaluation failed: {e}")
        correctness = None

    try:
        bleu = asyncio.run(bleu_evaluator.single_turn_ascore(sample))
    except Exception as e:
        print(f"⚠️ BLEU evaluation failed: {e}")
        bleu = None

    try:
        rouge = asyncio.run(rouge_evaluator.single_turn_ascore(sample))
    except Exception as e:
        print(f"⚠️ ROUGE evaluation failed: {e}")
        rouge = None

    return correctness, bleu, rouge

def main():
    input_path = INPUT_CSV_PATH
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"results_27B_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")

    # LLM-Setup
    ollama = Ollama(base_url="http://127.0.0.1:11434", model="gemma3:27B")
    wrapped_llm = LangchainLLMWrapper(ollama)
    correctness_evaluator = AspectCritic(
        name="correctness",
        definition="Ist die generierte Antwort korrekt im Vergleich zur Gold-Antwort?",
        llm=wrapped_llm
    )
    bleu_evaluator = BleuScore()
    rouge_evaluator = RougeScore(rouge_type="rougeL", mode="fmeasure")

    with open(input_path, newline="", encoding="utf-8") as infile, \
         open(output_path, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = ["id", "question", "bot_answer", "ref_answer", "correctness", "BLEU", "ROUGE"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            print(f"Frage {row['id']}: {row['question']}")

            sample = SingleTurnSample(
                user_input=row["question"],
                response=row["bot_answer"],
                reference=row["ref_answer"]
            )
            correctness, bleu, rouge = evaluate_sample(
                sample, correctness_evaluator, bleu_evaluator, rouge_evaluator
            )
            writer.writerow({
                "id": row["id"],
                "question": row["question"],
                "bot_answer": row["bot_answer"],
                "ref_answer": row["ref_answer"],
                "correctness": correctness,
                "BLEU": bleu,
                "ROUGE": rouge
            })
            print(f"Ausgewertet id={row['id']} → correctness={correctness}, BLEU={bleu}, ROUGE={rouge}")

    print(f"Ergebnisse gespeichert in: {output_path}")

if __name__ == "__main__":
    main()
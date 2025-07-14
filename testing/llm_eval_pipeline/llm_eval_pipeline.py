#!/usr/bin/env python3
# File: evaluate_facts.py

import argparse
from datetime import datetime
import pandas as pd
import requests

# ----------------
# CONFIG (anpassen)
ENDPOINT = "http://localhost:11434/v1/chat/completions"
MODEL    = "gemma3:4B"
CSV_FILE = "/Users/lelange/Uni/askTHWS/testing/llm_eval_pipeline/answered/answered_20250711_164137.csv"
# ----------------

PROMPT_TEMPLATE = """
Du bist ein neutraler Faktenprüfer.
Deine Aufgabe ist es, zwei Antworten zu vergleichen:

1. Die von einem Nutzer gestellte Frage: [FRAGE]
2. Die faktisch korrekte Antwort: [FAKTEN_ANTWORT]
3. Die Antwort des Chatbots: [CHATBOT_ANTWORT]

Überprüfe ausschließlich, ob die Fakten in der Chatbot-Antwort mit der faktisch korrekten Antwort übereinstimmen.
Ignoriere Stil, Ton, Grammatik oder andere Aspekte.
Gib eine Zahl zwischen 0 und 1 zurück, die angibt, wie gut die Fakten übereinstimmen.
0 bedeutet „keine Übereinstimmung“, 1 bedeutet „vollständige Übereinstimmung“.
Gib **nur die Zahl** aus, ohne Kommentar oder Erklärung.

[FRAGE]:
{question}

[FAKTEN_ANTWORT]:
{reference}

[CHATBOT_ANTWORT]:
{bot_answer}
""".strip()

def eval_row(row):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_TEMPLATE.format(
                question=row["question"],
                reference=row["ref_answer"],
                bot_answer=row["bot_answer"]
            )}
        ]
    }
    resp = requests.post(ENDPOINT, json=payload)
    resp.raise_for_status()
    txt = resp.json()["choices"][0]["message"]["content"].strip()
    try:
        return float(txt)
    except ValueError:
        raise RuntimeError(f"Ungültige Modellantwort: '{txt}'")

def main():
    df = pd.read_csv(CSV_FILE)
    if not {"id","question","bot_answer","ref_answer"}.issubset(df.columns):
        raise ValueError("Fehlende Spalten in der Eingabe-CSV. Erwartet: id, question, bot_answer, reference_answer")
    scores = []
    for idx, row in df.iterrows():
        score = eval_row(row)
        print(f"Row {idx} (id={row['id']}): Score = {score}")
        scores.append(score)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df["score"] = scores
    df.to_csv(f"/Users/lelange/Uni/askTHWS/testing/llm_eval_pipeline/scored/scored_{timestamp}.csv", index=False)
    print("Ergebnis gespeichert")

if __name__ == "__main__":
    main()
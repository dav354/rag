import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
import os


API_URL = "http://localhost:8000/ask"
METADATA_URL = "http://localhost:8000/metadata"
CSV_FILE = "/Users/lelange/Uni/askTHWS/testing/evaluation/fiw_fragenkatalog.csv"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# NLI (RoBERTa-large-MNLI) auf MPS laden und Pipeline bauen
MODEL_NAME = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16
            ).to(device)

nli = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1
)



def load_df():
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()

    df_eval = (
        df[["Id", "User_Input", "Response"]]
        .rename(columns={
            "Id": "id",
            "User_Input": "question",
            "Response": "reference_answer"
        })
    )

    print(df_eval.head(10))

    return df_eval

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
    

def semantic_score(pred: str, ref: str) -> float:
    emb_pred = embedder.encode(pred, convert_to_tensor=True)
    emb_ref  = embedder.encode(ref,  convert_to_tensor=True)
    # util.cos_sim gibt einen Tensor mit einem Wert zwischen –1 und 1 zurück
    return util.cos_sim(emb_pred, emb_ref).item()

def is_entailment(pred: str, ref: str, threshold: float = 0.9) -> bool:
    # Format: Premise </s></s> Hypothesis
    seq = f"{ref} </s></s> {pred}"
    out = nli(seq)[0]
    return out["label"] == "ENTAILMENT" and out["score"] >= threshold
    

def main():
    df = load_df()
    # Nur die ersten 5 Fragen verwenden
    df = df.head(5)
    THRESHOLD = 0.7
    results = []

    for _, row in df.iterrows():
        print(f"Bearbeite Frage {row['id']}: {row['question']}")
        question = row["question"]
        reference = row["reference_answer"]

        api_response, status_code = query_api(question)
        if status_code == 200:
            # Extrahiere den Text aus der verschachtelten Antwortstruktur
            predicted = api_response.get("answer", {}).get("answer", "")
        else:
            predicted = ""
        
        score = semantic_score(predicted, reference)
        entail  = is_entailment(predicted, reference)

        is_correct = score >= THRESHOLD or entail
        print(f"  -> sem_sim: {score:.3f}, entailment: {entail}, correct: {is_correct}")

        results.append({
            "id": row["id"],
            "question": question,
            "predicted": predicted,
            "reference": reference,
            "sem_sim": score,
            "entailment": entail,
            "correct": is_correct
        })
    
    report_df = pd.DataFrame(results)

    # Ergebnisse mit Zeitstempel im Zielordner speichern
    output_dir = "/Users/lelange/Uni/askTHWS/testing/evaluation/test_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.csv"
    report_path = os.path.join(output_dir, filename)
    report_df.to_csv(report_path, index=False)
    print(f"Saved results to {report_path}")


if __name__ == "__main__":
    main()
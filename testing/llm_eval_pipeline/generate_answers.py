import os
import re
import json
import requests
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:8000/ask"
METADATA_URL = "http://localhost:8000/metadata"
CSV_FILE = "/Users/lelange/Uni/askTHWS/testing/llm_eval_pipeline/fiw_fragenkatalog.csv"

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
    return df_eval

def query_api(question):
    """
    Queries the API and handles both successful and error responses.
    Returns only the answer text and the HTTP status code.
    """
    try:
        response = requests.post(API_URL, json={"query": question}, timeout=10000)
        if response.status_code == 200:
            data = response.json()
            return data.get("answer", ""), response.status_code
        else:
            return "", response.status_code
    except requests.exceptions.RequestException as e:
        return "", 503


def main():
    df = load_df()
    results = []

    for _, row in df.iterrows():
        print(f"Bearbeite Frage {row['id']}: {row['question']}")
        question = row["question"]
        reference = row["reference_answer"]

        res, status_code = query_api(question)
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

        print(answer)

        results.append({
            "id": row["id"],
            "question": question,
            "bot_answer": answer,
            "ref_answer": reference
        })

    df_result = pd.DataFrame(results)

    output_dir = "/Users/lelange/Uni/askTHWS/testing/llm_eval_pipeline/answered"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"answered_{timestamp}.csv"
    result_path = os.path.join(output_dir, filename)

    df_result.to_csv(result_path, index=False)

if __name__ == "__main__":
    main()
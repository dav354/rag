import requests
import re
import time
import os
import json  # This import is essential

from datetime import datetime
from typing import List, Dict, Any, Tuple, Union

MARKDOWN_FILE = "../docs/tests/fragen.md"
API_URL = "http://localhost:8000/ask"
METADATA_URL = "http://localhost:8000/metadata"


def extract_questions(md_file):
    """Extracts questions from a markdown file."""
    if not os.path.exists(md_file):
        print(f"ERROR: Markdown file not found at {md_file}")
        return []
    with open(md_file, "r", encoding="utf-8") as file:
        content = file.read()
    questions = re.findall(r"-\s+(.*?)\?", content)
    return [q.strip() + "?" for q in questions]


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


def write_header(f, metadata):
    """Writes the header for the results file."""
    commit_hash = metadata.get("git_commit", "unknown")
    embedding_model = metadata.get("embedding_model", "unknown")
    llm_model = metadata.get("llm_model", "unknown")
    device = metadata.get("device", "unknown")
    retrieval_mode = metadata.get("retrieval_mode", "unknown")

    f.write(f"Commit: {commit_hash}\n\n")
    f.write("# Automatischer Testlauf\n\n")
    f.write(f"- **Embedding-Modell**: `{embedding_model}`\n")
    f.write(f"- **LLM-Modell**: `{llm_model}`\n")
    f.write(f"- **Device**: `{device}`\n\n")
    f.write(f"- **Retrieval-Mode**: `{retrieval_mode}`\n\n")
    f.write("> Antworten aus dem ersten Lauf, keine manuelle Anpassung.\n\n")
    f.write("---\n")


def save_result(
        f, question: str, duration: float, api_response: Dict[str, Any], status_code: int
):
    """Saves a single test result or an error to the file."""
    f.write(f"### Frage: {question}\n\n")
    f.write(f"**Status**: `{status_code}` | **Dauer**: `{duration:.2f}s`\n\n")

    if status_code == 200:
        # The main 'answer' key from the API response contains the nested dictionary
        nested_data = api_response.get("answer", {})

        # Get the actual answer text
        answer_text = nested_data.get("answer", "No answer text found.")

        # --- THIS IS THE FIX ---
        # 1. Get the sources, which is a LIST of dictionaries, not a string.
        sources_list = nested_data.get("sources", [])

        # 2. Convert the list of source dictionaries into a formatted JSON string.
        #    - indent=2 makes it readable.
        #    - ensure_ascii=False correctly handles special characters like umlauts.
        sources_as_json_string = json.dumps(sources_list, indent=2, ensure_ascii=False)
        # --- END OF FIX ---

        f.write(f"**Antwort:**\n```\n{answer_text.strip()}\n```\n\n")
        f.write("#### 🔗 Quellen (nach Reranking):\n")
        f.write("```json\n") # Add json language tag for nice formatting
        f.write(sources_as_json_string) # Write the correctly formatted JSON string
        f.write("\n```\n")

    else:
        # Handle error case
        error_detail = api_response.get("detail", "An unknown error occurred on the server side.")
        # Pretty print the error detail if it's a dict/list
        if isinstance(error_detail, (dict, list)):
            error_detail_str = json.dumps(error_detail, indent=2, ensure_ascii=False)
        else:
            error_detail_str = str(error_detail)
        f.write(f"**Fehler:**\n```\n{error_detail_str}\n```\n")

    f.write("\n---\n\n")
    f.flush()


def run_tests():
    """Main testing routine."""
    metadata = get_metadata()
    if not metadata:
        print("Aborting tests due to failed metadata fetch.")
        return

    questions = extract_questions(MARKDOWN_FILE)
    if not questions:
        print("Aborting tests because no questions were found.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = os.path.join(
        "test_results", f"test_results_{timestamp}.md"
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    with open(result_file, "w", encoding="utf-8") as f:
        write_header(f, metadata)

        for i, q in enumerate(questions):
            print(f"🔍 Frage {i + 1}/{len(questions)}: {q}")
            start_time = time.time()
            res, status_code = query_api(q)
            duration = round(time.time() - start_time, 2)
            save_result(f, q, duration, res, status_code)

    print(f"\n✅ Test abgeschlossen. Ergebnisse gespeichert in: {result_file}")


if __name__ == "__main__":
    run_tests()
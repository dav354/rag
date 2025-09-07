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
from ragas.metrics import FactualCorrectness, SemanticSimilarity
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.exceptions import RagasOutputParserException

# Try to import Ragas HF embeddings; fall back to a local adapter using sentence-transformers
try:
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmb
except ImportError:
    RagasHFEmb = None
    from typing import List
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as _e:
        raise ImportError("sentence-transformers is required for SemanticSimilarity when ragas.embeddings.HuggingFaceEmbeddings is unavailable. Install via `pip install sentence-transformers`.") from _e

    class SBERTEmbeddings:
        def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
            self._model = SentenceTransformer(model, device=device)
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self._model.encode(texts, normalize_embeddings=True).tolist()
        def embed_query(self, text: str) -> List[float]:
            return self._model.encode([text], normalize_embeddings=True)[0].tolist()
        async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
            return self.embed_documents(texts)
        async def aembed_query(self, text: str) -> List[float]:
            return self.embed_query(text)

        async def embed_text(self, text: str) -> list[float]:
            # some ragas versions `await` embed_text directly
            return self.embed_query(text)

        async def aembed_text(self, text: str) -> list[float]:
            # async counterpart (kept for compatibility)
            return self.embed_query(text)

from datetime import datetime


API_URL = "http://localhost:8000/ask"
METADATA_URL = "http://localhost:8000/metadata"
CSV_FILE = "/Users/lelange/Uni/askTHWS/testing/aspect_critic_test/fragenkatalog_askTHWS_fiw_neu.csv"


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
    cases = all_cases[:100]
    if not cases:
        print("No test cases found.")
        return

    os.makedirs("test_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_csv = os.path.join("test_results", f"results_fiw_v4_gemini_mix_DC7_KG4.csv")

    # Initialize Gemini LLM client (requires: pip install langchain-google-genai; env var GOOGLE_API_KEY)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Please export your Google API key to use the Gemini models.")
    gemini_client = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=google_api_key,  # pass API key explicitly to avoid ADC
    )
    wrapped_llm = LangchainLLMWrapper(gemini_client)
    # Initialize embeddings for similarity metrics (robust against ragas version differences)
    if RagasHFEmb is not None:
        hf_embeddings = RagasHFEmb(model="sentence-transformers/all-MiniLM-L6-v2")
    else:
        hf_embeddings = SBERTEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    correctness_evaluator = AspectCritic(
        name="correctness",
        definition="Ist die generierte Antwort korrekt im Vergleich zur Gold-Antwort?",
        llm=wrapped_llm
    )
    bleu_evaluator = BleuScore()
    f1_evaluator = RougeScore(rouge_type="rougeL", mode="fmeasure")
    factual_correctness_evaluator = FactualCorrectness(llm=wrapped_llm)
    try:
        semantic_similarity_evaluator = SemanticSimilarity(embeddings=hf_embeddings)
    except TypeError:
        # fallback for older ragas versions that expect `embedding` (singular)
        semantic_similarity_evaluator = SemanticSimilarity(embedding=hf_embeddings)

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question', 'bot_answer', 'gold_answer', 'genauigkeit', 'F1', 'BLEU', 'factual_correctness', 'semantic_similarity', 'duration_seconds'])
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
                duration_seconds = res.get("duration_seconds", None)

                # Extract retrieved contexts from API response for Faithfulness
                retrieved_contexts = None
                try:
                    ans_obj = res.get("answer", {})
                    srcs = ans_obj.get("sources", [])
                    if isinstance(srcs, list) and srcs:
                        tmp = []
                        for s in srcs:
                            # Prefer 'content', fallback to 'description'
                            txt = (s.get("content") or s.get("description") or "").strip()
                            if txt:
                                tmp.append(txt)
                        if tmp:
                            retrieved_contexts = tmp
                except Exception:
                    retrieved_contexts = None
            else:
                answer = ''
                duration_seconds = None
                retrieved_contexts = None

            # Build a sample for evaluation
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                reference=gold_answer
            )
            # Attach contexts for Faithfulness: prefer retrieved contexts from API, otherwise use gold answer as fallback
            _contexts = retrieved_contexts if retrieved_contexts else [gold_answer]
            try:
                # Newer ragas versions
                sample.retrieved_contexts = _contexts
            except Exception:
                try:
                    # Older ragas versions
                    sample.contexts = _contexts
                except Exception:
                    pass

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

            # Factual Correctness
            try:
                factual_correctness_score = asyncio.run(factual_correctness_evaluator.single_turn_ascore(sample))
            except Exception as e:
                print(f"⚠️ Warning: Factual Correctness evaluation failed for question {i+1}: {e}")
                factual_correctness_score = None

            # Semantic Similarity
            try:
                semantic_similarity_score = asyncio.run(semantic_similarity_evaluator.single_turn_ascore(sample))
            except Exception as e:
                print(f"⚠️ Warning: Semantic Similarity evaluation failed for question {i+1}: {e}")
                semantic_similarity_score = None

            writer.writerow([
                question,
                answer,
                gold_answer,
                correctness_score,
                f1_score,
                bleu_score,
                factual_correctness_score,
                semantic_similarity_score,
                duration_seconds
            ])

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    run_tests()

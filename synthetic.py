import requests
import os
import time
from qdrant_client import PointStruct

SYVAI_TOKEN = os.getenv("SYVAI_TOKEN")
SYV_EMBED_URL = "https://api.syv.ai/v1/embeddings"
SYV_COMPLETION_URL = "https://api.syv.ai/v1/completions"

def embed_text(text: str) -> list:
    """
    Returnerer en embedding-vektor for en enkelt tekststreng via syv.ai proxy embedding endpoint
    """

    if not text or not text.strip():
        return []

    headers = {
        "Authorization": f"Bearer {SYVAI_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "input": [text],
        "model": "mistral/mistral-embed",
        "encoding_format": "float"
    }

    response = requests.post(SYV_EMBED_URL, headers=headers, json=data)

    if response.status_code != 200:
        print(f"⚠️ Fejl ({response.status_code}): {response.text[:200]}")
        response.raise_for_status()
    
    result = response.json()

    return result["data"][0]["embedding"]

def embed_batch(texts: list[str], batch_size: int = 32, max_retries: int = 5) -> list[list[float]]:
    """
    Batch embedder en liste af tekststykker.
    Returnerer en liste af embeddings i samme rækkefølge som input
    """

    if not texts:
        return []
    
    all_vectors = []

    # Opdeler tekstlisten i mindre batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {SYVAI_TOKEN}",
                    "Content-type": "application/json"
                }

                data = {
                    "input": batch,
                    "model": "mistral/mistral-embed",
                    "encoding_format": "float"
                }

                response = requests.post(
                    SYV_EMBED_URL,
                    headers=headers,
                    json=data,
                    timeout=30
                )

                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                result = response.json()

                for item in result["data"]:
                    all_vectors.append(item["embedding"])
                
                break
            
            except Exception as e:
                wait_time = 2 ** attempt
                print(f"Embed batch fejl (forsøg {attempt+1}/{max_retries}): {e}")
                print(f"Venter {wait_time} sekunder...")
                time.sleep(wait_time)

                if attempt == max_retries - 1:
                    raise RuntimeError("Embed_batch mislykkedes permanent.")

def generate_synthetic_questions(text: str, n_questions: int = 3) -> list:
    """
    Genererer syntetiske borger-spørgsmål baseret på chunkens tekst.
    Bruges til at forbedre retrieval
    """

    prompt = f"""
    Du skal omskrive følgende tekst til {n_questions} forskellige borger-spørgsmål.
    Spørgsmålene skal være korte, klare og formuleret i hverdagssprog.
    Eksempeltyper: hvad, hvorfor, hvordan, hvornår, hvem.

    Tekst:
    {text}

    Svar:
    """

    headers = {
        "Authorization": f"Bearer {SYVAI_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "danskgpt-v2.1",
        "prompt": prompt,
        "max_tokens": 250
    }

    response = requests.post(
        SYV_COMPLETION_URL,
        headers=headers,
        json=data
    )
    response.raise_for_status()

    raw_output = response.json()["choices"][0]["text"].strip()

    questions = [
        q.strip("-• ").strip() 
        for q in raw_output.split("\n")
        if len(q.strip()) > 3
        ]

    return questions[:n_questions]



def embed_synthetics(questions: list[str]) -> list[list[float]]:
    """
    Batch embedder en liste af syntetiske spørgsmål
    """
    return embed_batch(questions)


def build_synthetic_record(question: str, parent_record: dict, embedding: list):

    synthetic_id = (parent_record["chunk_id"]
                    + "_synthetic_"
                    + question[:20].replace(" ", "_")
                )

    return {
        "chunk_id": synthetic_id,
        "parent_chunk_id": parent_record["chunk_id"],
        "document_id": parent_record["document_id"],
        "committee": parent_record["committee"],
        "meeting_date": parent_record["meeting_date"],
        "meeting_year": parent_record.get("meeting_year"),
        "section_type": parent_record["section_type"],
        "section_category": parent_record["section_category"],
        "synthetic_question": question,
        "text": question,
    }


def upsert_synthetics(client, collection_name: str, chunk_record: dict, n_questions: int = 3):
    """
    Genererer syntetiske spørgsmål baseret på chunk_record["text],
    embedder dem, opbygger synthetic records og upserter dem i qdrant
    """

    text = chunk_record.get("text", "")
    if not text.strip():
        return 0
    
    # Genererer syntetiske spørgsmål
    questions = generate_synthetic_questions(text, n_questions=n_questions)
    if not questions:
        return 0
    
    # Embed synthetic points
    vectors = embed_synthetics(questions)

    # Upsert synthetic points
    synthetic_points = []

    for q, vec in zip(questions, vectors):
        synthetic_record = build_synthetic_record(q, chunk_record, vec)

        synthetic_points.append(PointStruct(
            id=synthetic_record["chunk_id"],
            vector=vec,
            payload=synthetic_record
        ))

    # Batch-upsert for effektivitet
    client.upsert(
        collection_name=collection_name,
        points=synthetic_points
    )

    return len(synthetic_points)
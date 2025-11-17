import os
import json
from pathlib import Path
from dotenv import load_dotenv
import re
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from chunk_data import normalize_committee

import requests

load_dotenv()

# Qdrant connection
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Syv.ai embedding setup
SYVAI_TOKEN = os.getenv("SYVAI_TOKEN")
if not SYVAI_TOKEN:
    raise ValueError("SYVAI_TOKEN mangler i .env")

SYV_EMBED_URL = "https://api.syv.ai/v1/embeddings"

COMMITTEE_LIST = [
    "Byrådet",
    "Plan- og Teknikudvalget",
    "Økonomiudvalget",
    "Social- og Sundhedsudvalget",
    "Børne- og Ungeudvalget",
    "Kultur- og Fritidsudvalget",
    "Klima- og Miljøudvalget",
    "Erhvervs- og Beskæftigelsesudvalget"
]


def extract_years(query: str):
    years = re.findall(r"\b(20[0-9]{2})\b", query)
    if not years:
        return None
    return max (int(y) for y in years)

def extract_section_intent(query: str):
    q = query.lower()

    if "besluttet" in q or "beslutning" in q:
        return "decision"
    if "baggrund" in q or "hvorfor" in q or "begrundelse" in q:
        return "background"
    if "resumé" in q or "resume" in q:
        return "resume"
    
    return None

def extract_committee(query: str):
    query_emb = embed_query(query)

    best_committee = None
    best_score = 0

    for committee in COMMITTEE_LIST:
        comm_emb = embed_query(committee)
        score = cosine_similarity(query_emb, comm_emb)

        if score > best_score and score > 0.75:
            best_score = score
            best_committee = committee

    return best_committee

def extract_query_filters (query: str):
    committee = extract_committee(query)
    year = extract_years(query)
    section = extract_section_intent(query)

    filters = {}

    if committee:
        filters["committee_normalized"] = normalize_committee(committee)

    if year:
        filters["meeting_year"] = year

    if section:
        filters["section_type"] = section
        filters["section_category"] = "core"

    return filters

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    return dot / (norm_a * norm_b)

def vector_to_score(distance: float) -> float:
    return 1 - distance

def bm25_search(query: str, collection_name: str, top_k: int = 10):
    return client.search(
        collection_name=collection_name,
        query=query,
        limit=top_k
    )

def hybrid_search(
        query: str,
        collection_name: str = "kommune-rag",
        top_k: int = 10,
        alpha: float = 0.7, #Vektor vægt
        beta: float = 0.3,  # keyword vægt
        gamma: float = 0.2,  # Recency score vægt
        filters: dict = None
):
    """
    Kombinerer vektor-søgning + bm25 søgning og reranker resultater - hybrid search
    """

    # 1/2 vektor søgningen
    vector_results = vector_search(
        query_text=query,
        collection_name=collection_name,
        top_k=top_k,
        filters=filters
    )

    # 1/2 Keyword søgning
    bm25_results = bm25_search(
        query=query,
        collection_name=collection_name,
        top_k=top_k
    )

    combined = {}

    # 2/2 vektor søgning
    for r in vector_results:
        chunk_id = r.payload.get("chunk_id")
        score = vector_to_score(r.score)
        combined[chunk_id] = {"payload": r.payload, "vector_score": score, "bm25_score": 0}

    # 2/2 BM25 søgning
    for r in bm25_results:
        chunk_id = r.payload.get("chunk_id")
        score = r.score

        if chunk_id in combined:
            combined[chunk_id]["bm25_score"] = score
        else:
            combined[chunk_id] = {"payload": r.payload, "vector_score": 0, "bm25_score": score}
    
    ranked = []
    for chunk_id, scores in combined.items():

        v = scores.get("vector_score", 0.0)
        bm_raw = scores.get("bm25_score", 0.0)

        bm = min(1.0, bm_raw / 10.0)

        chunk_year = scores["payload"].get("meeting_year")

        if chunk_year is None:
            r_score = 0.0
        else:
            r_score = recency_score(chunk_year)

        hybrid_score = (
            v * alpha
            + bm * beta
            + r_score * gamma
        )
        
        ranked.append((hybrid_score, scores["payload"]))

    ranked.sort(key=lambda x: x[0], reverse=True)

    return ranked[:top_k]

def embed_query(text: str) -> list:
    """
    Genererer embeddings for et bruger-spørgsmål via syv.ai proxyen
    returner en liste af floats (vektor)
    """

    headers = {
        "Authorization": f"Bearer {SYVAI_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "input": [text],                    # API forventer liste
        "model": "mistral/mistral-embed",   # samme model som indexing
        "encoding_format": "float"
    }

    response = requests.post(SYV_EMBED_URL, headers=headers, json=data)

    if response.status_code != 200:
        print(f"⚠️ Fejl ({response.status_code}): {response.text[:200]}")
        response.raise_for_status()
    
    result = response.json()
    return result["data"][0]["embedding"]

def vector_search(
        query_text: str,
        collection_name: str = "kommune-rag",
        top_k: int = 5,
        filters: dict = None
):
    """
    Semantisk søgning (vektor match) med valgfri metadata-filter.
    filters = {"committee": "Plan- og Teknikudvalget", "meeting_year": 2023}
    """

    query_vector = embed_query(query_text)

    qdrant_filter = None

    if filters:
        conditions = []

        for key, value in filters.items():
            if value is None:
                continue
        
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(any=value)
                    )
                )
                qdrant_filter = Filter(must=conditions)
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
                qdrant_filter = None

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter
    )

    return results

def get_chunk_neighbors(payload: dict, window: int = 1, collection_name: str = "kommune-rag"):
    """
    Finder 'naboerne' til chunks (n-1 og n+1) baseret på agenda_index og document_id.
    Returnerer en liste af payloads
    """

    doc_id = payload.get("document_id")
    idx = payload.get("agenda_index")

    neighbors = [payload]

    indices_to_fetch = list (range(idx - window, idx + window + 1))

    for neighbor_idx in indices_to_fetch:
        if neighbor_idx == idx:
            continue
        
        q_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=doc_id)
                ),
                FieldCondition(
                    key="agenda_index",
                    match=MatchValue(value=neighbor_idx)
                ),
            ]
        )

        result = client.search(
            collection_name=collection_name,
            query_vector=[0] * 1024, #dummy vektor da vi kun bruger filter
            limit=1,
            query_filter=q_filter
        )

        if result:
            neighbors.append(result[0].payload)
    
    return neighbors

def apply_windowing(hybrid_results, window=1):
    """
    Tilføjer nabo-chunks til hvert hybrid resultat.
    Returnerer en liste af payload grupper
    """

    windowed = []

    for score, payload in hybrid_results:
        group = get_chunk_neighbors(payload, window=window)
        windowed.append({
            "score": score,
            "chunks": group
        })
    
    return windowed

def recency_score(chunk_year: int, min_year: int = 2018, max_year: int= 2025):
    """
    Normaliseret recency-score mellem 0-1.
    Nyere dokumenter scorer højere
    """
    if chunk_year is None:
        return 0
    if chunk_year < min_year:
        chunk_year = min_year
    if chunk_year > max_year:
        chunk_year = max_year
    
    return 1 - ((max_year - chunk_year) / (max_year - min_year))

if __name__ == "__main__":
    query = "Hvad er planerne for cykelstier i Greve?"
    results = vector_search(query)

    for r in results:
        print("\nScore:", r.score)
        print("Tekst:", r.payload.get("text")[:250], "...")
        print("Dato:", r.payload.get("meeting_date"))
        print("Udvalg:", r.payload.get("committee"))

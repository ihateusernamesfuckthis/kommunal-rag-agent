import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from query_engine import (
    vector_search,
    hybrid_search,
    apply_windowing,
    extract_query_filters
)

load_dotenv()

SYVAI_TOKEN = os.getenv("SYVAI_TOKEN")
SYV_EMBED_URL = "https://api.syv.ai/v1/embeddings"
SYV_COMPLETION_URL = "https://api.syv.ai/v1/completions"

if not SYVAI_TOKEN:
    raise ValueError("SYVAI_TOKEN mangler i .env")

if not SYV_COMPLETION_URL:
    raise ValueError("SYV_COMPLETION_URL mangler i .env")

if not SYV_EMBED_URL:
    raise ValueError("SYV_EMBED_URL manger i .env")



def embed_query(text: str) -> List[float]:
    """
    Embedder en tekststring (brugerens spørgsmål)
    via embedding modellen fra syv platformen
    """

    headers = {
        "Authorization": f"Bearer {SYVAI_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "input": [text],
        "model": "mistral/mistral-embed",
        "encoding_format": "float"
    }

    response = requests.post(
        SYV_EMBED_URL,
        headers=headers,
        json=data
    )

    response.raise_for_status()

    result = response.json()
    return result["data"][0]["embedding"]

def build_retrieval_pipeline(query_text: str) -> List[Dict[str, Any]]:
    """
    Samler hele retrieval fasen for et bruger spørgsmål:
    - metadata-filtre (committee, årstal, section)
    - hybrid search (vektor + bm25 + recency)
    - windowing -> n-1, n, n+1 omkring valgte "vundende" chunks (de chunks med højest score)

    Returnerer en liste af "windowed chunks":
    [
        {
            "score": float,
            "chunks": [payload1, payload2, payload3]
        },
        ...
    ]
    """

    # 1) udtrækker metadata filtre
    filters = extract_query_filters(query_text)

    # 2) hybrid search -> semantik + keywords(bm25) + recency
    hybrid_results = hybrid_search(
        query=query_text,
        filters=filters,
        top_k=8
    )

    # 3) Windowing -> henter nabo-chunks
    windowed_results = apply_windowing(hybrid_results, window=1)

    return windowed_results


def build_context(windowed_results: List[Dict[str, Any]]) -> str:
    """
    Konverterer windowed retrieval-resultater til en struktureret tekstblok,
    som skal sendes til LLM'en i prompten.

    Output bliver en stor string med flere kontekstafsnit
    """

    blocks = []

    for idx, window in enumerate(windowed_results, start=1):
        score = window.get("score", 0)
        chunks = window.get("chunks", [])

        # Header for window
        block_lines = [
            f"### RESULTAT {idx} (score: {round(score, 3)})",
            ""
        ]

        # Tilføjer alle chunks i vinduet (n-1, n, n+1)
        for c in chunks:
            committee = c.get("committee", "Ukendt udvalg")
            date = c.get("meeting_date", "ukendt dato")
            section_type = c.get("section_type", "ukendt sektion")
            source = c.get("source_path", "ukendt kilde")
            text = c.get("text", "").strip()

            block_lines.append(f"[Chunk • {section_type} • {committee} • {date}]")
            block_lines.append(text)
            block_lines.append(f"(Kilde: {source})")
            block_lines.append("")  # tilføjer tom linje mellem chunks
        
        blocks.append("\n".join(block_lines))

        full_context = "\n\n".join(blocks)
        return full_context
    
def build_prompt(query: str, context: str) -> str:
    """
    Bygger et LLM optimeret prompt til danskgpt-v2.1

    regler:
    - Giv et præcist, faktuelt svar baseret på kontekst
    - Hvis konteksten ikke indeholder svaret - sig det
    - inkluder kildehenviser i bunden
    """

    def build_prompt(query: str, context: str) -> str:
        """
        Bygger en klar, robust og LLM-optimeret prompt til danskgpt-v2.1.
        
        Regler:
        - giv et præcist, faktuelt svar baseret på kontekst
        - hvis konteksten ikke indeholder svaret → sig det
        - inkludér kildehenvisninger i bunden
        """

        prompt = f"""

        Du er en kommunal informationsassistent for borgere. Du skal svare så præcist som muligt 
        på baggrund af de officielle kommunale dokumenter, som du har fået i konteksten.

        VIGTIGE REGLER FOR DIT SVAR:
        1. Brug KUN information fra konteksten herunder.
        2. Hvis konteksten ikke indeholder svaret, så skriv: 
        "Der findes ingen tilgængelig information om dette i de kommunale dokumenter."
        3. Giv et kort, klart og faktuelt svar.
        4. Til sidst skal du inkludere kildehenvisninger i følgende format:
        - “Kilde: <committee> · <meeting_date> · <source_path>”
        5. Du må aldrig opfinde fakta eller gætte.

        -------------------------------------
        BRUGERENS SPØRGSMÅL:
        {query}

        -------------------------------------
        RELEVANT KONTEKST FRA KOMMUNALE DOKUMENTER:
        {context}

        -------------------------------------
        SVAR:
        """

        return prompt.strip()

def call_llm(prompt: str, max_tokens: int = 500) -> str:
    """
    Sender prompten til danskgpt-v2.1 via syv platformen og returnerer tekst svar
    """

    url = SYV_COMPLETION_URL

    headers = {
        "Authorization": f"Bearer {SYVAI_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "danskgpt-v2.1",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.9
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["text"].strip()
    
    except requests.exceptions.Timeout:
        return "Fejl: LLM forespørgslen tog for lang tid og timede out"
    
    except requests.exceptions.RequestException as e:
        return f"Fejl i forbindelse med LLM-kaldet: {e}"
    
def format_answer(raw_answer: str, windowed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ekstraherer kildehenvisninger fra windowed_results.
    Kombiner rå LLM-svar med en struktureret liste af kilder

    output:

    {
        "answer": <tekst>,
        "sources": [
            { committee, meeting_date, section_type, source_path },
            ...
        ]
    }
    """

    sources = []

    for window in windowed_results:
        for chunk in window.get("chunks", []):
            sources.append({
                "committee": chunk.get("committee", "ukendt udvalg"),
                "meeting_date": chunk.get("meeting_date", "ukendt dato"),
                "section_type": chunk.get("section_type", "ukendt sektion"),
                "section_title": chunk.get("section_title", "ukendt titel"),
                "source_path": chunk.get("source_path", "")
            })
        
        # Fjerner dubletter ved at tjekke om source_path er unik
        unique_sources = []
        seen = set()

        for s in sources:
            key = (s["committee"], s["meeting_data"], s["source_path"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        
        return {
            "answer": raw_answer.strip(),
            "sources": unique_sources
        }
    
def answer_query(query_text: str) -> Dict[str, Any]:
    """
    Fuld RAG pipeline:
    1. Retrieval
    2. Context Assembly
    3. Prompt construction
    4. LLM call
    5. Formattering af output + kildehenvisninger
    """

    # 1. Retrieval
    windowed_results = build_retrieval_pipeline(query_text)

    if not windowed_results:
        return {
        "answer": "Der findes ingen tilgængelig information om dette i de kommunale dokumenter.",
        "sources": []
        }
    
    # 2. Context
    context = build_context(windowed_results)

    # 3. Prompt
    prompt = build_prompt(query_text, context)

    # 4. LLM svar
    raw_answer = call_llm(prompt)

    # 5 Formater output
    final = format_answer(raw_answer, windowed_results)
    
    return final

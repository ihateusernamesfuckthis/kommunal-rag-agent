# RAG Pipeline Code Review & Improvement Roadmap

**Date:** 2025-11-18
**Reviewer:** Claude
**Status:** Pre-Production Review

---

## Executive Summary

This is a strong first RAG implementation with several advanced techniques (hybrid search, windowing, synthetic questions). However, there are **critical bugs that must be fixed before running the embedding pipeline**. The architecture is sound, but needs production hardening.

**Overall Grade:** B+ (A- after critical fixes)

---

## 🔴 CRITICAL BUGS - Fix Before Embedding

These bugs will cause incorrect behavior, data corruption, or runtime crashes. **Do not proceed with embedding until these are fixed.**

### 1. Triple Upsert Bug in `index_chunks.py`

**Location:** `index_chunks.py:89-161`
**Severity:** CRITICAL - Causes data duplication and massive performance degradation
**Impact:** Each chunk gets upserted 3 times, wasting API calls and storage

**Problem:**
```python
# Lines 124-141: Batch upsert loop (CORRECT)
for chunk_record in reader:
    batch_texts.append(chunk_record["text"])
    batch_records.append(chunk_record)

    if len(batch_texts) >= batch_size:
        vectors = embed_batch(batch_texts)
        for vec, rec in zip(vectors, batch_records):
            upsert_chunk(client, collection_name, rec, vec)
            # ...
        batch_texts = []
        batch_records = []

    # Lines 143-152: DUPLICATE - This is still inside the for loop!
    if batch_texts:  # ← This runs after EVERY batch, not at the end
        vectors = embed_batch(batch_texts)
        # ... upserting again

    # Lines 154-160: TRIPLE upsert - individual upsert after batch
    result = upsert_chunk(client, collection_name, chunk_record)
```

**Fix:**
```python
# Lines 89-161 should be restructured like this:

def index_all_chunks(
        client: QdrantClient,
        collection_name: str = "kommune-rag",
        chunk_dir: Path = CHUNK_DIR
):
    print("--- Starter indeksering af chunks til Qdrant ---")

    chunk_files = list(chunk_dir.glob("*.jsonl"))
    if not chunk_files:
        print("Ingen chunk-filer fundet i /chunks")
        return

    total_chunks = 0
    total_synthetics = 0
    total_files = 0

    for file in chunk_files:
        print(f"\n--- Indekserer fil: {file.name} ---")
        chunks_in_file = 0

        with jsonlines.open(file, "r") as reader:
            batch_texts = []
            batch_records = []
            batch_size = 32

            for chunk_record in reader:
                batch_texts.append(chunk_record["text"])
                batch_records.append(chunk_record)

                # Process when batch is full
                if len(batch_texts) >= batch_size:
                    vectors = embed_batch(batch_texts)

                    for vec, rec in zip(vectors, batch_records):
                        upsert_chunk(client, collection_name, rec, vec)
                        chunks_in_file += 1
                        total_chunks += 1

                        num_syn = upsert_synthetics(client, collection_name, rec)
                        total_synthetics += num_syn

                    # Reset batch
                    batch_texts = []
                    batch_records = []

            # IMPORTANT: This is OUTSIDE the for loop - process remaining chunks
            if batch_texts:
                vectors = embed_batch(batch_texts)

                for vec, rec in zip(vectors, batch_records):
                    upsert_chunk(client, collection_name, rec, vec)
                    chunks_in_file += 1
                    total_chunks += 1

                    num_syn = upsert_synthetics(client, collection_name, rec)
                    total_synthetics += num_syn

        total_files += 1
        print(f"⮑ {chunks_in_file} chunks indekseret fra {file.name}")

    print("\n=== Indexing færdig! ===")
    print(f"Chunk-filer behandlet: {total_files}")
    print(f"{total_chunks} chunks indekseret.")
    print(f"{total_synthetics} syntetiske spørgsmål indekseret.")
```

**Action Items:**
- [ ] Rewrite `index_all_chunks()` with correct indentation
- [ ] Delete lines 154-160 entirely
- [ ] Test with 1 small file before running on all files

---

### 2. Broken Filter Logic in `query_engine.py`

**Location:** `query_engine.py:226-248`
**Severity:** CRITICAL - Filters are silently ignored
**Impact:** Metadata filtering doesn't work (year, committee, section filters fail)

**Problem:**
```python
if filters:
    conditions = []

    for key, value in filters.items():
        if value is None:
            continue

        if isinstance(value, list):
            conditions.append(...)
            qdrant_filter = Filter(must=conditions)  # ← Sets filter inside loop
        else:
            conditions.append(...)
            qdrant_filter = None  # ← BUG! Overwrites filter to None!
```

**Fix:**
```python
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
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )

        # Set filter AFTER loop, not inside it
        if conditions:
            qdrant_filter = Filter(must=conditions)

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter
    )

    return results
```

**Action Items:**
- [ ] Fix filter creation logic
- [ ] Move `qdrant_filter = Filter(must=conditions)` outside the loop
- [ ] Test with a filter like `{"meeting_year": 2023}` and verify results

---

### 3. Duplicate Function Definition in `raq_query_engine.py`

**Location:** `raq_query_engine.py:128-174`
**Severity:** CRITICAL - Code won't work as intended
**Impact:** The prompt function is defined twice (nested), inner function never executes

**Problem:**
```python
def build_prompt(query: str, context: str) -> str:
    """
    Bygger et LLM optimeret prompt til danskgpt-v2.1
    ...
    """

    def build_prompt(query: str, context: str) -> str:  # ← DUPLICATE!
        """
        Bygger en klar, robust og LLM-optimeret prompt til danskgpt-v2.1.
        ...
        """
        prompt = f"""..."""
        return prompt.strip()
```

**Fix:**
```python
# Delete lines 128-137, keep only lines 138-174

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
       - "Kilde: <committee> · <meeting_date> · <source_path>"
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
```

**Action Items:**
- [ ] Remove outer function definition (lines 128-137)
- [ ] Keep only the inner implementation

---

### 4. Typo Causing Runtime Crash in `raq_query_engine.py`

**Location:** `raq_query_engine.py:247`
**Severity:** CRITICAL - Will crash when generating answers
**Impact:** KeyError exception when formatting sources

**Problem:**
```python
key = (s["committee"], s["meeting_data"], s["source_path"])
#                          ^^^^^^^^^^^^^ Wrong key name!
```

**Fix:**
```python
key = (s["committee"], s["meeting_date"], s["source_path"])
#                          ^^^^^^^^^^^^^ Correct
```

**Action Items:**
- [ ] Fix typo: `meeting_data` → `meeting_date`
- [ ] Test `format_answer()` with mock data

---

### 5. Missing BM25 Implementation in `query_engine.py`

**Location:** `query_engine.py:105-110`
**Severity:** CRITICAL - Feature doesn't work
**Impact:** "Hybrid search" is actually just semantic search with fake BM25

**Problem:**
```python
def bm25_search(query: str, collection_name: str, top_k: int = 10):
    return client.search(
        collection_name=collection_name,
        query=query,  # ← Qdrant doesn't accept text queries like this
        limit=top_k
    )
```

This doesn't implement BM25. Qdrant's `.search()` requires a vector, not text.

**Fix Option A (Recommended):** Remove BM25 for now, rename to semantic search
```python
# In query_engine.py, rename hybrid_search to semantic_search_with_recency
# Remove bm25_search function
# Update weights to just alpha (vector) + gamma (recency)

def semantic_search_with_recency(
        query: str,
        collection_name: str = "kommune-rag",
        top_k: int = 10,
        alpha: float = 0.8,  # Vector weight
        gamma: float = 0.2,  # Recency weight
        filters: dict = None
):
    """
    Kombinerer vektor-søgning + recency reranking
    """

    # Vector search
    vector_results = vector_search(
        query_text=query,
        collection_name=collection_name,
        top_k=top_k * 2,  # Get more candidates for reranking
        filters=filters
    )

    ranked = []
    for r in vector_results:
        v_score = vector_to_score(r.score)
        chunk_year = r.payload.get("meeting_year")

        if chunk_year is None:
            r_score = 0.0
        else:
            r_score = recency_score(chunk_year)

        final_score = v_score * alpha + r_score * gamma
        ranked.append((final_score, r.payload))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:top_k]
```

**Fix Option B (Advanced):** Implement real BM25 with `rank-bm25` library

Add to `requirements.txt`:
```
rank-bm25==0.2.2
```

Create a new file `bm25_index.py`:
```python
from rank_bm25 import BM25Okapi
import jsonlines
from pathlib import Path
import pickle

def build_bm25_index(chunk_dir: Path):
    """
    Builds BM25 index from all chunks and saves it
    """
    corpus = []
    metadata = []

    for file in chunk_dir.glob("*.jsonl"):
        with jsonlines.open(file, "r") as reader:
            for chunk in reader:
                corpus.append(chunk["text"].lower().split())
                metadata.append(chunk)

    bm25 = BM25Okapi(corpus)

    # Save index
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "metadata": metadata}, f)

    print(f"BM25 index built with {len(corpus)} documents")
    return bm25, metadata

def load_bm25_index():
    """Load pre-built BM25 index"""
    with open("bm25_index.pkl", "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["metadata"]

def bm25_search(query: str, top_k: int = 10):
    """Search using BM25"""
    bm25, metadata = load_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top K
    top_indices = scores.argsort()[-top_k:][::-1]
    results = [(scores[i], metadata[i]) for i in top_indices]
    return results
```

Then update `query_engine.py` to use it:
```python
from bm25_index import bm25_search as bm25_search_impl

def bm25_search(query: str, collection_name: str, top_k: int = 10):
    return bm25_search_impl(query, top_k)
```

**Recommendation:** Start with Option A (remove BM25), then add Option B later after everything else works.

**Action Items:**
- [ ] Choose Option A or B
- [ ] If Option A: Remove BM25, rename functions, update weights
- [ ] If Option B: Add rank-bm25 dependency, implement index building
- [ ] Update `raq_query_engine.py` to use the correct function name

---

### 6. SECURITY: Exposed Credentials in `rag_overview.md`

**Location:** `rag_overview.md:1-3`
**Severity:** CRITICAL SECURITY ISSUE
**Impact:** API keys exposed in source code

**Problem:**
```markdown
qdrant key -> eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
qdrant endpoint -> https://9cb490fe-535e-4c3b-8977-d2ad1f6d243f.eu-central-1-0...
```

**Fix:**
1. Delete lines 1-4 from `rag_overview.md`
2. Ensure credentials are only in `.env` (which should be in `.gitignore`)
3. Check if this was committed to git:
   ```bash
   git log -p rag_overview.md | grep -i "qdrant key"
   ```
4. If committed, **rotate your Qdrant API key immediately** via Qdrant Cloud console
5. Consider using `git-secrets` or `gitleaks` to prevent future leaks

**Action Items:**
- [ ] Remove credentials from `rag_overview.md`
- [ ] Check git history for leaked credentials
- [ ] Rotate Qdrant API key if exposed
- [ ] Verify `.env` is in `.gitignore`

---

### 7. Missing Qdrant Payload Indexes

**Location:** `index_chunks.py:41-67`
**Severity:** HIGH - Required for filter performance
**Impact:** Filters will be slow/broken on large datasets

**Problem:**
You create the collection but don't create payload indexes for the metadata fields you filter on.

**Fix:**
```python
from qdrant_client.models import PayloadSchemaType

def create_or_reset_collection(
        client: QdrantClient,
        collection_name: str = "kommune-rag",
        vector_size: int = 1024
):
    """
    Opretter en qdrant collection hvis den ikke findes.
    Hvis den findes, droppes og genskabes (så man starter rent)
    """

    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name in existing_collections:
        print(f"Collection {collection_name} findes - sletter og genskaber")
        client.delete_collection(collection_name)

    print(f"Opretter Qdrant collection: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    print(f"Oprettet collection: {collection_name}")

    # Create payload indexes for filtering
    print("Opretter payload indexes...")

    client.create_payload_index(
        collection_name=collection_name,
        field_name="committee_normalized",
        field_schema=PayloadSchemaType.KEYWORD
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="meeting_year",
        field_schema=PayloadSchemaType.INTEGER
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="section_type",
        field_schema=PayloadSchemaType.KEYWORD
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="section_category",
        field_schema=PayloadSchemaType.KEYWORD
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="document_id",
        field_schema=PayloadSchemaType.KEYWORD
    )

    print("Payload indexes oprettet!")
```

**Action Items:**
- [ ] Add payload index creation to `create_or_reset_collection()`
- [ ] Verify indexes are created with `client.get_collection(collection_name)`

---

## 🟡 HIGH PRIORITY - Fix for Production Quality

These issues won't break the system but will cause poor performance or incorrect behavior in production.

### 8. Synthetic Question Generation is Too Slow

**Location:** `index_chunks.py:137-138` and `synthetic.py:93-136`
**Severity:** HIGH - Performance bottleneck
**Impact:** Generating 3 questions per chunk = 3 LLM calls per chunk. For 1000 chunks = 3000 LLM calls (will take hours)

**Current Code:**
```python
# This happens for EVERY chunk individually
num_syn = upsert_synthetics(client, collection_name, rec)
```

**Fix:** Batch process synthetic questions

Create a new function in `synthetic.py`:
```python
def generate_synthetic_questions_batch(chunks: list[dict], n_questions: int = 3) -> dict:
    """
    Genererer syntetiske spørgsmål for flere chunks i ét LLM kald
    Returnerer dict: {chunk_id: [question1, question2, ...]}
    """

    # Build prompt for multiple chunks
    prompt = f"""
    Du skal generere {n_questions} korte borger-spørgsmål for hver af følgende tekster.
    Format dit svar som:

    TEKST 1:
    - Spørgsmål 1
    - Spørgsmål 2
    - Spørgsmål 3

    TEKST 2:
    - Spørgsmål 1
    ...

    """

    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"][:500]  # Limit text length
        prompt += f"\nTEKST {i}:\n{text}\n"

    prompt += "\nGenerer nu spørgsmålene:"

    headers = {
        "Authorization": f"Bearer {SYVAI_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "danskgpt-v2.1",
        "prompt": prompt,
        "max_tokens": 150 * len(chunks)  # Scale with number of chunks
    }

    response = requests.post(SYV_COMPLETION_URL, headers=headers, json=data)
    response.raise_for_status()

    raw_output = response.json()["choices"][0]["text"].strip()

    # Parse output back into per-chunk questions
    result = {}
    current_text_idx = None
    current_questions = []

    for line in raw_output.split("\n"):
        line = line.strip()
        if line.startswith("TEKST"):
            # Save previous chunk's questions
            if current_text_idx is not None:
                chunk_id = chunks[current_text_idx]["chunk_id"]
                result[chunk_id] = current_questions[:n_questions]

            # Start new chunk
            current_text_idx = int(line.split()[1].rstrip(":")) - 1
            current_questions = []

        elif line.startswith("-") or line.startswith("•"):
            question = line.lstrip("-• ").strip()
            if len(question) > 3:
                current_questions.append(question)

    # Save last chunk's questions
    if current_text_idx is not None:
        chunk_id = chunks[current_text_idx]["chunk_id"]
        result[chunk_id] = current_questions[:n_questions]

    return result
```

Then update `index_chunks.py`:
```python
# Instead of calling upsert_synthetics for each chunk,
# batch process every 10 chunks:

batch_for_synthetics = []

for vec, rec in zip(vectors, batch_records):
    upsert_chunk(client, collection_name, rec, vec)
    chunks_in_file += 1
    total_chunks += 1

    batch_for_synthetics.append(rec)

    # Process synthetics in batches of 10
    if len(batch_for_synthetics) >= 10:
        questions_map = generate_synthetic_questions_batch(batch_for_synthetics)

        for chunk_rec in batch_for_synthetics:
            questions = questions_map.get(chunk_rec["chunk_id"], [])
            if questions:
                vectors = embed_synthetics(questions)
                # ... upsert synthetic points

        batch_for_synthetics = []
```

**Action Items:**
- [ ] Implement `generate_synthetic_questions_batch()`
- [ ] Update `index_chunks.py` to use batching
- [ ] Test with 10 chunks first to verify parsing works

---

### 9. No Deduplication of Chunks

**Location:** `chunk_data.py`
**Severity:** MEDIUM - Data quality issue
**Impact:** Identical sections (like "Godkendelse af dagsorden") appear in every meeting, polluting results

**Fix:**
Add semantic deduplication after chunking:

```python
import hashlib

def hash_text(text: str) -> str:
    """Create hash of normalized text"""
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()

def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Remove duplicate chunks based on text hash
    Keeps the most recent version
    """
    seen_hashes = {}

    for chunk in chunks:
        text_hash = hash_text(chunk["text"])

        if text_hash not in seen_hashes:
            seen_hashes[text_hash] = chunk
        else:
            # Keep the more recent version
            existing = seen_hashes[text_hash]
            existing_year = existing.get("meeting_year", 0)
            current_year = chunk.get("meeting_year", 0)

            if current_year > existing_year:
                seen_hashes[text_hash] = chunk

    return list(seen_hashes.values())
```

Add this to `chunk_data.py` and call it before writing:
```python
def process_all_chunks(input_dir: Path = PROCESSED_DIR, output_dir: Path = CHUNK_DIR):
    # ... existing code ...

    # Before writing, deduplicate across all chunks
    all_chunks = []
    for file in files:
        # ... chunk file ...
        all_chunks.extend(chunks_from_file)

    unique_chunks = deduplicate_chunks(all_chunks)
    print(f"Reduceret fra {len(all_chunks)} til {len(unique_chunks)} chunks efter deduplicering")

    # Write unique chunks
```

**Action Items:**
- [ ] Implement deduplication function
- [ ] Integrate into chunking pipeline
- [ ] Log statistics (how many duplicates removed)

---

### 10. Missing Table and List Handling

**Location:** `clean_data.py:138-152`
**Severity:** MEDIUM - Data quality issue
**Impact:** Tables and lists lose structure when converted to plain text

**Current Code:**
```python
soup = BeautifulSoup(html_content, "html.parser")
text = soup.get_text(separator=" ", strip=True)
# This flattens everything
```

**Fix:**
Use markdown conversion to preserve structure:

```bash
pip install markdownify
```

Update `clean_data.py`:
```python
from markdownify import markdownify as md

def clean_html_to_text(html_content: str, preserve_structure: bool = True) -> str:
    """
    Konverterer HTML til ren tekst og normaliserer whitespace.
    preserve_structure: Hvis True, konverterer til markdown for at bevare tabeller/lister
    """
    if not html_content:
        return ""

    if preserve_structure:
        # Convert to markdown to preserve tables, lists, headings
        text = md(html_content, heading_style="ATX", bullets="-")
    else:
        # Fallback to plain text
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

    # Normaliser whitespace og unicode
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines
    text = re.sub(r" +", " ", text)  # Single spaces

    return text.strip()
```

**Action Items:**
- [ ] Add `markdownify` to `requirements.txt`
- [ ] Update `clean_html_to_text()` to preserve structure
- [ ] Test on a document with tables to verify output quality

---

### 11. Missing Attachment Metadata Extraction

**Location:** `clean_data.py:154-223`
**Severity:** MEDIUM - Missing feature
**Impact:** Attachments mentioned in overview doc but not extracted

**Problem:**
Raw JSON has attachment data:
```json
"Bilag": [
    {
        "Navn": "Budget 2023",
        "Link": "https://..."
    }
]
```

But `extract_agenda_sections()` doesn't extract it.

**Fix:**
```python
def extract_agenda_sections(raw_meta: dict) -> list:
    """
    Parser dagsordenpunkter og udtrækker disse sektioner:
    Resume, Baggrund og Indstilling, Beslutning, Tekst, Felter.html, Bilag
    """

    sections = []
    agenda_items = raw_meta.get("raw_agenda_items", [])
    # ... existing code ...

    for idx, item in enumerate(agenda_items):
        # ... existing extraction ...

        # Extract attachments
        bilag = item.get("Bilag", [])
        attachments = []
        for b in bilag:
            if isinstance(b, dict):
                attachments.append({
                    "name": b.get("Navn", "Ukendt bilag"),
                    "link": b.get("Link", "")
                })

        # ... existing sections code ...

        # Add attachments to metadata
        for section_dict in sections:
            if section_dict["agenda_index"] == idx:
                section_dict["attachments"] = attachments

    return sections
```

**Action Items:**
- [ ] Update `extract_agenda_sections()` to include attachments
- [ ] Add attachment links to chunk metadata
- [ ] Update `format_answer()` to include attachment links in response

---

### 12. No Connection Pooling for HTTP Requests

**Location:** `synthetic.py`, `query_engine.py`, `raq_query_engine.py`
**Severity:** MEDIUM - Performance issue
**Impact:** Each API call creates a new HTTP connection (slow, wasteful)

**Fix:**
Create a shared session in each file:

```python
import requests

# At module level (top of file)
_session = None

def get_session():
    """Get or create shared requests session"""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "Authorization": f"Bearer {SYVAI_TOKEN}",
            "Content-Type": "application/json"
        })
    return _session

# Then replace all requests.post() with:
session = get_session()
response = session.post(SYV_EMBED_URL, json=data)
```

**Action Items:**
- [ ] Add session management to `synthetic.py`
- [ ] Add session management to `query_engine.py`
- [ ] Add session management to `raq_query_engine.py`
- [ ] Measure performance improvement (should see 20-30% faster API calls)

---

### 13. Add Retry Logic with Exponential Backoff

**Location:** All files making API calls
**Severity:** MEDIUM - Robustness issue
**Impact:** Transient API errors cause pipeline to crash

**Fix:**
Already have `tenacity` in requirements! Use it:

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import requests

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError))
)
def embed_text(text: str) -> list:
    """
    Returnerer en embedding-vektor for en enkelt tekststreng via syv.ai
    Automatically retries on failure with exponential backoff
    """
    if not text or not text.strip():
        return []

    session = get_session()

    data = {
        "input": [text],
        "model": "mistral/mistral-embed",
        "encoding_format": "float"
    }

    response = session.post(SYV_EMBED_URL, json=data, timeout=30)
    response.raise_for_status()

    result = response.json()
    return result["data"][0]["embedding"]
```

Apply to:
- `embed_text()`
- `embed_batch()`
- `generate_synthetic_questions()`
- `call_llm()`

**Action Items:**
- [ ] Add `@retry` decorator to all API-calling functions
- [ ] Add timeout parameter to all requests
- [ ] Test by temporarily breaking API credentials to verify retry works

---

## 🟢 MEDIUM PRIORITY - Production Hardening

These improve quality, maintainability, and observability but aren't blocking.

### 14. Create Configuration File

**Location:** Hardcoded values scattered across all files
**Severity:** MEDIUM - Maintainability issue

**Fix:**
Create `config.py`:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathConfig:
    """File paths"""
    BASE_DIR: Path = Path("data/greve")
    RAW_DIR: Path = BASE_DIR / "raw_dagsordener"
    PROCESSED_DIR: Path = BASE_DIR / "dagsordener"
    CHUNK_DIR: Path = BASE_DIR / "chunks"

@dataclass
class ChunkConfig:
    """Chunking parameters"""
    MAX_TOKENS: int = 1500
    OVERLAP_RATIO: float = 0.15
    APPROX_TOKEN_PER_WORD: float = 1.0

@dataclass
class EmbeddingConfig:
    """Embedding model config"""
    MODEL_NAME: str = "mistral/mistral-embed"
    VECTOR_SIZE: int = 1024
    BATCH_SIZE: int = 32

@dataclass
class RetrievalConfig:
    """Retrieval parameters"""
    TOP_K: int = 10
    VECTOR_WEIGHT: float = 0.7  # alpha
    BM25_WEIGHT: float = 0.3    # beta (if implemented)
    RECENCY_WEIGHT: float = 0.2 # gamma
    WINDOW_SIZE: int = 1
    MIN_YEAR: int = 2018
    MAX_YEAR: int = 2025

@dataclass
class SyntheticConfig:
    """Synthetic question generation"""
    NUM_QUESTIONS: int = 3
    BATCH_SIZE: int = 10
    MAX_RETRIES: int = 5

@dataclass
class LLMConfig:
    """LLM generation parameters"""
    MODEL_NAME: str = "danskgpt-v2.1"
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9

# Create global instances
paths = PathConfig()
chunk_config = ChunkConfig()
embedding_config = EmbeddingConfig()
retrieval_config = RetrievalConfig()
synthetic_config = SyntheticConfig()
llm_config = LLMConfig()
```

Then import and use:
```python
from config import chunk_config, paths

def chunk_text(text: str, max_tokens: int = None, overlap_ratio: float = None):
    max_tokens = max_tokens or chunk_config.MAX_TOKENS
    overlap_ratio = overlap_ratio or chunk_config.OVERLAP_RATIO
    # ...
```

**Action Items:**
- [ ] Create `config.py`
- [ ] Replace all hardcoded values with config references
- [ ] Document each parameter in docstrings

---

### 15. Add Logging Instead of Print Statements

**Location:** All files
**Severity:** MEDIUM - Production readiness

**Fix:**
Create `logger.py`:

```python
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Setup logger with file and console handlers"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(exist_ok=True, parents=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

Then in each file:
```python
from logger import setup_logger

logger = setup_logger(__name__, "logs/clean_data.log")

# Replace print() with:
logger.info(f"Skrev {len(sections)} sektioner -> {output_path.name}")
logger.warning(f"Ingen sektioner fundet i dokumentet: {document_id}")
logger.error(f"JSON parsing fejl i {path.name}: {e}")
```

**Action Items:**
- [ ] Create `logger.py`
- [ ] Replace all `print()` with `logger.info/warning/error()`
- [ ] Create `logs/` directory
- [ ] Add log rotation for production

---

### 16. Add Input Validation

**Location:** All public functions
**Severity:** MEDIUM - Security/robustness

**Fix:**
Create `validation.py`:

```python
def validate_query(query: str, max_length: int = 1000) -> str:
    """
    Validates and sanitizes user query
    Raises ValueError if invalid
    """
    if not query:
        raise ValueError("Query cannot be empty")

    query = query.strip()

    if len(query) > max_length:
        raise ValueError(f"Query too long (max {max_length} characters)")

    if len(query) < 3:
        raise ValueError("Query too short (min 3 characters)")

    # Basic injection prevention
    dangerous_patterns = ["<script", "javascript:", "onerror="]
    for pattern in dangerous_patterns:
        if pattern.lower() in query.lower():
            raise ValueError("Invalid characters in query")

    return query

def validate_chunk_record(chunk: dict) -> bool:
    """Validates chunk has required fields"""
    required_fields = ["text", "chunk_id", "document_id"]

    for field in required_fields:
        if field not in chunk:
            return False
        if not chunk[field]:
            return False

    return True

def validate_file_path(path: Path, must_exist: bool = True) -> Path:
    """Validates file path"""
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.exists() and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return path
```

Use in functions:
```python
from validation import validate_query

def answer_query(query_text: str) -> Dict[str, Any]:
    # Validate input
    query_text = validate_query(query_text)

    # ... rest of function
```

**Action Items:**
- [ ] Create `validation.py`
- [ ] Add validation to all user-facing functions
- [ ] Add validation to file operations
- [ ] Write unit tests for validation functions

---

### 17. Add Type Hints Consistently

**Location:** All files
**Severity:** LOW - Code quality

**Current State:** Inconsistent type hints

**Fix:**
Use type hints everywhere:

```python
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

def load_raw_document(path: Path) -> Optional[Dict[str, Any]]:
    """..."""

def extract_base_metadata(raw_doc: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """..."""

def chunk_text(
    text: str,
    max_tokens: int = 1500,
    overlap_ratio: float = 0.15
) -> List[str]:
    """..."""
```

Then run type checking:
```bash
pip install mypy
mypy clean_data.py chunk_data.py --strict
```

**Action Items:**
- [ ] Add type hints to all function signatures
- [ ] Add type hints to all return types
- [ ] Install and run `mypy` to catch type errors

---

## 🔵 TESTING SETUP - Learn Production Best Practices

Testing is critical for production systems. Here's how to set it up properly.

### 18. Testing Infrastructure Setup

**Goal:** Learn to write and run tests like in real-world production environments.

#### Step 1: Install Testing Dependencies

Add to `requirements.txt`:
```
pytest==8.3.4
pytest-cov==6.0.0
pytest-mock==3.14.0
pytest-asyncio==0.25.2
```

Install:
```bash
pip install -r requirements.txt
```

#### Step 2: Create Test Directory Structure

```bash
mkdir -p tests
touch tests/__init__.py
touch tests/test_clean_data.py
touch tests/test_chunk_data.py
touch tests/test_query_engine.py
touch tests/test_synthetic.py
touch tests/test_integration.py
touch tests/conftest.py  # Pytest fixtures
```

#### Step 3: Create Test Fixtures

`tests/conftest.py`:
```python
import pytest
from pathlib import Path
import json
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Creates a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_raw_json():
    """Sample raw municipal document"""
    return {
        "Id": "test-123",
        "Dokumenttype": "dagsorden",
        "Udvalg": {"Navn": "Byrådet"},
        "MødeDato": "2023-01-15T10:00:00+01:00",
        "Titel": "Test Dagsorden",
        "Dagsordenpunkter": [
            {
                "Navn": "Test Punkt",
                "Sagsnummer": "23-001",
                "Resume": "<p>Dette er en test resume</p>",
                "BaggrundOgIndstilling": "<p>Test baggrund</p>",
                "Beslutning": "<p>Godkendt</p>",
                "Bilag": [
                    {"Navn": "Bilag 1", "Link": "https://example.com/bilag1"}
                ]
            }
        ]
    }

@pytest.fixture
def sample_chunk():
    """Sample processed chunk"""
    return {
        "document_id": "test-123",
        "chunk_id": "test-123_0_resume_chunk0",
        "section_type": "resume",
        "section_title": "Test Punkt",
        "committee": "Byrådet",
        "meeting_date": "2023-01-15T10:00:00+01:00",
        "case_number": "23-001",
        "agenda_index": 0,
        "text": "Dette er test tekst for chunken.",
        "source_path": "test.json",
        "meeting_year": 2023,
        "meeting_month": 1,
        "committee_normalized": "byraad",
        "section_category": "core"
    }

@pytest.fixture
def mock_embedding():
    """Mock embedding vector (1024 dimensions)"""
    return [0.1] * 1024

@pytest.fixture
def mock_qdrant_client(mocker):
    """Mock Qdrant client to avoid real API calls in tests"""
    mock_client = mocker.MagicMock()
    mock_client.search.return_value = []
    mock_client.upsert.return_value = True
    return mock_client
```

#### Step 4: Write Unit Tests

`tests/test_clean_data.py`:
```python
import pytest
from pathlib import Path
import json
from clean_data import (
    load_raw_document,
    extract_base_metadata,
    clean_html_to_text,
    extract_agenda_sections
)

def test_clean_html_to_text():
    """Test HTML cleaning"""
    html = "<p>Dette er <strong>fed</strong> tekst</p>"
    result = clean_html_to_text(html)
    assert result == "Dette er fed tekst"
    assert "<" not in result
    assert ">" not in result

def test_clean_html_handles_empty():
    """Test empty HTML input"""
    assert clean_html_to_text("") == ""
    assert clean_html_to_text(None) == ""

def test_clean_html_normalizes_whitespace():
    """Test whitespace normalization"""
    html = "<p>Test    med   mange     mellemrum</p>"
    result = clean_html_to_text(html)
    assert "  " not in result  # No double spaces

def test_load_raw_document(temp_dir, sample_raw_json):
    """Test loading raw JSON document"""
    # Write sample JSON to temp file
    test_file = temp_dir / "test.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(sample_raw_json, f)

    # Load it
    result = load_raw_document(test_file)

    assert result is not None
    assert result["Id"] == "test-123"
    assert result["Dokumenttype"] == "dagsorden"

def test_load_raw_document_missing_file(temp_dir):
    """Test loading non-existent file"""
    result = load_raw_document(temp_dir / "missing.json")
    assert result is None

def test_extract_base_metadata(sample_raw_json, temp_dir):
    """Test metadata extraction"""
    test_path = temp_dir / "test.json"

    metadata = extract_base_metadata(sample_raw_json, test_path)

    assert metadata["document_id"] == "test-123"
    assert metadata["document_type"] == "dagsorden"
    assert metadata["committee"] == "Byrådet"
    assert metadata["title"] == "Test Dagsorden"
    assert len(metadata["raw_agenda_items"]) == 1

def test_extract_agenda_sections(sample_raw_json, temp_dir):
    """Test section extraction"""
    test_path = temp_dir / "test.json"
    metadata = extract_base_metadata(sample_raw_json, test_path)
    sections = extract_agenda_sections(metadata)

    assert len(sections) > 0

    # Check first section
    section = sections[0]
    assert "text" in section
    assert "section_type" in section
    assert "committee" in section
    assert section["committee"] == "Byrådet"
```

`tests/test_chunk_data.py`:
```python
import pytest
from chunk_data import (
    approximate_token_count,
    split_text_into_sentences,
    chunk_text,
    normalize_committee,
    add_metadata_fields
)

def test_approximate_token_count():
    """Test token counting"""
    assert approximate_token_count("hello world") == 2
    assert approximate_token_count("") == 0
    assert approximate_token_count("one") == 1

def test_split_text_into_sentences():
    """Test sentence splitting"""
    text = "Dette er sætning 1. Dette er sætning 2! Er dette sætning 3?"
    sentences = split_text_into_sentences(text)

    assert len(sentences) == 3
    assert sentences[0] == "Dette er sætning 1."
    assert sentences[1] == "Dette er sætning 2!"
    assert sentences[2] == "Er dette sætning 3?"

def test_chunk_text_small():
    """Test chunking small text"""
    text = "Dette er en kort tekst."
    chunks = chunk_text(text, max_tokens=100)

    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_with_overlap():
    """Test chunking with overlap"""
    # Create text with multiple sentences
    sentences = ["Dette er sætning {}.".format(i) for i in range(1, 11)]
    text = " ".join(sentences)

    chunks = chunk_text(text, max_tokens=10, overlap_ratio=0.2)

    # Should create multiple chunks
    assert len(chunks) > 1

    # Check overlap exists (last words of chunk N should appear in chunk N+1)
    if len(chunks) >= 2:
        # This is approximate due to word-based overlap
        assert len(chunks[0]) > 0
        assert len(chunks[1]) > 0

def test_normalize_committee():
    """Test committee name normalization"""
    assert normalize_committee("Plan- og Teknikudvalget") == "planteknik"
    assert normalize_committee("Byrådet") == "byråd"
    assert normalize_committee("Økonomiudvalget") == "økonomi"
    assert normalize_committee(None) == "unknown"
    assert normalize_committee("") == "unknown"

def test_add_metadata_fields(sample_chunk):
    """Test metadata enrichment"""
    result = add_metadata_fields(sample_chunk)

    assert result["meeting_year"] == 2023
    assert result["meeting_month"] == 1
    assert result["committee_normalized"] == "byråd"
    assert result["section_category"] == "core"
    assert result["is_decision_section"] == False
    assert result["is_background_section"] == False
```

`tests/test_query_engine.py`:
```python
import pytest
from query_engine import (
    extract_years,
    extract_section_intent,
    normalize_committee,
    recency_score,
    vector_to_score
)

def test_extract_years():
    """Test year extraction from query"""
    assert extract_years("Hvad skete der i 2023?") == 2023
    assert extract_years("Sammenlign 2022 og 2024") == 2024  # Returns max
    assert extract_years("Hvad er planerne?") is None

def test_extract_section_intent():
    """Test section intent extraction"""
    assert extract_section_intent("Hvad blev besluttet?") == "decision"
    assert extract_section_intent("Hvorfor blev det gjort?") == "background"
    assert extract_section_intent("Hvad er resuméet?") == "resume"
    assert extract_section_intent("Generel spørgsmål") is None

def test_recency_score():
    """Test recency scoring"""
    # Recent year should score high
    assert recency_score(2025, min_year=2020, max_year=2025) == 1.0

    # Middle year should score around 0.5
    mid_score = recency_score(2022, min_year=2020, max_year=2024)
    assert 0.4 <= mid_score <= 0.6

    # Old year should score low
    assert recency_score(2020, min_year=2020, max_year=2025) == 0.0

def test_vector_to_score():
    """Test distance to score conversion"""
    assert vector_to_score(0.0) == 1.0  # Perfect match
    assert vector_to_score(1.0) == 0.0  # Worst match
    assert vector_to_score(0.5) == 0.5  # Medium match
```

#### Step 5: Integration Tests

`tests/test_integration.py`:
```python
import pytest
from pathlib import Path
import json
import jsonlines
from clean_data import preprocess_all_documents
from chunk_data import process_all_chunks

def test_full_pipeline_small_dataset(temp_dir, sample_raw_json):
    """
    Integration test: raw JSON → preprocessed JSONL → chunks
    """
    # Setup directories
    raw_dir = temp_dir / "raw_dagsordener"
    processed_dir = temp_dir / "dagsordener"
    chunk_dir = temp_dir / "chunks"

    raw_dir.mkdir()
    processed_dir.mkdir()
    chunk_dir.mkdir()

    # Create sample raw file
    raw_file = raw_dir / "test_doc.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(sample_raw_json, f)

    # Step 1: Preprocess
    # (You'd need to modify preprocess_all_documents to accept custom dirs)
    # For now, this is a placeholder

    # Step 2: Chunk
    # (Same - would need to modify chunk functions)

    # Verify output
    # assert chunk_dir has files
    # assert chunks have correct structure

    # This is a template - you'll need to adapt your functions
    # to accept directory parameters for testing
```

#### Step 6: Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_clean_data.py

# Run specific test
pytest tests/test_clean_data.py::test_clean_html_to_text

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "clean_html"
```

#### Step 7: Continuous Testing During Development

Create a test script: `run_tests.sh`
```bash
#!/bin/bash

# Run tests with coverage
pytest --cov=. --cov-report=term-missing --cov-report=html

# Open coverage report
open htmlcov/index.html  # On macOS
# xdg-open htmlcov/index.html  # On Linux
```

Make it executable:
```bash
chmod +x run_tests.sh
./run_tests.sh
```

#### Step 8: Mock External APIs for Testing

`tests/test_synthetic.py`:
```python
import pytest
from unittest.mock import Mock, patch
from synthetic import embed_text, generate_synthetic_questions

@patch('synthetic.requests.post')
def test_embed_text_success(mock_post):
    """Test embedding with mocked API response"""
    # Setup mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1] * 1024}]
    }
    mock_post.return_value = mock_response

    # Call function
    result = embed_text("test text")

    # Verify
    assert len(result) == 1024
    assert all(x == 0.1 for x in result)
    mock_post.assert_called_once()

@patch('synthetic.requests.post')
def test_embed_text_api_failure(mock_post):
    """Test embedding handles API failures"""
    # Setup mock to raise exception
    mock_post.side_effect = Exception("API Error")

    # Should raise or handle gracefully
    with pytest.raises(Exception):
        embed_text("test text")

@patch('synthetic.requests.post')
def test_generate_synthetic_questions(mock_post):
    """Test synthetic question generation"""
    # Mock LLM response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "text": "- Hvad handler dette om?\n- Hvorfor er det vigtigt?\n- Hvem er ansvarlig?"
        }]
    }
    mock_post.return_value = mock_response

    result = generate_synthetic_questions("Test tekst", n_questions=3)

    assert len(result) <= 3
    assert all(isinstance(q, str) for q in result)
```

#### Step 9: Test Coverage Goals

Aim for these coverage targets:
- **Critical functions:** 100% (preprocessing, chunking)
- **Business logic:** 90% (retrieval, query parsing)
- **Utility functions:** 80%
- **Overall:** 85%+

Check coverage:
```bash
pytest --cov=. --cov-report=term-missing
```

#### Step 10: Pre-commit Testing Hook

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash

echo "Running tests before commit..."
pytest --quiet

if [ $? -ne 0 ]; then
    echo "Tests failed! Commit aborted."
    exit 1
fi

echo "Tests passed!"
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

**Action Items for Testing:**
- [ ] Install pytest and dependencies
- [ ] Create test directory structure
- [ ] Write `conftest.py` with fixtures
- [ ] Write unit tests for `clean_data.py`
- [ ] Write unit tests for `chunk_data.py`
- [ ] Write unit tests for `query_engine.py`
- [ ] Write mocked tests for API calls
- [ ] Create integration test for full pipeline
- [ ] Set up coverage reporting
- [ ] Add pre-commit hook for tests
- [ ] Aim for 85%+ test coverage

---

## 📋 IMPLEMENTATION CHECKLIST

Use this as your step-by-step guide. Complete in order for best results.

### Phase 1: Critical Bug Fixes (DO THIS FIRST)
- [ ] **Bug #1:** Fix triple upsert in `index_chunks.py:89-161`
- [ ] **Bug #2:** Fix filter logic in `query_engine.py:226-248`
- [ ] **Bug #3:** Remove duplicate function in `raq_query_engine.py:128-174`
- [ ] **Bug #4:** Fix typo in `raq_query_engine.py:247` (meeting_data → meeting_date)
- [ ] **Bug #5:** Decide on BM25 (remove or implement properly)
- [ ] **Bug #6:** Add Qdrant payload indexes in `index_chunks.py`
- [ ] **Security:** Remove credentials from `rag_overview.md`
- [ ] **Security:** Check git history for leaked credentials
- [ ] **Security:** Rotate Qdrant API key if exposed

**Test after Phase 1:** Run embedding on 10 documents only, verify results in Qdrant

### Phase 2: Testing Infrastructure
- [ ] Install pytest and dependencies
- [ ] Create `tests/` directory structure
- [ ] Write `tests/conftest.py` with fixtures
- [ ] Write unit tests for `clean_data.py` (min 5 tests)
- [ ] Write unit tests for `chunk_data.py` (min 5 tests)
- [ ] Write unit tests for `query_engine.py` (min 5 tests)
- [ ] Mock API calls in tests
- [ ] Run tests and verify they pass
- [ ] Set up coverage reporting
- [ ] Achieve 70%+ coverage

**Test after Phase 2:** `pytest --cov=.` should show green passing tests

### Phase 3: Performance Improvements
- [ ] Batch synthetic question generation
- [ ] Add HTTP session pooling
- [ ] Add retry logic with `@retry` decorator
- [ ] Test with 50 documents, measure speed improvement

**Test after Phase 3:** Time full pipeline, should be 30-50% faster

### Phase 4: Code Quality
- [ ] Create `config.py` with all parameters
- [ ] Create `logger.py` and replace all prints
- [ ] Create `validation.py` for input validation
- [ ] Add consistent type hints across all files
- [ ] Run `mypy` for type checking
- [ ] Add docstrings in consistent format

**Test after Phase 4:** Code should be more readable, `mypy` passes

### Phase 5: Data Quality
- [ ] Add deduplication in `chunk_data.py`
- [ ] Add table/list preservation in `clean_data.py`
- [ ] Extract attachment metadata
- [ ] Test on sample documents with tables

**Test after Phase 5:** Check chunk quality manually on 5 documents

### Phase 6: Full Pipeline Test
- [ ] Run full embedding on all 370 documents
- [ ] Verify Qdrant collection has correct count
- [ ] Test 10 sample queries
- [ ] Measure retrieval quality manually
- [ ] Document any issues found

### Phase 7: Advanced Features (Optional)
- [ ] Add reranking with cross-encoder
- [ ] Add query decomposition
- [ ] Add caching layer
- [ ] Add observability/metrics
- [ ] Implement multi-document synthesis

---

## 📚 Learning Resources

To understand the concepts better:

### RAG Fundamentals
- [Anthropic RAG Guide](https://www.anthropic.com/research/retrieval-augmented-generation)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Pinecone Learning Center](https://www.pinecone.io/learn/rag/)

### Testing in Python
- [Pytest Documentation](https://docs.pytest.org/)
- [Real Python - Testing](https://realpython.com/pytest-python-testing/)
- [Test-Driven Development with pytest](https://testdriven.io/blog/tdd-python/)

### Production Best Practices
- [The Twelve-Factor App](https://12factor.net/)
- [Logging Best Practices](https://docs.python.org/3/howto/logging.html)
- [Python Packaging Guide](https://packaging.python.org/)

---

## 💡 Final Thoughts

You've built something genuinely impressive for a first RAG system. The bugs I found are typical of rapid development - they're easy to fix and won't take long.

**Your strengths:**
- Strong understanding of RAG architecture
- Clean, modular code structure
- Good instincts for advanced techniques

**Focus areas:**
- Testing (this is where juniors typically struggle)
- Error handling and edge cases
- Production hardening

Work through this checklist in order. Don't skip to the "fun" advanced features until you've fixed the critical bugs and added tests. That's what separates learning projects from production systems.

Good luck! You're on the right track.

---

**Generated:** 2025-11-18
**Reviewed by:** Claude (Anthropic)

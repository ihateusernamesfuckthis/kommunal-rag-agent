from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
import jsonlines
from pathlib import Path
from dotenv import load_dotenv
import requests
from synthetic import upsert_synthetics, embed_text, embed_batch
f

BASE_DIR = Path("data/greve")
RAW_DIR = BASE_DIR / "raw_dagsordener"
PROCESSED_DIR = BASE_DIR / "dagsordener"
CHUNK_DIR = BASE_DIR / "chunks"

# De her linjer gør projektet reproducerbart - selvom man har oprettet mapperne, så er det stadig et unødvendigt weakpoint for koden
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
SYVAI_TOKEN = os.getenv("SYVAI_TOKEN")

if not SYVAI_TOKEN:
    raise ValueError("SYVAI_TOKEN mangler i .env")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL mangler i .env")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY mangler i .env")


SYV_EMBED_URL = "https://api.syv.ai/v1/embeddings"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def create_or_reset_collection(
        client: QdrantClient,
        collection_name: str = "kommune-rag",
        vector_size: int = 1024
):
    """
    Opretter en qdrant collection hvis den ikke findes.
    Hvis den findes, droppes og genskabes (så man starter rent)
    Qdrant lader ikke en ændre et schema live
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


def upsert_chunk(
        client: QdrantClient,
        collection_name: str,
        chunk_record: dict,
        vector: list
):
    """
    Upserter færdigt embedded chunks til Qdrant.
    """

    point = PointStruct(
        id = chunk_record["chunk_id"],
        vector=vector,
        payload=chunk_record
    )

    client.upsert(collection_name=collection_name, points=[point])


def index_all_chunks(
        client: QdrantClient,
        collection_name: str = "kommune-rag",
        chunk_dir: Path = CHUNK_DIR
):
    """
    Batch indekserer:
    - Alle chunk-jsonl filer
    - Batch embedder chunk-tekster
    - Upserter chunks
    - Upserter syntetiske spørgsmål
    """

    print("--- Starter indeksering af chunks til Qdrant ---")

    chunk_files = list(chunk_dir.glob("*.jsonl"))
    if not chunk_files:
        print("Ingen chunk-filer fundet i /chunks")


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

                if len(batch_texts) >= batch_size:
                    vectors = embed_batch(batch_texts)

                    for vec, rec in zip(vectors, batch_records):
                        upsert_chunk(client, collection_name, rec, vec)
                        chunks_in_file += 1
                        total_chunks += 1

                        num_syn = upsert_synthetics(client, collection_name, rec)
                        total_synthetics += num_syn
                    
                    batch_texts = []
                    batch_records = []
                
                if batch_texts:
                    vectors = embed_batch(batch_texts)

                    for vec, rec in zip(vectors, batch_records):
                        upsert_chunk(client, collection_name, rec, vec)
                        chunks_in_file += 1
                        total_chunks += 1

                        num_syn = upsert_synthetics(client, collection_name, rec)
                        total_synthetics = num_syn

                result = upsert_chunk(client, collection_name, chunk_record)
                if result:
                    total_chunks += 1
                    chunks_in_file += 1

                num_synthetics = upsert_synthetics(client, collection_name, chunk_record)
                total_synthetics += num_synthetics

    print("\n=== Indexing færdig! ===")
    print(f"Chunk-filer behandlet: {total_files}")
    print(f"{total_chunks} chunks indekseret.")
    print(f"{total_synthetics} syntetiske spørgsmål indekseret.")


if __name__ == "__main__":
    create_or_reset_collection(client, "greve_rag", 1024)
    index_all_chunks(client, "greve_rag")


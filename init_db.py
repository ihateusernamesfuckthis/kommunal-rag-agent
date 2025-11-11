"""
Initializes ChromaDB from CSV data if it doesn't exist.
This runs automatically on Render deployment.
"""
import os
import pandas as pd
import chromadb
from embedding import SyvMistralEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("SYVAI_TOKEN")

def init_database():
    """Creates ChromaDB from CSV if it doesn't already exist"""

    # Create ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_data")
    embedding_fn = SyvMistralEmbeddingFunction(token)

    # Check if collection already exists and has data
    try:
        collection = client.get_collection(name="greve_referater", embedding_function=embedding_fn)
        count = collection.count()
        if count > 0:
            print(f"✅ ChromaDB already exists with {count} documents, skipping initialization")
            return
        else:
            print("📊 Collection exists but is empty, re-initializing...")
    except:
        print("🔄 Collection not found, initializing ChromaDB from CSV...")
        collection = client.get_or_create_collection(
            name="greve_referater",
            embedding_function=embedding_fn
        )

    # Load CSV
    CSV_PATH = "data/greve/cleaned_dagsordener.csv"
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"📊 Loading {len(df)} documents from {CSV_PATH}")

    # Insert in batches
    batch_size = 10
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_num = i // batch_size + 1

        try:
            collection.add(
                ids=batch["punkt_id"].astype(str).tolist(),
                documents=batch["tekst"].astype(str).tolist(),
                metadatas=[
                    {
                        "kommune": row["kommune"],
                        "udvalg": row["udvalg"],
                        "dato": row["dato"],
                        "punkt_navn": row["punkt_navn"]
                    }
                    for _, row in batch.iterrows()
                ]
            )
            print(f"✓ Batch {batch_num}/{total_batches} completed")
        except Exception as e:
            print(f"⚠️ Error at batch {batch_num}: {e}")
            continue

    print("✅ ChromaDB initialization complete!")

if __name__ == "__main__":
    init_database()

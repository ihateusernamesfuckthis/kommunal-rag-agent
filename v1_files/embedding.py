import os
import pandas as pd
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb

# --- Setup ---
load_dotenv()
token = os.getenv("SYVAI_TOKEN")

if not token:
    raise ValueError("SYVAI_TOKEN mangler i .env")

class SyvMistralEmbeddingFunction:
    def __init__(self, token):
        self.token = token
        self.url = "https://api.syv.ai/v1/embeddings"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Chroma kalder denne når du bruger collection.query()"""
        return self._embed(input)

    def _embed(self, input: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {
            "input": input,
            "model": "mistral/mistral-embed",
            "encoding_format": "float"
        }

        response = requests.post(self.url, headers=headers, json=data)
        if response.status_code != 200:
            print(f"⚠️ Fejl ({response.status_code}): {response.text[:200]}")
            response.raise_for_status()

        result = response.json()
        return [item["embedding"] for item in result["data"]]

    def name(self):
        return "syvai-mistral-embed"


def main():
    # Opretter chroma vektor database
    client = chromadb.PersistentClient(path="./chroma_data")

    embedding_fn = SyvMistralEmbeddingFunction(token)
    collection = client.get_or_create_collection(
        name="greve_referater",
        embedding_function=embedding_fn
    )

    # Loader data
    CSV_PATH = "data/greve/cleaned_dagsordener.csv"
    df = pd.read_csv(CSV_PATH)
    print(f"Indlæser {len(df)} dokumenter fra {CSV_PATH}")

    # Indsætter i batches
    batch_size = 10

    existing_ids = set(collection.get(include=[])["ids"])

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_ids = batch["punkt_id"].astype(str).tolist()

        if all(p_id in existing_ids for p_id in batch_ids):
            print(f"⏭️  Spring batch {i} over (allerede embedded)")
            continue

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
        except Exception as e:
            print(f"⚠️ Fejl ved batch {i}: {e}")
            continue

    print("✅ Embeddings gemt i ChromaDB")

    # Hurtig test
    query = "affaldshåndtering og genbrug"
    results = collection.query(query_texts=[query], n_results=3)

    print("\n🔎 Eksempel på semantisk søgning:")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"\nUdvalg: {meta['udvalg']}\nDato: {meta['dato']}")
        print(f"Punkt: {meta['punkt_navn']}")
        print(f"Uddrag: {doc[:250]}...")

if __name__ == "__main__":
    main()

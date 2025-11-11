import os
import requests
import chromadb
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from embedding import SyvMistralEmbeddingFunction


load_dotenv()
token = os.getenv("SYVAI_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

embedding_fn = SyvMistralEmbeddingFunction (token)
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection(name="greve_referater", embedding_function=embedding_fn)

def retrieve_context(question, n_results=5):
    """Finder relevante tekst udrag fra Chroma Databasen"""
    results = collection.query(query_texts=[question], n_results=n_results)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    blocks = []
    for doc, meta in zip(docs, metas):
        blocks.append(
            f"Udvalg: {meta['udvalg']}\nDato: {meta['dato']}\nPunkt: {meta['punkt_navn']}\n\n{doc[:1000]}"
        )
    return "\n\n---\n\n".join(blocks)

def ask_danskgpt(question, context):
    """API kald til DanskGPT gennem syv.ai platformen"""
    url = "https://api.syv.ai/v1/chat/completions"
    payload = {
        "model": "syvai/danskgpt-v2.1",
        "messages": [
            {"role": "system", "content": "Du er en hjælpsom dansk assistent, der opsummerer dagsordener og mødereferater fra danske kommuner. Dine svar er korte, tydelige og i et let forståeligt sprog."},
            {"role": "user", "content": f"Spørgsmål: {question}\n\nKontekst:\n{context}\n\nBesvar kort og klart på dansk."}
        ]
    }
    r = requests.post(url, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


app = FastAPI(title="Kommunal Rag API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):
    """Frontend kalder dette endpoint med et spørgsmål"""
    try:
        question = payload.question
        context = retrieve_context(question)
        if not context:
            return {"Answer": "Jeg kunne ikke finde noget relevant i databasen"}
        
        answer = ask_danskgpt(question, context)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/")
def root():
    """Serverer HTML frontend"""
    return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

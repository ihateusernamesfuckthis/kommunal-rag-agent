import requests, os
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("SYVAI_TOKEN")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

data = {
    "model": "syvai/danskGPT-v2.1",
    "messages": [
        {"role": "system", "content": "Du er en hjælpsom dansk assistent."},
        {"role": "user", "content": "Skriv en kort hilsen på dansk."}
    ]
}

r = requests.post("https://api.syv.ai/v1/chat/completions", headers=headers, json=data)
print(r.status_code, r.text[:300])

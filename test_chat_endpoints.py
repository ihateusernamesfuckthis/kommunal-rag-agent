import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("SYVAI_TOKEN")

if not token:
    raise ValueError("⚠️ SYVAI_TOKEN mangler i .env")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Mulige modelnavne at teste
models = [
    "syvai/danskGPT-v2.1",
    "danskGPT-v2.1",
    "syvai/dansk-gpt-v2.1",
    "syvai/danskGPT"
]

url = "https://api.syv.ai/v1/chat/completions"

for model in models:
    print(f"\n🔍 Tester model: {model}")
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Du er en hjælpsom dansk assistent."},
            {"role": "user", "content": "Skriv en kort hilsen på dansk."}
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=20)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            print("✅ Svar:", content)
            break
        else:
            print("Fejl:", r.text[:250])
    except Exception as e:
        print("⚠️ Request fejl:", str(e))

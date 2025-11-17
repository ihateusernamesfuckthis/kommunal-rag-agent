import requests
import json
import os
import time

BASE_URL = "https://dagsordener.greve.dk/api/agenda"
DATA_DIR = "data/greve"


def save_json(data, filepath):
    """Gemmer data som json-fil lokalt"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_udvalgsliste():
    """Henter liste over udvalg i kommunen"""
    url = f"{BASE_URL}/udvalgsliste"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def fetch_dagsorden(moede_id):
    """Henter hele dagsorden for et møde"""
    url = f"{BASE_URL}/dagsorden/{moede_id}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()
    
def main():
    os.makedirs(f"{DATA_DIR}/dagsordener", exist_ok=True)

    data = fetch_udvalgsliste()
    udvalg = data["Udvalg"]["Aktuelle politiske udvalg"]
    print (f"Hentede {len(udvalg)} udvalg")

    for u in udvalg:
        udvalg_navn = u["Navn"]
        udvalg_id = u["Id"]
        moeder = u.get("Moeder", [])
        print(f"\n--- {udvalg_navn} ({len(moeder)} møder) ---")

        for m in moeder:
            moede_id = m["Id"]
            dato = m.get("Dato", "ukendt_dato")
            path = f"{DATA_DIR}/dagsordener/{udvalg_navn}_{dato}_{moede_id}.json"

            if os.path.exists(path):
                continue

            try:
                dagsorden = fetch_dagsorden(moede_id)
                save_json(dagsorden, path)
                print(f"Gemte {udvalg_navn} ({dato})")
                time.sleep(0.2)
            except Exception as e:
                print(f"fejl ved {moede_id}: {e}")

if __name__ == "__main__":
    main()



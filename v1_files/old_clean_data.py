import os
import json
import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = "data/greve/dagsordener"
OUTPUT_FILE = "data/greve/cleaned_dagsordener.csv"

def extract_text_from_html(html_content):
    """Fjerner HTML tags og returnerer ren tekst"""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())

def parse_json_file(filepath):
    """Parser en dagsorden-fil og returnerer liste af tekst-punkter"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Vælger kommune 
    kommune = "Greve"

    # Renser udvalg
    udvalg_data = data.get("Udvalg", {})
    if isinstance(udvalg_data, dict):
        udvalg = udvalg_data.get("Navn", "Ukendt udvalg")
    else:
        udvalg = udvalg_data or "Ukendt udvalg"

    # Renser dato
    moede_info = data.get("Moede") or {}
    dato = moede_info.get("Dato") or data.get("Dato") or data.get("MødeDato", "Ukendt dato")


    punkter = []
    dagsordenpunkter = data.get("Dagsordenpunkter", [])
    for p in dagsordenpunkter:
        navn = p.get("Navn", "Ukendt punkt")
        felter = p.get("Felter", [])

        tekstsammen = " ".join(
            extract_text_from_html(f.get("Html", "")) for f in felter if f.get("Html")
        ).strip()

        if not tekstsammen:
            continue

        punkt_id = f"{udvalg}_{dato}_{navn[:40]}".replace(" ", "_")

        punkter.append({
            "punkt_id": punkt_id,
            "kommune": kommune,
            "udvalg": udvalg,
            "dato": dato,
            "punkt_navn": navn,
            "tekst": tekstsammen
            })
        
    return punkter

def main():
    alle_punkter = []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    print(f"Behandler {len(files)} filer")

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            parsed_punkter = parse_json_file(filepath)
            alle_punkter.extend(parsed_punkter)
        except Exception as e:
            print(f"⚠️ Fejl i {filename}: {e}")
    
    df = pd.DataFrame(alle_punkter)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"✅ Færdig! Gemte {len(df)} rensede punkter i {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

import os
import json
import html
import re
import unicodedata

# json support -> understøtter RAG systemer
import jsonlines
from pathlib import Path

# bruges til at konvertere HTML til ren tekst
from bs4 import BeautifulSoup

# dato parsin og normalisering
from datetime import datetime


# Rodmappe til databehandling
BASE_DIR = Path("data/greve")

# Denne folder er den eneste der indeholder de rå data der er hentet via API kaldet fra fetch_data.py
RAW_DIR = BASE_DIR / "raw_dagsordener"

# Denne folder indeholder output fra første fase af preprocessing -> et JSONL dokument pr dagsorden
PROCESSED_DIR = BASE_DIR / "dagsordener"


CHUNK_DIR = BASE_DIR / "chunks"

# De her linjer gør projektet reproducerbart - selvom man har oprettet mapperne, så er det stadig et unødvendigt weakpoint for koden
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_document(path: Path) -> dict:
    """
    Loader et råt JSON dokument fra kommunenes API.
    Returnerer en dict med rå data eller None hvis der opstår en fejl
    """

    # Fejlhåndtering: Filen findes ikke -> stop tidligt og undgå FileNotFound
    if not path.exists():
        print(f"Filen findes ikke: {path}")
        return None
    
    try:
        # åbner som UTF-8 (normal encoding)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Fejlhåndtering: Json indhold er ikke et objekt -> dokument kan ikke behandles
        if not isinstance(data, dict):
            print(f"Uventet JSON struktur i {path.name}")
            return None
        
        return data
    
    # Fejlhåndtering: Filen er korrupt -> undgår crash
    except json.JSONDecodeError as e:
        print(f"JSON parsing fejl i {path.name}: {e}")
        return None
    
    # Fejlhåndtering: Encoding fejl -> fallback til latin-1 som før er brugt af kommunale system
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="latin-1") as f:
                data = json.load(f)
                return data
        except Exception:
            # Fallback mislykkedes også - filen kan ikke læses
            print(f"Kan ikke læse fil - encoding fejl: {path.name}")
            return None
    
    # Fejlhåndtering: Ukendte og uventede fejl -> vi 'fanger' dem for at holde pipeline kørende
    except Exception as e:
        print(f"Ukendt fejl ved load af {path.name}: {e}")
        return None
    

def extract_base_metadata(raw_doc: dict, path: Path) -> dict:
    """
    Udtrækker top-level metadata fra et råt kommunalt json dokument.
    Returnerer en dict med ren, struktureret og forudsigelig metadata containers.
    Funktionen har et overordnet formål om at skabe en masse "base containers" der udfyldes med metadata
    """

    # Metadata container til dokument id -> fallback er filnavnet
    document_id = raw_doc.get("Id") or path.stem

    # Metadata container til dokumenttypen
    document_type = raw_doc.get("Dokumenttype", "dagsorden").lower()

    # Metadata container til udvalg/komiteer 
    committee = None
    udvalg_data = raw_doc.get("Udvalg") or raw_doc.get("Committee")
    if isinstance(udvalg_data, dict):
        committee = udvalg_data.get("Navn") or udvalg_data.get("Name")
    elif isinstance(udvalg_data, str):
        committee = udvalg_data
    else:
        committee = "Ukendt udvalg"

    # Metadata container til mødedatoer

    meeting_date_raw = (
        raw_doc.get("MødeDato")
        or raw_doc.get("Dato")
        or raw_doc.get("Moede", {}).get("Dato")
        or None
    )

    meeting_date = None
    if meeting_date_raw:
        try:
            meeting_date = datetime.fromisoformat(meeting_date_raw.replace("Z", "+00:00"))
        except Exception:
            meeting_date = meeting_date_raw
        
    title = (
        raw_doc.get("Titel")
        or raw_doc.get ("Title")
        or f"Dagsorden - {committee}"
    )

    agenda_items = raw_doc.get("Dagsordenpunkter") or raw_doc.get("Punkter") or []    

    return {
        "document_id": document_id,
        "document_type": document_type,
        "committee": committee,
        "meeting_date": meeting_date,
        "title": title,
        "source_path": str(path),
        "raw_agenda_items": agenda_items
    }

def clean_html_to_text(html_content: str) -> str:
    """
    Dette er en helper-funktion der konverterer HTML til ren tekst og normaliserer whitespace.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    # Normaliser whitespace og unicode (fjerner dobbeltspaces, specialtegn osv.)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def extract_agenda_sections(raw_meta: dict) -> list:
    """
    Parser dagsordenpunkter og udtrækker disse sektioner:
    Resume, Baggrund og Indstilling, Beslutning, Tekst, Felter.html.
    Returnerer en liste af strukturede sektioner
    """

    sections = []

    agenda_items = raw_meta.get("raw_agenda_items", [])
    document_id = raw_meta["document_id"]

    committee = raw_meta["committee"]
    meeting_date = raw_meta["meeting_date"]
    # Konverter datetime til string for JSON serialisering
    if isinstance(meeting_date, datetime):
        meeting_date = meeting_date.isoformat()
    source_path = raw_meta["source_path"]

    for idx, item in enumerate(agenda_items):

        # Titel og metadata
        section_title = item.get("Navn") or item.get("Title") or "Ukendt punkt"
        case_number = item.get("Sagsnummer") or item.get("CaseNumber")

        # Udtrækker sektioner
        raw_resume = item.get("Resume")
        raw_background = item.get("BaggrundOgIndstilling")
        raw_decision = item.get("Beslutning")
        raw_text = item.get("Tekst")

        # Felter.Html (typiske html fragmenter)
        felter_html = ""
        felter = item.get("Felter", [])
        if isinstance(felter, list):
            html_blocks = []
            for f in felter:
                if f.get("Html"):
                    html_blocks.append(f.get("Html"))
            felter_html = " ".join(html_blocks)
        
        # Liste over sektioner med label og rå tekst
        raw_sections = [
            ("resume", raw_resume),
            ("background", raw_background),
            ("decision", raw_decision),
            ("text", raw_text),
            ("html_fragment", felter_html),
        ]

        for section_type, raw_content in raw_sections:
            if not raw_content:
                continue

            cleaned_text = clean_html_to_text(raw_content)

            if cleaned_text:
                sections.append({
                    "document_id": document_id,
                    "committee": committee,
                    "meeting_date": meeting_date,
                    "section_type": section_type,
                    "section_title": section_title,
                    "case_number": case_number,
                    "agenda_index": idx,
                    "text": cleaned_text,
                    "source_path": source_path
                })

    return sections
            
        
def write_processed_jsonl(raw_meta: dict, sections: list):
    """
    Skriver et preprocessed dokument til JSONL-format.
    Et råt dokumentt = en JSONL fil
    En sektion = en linje i filen
    """

    document_id = raw_meta["document_id"]
    output_path = PROCESSED_DIR / f"{document_id}.jsonl"

    if not sections:
        print(f"Ingen sektioner fundet i dokumentet: {document_id}")
        return
    
    with jsonlines.open(output_path, mode="w") as writer:
        for sec in sections:
            writer.write(sec)
    
    print (f"Skrev {len(sections)} sektioner -> {output_path.name}")


def preprocess_all_documents():
    """
    Kører hele preprocess-pipeline:
    1. Load rå JSON
    2. Extract base metadata
    3. Extract agenda sections
    4. Write JSONL output
    """
    
    print("=== Starter preprocessing af dagsordener ===")

    files = list(RAW_DIR.glob("*.json"))
    if not files:
        print("ingen rå json dokumenter fundet")
        return

    total_docs = 0
    total_sections = 0

    for file in files:
        print(f"\n--- Behandler: {file.name} ---")

        # step 1 -> loader rå dokument
        raw_doc = load_raw_document(file)
        if raw_doc is None:
            print (f"Springer over (kan ikke læse): {file.name}")
            continue
        
        # step 2 -> ekstraherer metadata
        raw_meta = extract_base_metadata(raw_doc, file)

        # step 3 -> ekstraherer sektioner
        sections = extract_agenda_sections(raw_meta)

        if not sections:
            print(f"Ingen sektioner fundet i dokument: {file.name}")
            continue

        # step 4 -> skriver JSONL output
        write_processed_jsonl(raw_meta, sections)
        print (f"Parsed {len(sections)} sektioner")

        total_docs += 1
        total_sections += len(sections) 

    print("\n=== Preprocessing færdig! ===")
    print(f"📄 Dokumenter behandlet: {total_docs}")
    print(f"📦 Sektioner genereret : {total_sections}")

if __name__ == "__main__":
    preprocess_all_documents()

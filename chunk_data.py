from pathlib import Path
from datetime import datetime
import jsonlines
import re

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


def approximate_token_count(text: str) -> int:
    """
    Det her er en meget simpel "tokenizer".
    Estimerer antal tokens baseret på antal ord.
    Bruges til at styre størrelsen på chunks uden en tung tokenizer
    """
    if not text:
        return 0

    # Simpel estimering: antal ord som proxy for tokens
    # Typisk er 1 token ≈ 0.75 ord, men vi bruger 1:1 for at være konservative
    words = text.split()
    return len(words)

def split_text_into_sentences(text: str) -> list:
    """
    Splitter tekst i sætninger baseret på punktum, spørgsmåltegn og udråbstegn.
    Tegnsætningen bliver bevaret
    """

    if not text:
        return []
    
    # Regex-forklaring:

    # (?<=[.!?]) -> splitter efter ., ! eller ?
    # \s+        -> ét eller flere mellemrum
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Fjerner tomme elementer og whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def chunk_text(
        text: str,
        max_tokens: int = 1500,
        overlap_ratio: float = 0.15
) -> list:
    
    """
    Chunker tekst i semantiske stykker baseret på sætniger.
    max_tokens er sat efter mistral embedding-modellers konvertering, som er omkring 4k tokens.
    overlap_ratio sikrer at vi ikke mister kontekst mellem chunks - der er altså et overlap på 15% af hver chunks indhold
    """

    if not text:
        return []
    
    sentences = split_text_into_sentences(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentences_tokens = approximate_token_count(sentence)

        # Hvis en sætning er længere end max_tokens -> hårdt split
        if sentences_tokens > max_tokens:
            chunks.append(sentence.strip())
            continue

        # 1/2 Hvis vi ikke kan tilføje sætningen til den nuværende chunk -> afslut chunk
        if current_tokens + sentences_tokens > max_tokens:
            chunk_text = " ".join(current_chunk).strip()
            chunks.append(chunk_text)

            # Laver overlap mellem chunks
            overlap_tokens = int(max_tokens * overlap_ratio)
            overlap_words = chunk_text.split()[-overlap_tokens:]

            # starter næste chunk med overlap + ny sætning
            current_chunk = overlap_words + [sentence]
            current_tokens = len(overlap_words) + sentences_tokens
        
        else:
            # 2/2 ellers tilføjer vi sætningen til nuværende chunk
            current_chunk.append(sentence)
            current_tokens += sentences_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    
    return chunks

def normalize_committee(name: str) -> str:
    """
    Normaliserer udvalg-navne til lowercase uden specialtegn - forbedrer retrieval.
    Eksempel er "Plan- og Teknikudvalget" -> "planteknik"
    """

    if not name:
        return "unknown"
    
    name = name.lower()
    name = name.replace(" ", "")
    name = name.replace("-", "")
    name = name.replace("&", "og")
    name = name.replace("udvalget", "")
    return name.strip()

def add_metadata_fields(chunk: dict) -> dict:
    """
    Tilføjer ekstra metadata til chunk:
    - meeting_year
    - meeting_month
    - comittee_normalized
    - section_category
    - is_decision_section
    - is_background_section
    - document_period
    """

    meeting_date = chunk.get("meeting_date")
    meeting_year = None
    meeting_month = None

    if meeting_date:
        try:
            dt = datetime.fromisoformat(meeting_date.replace("Z", "+00:00"))
            meeting_year = dt.year
            meeting_month = dt.month
        except:
            pass
    
    committee = chunk.get("committee")
    committee_norm = normalize_committee(committee)

    st = chunk.get("section_type")

    if st in ["resume", "background", "decision"]:
        section_category = "core"
    
    else:
        section_category = "other"

    is_decision = st == "decision"
    is_background = st == "background"
    is_html = st == "html_fragment"

    document_period = meeting_year or 0

    chunk["meeting_year"] = meeting_year
    chunk["meeting_month"] = meeting_month
    chunk["committee_normalized"] = committee_norm
    chunk["section_category"] = section_category
    chunk["is_decision_section"] = is_decision
    chunk["is_background_section"] = is_background
    chunk["is_html_fragtment"] = is_html
    chunk["document_period"] = document_period

    return chunk

def process_jsonl_file_into_chunks(input_path: Path, output_dir: Path = CHUNK_DIR):
    """
    Læser én preprocessed JSONL-fil fra /dagsordener.
    Hver sektion bliver chunked og chunked resultat skrives til en ny JSONL-fil i /chunks
    """

    # Ekstraherer dokument-id ud fra filnavnet (uden .jsonl til sidst)
    # bruges til chunk_id og outputfilens navn
    document_id = input_path.stem

    # Definerer hvor det chunkede resultat skal gemmes
    output_path = output_dir / f"{document_id}.jsonl"

    chunks_written = 0

    # Åbner inpu (til læsning) og output (til at skrive ) filerne
    with jsonlines.open(input_path, "r") as reader, \
         jsonlines.open(output_path, "w") as writer:
        
        # iterer over hver sektion i dagsordenens JSONL fil
        for section in reader:

            # Henter sektionens teksts (selve indholdet der skal chunkes)
            section_text = section.get("text", "")

            # Henter metadata som skal følge med hver chunk
            section_type = section.get("section_type")
            section_title = section.get("section_title")
            committee = section.get("committee")
            meeting_date = section.get("meeting_date")
            case_number = section.get("case_number")
            agenda_index = section.get("agenda_index")
            source = section.get("source_path")

            # Chunker selve teksen - her bruger vi den funktion vi byggede ovenover
            text_chunks = chunk_text(section_text)

            # For hver chunk, opretter vi en ny JSONL-linje med metadata
            for i, chunk in enumerate(text_chunks):

                chunk_record = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_{agenda_index}_{section_type}_chunk{i}",
                    "section_type": section_type,
                    "section_title": section_title,
                    "committee": committee,
                    "meeting_date": meeting_date,
                    "case_number": case_number,
                    "agenda_index": agenda_index,
                    "text": chunk,
                    "source_path": source,
                }

                # Her samler vi metadata + selve chunk teksten i et objekt
                chunk_record = add_metadata_fields(chunk_record)

                # Skriv chunket til output JSONL
                writer.write(chunk_record)
                chunks_written += 1
    
    # mest bare til logging -> skriver hvor mange chunks der blev generet fra det ene dokument
    print(f"{document_id}: {chunks_written} chunks skrevet til {output_path.name}")


def process_all_chunks(input_dir: Path = PROCESSED_DIR, output_dir: Path = CHUNK_DIR):
    """
    Chunker alle preprocessed JSONL-dokumenter fra /dagsordener
    og gemmer resultatet i /chunks
    """

    files = list(input_dir.glob("*.jsonl"))

    if not files:
        print("Ingen preprocessed JSONL filer fundet i /dagsordener")
        return
    
    total_docs = 0
    total_chunks = 0

    print("=== Starter chunking af preprocessed dagsordener ===")

    # Looper over alle filer, chunker en dagsorden ad gangen
    for file in files:
        print(f"\n--- Chunker dokument: {file.name} ---")

        process_jsonl_file_into_chunks(file, output_dir)
        total_docs += 1

        # beregner hvor mange chunks det specifikke dokument tilføjede til /chunks
        chunks_after = 0
        try:
            with jsonlines.open(output_dir / file.name, "r") as reader:
                for _ in reader:
                    chunks_after += 1
        except FileNotFoundError:
            print(f"⚠️ Ingen output for {file.name} — blev filen skrevet korrekt?")
        
        total_chunks += chunks_after
        print(f"⮑ {chunks_after} chunks genereret fra {file.name}")
    
    print("\n=== Chunking færdig! ===")
    print(f"📄 Dokumenter chunket: {total_docs}")
    print(f"📦 Total antal chunks: {total_chunks}")

if __name__ == "__main__":
    process_all_chunks()


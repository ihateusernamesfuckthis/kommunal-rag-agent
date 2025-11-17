qdrant key -> eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.rAUsOqeI7PVwI7-DaGFlrGgn0DltPmC8wJraZcVBCog

qdrant endpoint -> https://9cb490fe-535e-4c3b-8977-d2ad1f6d243f.eu-central-1-0.aws.cloud.qdrant.io



# RAG System Upgrade Overview

Denne fil giver et overblik over de vigtigste komponenter i et moderne Retrieval-Augmented Generation (RAG) system. Hver sektion forklarer kort, hvad emnet handler om, og hvordan forbedringen påvirker det samlede system.

---

## 1. Preprocessing

**Formål:** Gøre rå tekst mere struktureret, forståelig og maskinlæsbar før chunking, embeddings og retrieval.

Preprocessing er det vigtigste lag i hele RAG-systemet. Alle downstream-features (chunking, hybrid search, syntetiske spørgsmål, periodematching, multi-doc synthesis) fungerer kun korrekt, hvis preprocessing skaber **rene**, **strukturerede** og **metadata-rige** dokumenter.

**Preprocessing skal opnå følgende:**

### **1. Fjernelse af støj (Noise Reduction)**

* Fjern sidehoveder, sidefødder, sidetal, PDF-artefakter.
* Fjern gentagne sektioner (fx "Greve Kommune – Teknik & Miljø" på hver side).
* Normaliser whitespace, punktopstillinger og linjeskift.
* Kombinér linjer som hører sammen, så sætninger ikke bliver fragmenterede.

### **2. Strukturel segmentering (Logical Structure Parsing)**

* Identificér og bevar dokumentets **hierarki**:

  * dokumenttitel
  * udvalg
  * sagsnummer
  * sektionstitler
  * underafsnit
  * bullet lists
* Konverter dette til en struktureret JSON-struktur.

Eksempel:

```json
{
  "document_title": "Referat – Teknik & Miljø",
  "sections": [
    {
      "title": "Cykelstier",
      "content": "...",
      "subsections": [ ... ]
    }
  ]
}
```

### **3. Dokumentklassificering (Document Typing)**

For hvert dokument skal du identificere:

* referat
* dagsorden
* bilag
* projektoversigt / statusdokument
* rapport

Dette lag er kritisk for hybrid search og multi-document synthesis.

### **4. Metadata-ekstraktion (Metadata Extraction)**

Preprocessing skal udtrække og ensrette metadata, fx:

* `doc_date`
* `period_start` / `period_end` (for oversigter)
* `udvalg`
* `sagsnummer`
* `source_url`
* `document_type`
* `attachment_links`

Alt skal gemmes i standardiserede formater.

### **5. Normalisering (Text Normalization)**

* Konverter formatering (fx PDF → plaintext → JSON).
* Ret OCR-fejl.
* Ensret specialtegn, citationstegn, punktopstillinger.
* Sørg for korrekt unicode og encoding.

### **6. Semantik-bevaring (Contextual Integrity)**

Sørg for at **hver tekstblok giver mening alene**, fx:

* tilføj sektionstitel til chunk metadata
* tilføj dokumenttitel
* tilføj udvalg

Dette gør chunking og embedding mere meningsfuld.

### **7. Forberedelse til context-aware chunking**

Preprocessing skal sikre:

* tydelige grænser for afsnit
* klare sektionstitler
* ingen ulogiske merges af tekst
* ingen fragmenterede sætninger

Chunkeren må aldrig "gætte" – den skal få perfekte input.

---

God preprocessing skaber **rene, strukturerede, timestampede og semantisk forståelige dokumenter**, som gør alle senere features markant bedre og mere stabile.

---

## 2. Konvertering til JSONL

**Formål:** Gemme dokumenter i et format, der naturligt understøtter metadata og nested struktur.

**Hvad det bidrager med:**

* Ét dokument per linje → nem versionering og streaming.
* Understøtter metadata (fx titel, dato, udvalg, bilag).
* Bedre kompatibilitet med moderne RAG frameworks (HF, LangChain, LlamaIndex).

JSONL er standarden i RAG systemer.

---

## 3. Context-Aware Chunking

**Formål:** Opdele dokumenter i chunks baseret på semantik i stedet for faste længder.

**Hvad det bidrager med:**

* Chunks følger naturlige grænser (fx ny overskrift).
* Bevarer kontekst i hver chunk.
* Reducerer risiko for at modellens svar bliver uklare pga. forkerte split.

Smartere chunking → markant bedre recall og svarpræcision.

---

## 4. Hybrid Search (Semantic + BM25)

**Formål:** Kombinere styrkerne fra semantisk søgning og klassisk nøgleordsbaseret søgning.

**Hvad det bidrager med:**

* Embeddings forstår parafraser og betydning.
* BM25 fanger specifikke termer som navne, adresser og paragraffer.
* Kombineret score giver robust dokumentretrieval.

Hybrid search matcher virkelige brugerbehov langt bedre end kun embeddings.

---

## 5. Syntetiske Spørgsmål

**Formål:** Øge systemets evne til at forstå brugerens spørgsmål ved at generere eksempler på potentielle queries.

**Hvad det bidrager med:**

* Fremhæver, hvad chunken *kan* besvare.
* Forbedrer BM25 ved at tilføje relevant kontekst.
* Gør det lettere for systemet at matche borgeres formuleringer.

Især nyttigt i kommunale referater, hvor sproget ofte er implicit.

---

## 6. Kildehenvisninger og Bilag

**Formål:** Muliggøre at brugeren kan navigere tilbage til originalt dokument og eventuelle bilag.

**Hvad det bidrager med:**

* Øger transparens og troværdighed.
* Gør UI mere brugbart (fx "Åbn bilag 2").
* Gemmes som metadata og bruges i frontend.

Kritisk for myndighedsafhængige RAG systemer.

---

## 7. Metadata Filtering

**Formål:** Filtrere søgning baseret på strukturelle metadata.

**Hvad det bidrager med:**

* Mere præcis søgning.
* Mulighed for facettering (fx udvalg, årstal, sagsnummer).
* Hurtigere retrieval når dataset vokser.

Gør søgesystemet mere fleksibelt og skalerbart.

---

## 8. Overordnet Arkitekturforbedring

Samlet set opgraderer disse forbedringer systemet fra en simpel "embedding + query" pipeline til et mere robust, skalerbart og præcist RAG-system, der matcher moderne best practices.

Effekten er tydelig på:

* Recall
* Svarpræcision
* Brugeroplevelse
* Skalerbarhed
* Dokumentation og maintainability.

---

*Denne fil fungerer som en hurtig reference, når du arbejder i dybden med hver komponent i systemet.*

## 9. Multi-Document Retrieval & Synthesis (Strategy B)

**Formål:** Håndtere situationer hvor flere dokumenter indeholder overlappende eller modsatrettet information — fx ældre referater og nyere statusdokumenter.

**Hvad det bidrager med:**

* Systemet henter **flere relevante dokumenter** (fx top 5–10) i stedet for ét.
* LLM'en får alle dokumenter samtidigt og instrueres i at:

  * identificere forskelle
  * prioritere nyere information
  * markere forældede oplysninger
  * sammenfatte den mest korrekte og aktuelle status.
* Gør RAG systemet robust overfor:

  * informationskonflikter
  * opdateringer
  * bilag der kun udgives periodisk

Dette skaber en mere "menneskelig" forståelse af datasættet — modellen lærer at krydstjekke, sammenligne og lave et samlet konkluderende svar baseret på alle kilder i stedet for at stole på ét enkelt dokument.

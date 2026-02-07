# Current Data Extraction Workflow

## Overview
A **dual-mode intelligent data extraction pipeline** with two strategies:

| Mode | Description | Speed | Entity Coverage | Relationship Coverage |
|------|-------------|-------|-----------------|----------------------|
| **Traditional** (7-layer) | NER + proximity + LLM → schema filter | ⚡ Fast (~60s) | 100% | 71–100% |
| **Semantic** (embedding-first) | Chunk → embed → per-chunk LLM → cross-chunk discovery | 🔬 Deep (~180s) | 94% | 86% |

Both modes use LangGraph orchestration, LLM-powered extraction, and multi-store persistence (Weaviate + NebulaGraph).

---

## Dual-Mode Architecture

### **Mode Selection**
```python
# Traditional (default — fast, proven)
pipeline = IntegratedPipeline(mode="traditional")

# Semantic (embedding-first — deeper cross-chunk discovery)
pipeline = IntegratedPipeline(mode="semantic")
```

---

## Semantic Pipeline (NEW — `semantic_extraction/`)

### **Why Semantic?**
Instead of sending the full document to the LLM in one shot, the semantic pipeline:
1. **Embeds** text into vector space → captures *meaning*, not just keywords
2. **Extracts per-chunk** → LLM gets focused 3-sentence windows → higher recall
3. **Deduplicates via cosine similarity** → "Meridian" ≈ "Meridian Infrastructure Solutions Pvt. Ltd."
4. **Discovers cross-chunk relationships** → finds implicit connections spanning paragraphs

### **Semantic Pipeline Flow**
```
┌─────────────────┐
│ Unstructured Text│
└───────┬─────────┘
        ▼
┌──────────────────────────────────┐
│ Step 1: Sentence Splitter        │  Abbreviation-aware (Pvt., Ltd., Dr.)
│         → Overlapping Chunks     │  3-sentence windows, 1-sentence overlap
│         → Sentence-Transformer   │  all-MiniLM-L6-v2 → 384-dim vectors
└───────┬──────────────────────────┘
        ▼
┌──────────────────────────────────┐
│ Step 2: Per-Chunk LLM Extraction │  Focused context → fewer missed entities
│         (Ollama llama3)          │  9 chunks processed independently
└───────┬──────────────────────────┘
        ▼
┌──────────────────────────────────┐
│ Step 3: Embedding-Based Dedup    │  Cosine similarity > 0.75 → merge
│         67 mentions → 34 unique  │  Keeps longer (more specific) form
└───────┬──────────────────────────┘
        ▼
┌──────────────────────────────────┐
│ Step 4: Semantic Relationship    │  Co-occurring entities in same chunk
│         Discovery                │  Constrained by VALID_TRIPLES schema
└───────┬──────────────────────────┘
        ▼
┌──────────────────────────────────┐
│ Step 5: Cross-Chunk Implicit     │  Entity pairs in DIFFERENT chunks
│         Relationships            │  whose chunk embeddings are similar
│         (cosine ≥ 0.45)          │  → infer indirect relationships
└───────┬──────────────────────────┘
        ▼
┌──────────────────────────────────┐
│ Weaviate (vectors) + NebulaGraph │
└──────────────────────────────────┘
```

### **Key Files**
- `semantic_extraction/semantic_chunker.py` — Sentence splitting, overlapping windows, embedding, Weaviate chunk storage
- `semantic_extraction/semantic_extractor.py` — Per-chunk extraction, embedding dedup, cross-chunk relationship discovery

### **Dependencies**
- `sentence-transformers` — all-MiniLM-L6-v2 (384-dim, ~90 MB)
- `numpy` — cosine similarity computation
- Weaviate — vector storage with `nearVector` search

---

## Complete Workflow Architecture

### **Entry Point: `server.py` (Flask REST API)**
- Listens on `http://localhost:5000`
- Exposes endpoints for extraction, storage, and querying
- CORS-enabled for React frontend (`text-data-weaver/`)

---

## Layer-by-Layer Pipeline

### **Phase 0: Text Input**
User provides unstructured business text (contracts, agreements, reports)

### **Phase 1: LangGraph Agentic Workflow** (`agentic_workflow/workflow.py`)
Built with `langgraph.graph.StateGraph` — orchestrates the entire pipeline

**Node 1 – Extraction Node**
- Calls `entity_extraction/entity_extractor.py → extract_from_text()`
- Processes 4 extraction strategies (ordered by priority):
  1. **LLM Extraction** (Primary)
     - Model: `meta-llama/llama-3.3-70b-instruct:free` (via OpenRouter)
     - Fallback: `liquid/lfm-2.5-1.2b-thinking:free` 
     - API Key: `utils/.env → data_extraction_LiquidAi_api_key`
  2. **Regex Fallback** (Only if LLM fails)
  3. **Validation & Type Reclassification**
     - Uses `utils/domain_schema.py → identify_entity_type()`
     - Fixes misclassifications (e.g., company names wrongly marked as persons)
  4. **Returns** `ExtractionResult` with list of `Entity` objects

**Node 2 – Validation Node**
- Validates entity structure (type, value, confidence)
- Checks confidence threshold ≥ 0.3
- Marks workflow state: `validation_passed = True/False`

**Node 3 – Storage Node**
- Prepares final result for downstream storage
- Logs entity count and status

---

### **Phase 2: 7-Layer Relationship Generation** (`integration_demo/integrated_pipeline.py`)
After entities are extracted, the `IntegratedPipeline` class generates relationships:

**Layer 1 – Candidate Extraction**
- ✅ Already done by entity extractor

**Layer 2 – Type Identification & Deduplication**
- Reclassifies entity types using domain schema
- Deduplicates similar entities (merge "Acme Corp" + "ACME CORPORATION")
- Groups entities by type (person, organization, location, project, invoice, agreement, role)

**Layer 3 – Relation Mapping (3 methods)**
- **3a. LLM-based** (Primary)
  - Prompt: "Extract relationships from this text using VALID_TRIPLES"
  - Returns structured relationships
  - Handles: WORKS_AT, MANAGES, PARTNERS_WITH, etc.

- **3b. Proximity-based** (Co-occurrence)
  - Splits text into sentences
  - For each VALID_TRIPLE in schema:
    - Finds entities of source & target type in same/nearby sentences
    - Tight window (2 sentences) for WORKS_AT (person→org)
    - Wider window (3 sentences) for other relationships
  - Avoids cartesian products via schema constraints

- **3c. Verb-pattern Matching** (NLP-lite)
  - Regex patterns: "X works at Y", "X manages Y", "X is based in Y"
  - Lower confidence than LLM

**Layer 4 – Domain Schema Filter**
- Validates each relationship triple against `VALID_TRIPLES` (30+ allowed triples)
- Example valid: (person, WORKS_AT, organization)
- Example invalid: (organization, WORKS_AT, organization) → REJECTED

**Layer 5 – Confidence Scoring & Threshold**
- Scores each relationship 0.0–1.0
- Filters by thresholds:
  - `REL_ACCEPT_THRESHOLD = 0.65` → auto-accept
  - `REL_DISCARD_THRESHOLD = 0.45` → auto-discard
  - Between: mark as "needs_review"

**Layer 6 – Deduplication**
- Merges duplicate relationships (same source, type, target)
- Keeps highest confidence score

**Layer 7 – Output**
- Returns cleaned, deduplicated relationships ready for graph storage

---

## Multi-Store Persistence

### **Phase 3: Weaviate (Vector Database)**
- `vector_database/weaviate_handler.py`
- Stores documents with embeddings
- Enables semantic search
- Port: 8080 (browser) / 50051 (gRPC)

### **Phase 4: NebulaGraph (Knowledge Graph)**
- `knowledge_graph/nebula_handler.py`
- Stores entities as nodes
- Stores relationships as edges
- Runs within Docker (`nebula-docker-compose/`)
- Enables graph traversal queries

---

## Model Strategy (Updated)

**OpenRouter LLM Handler** (`utils/llm_handler.py`)
- **Priority List** (automatic fallback):
  1. `meta-llama/llama-3.3-70b-instruct:free` (Llama 3.3 70B) ← **PRIMARY**
  2. `liquid/lfm-2.5-1.2b-thinking:free` (Liquid LFM) ← **FALLBACK**

- **Fallback Logic**: 
  - If Llama returns HTTP error (429, 402, 5xx) → tries Liquid
  - If Llama times out → tries Liquid
  - If Llama returns empty content → tries Liquid
  - If all models fail → raises error

- **Both Support**:
  - Sync: `llm.chat(prompt)` (blocks until response)
  - Async: `llm.stream_chat(prompt)` (streams tokens as AsyncGenerator)

- **Auto-tracking**: `llm.active_model` shows which model was used

---

## Domain Schema Layer** (`utils/domain_schema.py`)

**30+ VALID_TRIPLES** define allowed relationships:
```
(person, WORKS_AT, organization)
(person, BASED_IN, location)
(person, LEADS, organization)
(organization, PARTNERS_WITH, organization)
(organization, LOCATED_IN, location)
(organization, MANAGES, project)
(invoice, HAS_AMOUNT, amount)
(agreement, PARTY_TO, organization)
... (and 22 more)
```

**Entity Type Identification**
- Regex patterns for person names, companies, locations
- Organization indicators: "Ltd", "Inc", "Corp", "Solutions", "Infrastructure", "Engineering", "Services"
- Role indicators: "Manager", "Director", "CEO", "CFO", "Architect"
- Location indicators: cities, states, countries

---

## Test Scripts (Verification Tools)

| Test | File | Purpose |
|------|------|---------|
| **LLM Connectivity** | `test_llama.py` | Verify Llama 3.3 70B is reachable |
| **LangGraph Workflow** | `test_langgraph.py` | Verify state machine orchestration |
| **Full Pipeline** | `test_llm_extraction.py` | 7-layer pipeline quality check |

---

## Data Flow Diagram

```
User Input (unstructured text)
        ↓
    [LangGraph Workflow]
        ├─ Extraction Node
        │   ├─ LLM (Llama 3.3 70B)
        │   ├─ Fallback (Liquid LFM)
        │   └─ Regex (if both fail)
        ├─ Validation Node
        │   └─ Type reclassification (domain schema)
        └─ Storage Node
        ↓
    [7-Layer Relationship Pipeline]
        ├─ Layer 2: Type ID + Dedup
        ├─ Layer 3: LLM + Proximity + Verb matching
        ├─ Layer 4: Schema validation
        ├─ Layer 5: Confidence scoring
        ├─ Layer 6: Relationship dedup
        └─ Layer 7: Output
        ↓
    [Multi-Store Persistence]
        ├─ Weaviate (vectors + semantic search)
        └─ NebulaGraph (knowledge graph)
        ↓
    [Query/Retrieval]
        ├─ GraphTraversal
        ├─ SemanticSearch
        └─ REST API endpoints
```

---

## Current Status

✅ **Implemented & Tested**
- LangGraph orchestration with 3-node workflow
- Llama 3.3 70B integration with automatic fallback
- 7-layer relationship generation pipeline
- Domain schema validation (30+ valid triples)
- Entity deduplication & type reclassification
- Regex fallback when LLM fails
- Weaviate vector database integration
- NebulaGraph knowledge graph integration

⚠️ **Known Constraints**
- Llama 3.3 70B free-tier has rate limits (429 errors) → Liquid LFM fallback activates
- Regex extraction only runs when LLM fails (not merged) → lower entity coverage on small LLM failures

✨ **Quality Metrics (Latest Test)**
- Entity Coverage: 100% (when LLM works) / fallback to regex
- Relationship Coverage: 100% (7/7 key relationships found)
- Type Misclassifications: 0
- Schema Violations: 0
- Logic Errors: 0

---

## How to Run the Full Pipeline

```bash
# 1. Start Docker services
docker-compose -f nebula-docker-compose/docker-compose-lite.yaml up -d

# 2. Start Flask server
python server.py

# 3. (Optional) Run tests
python test_langgraph.py          # Test workflow
python test_llama.py              # Test LLM connectivity
python test_llm_extraction.py     # Test full 7-layer pipeline

# 4. Use React frontend
cd text-data-weaver
npm start  # runs on http://localhost:3000
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│               React Frontend (Port 3000)                │
│          (text-data-weaver/)                            │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/REST
┌──────────────────▼──────────────────────────────────────┐
│             Flask Server (Port 5000)                    │
│               (server.py)                               │
└──────┬──────────────────────┬──────────────────────────┘
       │                      │
       │ Orchestrates         │ Queries
       ▼                      ▼
┌──────────────────────────┐  ┌────────────────────────┐
│ LangGraph Workflow       │  │ Knowledge Graph Access │
│ - Extract               │  │ - Traversal            │
│ - Validate              │  │ - Semantic Search      │
│ - Store                 │  │ - Graph Queries        │
└──────┬──────────────────┘  └────────────────────────┘
       │
       ├─ 7-Layer Pipeline
       │  (relationships)
       │
       ├─▶ Weaviate (Port 8080)
       │   Vector DB + Semantic Search
       │
       └─▶ NebulaGraph (Port 9669)
           Knowledge Graph DB
```

---

## Next Steps (Optional Enhancements)

1. **Rate-limit handling**: Implement request queuing for Llama
2. **Hybrid extraction**: Smart model selection based on text length/domain
3. **Custom LLM**: Fine-tune Llama on domain-specific data
4. **Enhanced relationships**: Add confidence-weighted aggregation
5. **UI improvements**: Real-time extraction progress in React frontend

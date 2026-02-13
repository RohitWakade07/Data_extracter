# Semantic Extractor — Embedding-First Entity & Relationship Extraction
#
# Instead of sending the full document to the LLM in one shot, this module:
#
#   1. Chunks the document into overlapping semantic segments
#   2. Embeds each chunk with sentence-transformers
#   3. For entity extraction:
#      - Extracts entities from EACH chunk independently (focused context)
#      - Uses embedding similarity to deduplicate & merge across chunks
#      - Resolves coreferences (e.g. "the company" → Meridian) via cosine distance
#   4. For relationship extraction:
#      - Finds chunks where TWO entities co-occur semantically (not just textually)
#      - Builds focused context windows for the LLM
#      - Cross-chunk relationships discovered via shared embedding space
#   5. Returns a unified, deduplicated entity+relationship set
#
# Why this is better than single-pass extraction:
#   - LLM gets focused, relevant context → fewer missed entities
#   - Semantic similarity catches implicit/indirect mentions
#   - Cross-chunk analysis finds relationships spanning paragraphs
#   - Embedding-based dedup catches "Meridian" vs "Meridian Infrastructure Solutions Pvt. Ltd."

import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field

from semantic_extraction.semantic_chunker import (
    SemanticChunker,
    SemanticChunk,
    WeaviateChunkStore,
    embed_texts,
    split_sentences,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SemanticEntity:
    """Entity extracted with semantic context."""
    type: str
    value: str
    confidence: float
    source_chunks: List[str] = field(default_factory=list)  # chunk_ids where found
    embedding: Optional[List[float]] = None  # entity name embedding


@dataclass
class SemanticRelationship:
    """Relationship discovered via semantic similarity."""
    source: str
    source_type: str
    target: str
    target_type: str
    relationship: str
    confidence: float
    evidence_chunks: List[str] = field(default_factory=list)
    semantic_score: float = 0.0   # cosine similarity of source↔target context


@dataclass
class SemanticExtractionResult:
    """Full result of semantic extraction pipeline."""
    entities: List[SemanticEntity]
    relationships: List[SemanticRelationship]
    chunks: List[SemanticChunk]
    stats: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Semantic Extraction Pipeline
# ---------------------------------------------------------------------------

class SemanticExtractor:
    """
    Embedding-first extraction pipeline.

    Flow:
    ┌─────────────────┐
    │ Unstructured Text│
    └───────┬─────────┘
            ▼
    ┌─────────────────┐
    │ Sentence Splitter│
    └───────┬─────────┘
            ▼
    ┌─────────────────────────┐
    │ Overlapping Chunk Windows│  (3-sentence windows, 1-sentence overlap)
    └───────┬─────────────────┘
            ▼
    ┌─────────────────────┐
    │ Sentence-Transformer │  all-MiniLM-L6-v2  →  384-dim vectors
    │  Embedding           │
    └───────┬─────────────┘
            ▼
    ┌──────────────────────────┐
    │ Per-Chunk LLM Extraction │  Focused context → better recall
    └───────┬──────────────────┘
            ▼
    ┌────────────────────────────────┐
    │ Cross-Chunk Entity Merging     │  Embedding similarity dedup
    │  "Meridian" ≈ "Meridian Infra" │  cosine > 0.75 → merge
    └───────┬────────────────────────┘
            ▼
    ┌───────────────────────────────────┐
    │ Semantic Relationship Discovery   │
    │  Find entity pairs that share     │
    │  semantically similar chunk       │
    │  contexts → infer relationships   │
    └───────┬───────────────────────────┘
            ▼
    ┌─────────────────────────────┐
    │ Schema Validation & Output  │
    └─────────────────────────────┘
    """

    def __init__(
        self,
        window_size: int = 3,
        overlap: int = 1,
        similarity_threshold: float = 0.75,
        weaviate_url: str = "http://localhost:8080",
        store_in_weaviate: bool = True,
    ):
        self.chunker = SemanticChunker(window_size=window_size, overlap=overlap)
        self.similarity_threshold = similarity_threshold
        self.weaviate_store = WeaviateChunkStore(weaviate_url) if store_in_weaviate else None
        self._llm = None

    # ------------------------------------------------------------------
    # LLM initialization (Ollama → OpenRouter fallback)
    # ------------------------------------------------------------------

    def _get_llm(self):
        """Lazy-load LLM (Ollama first, OpenRouter fallback)."""
        if self._llm is not None:
            return self._llm
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'utils', '.env'))

            try:
                from utils.ollama_handler import OllamaLLM
                model = os.getenv("OLLAMA_MODEL", "llama3")
                self._llm = OllamaLLM(model=model)
                return self._llm
            except Exception:
                api_key = os.getenv("data_extraction_LiquidAi_api_key")
                if api_key:
                    from utils.llm_handler import OpenRouterLLM
                    self._llm = OpenRouterLLM(api_key=api_key)
                    return self._llm
        except Exception as e:
            print(f"⚠ LLM init failed: {e}")
        return None

    # ==================================================================
    # MAIN PIPELINE
    # ==================================================================

    def extract(self, text: str, doc_id: str = "doc") -> SemanticExtractionResult:
        """
        Run the full semantic extraction pipeline.

        Parameters
        ----------
        text : str
            Unstructured document text.
        doc_id : str
            Document identifier.

        Returns
        -------
        SemanticExtractionResult
            Entities, relationships, chunks, and stats.
        """
        print(f"\n{'═'*70}")
        print("SEMANTIC EXTRACTION PIPELINE  (embedding-first)")
        print(f"{'═'*70}")

        # ── Step 1: Chunk & Embed ──────────────────────────────────
        print("\n▶ Step 1 — Chunking & embedding …")
        chunks = self.chunker.chunk_and_embed(text, doc_id)
        n_sent = len(split_sentences(text))
        print(f"  {n_sent} sentences → {len(chunks)} overlapping chunks  "
              f"(window={self.chunker.window_size}, overlap={self.chunker.overlap})")

        # Optionally persist to Weaviate
        if self.weaviate_store:
            stored = self.weaviate_store.store_chunks(chunks)
            if stored:
                print(f"  ✓ {stored} chunks stored in Weaviate (vector index)")

        # ── Step 2: Per-Chunk Entity Extraction ────────────────────
        print("\n▶ Step 2 — Per-chunk entity extraction (focused context) …")
        raw_entities = self._extract_entities_per_chunk(chunks)
        print(f"  {len(raw_entities)} raw entity mentions across all chunks")

        # ── Step 2.5: LLM-Based Entity Validation ─────────────────
        print("\n▶ Step 2.5 — LLM-based entity validation agent …")
        raw_entities = self._validate_entities_with_llm(raw_entities, text)
        print(f"  {len(raw_entities)} entities after validation")

        # ── Step 3: Embedding-Based Entity Merging ─────────────────
        print("\n▶ Step 3 — Embedding-based entity deduplication …")
        merged_entities = self._merge_entities_by_embedding(raw_entities)
        type_counts = {}
        for e in merged_entities:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1
        print(f"  {len(raw_entities)} → {len(merged_entities)} unique entities")
        for t, c in sorted(type_counts.items()):
            print(f"    {t}: {c}")

        # ── Step 4: Semantic Relationship Discovery ────────────────
        print("\n▶ Step 4 — Semantic relationship discovery …")
        relationships = self._discover_relationships(merged_entities, chunks, text)
        rel_counts = {}
        for r in relationships:
            rel_counts[r.relationship] = rel_counts.get(r.relationship, 0) + 1
        print(f"  {len(relationships)} relationships discovered")
        for rt, c in sorted(rel_counts.items()):
            print(f"    {rt}: {c}")

        # ── Step 5: Cross-Chunk Implicit Relationships ─────────────
        print("\n▶ Step 5 — Cross-chunk implicit relationship discovery …")
        implicit_rels = self._cross_chunk_relationships(merged_entities, chunks)
        if implicit_rels:
            print(f"  {len(implicit_rels)} cross-chunk relationships found")
            relationships.extend(implicit_rels)
        else:
            print("  (no additional cross-chunk relationships)")

        # Deduplicate relationships
        relationships = self._deduplicate_relationships(relationships)

        # ── Stats ──────────────────────────────────────────────────
        stats = {
            "total_sentences": n_sent,
            "total_chunks": len(chunks),
            "raw_entity_mentions": len(raw_entities),
            "unique_entities": len(merged_entities),
            "total_relationships": len(relationships),
            "entity_type_counts": type_counts,
            "relationship_type_counts": rel_counts,
        }

        print(f"\n{'═'*70}")
        print(f"SEMANTIC EXTRACTION COMPLETE")
        print(f"  Entities:      {len(merged_entities)}")
        print(f"  Relationships: {len(relationships)}")
        print(f"{'═'*70}")

        return SemanticExtractionResult(
            entities=merged_entities,
            relationships=relationships,
            chunks=chunks,
            stats=stats,
        )

    # ==================================================================
    # STEP 2 — Per-Chunk Entity Extraction
    # ==================================================================

    def _extract_entities_per_chunk(
        self, chunks: List[SemanticChunk]
    ) -> List[SemanticEntity]:
        """
        Extract entities from each chunk independently.
        The LLM gets a small, focused context window → higher recall.
        """
        llm = self._get_llm()
        all_entities: List[SemanticEntity] = []

        for i, chunk in enumerate(chunks):
            chunk_entities = self._extract_from_single_chunk(chunk, llm)
            all_entities.extend(chunk_entities)
            if (i + 1) % 5 == 0:
                print(f"    … processed {i + 1}/{len(chunks)} chunks ({len(all_entities)} entities so far)")

        return all_entities

    def _extract_from_single_chunk(
        self, chunk: SemanticChunk, llm: Any
    ) -> List[SemanticEntity]:
        """Extract entities from one chunk using the LLM with dynamic type discovery."""
        prompt = f"""You are a named entity recognition expert. Extract ALL meaningful entities from this text.

For EACH entity, you must:
1. Identify the EXACT phrase from the text
2. Classify it with the MOST SPECIFIC type that fits its real-world meaning
3. Assign a confidence score (0.0 to 1.0)

CLASSIFICATION RULES (apply strictly):
- A PERSON is ONLY a real human name (e.g. "John Smith"). Job titles, roles, pronouns, document terms are NOT persons.
- An ORGANIZATION is a named company, firm, bank, government body, institution.
- A LOCATION is a city, state, country, address, or geographic place.
- A DATE is a specific date, deadline, or time reference.
- An AMOUNT is any monetary value, whether numeric ("$25,000") or written ("Twenty-Five Million Dollars"). Include currency.
- An INVOICE is a specific invoice identifier (e.g. "INV-78421", "Invoice #1234").
- An AGREEMENT / CONTRACT is a named contract, agreement, or legal document.
- A PERCENTAGE is any percentage value (e.g. "18%", "2%").
- A TAX is a tax type or reference (e.g. "GST", "VAT", "Sales Tax").
- If an entity doesn't fit the above, assign the most descriptive type (e.g. PROJECT, ROLE, REGULATION, CLAUSE, ASSET, ACCOUNT, PRODUCT, EVENT, DURATION, TERM, PENALTY, etc.)

DO NOT extract:
- Generic document structure words ("Section", "Article", "Clause" by themselves)
- Pronouns or references ("this", "that", "the buyer")
- Partial phrases that aren't meaningful standalone entities

Return a JSON array. Each item: {{"type": "TYPE_IN_CAPS", "value": "exact text", "confidence": 0.95}}

TEXT:
{chunk.text}

JSON array:"""

        try:
            if llm and hasattr(llm, "chat"):
                raw = llm.chat(prompt)
            else:
                return self._regex_extract_chunk(chunk)

            # Parse JSON
            clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if clean.startswith('```'):
                lines = clean.split('\n')
                lines = [l for l in lines if not l.strip().startswith('```')]
                clean = '\n'.join(lines)

            j_start = clean.find('[')
            j_end = clean.rfind(']') + 1
            if j_start == -1 or j_end <= j_start:
                return self._regex_extract_chunk(chunk)

            data = json.loads(clean[j_start:j_end])
            entities: List[SemanticEntity] = []
            for item in data:
                raw_type = str(item.get("type", "unknown")).strip().lower()
                value = str(item.get("value", "")).strip()
                if not value or len(value) < 2:
                    continue
                conf = float(item.get("confidence", 0.8))
                
                entities.append(SemanticEntity(
                    type=raw_type,
                    value=value,
                    confidence=conf,
                    source_chunks=[chunk.chunk_id],
                ))
            return entities

        except Exception:
            return self._regex_extract_chunk(chunk)

    # ==================================================================
    # STEP 2.5 — LLM-Based Entity Validation Agent
    # ==================================================================

    def _validate_entities_with_llm(
        self, entities: List[SemanticEntity], text: str
    ) -> List[SemanticEntity]:
        """
        Agentic LLM-based entity validation.

        Instead of rule-based correction, we send the entire batch of
        extracted entities back to the LLM along with the source text
        and ask it to:
          1. Verify each entity's type is correct
          2. Reclassify mistyped entities
          3. Remove junk / non-entity extractions
          4. Merge obvious duplicates it can spot

        This is fully dynamic — the LLM adapts to ANY domain (legal,
        medical, finance, HR, engineering …) without hardcoded lists.
        """
        llm = self._get_llm()
        if not llm or not hasattr(llm, "chat"):
            return entities  # Can't validate without LLM

        if not entities:
            return entities

        # Build a compact table for the LLM
        entity_rows = []
        for i, e in enumerate(entities):
            entity_rows.append(f'{i}|{e.type}|{e.value}|{e.confidence:.2f}')
        entity_table = '\n'.join(entity_rows)

        # Use a truncated version of the text for context (first 2000 chars)
        context_text = text[:2000] if len(text) > 2000 else text

        prompt = f"""You are an entity validation agent. Below is a list of entities extracted from a document, and the source text for context.

Your job:
1. CHECK each entity's TYPE — is it correct given the entity value and document context?
2. RECLASSIFY any entity whose type is wrong. Use the most specific, accurate type.
3. REMOVE entities that are not real named entities (generic phrases, pronouns, structural terms).

STRICT TYPE RULES:
- PERSON: ONLY real human names (first + last name). NOT titles, roles, descriptions, document terms, or concepts.
- ORGANIZATION: Named companies, firms, banks, institutions, government bodies.
- LOCATION: Cities, states, countries, addresses, geographic regions.
- DATE: Specific dates, deadlines, time periods, durations.
- AMOUNT: Any monetary value — numeric ($5M) or written (Five Million Dollars). Include currency symbol/code.
- INVOICE: Invoice identifiers (INV-1234, Invoice #5678). NOT the word "invoice" alone.
- AGREEMENT: Named contracts, agreements, terms, clauses, legal documents.
- PERCENTAGE: Percentage values (18%, 2%, 5.5%).
- TAX: Tax types (GST, VAT, Sales Tax, Income Tax).
- ASSET: Property, real estate, equipment, intellectual property being transacted.
- ROLE: Job titles, positions, designations (CEO, Managing Director, etc.).
- DURATION: Time spans (30 days, 6 months, 2 years).
- REGULATION: Laws, acts, regulatory requirements.
- TERM: Contractual terms, conditions, penalties, clauses.
- Use any other specific type if none of the above fits (PROJECT, EVENT, PRODUCT, ACCOUNT, etc.).

ENTITIES (index|current_type|value|confidence):
{entity_table}

SOURCE TEXT (for context):
{context_text}

Return a JSON array with the validated entities. For each entity return:
{{"index": 0, "type": "CORRECT_TYPE", "value": "exact value", "keep": true}}

Set "keep": false for entities that should be removed (not real entities).
Only change "type" if the current type is WRONG.

JSON array:"""

        try:
            raw = llm.chat(prompt)

            # Parse response
            clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if clean.startswith('```'):
                lines = clean.split('\n')
                lines = [l for l in lines if not l.strip().startswith('```')]
                clean = '\n'.join(lines)

            j_start = clean.find('[')
            j_end = clean.rfind(']') + 1
            if j_start == -1 or j_end <= j_start:
                print("    ⚠ Validation agent returned unparseable response — keeping originals")
                return entities

            corrections = json.loads(clean[j_start:j_end])

            # Build lookup from corrections
            correction_map: Dict[int, Dict[str, Any]] = {}
            for item in corrections:
                idx = item.get("index")
                if idx is not None:
                    correction_map[int(idx)] = item

            validated: List[SemanticEntity] = []
            changed = 0
            removed = 0

            for i, entity in enumerate(entities):
                corr = correction_map.get(i)
                if corr is None:
                    # Not in corrections → keep as-is
                    validated.append(entity)
                    continue

                if not corr.get("keep", True):
                    removed += 1
                    continue

                new_type = str(corr.get("type", entity.type)).strip().lower()
                if new_type != entity.type:
                    changed += 1
                    entity.type = new_type

                # Optionally update value if the LLM corrected it
                new_value = corr.get("value", entity.value)
                if new_value and len(str(new_value).strip()) >= 2:
                    entity.value = str(new_value).strip()

                validated.append(entity)

            print(f"    Validation agent: {changed} types corrected, {removed} entities removed")
            return validated

        except Exception as e:
            print(f"    ⚠ Validation agent error: {e} — keeping originals")
            return entities

    def _regex_extract_chunk(self, chunk: SemanticChunk) -> List[SemanticEntity]:
        """Fallback regex extraction for a single chunk."""
        entities: List[SemanticEntity] = []
        text = chunk.text

        # Persons (two+ capitalized words)
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            name = m.group(1)
            if len(name.split()) <= 3:
                entities.append(SemanticEntity(
                    type='person', value=name, confidence=0.7,
                    source_chunks=[chunk.chunk_id],
                ))

        # Amounts
        for m in re.finditer(r'(?:INR|Rs\.?|USD|\$)\s*[\d,.]+\s*(?:crore|lakh|million|billion)?', text, re.I):
            entities.append(SemanticEntity(
                type='amount', value=m.group().strip(), confidence=0.95,
                source_chunks=[chunk.chunk_id],
            ))

        # Dates
        for m in re.finditer(
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            text, re.I
        ):
            entities.append(SemanticEntity(
                type='date', value=m.group().strip(), confidence=0.95,
                source_chunks=[chunk.chunk_id],
            ))

        return entities

    # ==================================================================
    # STEP 3 — Embedding-Based Entity Deduplication
    # ==================================================================

    def _merge_entities_by_embedding(
        self, entities: List[SemanticEntity]
    ) -> List[SemanticEntity]:
        """
        Merge entity mentions that refer to the same real-world entity.

        Uses cosine similarity of entity VALUE embeddings:
          "Meridian" ≈ "Meridian Infrastructure Solutions Pvt. Ltd."
          → cosine > threshold → merge (keep longer form)
        """
        if not entities:
            return []

        # Group by type first (only merge within same type)
        by_type: Dict[str, List[SemanticEntity]] = {}
        for e in entities:
            by_type.setdefault(e.type, []).append(e)

        merged_all: List[SemanticEntity] = []

        for etype, ents in by_type.items():
            # Deduplicate exact matches first
            unique: Dict[str, SemanticEntity] = {}
            for e in ents:
                key = e.value.lower().strip()
                if key in unique:
                    # Merge: keep higher confidence, union chunks
                    existing = unique[key]
                    existing.confidence = max(existing.confidence, e.confidence)
                    existing.source_chunks = list(set(existing.source_chunks + e.source_chunks))
                else:
                    unique[key] = SemanticEntity(
                        type=e.type, value=e.value, confidence=e.confidence,
                        source_chunks=list(e.source_chunks),
                    )

            ent_list = list(unique.values())
            if len(ent_list) <= 1:
                merged_all.extend(ent_list)
                continue

            # Embed all entity values in this type group
            values = [e.value for e in ent_list]
            embeddings = embed_texts(values)

            # Compute pairwise similarities
            merged_flags = [False] * len(ent_list)
            for i in range(len(ent_list)):
                if merged_flags[i]:
                    continue
                for j in range(i + 1, len(ent_list)):
                    if merged_flags[j]:
                        continue
                    sim = float(np.dot(embeddings[i], embeddings[j]) /
                                (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9))
                    if sim >= self.similarity_threshold:
                        # Merge j into i — keep the longer (more specific) value
                        longer = ent_list[i] if len(ent_list[i].value) >= len(ent_list[j].value) else ent_list[j]
                        ent_list[i].value = longer.value
                        ent_list[i].confidence = max(ent_list[i].confidence, ent_list[j].confidence)
                        ent_list[i].source_chunks = list(
                            set(ent_list[i].source_chunks + ent_list[j].source_chunks)
                        )
                        merged_flags[j] = True

            for i, ent in enumerate(ent_list):
                if not merged_flags[i]:
                    # Attach embedding
                    ent.embedding = embeddings[i].tolist()
                    merged_all.append(ent)

        return merged_all

    # ==================================================================
    # STEP 4 — Semantic Relationship Discovery
    # ==================================================================

    def _discover_relationships(
        self,
        entities: List[SemanticEntity],
        chunks: List[SemanticChunk],
        full_text: str,
    ) -> List[SemanticRelationship]:
        """
        Discover relationships using LLM — fully dynamic, works with any entity types.

        Instead of hardcoded directional constraints, we send entity pairs
        that co-occur in the same chunk context to the LLM and ask it to
        identify relationships between them.
        """
        if not entities or len(entities) < 2:
            return []

        llm = self._get_llm()
        if not llm or not hasattr(llm, "chat"):
            return []

        # Build entity list for the prompt
        entity_list = []
        for i, e in enumerate(entities):
            entity_list.append(f"{i}. [{e.type.upper()}] {e.value}")
        entity_block = "\n".join(entity_list)

        # Use truncated text for context
        context_text = full_text[:3000] if len(full_text) > 3000 else full_text

        prompt = f"""You are a relationship extraction expert for business documents. Given the entities and source document below, identify ALL meaningful relationships.

ENTITIES:
{entity_block}

SOURCE TEXT:
{context_text}

DOCUMENT-AWARE RELATIONSHIP RULES:

1. **INVOICES** — The invoice entity (e.g. INV-78421) is the CENTRAL node:
   - Invoice → ISSUED_BY → Organization (the issuer/sender)
   - Invoice → BILLED_TO → Organization (the recipient/buyer)
   - Invoice → HAS_SUBTOTAL → Amount (subtotal before tax)
   - Invoice → HAS_TAX → Amount (tax amount, GST, VAT)
   - Invoice → HAS_TOTAL → Amount (total payable)
   - Invoice → INVOICE_DATE → Date (the issue date)
   - Invoice → DUE_DATE → Date (the payment deadline)
   - Invoice → HAS_TAX_RATE → Percentage (tax rate like 18%)
   - Invoice → HAS_PENALTY → Term/Percentage (late fees, penalties)
   - Organization (issuer) → BILLS → Organization (recipient)

2. **CONTRACTS / AGREEMENTS** — The agreement is the CENTRAL node:
   - Agreement → SIGNED_BY → Person/Organization (each party)
   - Agreement → EFFECTIVE_DATE → Date
   - Agreement → EXPIRY_DATE → Date
   - Agreement → HAS_VALUE → Amount (contract value)
   - Agreement → GOVERNED_BY → Regulation/Location (jurisdiction)
   - Person → PARTY_TO → Agreement
   - Organization → PARTY_TO → Agreement

3. **GENERAL RELATIONSHIPS** (use when appropriate):
   - Person → WORKS_AT → Organization
   - Organization → LOCATED_IN → Location
   - Person → SIGNED_ON → Date
   - Organization → OWNS → Asset
   - Organization → SELLS_TO / SUPPLIES_TO → Organization

CRITICAL RULES:
- Dates DON'T "have values" of amounts. Amounts belong to invoices/agreements.
- Always connect amounts and dates THROUGH the document entity (invoice/agreement), not directly to each other.
- Differentiate between dates by their ROLE (invoice date vs due date vs effective date).
- Differentiate between amounts by their ROLE (subtotal vs tax vs total).
- Only extract relationships EXPLICITLY stated or strongly implied in the text.
- Confidence: 0.85-0.95 for explicit, 0.70-0.84 for implied.

Return a JSON array. Each item:
{{"source": 0, "target": 1, "relationship": "RELATIONSHIP_TYPE", "confidence": 0.85}}

Use entity index numbers for source and target.

JSON array:"""

        try:
            raw = llm.chat(prompt)

            # Parse response
            clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if clean.startswith('```'):
                lines = clean.split('\n')
                lines = [l for l in lines if not l.strip().startswith('```')]
                clean = '\n'.join(lines)

            j_start = clean.find('[')
            j_end = clean.rfind(']') + 1
            if j_start == -1 or j_end <= j_start:
                print("    ⚠ Relationship agent returned unparseable response")
                return []

            data = json.loads(clean[j_start:j_end])
            relationships: List[SemanticRelationship] = []
            seen: Set[Tuple[str, str, str]] = set()

            for item in data:
                src_idx = int(item.get("source", -1))
                tgt_idx = int(item.get("target", -1))
                rel_type = str(item.get("relationship", "RELATED_TO")).strip().upper()
                conf = float(item.get("confidence", 0.7))

                if src_idx < 0 or src_idx >= len(entities):
                    continue
                if tgt_idx < 0 or tgt_idx >= len(entities):
                    continue
                if src_idx == tgt_idx:
                    continue

                src_ent = entities[src_idx]
                tgt_ent = entities[tgt_idx]

                key = (src_ent.value.lower(), tgt_ent.value.lower(), rel_type)
                if key in seen:
                    continue
                seen.add(key)

                # Compute semantic similarity
                sem_score = 0.0
                if src_ent.embedding and tgt_ent.embedding:
                    a = np.array(src_ent.embedding, dtype=np.float32)
                    b = np.array(tgt_ent.embedding, dtype=np.float32)
                    sem_score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

                relationships.append(SemanticRelationship(
                    source=src_ent.value,
                    source_type=src_ent.type,
                    target=tgt_ent.value,
                    target_type=tgt_ent.type,
                    relationship=rel_type,
                    confidence=round(conf, 3),
                    evidence_chunks=[],
                    semantic_score=round(sem_score, 3),
                ))

            print(f"    LLM discovered {len(relationships)} relationships")
            return relationships

        except Exception as e:
            print(f"    ⚠ Relationship discovery error: {e}")
            return []

        return relationships

    # ==================================================================
    # STEP 5 — Cross-Chunk Implicit Relationships
    # ==================================================================

    def _cross_chunk_relationships(
        self,
        entities: List[SemanticEntity],
        chunks: List[SemanticChunk],
    ) -> List[SemanticRelationship]:
        """
        Cross-chunk relationship discovery is DISABLED in production mode.
        
        Reason: Cross-chunk relationships are high-risk for hallucination.
        When entities don't appear in the same sentence/paragraph, we cannot
        reliably infer a relationship between them without explicit evidence.
        
        This prevents:
          - Role leakage (Tim Cook LEADS OpenAI)
          - False partnerships (Apple PARTNERS_WITH Microsoft)
          - Hallucinated contracts across document sections
        
        If cross-chunk relationships are needed, they should be discovered
        through explicit multi-hop graph queries after initial extraction.
        """
        # Return empty — cross-chunk implicit relationships are too risky
        # Enable only if you have a reliable evidence extraction mechanism
        return []

    # ==================================================================
    # Deduplication
    # ==================================================================

    def _deduplicate_relationships(
        self, rels: List[SemanticRelationship]
    ) -> List[SemanticRelationship]:
        """Keep highest-confidence per (source, target, relationship)."""
        best: Dict[Tuple[str, str, str], SemanticRelationship] = {}
        for r in rels:
            key = (r.source.lower(), r.target.lower(), r.relationship)
            existing = best.get(key)
            if existing is None or r.confidence > existing.confidence:
                best[key] = r
        return list(best.values())


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def semantic_extract(text: str, doc_id: str = "doc", **kwargs) -> SemanticExtractionResult:
    """One-call entry point for semantic extraction."""
    extractor = SemanticExtractor(**kwargs)
    return extractor.extract(text, doc_id)

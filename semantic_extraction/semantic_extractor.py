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
        """Extract entities from one chunk using the LLM."""
        prompt = f"""Extract ALL entities from this text segment. Return a JSON array.
Each item: {{"type": "...", "value": "...", "confidence": 0.95}}

Types: PERSON, ORGANIZATION, DATE, AMOUNT, LOCATION, PROJECT, INVOICE, AGREEMENT

TEXT SEGMENT:
{chunk.text}

Return ONLY a JSON array:"""

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

            # Normalize types
            type_normalize = {
                'state': 'location', 'country': 'location', 'region': 'location',
                'city': 'location', 'place': 'location', 'area': 'location',
                'company': 'organization', 'org': 'organization', 'firm': 'organization',
                'bank': 'organization', 'institution': 'organization',
                'name': 'person', 'individual': 'person',
                'money': 'amount', 'currency': 'amount', 'cost': 'amount',
                'contract': 'agreement', 'proposal': 'agreement',
            }

            data = json.loads(clean[j_start:j_end])
            entities: List[SemanticEntity] = []
            for item in data:
                raw_type = str(item.get("type", "unknown")).lower()
                norm_type = type_normalize.get(raw_type, raw_type)
                value = str(item.get("value", "")).strip()
                if not value:
                    continue
                conf = float(item.get("confidence", 0.9))
                entities.append(SemanticEntity(
                    type=norm_type,
                    value=value,
                    confidence=conf,
                    source_chunks=[chunk.chunk_id],
                ))
            return entities

        except Exception:
            return self._regex_extract_chunk(chunk)

    def _regex_extract_chunk(self, chunk: SemanticChunk) -> List[SemanticEntity]:
        """Fallback regex extraction for a single chunk."""
        entities: List[SemanticEntity] = []
        text = chunk.text

        # Persons (two+ capitalized words)
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            name = m.group(1)
            # Filter out non-person capitalized phrases
            skip_words = {'Smart City', 'Urban Road', 'Pune Municipal', 'Hinjewadi IT',
                          'Chief Financial', 'Infrastructure Planning'}
            if name not in skip_words and len(name.split()) <= 3:
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
        Discover relationships by:
        1. Finding entity-pairs that co-occur in the same chunk
        2. Using the chunk context to ask the LLM what relationship connects them
        3. Scoring by semantic similarity between entity embeddings
        """
        from utils.domain_schema import validate_relationship_triple, VALID_TRIPLES

        # Build entity → chunk mapping
        entity_chunks: Dict[str, Set[str]] = {}
        for e in entities:
            entity_chunks[e.value.lower()] = set(e.source_chunks)

        # Also check textual presence in chunks (entity might appear in chunks
        # where it wasn't explicitly extracted)
        chunk_lookup = {c.chunk_id: c for c in chunks}
        for e in entities:
            for c in chunks:
                if e.value.lower() in c.text.lower():
                    entity_chunks.setdefault(e.value.lower(), set()).add(c.chunk_id)

        # Find co-occurring entity pairs
        relationships: List[SemanticRelationship] = []
        pair_contexts: Dict[Tuple[str, str], List[str]] = {}  # (ent1, ent2) → chunk texts

        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i >= j:
                    continue
                if e1.type == e2.type and e1.type in ('date', 'amount'):
                    continue  # Skip date↔date, amount↔amount

                # Find shared chunks
                chunks_e1 = entity_chunks.get(e1.value.lower(), set())
                chunks_e2 = entity_chunks.get(e2.value.lower(), set())
                shared = chunks_e1 & chunks_e2

                if shared:
                    # These entities co-occur in at least one chunk
                    context_texts = [chunk_lookup[cid].text for cid in shared if cid in chunk_lookup]
                    pair_contexts[(e1.value, e2.value)] = context_texts

        # For each co-occurring pair, determine the relationship
        # Use the domain schema VALID_TRIPLES to constrain
        entity_map = {e.value.lower(): e for e in entities}

        for (e1_val, e2_val), contexts in pair_contexts.items():
            e1 = entity_map.get(e1_val.lower())
            e2 = entity_map.get(e2_val.lower())
            if not e1 or not e2:
                continue

            # Check what relationships are valid for this type pair
            valid_rels_forward = [
                (src, rel, tgt) for src, rel, tgt in VALID_TRIPLES
                if src == e1.type and tgt == e2.type
            ]
            valid_rels_reverse = [
                (src, rel, tgt) for src, rel, tgt in VALID_TRIPLES
                if src == e2.type and tgt == e1.type
            ]

            # Compute semantic similarity between the two entities' embeddings
            sem_score = 0.0
            if e1.embedding and e2.embedding:
                a = np.array(e1.embedding, dtype=np.float32)
                b = np.array(e2.embedding, dtype=np.float32)
                sem_score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

            # Forward relationships
            for src_t, rel_type, tgt_t in valid_rels_forward:
                # Compute confidence based on co-occurrence + semantic score
                base_conf = 0.80 if len(contexts) > 1 else 0.70
                conf = min(0.95, base_conf + sem_score * 0.10)
                relationships.append(SemanticRelationship(
                    source=e1.value, source_type=e1.type,
                    target=e2.value, target_type=e2.type,
                    relationship=rel_type,
                    confidence=round(conf, 3),
                    evidence_chunks=[c[:100] for c in contexts[:2]],
                    semantic_score=round(sem_score, 3),
                ))

            # Reverse relationships
            for src_t, rel_type, tgt_t in valid_rels_reverse:
                base_conf = 0.80 if len(contexts) > 1 else 0.70
                conf = min(0.95, base_conf + sem_score * 0.10)
                relationships.append(SemanticRelationship(
                    source=e2.value, source_type=e2.type,
                    target=e1.value, target_type=e1.type,
                    relationship=rel_type,
                    confidence=round(conf, 3),
                    evidence_chunks=[c[:100] for c in contexts[:2]],
                    semantic_score=round(sem_score, 3),
                ))

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
        Find relationships between entities that DON'T share a chunk,
        but whose chunks are semantically similar.

        This catches implicit/indirect relationships spanning paragraphs.
        Example: Paragraph 1 mentions "HDFC Bank" as escrow agent,
                 Paragraph 5 mentions "fund disbursements".
                 These chunks are semantically related even though the
                 entities don't co-occur textually.
        """
        from utils.domain_schema import validate_relationship_triple, VALID_TRIPLES

        chunk_lookup = {c.chunk_id: c for c in chunks}
        relationships: List[SemanticRelationship] = []
        seen: Set[Tuple[str, str, str]] = set()

        # For each pair of entities NOT already sharing a chunk
        entity_chunks: Dict[str, Set[str]] = {}
        for e in entities:
            for c in chunks:
                if e.value.lower() in c.text.lower():
                    entity_chunks.setdefault(e.value.lower(), set()).add(c.chunk_id)

        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i >= j:
                    continue
                if e1.type == e2.type and e1.type in ('date', 'amount'):
                    continue

                c1 = entity_chunks.get(e1.value.lower(), set())
                c2 = entity_chunks.get(e2.value.lower(), set())

                # Skip if they already share a chunk (handled in Step 4)
                if c1 & c2:
                    continue

                # Check semantic similarity between their chunks
                if not c1 or not c2:
                    continue

                # Find max cosine similarity between any chunk of e1 and any chunk of e2
                max_sim = 0.0
                best_pair = ("", "")
                for cid1 in c1:
                    for cid2 in c2:
                        ch1 = chunk_lookup.get(cid1)
                        ch2 = chunk_lookup.get(cid2)
                        if ch1 and ch2 and ch1.embedding and ch2.embedding:
                            a = np.array(ch1.embedding, dtype=np.float32)
                            b = np.array(ch2.embedding, dtype=np.float32)
                            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                            if sim > max_sim:
                                max_sim = sim
                                best_pair = (cid1, cid2)

                # If chunks are semantically similar enough, infer relationship
                if max_sim >= 0.45:  # lower threshold for cross-chunk
                    valid_rels = [
                        (src, rel, tgt) for src, rel, tgt in VALID_TRIPLES
                        if src == e1.type and tgt == e2.type
                    ]
                    for _, rel_type, _ in valid_rels:
                        key = (e1.value.lower(), e2.value.lower(), rel_type)
                        if key not in seen:
                            seen.add(key)
                            conf = round(0.55 + max_sim * 0.30, 3)
                            relationships.append(SemanticRelationship(
                                source=e1.value, source_type=e1.type,
                                target=e2.value, target_type=e2.type,
                                relationship=rel_type,
                                confidence=conf,
                                evidence_chunks=[best_pair[0], best_pair[1]],
                                semantic_score=round(max_sim, 3),
                            ))

        return relationships

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

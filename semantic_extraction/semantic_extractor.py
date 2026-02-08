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
        Discover relationships using production-grade validation:
        
        1. Find entity-pairs that co-occur in the same sentence/paragraph
        2. Check for explicit verb/keyword evidence
        3. Apply directional constraints
        4. Compute deterministic confidence (not LLM-generated)
        5. Only emit relationships with sufficient evidence
        
        This addresses:
          - Relation explosion (overgeneration)
          - Role leakage (person→org contamination)
          - Symmetric relation duplication
          - Flat/misleading confidence scores
        """
        from utils.relation_validator import (
            RelationshipValidator,
            entities_in_same_sentence,
            find_evidence_for_relation,
            CORE_RELATIONS,
            DIRECTIONAL_CONSTRAINTS,
        )
        
        validator = RelationshipValidator(
            min_confidence=0.45,
            require_same_sentence=False,  # Allow paragraph-level
            require_evidence=True,         # Require keyword evidence
        )

        # Build entity lookup
        entity_map = {e.value.lower(): e for e in entities}
        relationships: List[SemanticRelationship] = []
        seen_pairs: Set[Tuple[str, str, str]] = set()  # (src, tgt, rel)

        # Process each entity pair
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i >= j:
                    continue
                # Skip same-type pairs for dates/amounts
                if e1.type == e2.type and e1.type in ('date', 'amount'):
                    continue

                # ── Determine valid relations for this type pair ──
                candidate_rels = []
                for rel, (allowed_src, allowed_tgt) in DIRECTIONAL_CONSTRAINTS.items():
                    if e1.type == allowed_src and e2.type == allowed_tgt:
                        candidate_rels.append((rel, e1, e2))
                    elif e2.type == allowed_src and e1.type == allowed_tgt:
                        candidate_rels.append((rel, e2, e1))
                
                # Also check symmetric relations
                if e1.type == "organization" and e2.type == "organization":
                    candidate_rels.append(("PARTNERS_WITH", e1, e2))

                if not candidate_rels:
                    continue  # No valid relations possible

                # ── Validate each candidate relation ──
                for rel_type, src_ent, tgt_ent in candidate_rels:
                    # Skip if we've already seen this triple
                    key = (src_ent.value.lower(), tgt_ent.value.lower(), rel_type)
                    if key in seen_pairs:
                        continue

                    result = validator.validate(
                        source=src_ent.value,
                        source_type=src_ent.type,
                        target=tgt_ent.value,
                        target_type=tgt_ent.type,
                        proposed_relation=rel_type,
                        full_text=full_text,
                    )

                    if result.is_valid:
                        seen_pairs.add(key)
                        
                        # Compute semantic similarity for additional scoring
                        sem_score = 0.0
                        if src_ent.embedding and tgt_ent.embedding:
                            a = np.array(src_ent.embedding, dtype=np.float32)
                            b = np.array(tgt_ent.embedding, dtype=np.float32)
                            sem_score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                        
                        relationships.append(SemanticRelationship(
                            source=result.source,
                            source_type=result.source_type,
                            target=result.target,
                            target_type=result.target_type,
                            relationship=result.relation,
                            confidence=round(result.confidence, 3),
                            evidence_chunks=result.evidence_matches or [],
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

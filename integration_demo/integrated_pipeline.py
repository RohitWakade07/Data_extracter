# Phase 5 - Integration & Demo
# End-to-end 7-layer extraction pipeline:
#   1. Candidate Extraction  (NER + regex)
#   2. Type Identification   (domain schema reclassification)
#   3. Relation Mapping       (LLM + proximity + verb extraction)
#   4. Domain Schema Filter   (only valid triples allowed)
#   5. Confidence Scoring     (entity + relationship confidence)
#   6. Deduplication          (merge similar entities)
#   7. Graph Storage          (NebulaGraph + Weaviate)

import json
import re
import os
from typing import Dict, Any, List, Tuple
from entity_extraction.entity_extractor import extract_from_text
from agentic_workflow.workflow import run_workflow
from vector_database.weaviate_handler import store_in_weaviate, Document
from knowledge_graph.nebula_handler import store_in_nebula
from utils.domain_schema import (
    identify_entity_type,
    validate_relationship_triple,
    get_allowed_relationships,
    deduplicate_entities,
    filter_by_confidence,
    VALID_TRIPLES,
    ENTITY_ACCEPT_THRESHOLD,
    REL_ACCEPT_THRESHOLD,
    REL_DISCARD_THRESHOLD,
)

class IntegratedPipeline:
    """Complete end-to-end extraction pipeline.

    Supports two extraction modes:
      - ``mode="traditional"`` — 7-layer NER + proximity + LLM pipeline (default, faster)
      - ``mode="semantic"``    — embedding-first pipeline: chunk → embed → per-chunk LLM
                                  → embedding dedup → cross-chunk relationship discovery
    """
    
    def __init__(self, mode: str = "traditional"):
        self.results = {}
        self.mode = mode.lower()   # "traditional" or "semantic"
    
    def run_complete_pipeline(self, unstructured_text: str) -> Dict[str, Any]:
        """
        Execute the complete integrated pipeline
        
        Phase 1: Extract entities (+ relationships via semantic or traditional)
        Phase 2: Agentic workflow validation
        Phase 3: Store in vector database
        Phase 4: Store in knowledge graph
        Phase 5: Execute queries
        """
        
        print("\n" + "="*70)
        print(f"COMPLETE AUTOMATED DATA EXTRACTION PIPELINE  (mode={self.mode})")
        print("="*70)

        # ── Semantic mode: use embedding-first pipeline ────────────
        if self.mode == "semantic":
            return self._run_semantic_pipeline(unstructured_text)

        # ── Traditional mode (default) ─────────────────────────────
        
        # Phase 2: Run agentic workflow
        print("\n== PHASE 2: AGENTIC WORKFLOW")
        workflow_result = run_workflow(unstructured_text)
        self.results["workflow"] = workflow_result
        
        entities = workflow_result.get("entities", [])
        
        if not entities:
            print("No entities extracted. Pipeline incomplete.")
            return self.results
        
        # Phase 3: Store in Weaviate
        print("\n== PHASE 3: VECTOR DATABASE (WEAVIATE)")
        doc = Document(
            id="doc_001",
            content=unstructured_text,
            entities=entities,
            metadata={"type": "document"}
        )
        vector_result = store_in_weaviate([doc])
        self.results["vector_storage"] = vector_result
        
        # Phase 4: Store in NebulaGraph
        print("\n== PHASE 4: KNOWLEDGE GRAPH (NEBULA)")
        relationships = self._generate_relationships(entities, unstructured_text)
        self.results["relationships"] = relationships
        graph_result = store_in_nebula(entities, relationships)
        self.results["graph_storage"] = graph_result
        
        # Phase 5: Execute queries
        print("\n== PHASE 5: EXECUTE QUERIES")
        self._demonstrate_queries(entities)
        
        return self.results

    # ==================================================================
    # SEMANTIC MODE — embedding-first pipeline
    # ==================================================================

    def _run_semantic_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Run the embedding-first semantic extraction pipeline.

        Flow:
          1. Chunk text → embed with sentence-transformers
          2. Per-chunk LLM extraction (focused context)
          3. Embedding-based entity deduplication
          4. Semantic co-occurrence + cross-chunk relationship discovery
          5. Store in Weaviate (with vectors) and NebulaGraph
        """
        from semantic_extraction.semantic_extractor import SemanticExtractor

        print("\n== PHASE 1: SEMANTIC EXTRACTION (embedding-first)")
        extractor = SemanticExtractor(
            window_size=3,
            overlap=1,
            similarity_threshold=0.75,
            store_in_weaviate=True,
        )
        sem_result = extractor.extract(text, doc_id="doc_001")

        # Convert semantic entities to dict format for downstream
        entities = [
            {"type": e.type, "value": e.value, "confidence": e.confidence}
            for e in sem_result.entities
        ]
        relationships = [
            {
                "from_id": f"{r.source_type}_{r.source.replace(' ', '_')}",
                "to_id": f"{r.target_type}_{r.target.replace(' ', '_')}",
                "from_type": r.source_type,
                "to_type": r.target_type,
                "type": r.relationship,
                "confidence": r.confidence,
            }
            for r in sem_result.relationships
        ]

        self.results["semantic_extraction"] = sem_result.stats
        self.results["entities"] = entities
        self.results["relationships"] = relationships

        # Phase 2: Agentic workflow validation
        print("\n== PHASE 2: AGENTIC WORKFLOW VALIDATION")
        workflow_result = run_workflow(text)
        self.results["workflow"] = workflow_result

        # Phase 3: Store in Weaviate (chunks already stored by SemanticExtractor)
        print("\n== PHASE 3: VECTOR DATABASE (Weaviate)")
        doc = Document(
            id="doc_001", content=text, entities=entities,
            metadata={"type": "document", "mode": "semantic"},
        )
        vector_result = store_in_weaviate([doc])
        self.results["vector_storage"] = vector_result

        # Phase 4: Store in NebulaGraph
        print("\n== PHASE 4: KNOWLEDGE GRAPH (NebulaGraph)")
        graph_result = store_in_nebula(entities, relationships)
        self.results["graph_storage"] = graph_result

        # Phase 5: Queries
        print("\n== PHASE 5: EXECUTE QUERIES")
        self._demonstrate_queries(entities)

        return self.results
    
    def _generate_relationships(self, entities: List[Dict[str, Any]], text: str = "") -> List[Dict[str, str]]:
        """
        Generate relationships using the full 7-layer pipeline:

        Layer 1 – Candidate Extraction   (already done by entity_extractor)
        Layer 2 – Type Identification    (reclassify entities via domain schema)
        Layer 3 – Relation Mapping       (LLM + proximity + verb patterns)
        Layer 4 – Domain Schema Filter   (reject invalid triples)
        Layer 5 – Confidence Scoring     (score & threshold)
        Layer 6 – Deduplication          (merge duplicate entities/rels)
        Layer 7 – Output                 (ready for graph storage)
        """

        # ── Helper ─────────────────────────────────────────────────────
        def make_entity_id(entity_type: str, entity_value: str) -> str:
            clean = entity_value.replace('\n', ' ').strip().replace(' ', '_')
            while '__' in clean:
                clean = clean.replace('__', '_')
            return f"{entity_type}_{clean}"

        # ══════════════════════════════════════════════════════════════
        # LAYER 2 – TYPE IDENTIFICATION (reclassify wrong types)
        # ══════════════════════════════════════════════════════════════
        reclassified_count = 0
        for e in entities:
            old_type = e.get("type", "unknown").lower()
            new_type = identify_entity_type(e.get("value", ""), old_type)
            if new_type != old_type:
                reclassified_count += 1
                e["type"] = new_type
                e["confidence"] = max(0.55, e.get("confidence", 0.5) - 0.10)

        # ══════════════════════════════════════════════════════════════
        # LAYER 6a – ENTITY DEDUPLICATION (merge similar entities)
        # ══════════════════════════════════════════════════════════════
        entities = deduplicate_entities(entities)

        # ── Group entities by type ─────────────────────────────────
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for e in entities:
            t = e.get("type", "unknown").lower()
            by_type.setdefault(t, []).append(e)

        n_per  = len(by_type.get("person", []))
        n_org  = len(by_type.get("organization", []))
        n_loc  = len(by_type.get("location", []))
        n_proj = len(by_type.get("project", []))
        n_inv  = len(by_type.get("invoice", []))
        n_agr  = len(by_type.get("agreement", []))
        n_role = len(by_type.get("role", []))

        print(f"\n🔎 Relationship Generation (7-Layer Pipeline)")
        print(f"   Layer 2 – Reclassified {reclassified_count} entity types")
        print(f"   Entities: {n_per} persons | {n_org} orgs | {n_loc} locations | "
              f"{n_proj} projects | {n_inv} invoices | {n_agr} agreements | {n_role} roles")

        relationships: List[Dict[str, str]] = []

        # ══════════════════════════════════════════════════════════════
        # LAYER 3 – RELATION MAPPING
        #   3a. LLM-based extraction (most accurate)
        #   3b. Text proximity matching (co-occurrence in sentences)
        #   3c. Verb/pattern matching (NLP-lite)
        # ══════════════════════════════════════════════════════════════

        # --- 3a. LLM-based ---
        llm_rels = self._extract_relationships_with_llm(text, entities)
        if llm_rels:
            print(f"   Layer 3a – LLM extracted {len(llm_rels)} relationship candidates")
            relationships.extend(llm_rels)

        # --- 3b. Proximity-based ---
        sentences = self._split_into_sentences(text) if text else []
        if sentences:
            # For every VALID TRIPLE in the schema, run proximity matching
            # but ONLY for type-pairs that actually have entities.
            type_pair_done: set = set()
            for src_type, rel_type, tgt_type in VALID_TRIPLES:
                pair_key = (src_type, rel_type, tgt_type)
                if pair_key in type_pair_done:
                    continue
                type_pair_done.add(pair_key)

                src_entities = by_type.get(src_type, [])
                tgt_entities = by_type.get(tgt_type, [])
                if not src_entities or not tgt_entities:
                    continue

                # Tight window for person→org (WORKS_AT), wider for others
                window = 2 if rel_type == "WORKS_AT" else 3
                prox_rels = self._proximity_based_relationships(
                    src_entities, tgt_entities, sentences,
                    rel_type, src_type, tgt_type, window=window,
                )
                relationships.extend(prox_rels)

        # --- 3c. Verb-pattern matching (NLP-lite) ---
        if text and sentences:
            verb_rels = self._verb_pattern_relationships(entities, sentences)
            relationships.extend(verb_rels)

        # ══════════════════════════════════════════════════════════════
        # LAYER 4 – DOMAIN SCHEMA FILTER
        #   Reject any relationship whose triple is NOT in VALID_TRIPLES.
        # ══════════════════════════════════════════════════════════════
        before = len(relationships)
        relationships = [
            r for r in relationships
            if validate_relationship_triple(
                r.get("from_type", ""),
                r.get("type", ""),
                r.get("to_type", ""),
            )
        ]
        rejected = before - len(relationships)
        if rejected:
            print(f"   Layer 4 – Schema filter rejected {rejected} invalid triples")

        # ══════════════════════════════════════════════════════════════
        # LAYER 5 – CONFIDENCE SCORING & THRESHOLD
        # ══════════════════════════════════════════════════════════════
        accepted, needs_review = filter_by_confidence(
            relationships, REL_ACCEPT_THRESHOLD, REL_DISCARD_THRESHOLD
        )
        if needs_review:
            print(f"   Layer 5 – {len(needs_review)} relationships below threshold (discarded)")
        relationships = accepted

        # ══════════════════════════════════════════════════════════════
        # LAYER 6b – RELATIONSHIP DEDUPLICATION
        # ══════════════════════════════════════════════════════════════
        relationships = self._deduplicate_relationships(relationships)

        print(f"   ✅ Final: {len(relationships)} relationships")
        return relationships

    # ──────────────────────────────────────────────────────────────────
    # UTILITY: Sentence splitting (abbreviation-aware)
    # ──────────────────────────────────────────────────────────────────
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for proximity analysis.
        
        Handles abbreviations (Pvt., Ltd., Dr., Mr., etc.) to avoid
        false sentence breaks.
        """
        # First, collapse soft line breaks within paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        result = []
        
        # Abbreviations that should NOT trigger sentence breaks
        abbrevs = r'(?:Pvt|Ltd|Inc|Corp|Mr|Mrs|Ms|Dr|Jr|Sr|St|Ave|Blvd|No|vs|etc|approx|dept|govt|est)'
        
        for para in paragraphs:
            para_clean = ' '.join(para.split())
            if not para_clean:
                continue
            # Temporarily protect abbreviation dots
            protected = re.sub(rf'({abbrevs})\.\s', r'\1<DOT> ', para_clean, flags=re.IGNORECASE)
            # Split on real sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', protected)
            # Restore dots
            sentences = [s.replace('<DOT>', '.') for s in sentences]
            result.extend(s.strip() for s in sentences if s.strip())
        return result
    
    def _entity_in_text(self, entity_value: str, text: str) -> bool:
        """Check if entity value appears in a text segment (case-insensitive)"""
        return entity_value.lower() in text.lower()
    
    def _proximity_based_relationships(
        self,
        source_entities: List[Dict[str, Any]],
        target_entities: List[Dict[str, Any]],
        sentences: List[str],
        rel_type: str,
        source_type: str,
        target_type: str,
        window: int = 3  # number of neighboring sentences to consider
    ) -> List[Dict[str, str]]:
        """
        Create relationships only between entities that appear close to each other in the text.
        
        Uses a sliding window of sentences to determine proximity.
        Entities in the same sentence get highest confidence.
        Entities in adjacent sentences get lower confidence.
        """
        relationships = []
        seen = set()
        
        def make_entity_id(entity_type: str, entity_value: str) -> str:
            clean = entity_value.replace('\n', ' ').strip().replace(' ', '_')
            while '__' in clean:
                clean = clean.replace('__', '_')
            return f"{entity_type}_{clean}"
        
        for i, sentence in enumerate(sentences):
            # Build a context window (current sentence + nearby sentences)
            context_sentences = sentences[max(0, i - window):i + window + 1]
            
            # Find source entities in current sentence
            sources_in_sentence = [
                e for e in source_entities
                if self._entity_in_text(e.get('value', ''), sentence)
            ]
            
            if not sources_in_sentence:
                continue
            
            # Find target entities in context window
            context_text = ' '.join(context_sentences)
            for source in sources_in_sentence:
                for target in target_entities:
                    target_val = target.get('value', '')
                    source_val = source.get('value', '')
                    
                    # Skip if source and target are the same
                    if source_val.lower() == target_val.lower():
                        continue
                    
                    if self._entity_in_text(target_val, context_text):
                        # Determine confidence based on distance
                        if self._entity_in_text(target_val, sentence):
                            # Same sentence = highest confidence
                            confidence = 0.92
                        else:
                            # Adjacent sentences = lower confidence
                            confidence = 0.75
                        
                        # Boost from entity confidence
                        source_conf = source.get('confidence', 0.5)
                        target_conf = target.get('confidence', 0.5)
                        confidence = min(0.95, confidence * (source_conf + target_conf) / 2.0)
                        
                        rel_key = (source_val.lower(), target_val.lower(), rel_type)
                        if rel_key not in seen:
                            seen.add(rel_key)
                            relationships.append({
                                "from_id": make_entity_id(source_type, source_val),
                                "to_id": make_entity_id(target_type, target_val),
                                "from_type": source_type,
                                "to_type": target_type,
                                "type": rel_type,
                                "confidence": round(confidence, 3)
                            })
        
        return relationships
    
    def _extract_relationships_with_llm(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Layer 3a — Use LLM to extract meaningful relationships from text.

        The prompt now includes the VALID_TRIPLES so the LLM only produces
        schema-compliant relationships.
        """
        if not text:
            return []

        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'utils', '.env'))

            # Priority: Ollama (local, free) → OpenRouter (cloud, fallback)
            llm = None
            try:
                from utils.ollama_handler import OllamaLLM
                model = os.getenv("OLLAMA_MODEL", "llama3")
                llm = OllamaLLM(model=model)
                print(f"   Layer 3a – Using local Ollama ({model})")
            except Exception as ollama_err:
                print(f"   ⚠ Ollama unavailable ({str(ollama_err)[:50]}), trying OpenRouter…")
                api_key = os.getenv("data_extraction_LiquidAi_api_key")
                if not api_key:
                    return []
                from utils.llm_handler import OpenRouterLLM
                llm = OpenRouterLLM(api_key=api_key)

            # Build the allowed triples list for the prompt
            triple_lines = []
            for src, rel, tgt in sorted(VALID_TRIPLES):
                triple_lines.append(f"  ({src}, {rel}, {tgt})")
            triple_str = "\n".join(triple_lines)

            # Build entity list for context
            entity_summary = []
            for e in entities:
                if e.get('confidence', 0) >= 0.60:
                    entity_summary.append(f"  {e.get('type', 'unknown')}: {e.get('value', '')}")
            entity_str = "\n".join(entity_summary[:40])

            prompt = f"""Extract relationships from this business text. Return a JSON array.

RULES:
1. ONLY use relationship triples from this ALLOWED list (source_type, relationship, target_type):
{triple_str}

2. A person can WORKS_AT exactly ONE organization — pick the one they belong to.
3. An organization can PARTNERS_WITH another organization.
4. NEVER create WORKS_AT between two organizations or two locations.
5. LOCATED_IN is for (organization → location) only.
6. BASED_IN is for (person → location) only.

KNOWN ENTITIES:
{entity_str}

TEXT:
{text[:3000]}

Output format — JSON array:
[{{"source":"John","source_type":"person","target":"Acme Corp","target_type":"organization","relationship":"WORKS_AT"}}]

Return ONLY the JSON array:"""

            raw = llm.chat(prompt)

            # Parse — remove <think>…</think> blocks
            clean_raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

            json_start = clean_raw.find('[')
            json_end = clean_raw.rfind(']') + 1
            if json_start == -1 or json_end <= json_start:
                return []

            json_str = clean_raw[json_start:json_end]
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                json_str = re.sub(r',\s*]', ']', json_str)
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"   ⚠ LLM JSON parse failed, skipping")
                    return []

            def make_entity_id(etype: str, evalue: str) -> str:
                c = evalue.replace('\n', ' ').strip().replace(' ', '_')
                while '__' in c:
                    c = c.replace('__', '_')
                return f"{etype}_{c}"

            relationships = []
            for item in data:
                source = item.get('source', '')
                target = item.get('target', '')
                rel_type = item.get('relationship', '')
                src_type = item.get('source_type', 'entity').lower()
                tgt_type = item.get('target_type', 'entity').lower()

                if not (source and target and rel_type):
                    continue

                # ── Layer 4 pre-filter: only accept valid triples ──
                if not validate_relationship_triple(src_type, rel_type, tgt_type):
                    continue

                relationships.append({
                    "from_id": make_entity_id(src_type, source),
                    "to_id": make_entity_id(tgt_type, target),
                    "from_type": src_type,
                    "to_type": tgt_type,
                    "type": rel_type.upper(),
                    "confidence": 0.90,
                })

            return relationships

        except Exception as e:
            print(f"   ⚠ LLM relationship extraction failed: {str(e)[:80]}")
            return []

    # ──────────────────────────────────────────────────────────────────
    # LAYER 3c – VERB/PATTERN-BASED RELATIONSHIP EXTRACTION
    # ──────────────────────────────────────────────────────────────────

    # Verb patterns: compiled regex → (relationship_type, source_group_type, target_group_type)
    VERB_PATTERNS = [
        # person works/employed at organization
        (re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+"
            r"(?:works?\s+(?:at|for)|employed\s+(?:at|by)|serves?\s+(?:at|for|as\s+\w+\s+at))\s+"
            r"(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ), "WORKS_AT", "person", "organization"),

        # person, <Role> at organization
        (re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*,\s*"
            r"(?:CEO|CTO|CFO|COO|Director|Manager|Lead|Head|Partner|Officer|Advocate|Consultant)\s+"
            r"(?:at|of)\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ), "WORKS_AT", "person", "organization"),

        # organization headquartered/based in location
        (re.compile(
            r"\b(.+?)\s*,?\s+"
            r"(?:headquartered|based|located|situated)\s+in\s+"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.IGNORECASE,
        ), "LOCATED_IN", "organization", "location"),

        # organization has partnered with organization
        (re.compile(
            r"\b(.+?)\s+(?:has\s+)?partner(?:ed|s)?\s+with\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ), "PARTNERS_WITH", "organization", "organization"),

        # organization awarded contract/project
        (re.compile(
            r"\b(.+?)\s+(?:has\s+been\s+)?awarded\s+(?:a\s+)?(?:contract|project)\b",
            re.IGNORECASE,
        ), "AWARDED", "organization", "project"),

        # organization audited by organization
        (re.compile(
            r"\b(.+?)\s+(?:audited|reviewed)\s+(?:by|quarterly\s+by)\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ), "AUDITED_BY", "organization", "organization"),

        # organization engaged organization
        (re.compile(
            r"\b(.+?)\s+(?:has\s+)?(?:also\s+)?engaged\s+(.+?)(?:\.|,|$)",
            re.IGNORECASE,
        ), "ENGAGED", "organization", "organization"),
    ]

    def _verb_pattern_relationships(
        self, entities: List[Dict[str, Any]], sentences: List[str]
    ) -> List[Dict[str, str]]:
        """
        Layer 3c — Extract relationships by matching verb/preposition patterns
        in sentences, then resolving the captured groups to known entities.
        """
        relationships = []
        seen: set = set()

        # Build entity lookup: value_lower → entity dict
        entity_lookup: Dict[str, Dict[str, Any]] = {}
        for e in entities:
            val = e.get("value", "").strip()
            if val:
                entity_lookup[val.lower()] = e

        def make_entity_id(etype: str, evalue: str) -> str:
            c = evalue.replace('\n', ' ').strip().replace(' ', '_')
            while '__' in c:
                c = c.replace('__', '_')
            return f"{etype}_{c}"

        def resolve_entity(captured: str, expected_type: str) -> Dict[str, Any] | None:
            """Try to match a regex-captured string to a known entity."""
            cap_lower = captured.strip().lower()
            # Exact match
            if cap_lower in entity_lookup:
                ent = entity_lookup[cap_lower]
                if ent.get("type", "").lower() == expected_type:
                    return ent
            # Substring containment
            for val, ent in entity_lookup.items():
                if ent.get("type", "").lower() != expected_type:
                    continue
                if cap_lower in val or val in cap_lower:
                    return ent
            return None

        for sentence in sentences:
            for pattern, rel_type, src_type, tgt_type in self.VERB_PATTERNS:
                for match in pattern.finditer(sentence):
                    groups = match.groups()
                    if len(groups) < 1:
                        continue

                    src_captured = groups[0].strip()
                    tgt_captured = groups[1].strip() if len(groups) > 1 else ""

                    src_ent = resolve_entity(src_captured, src_type)
                    tgt_ent = resolve_entity(tgt_captured, tgt_type) if tgt_captured else None

                    if src_ent and tgt_ent:
                        src_val = src_ent.get("value", "")
                        tgt_val = tgt_ent.get("value", "")
                        key = (src_val.lower(), tgt_val.lower(), rel_type)
                        if key not in seen:
                            seen.add(key)
                            relationships.append({
                                "from_id": make_entity_id(src_type, src_val),
                                "to_id": make_entity_id(tgt_type, tgt_val),
                                "from_type": src_type,
                                "to_type": tgt_type,
                                "type": rel_type,
                                "confidence": 0.88,
                            })

        return relationships
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships, keeping the one with highest confidence"""
        seen: Dict[Tuple[str, str, str], Dict] = {}
        for rel in relationships:
            key = (rel.get("from_id", ""), rel.get("to_id", ""), rel.get("type", ""))
            existing = seen.get(key)
            if existing is None or rel.get("confidence", 0) > existing.get("confidence", 0):
                seen[key] = rel
        return list(seen.values())
    
    def _demonstrate_queries(self, entities: List[Dict[str, Any]]):
        """Execute semantic and graph queries"""
        
        # Semantic Query - Weaviate
        print("\n" + "-"*70)
        print("SEMANTIC QUERY (Weaviate)")
        print("-"*70)
        query_text = "contracts and agreements"
        print(f"Query: '{query_text}'")
        print("Expected: Return similar documents using vector similarity")
        
        try:
            from vector_database.weaviate_handler import query_weaviate
            results = query_weaviate(query_text)
            if results:
                print(f"\n✓ Found {len(results)} similar documents:")
                for i, result in enumerate(results[:3], 1):  # Show top 3
                    print(f"  {i}. Score: {result.get('score', 'N/A')}")
            else:
                print("(No results or Weaviate not responding)")
        except Exception as e:
            print(f"(Query execution error: {str(e)[:50]}...)")
        
        # Graph Query - NebulaGraph
        print("\n" + "-"*70)
        print("GRAPH QUERY (NebulaGraph)")
        print("-"*70)
        
        persons = [e for e in entities if e.get("type") == "person"]
        if persons:
            try:
                from knowledge_graph.nebula_handler import execute_graph_query
                person_name = persons[0].get("value", "")
                # Build entity ID like the storage uses
                person_id = f'person_{person_name.replace(" ", "_")}'
                
                # NebulaGraph 3.x requires YIELD clause in FETCH
                query = f'FETCH PROP ON Person "{person_id}" YIELD properties(vertex) AS props;'
                print(f"Query: Fetch all properties of person: {person_name}")
                print("Expected: Return person node with all attributes")
                
                result = execute_graph_query(query)
                if result:
                    print(f"✓ Query executed successfully")
                    print(f"Result: {result}")
                else:
                    print("(Query returned no results)")
            except Exception as e:
                print(f"(Query execution error: {str(e)[:50]}...)")
        else:
            print("No persons found to query")
        
        # Relationship Query - NebulaGraph
        print("\n" + "-"*70)
        print("RELATIONSHIP QUERY (NebulaGraph)")
        print("-"*70)
        print("Query: Find all organizations associated with extracted persons")
        print("Expected: Return graph paths showing person->works_at->organization")
        
        try:
            from knowledge_graph.nebula_handler import execute_graph_query
            
            # Find all person->works_at->organization relationships
            # In NebulaGraph 3.x, use tag-qualified property access: p.Person.name
            query = 'MATCH (p:Person)-[e:WORKS_AT]->(o:Organization) RETURN p.Person.name AS person, o.Organization.name AS org LIMIT 10;'
            print(f"\n✓ Executing relationship query...")
            result = execute_graph_query(query)
            if result:
                print(f"Found relationships:")
                print(result)
            else:
                print("(No relationships found or NebulaGraph not responding)")
        except Exception as e:
            print(f"(Query execution error: {str(e)[:50]}...)")


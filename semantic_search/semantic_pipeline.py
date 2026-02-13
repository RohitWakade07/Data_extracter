# Semantic Graph Pipeline
# Combines semantic search (Weaviate) with knowledge graph (NebulaGraph)
# for intelligent text-to-insight conversion

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

from entity_extraction.entity_extractor import extract_from_text, ExtractionResult
from vector_database.weaviate_handler import Document, WeaviateClient
from knowledge_graph.nebula_handler import NebulaGraphClient, store_in_nebula
from utils.domain_schema import clean_entity_value, deduplicate_entities


@dataclass
class ProcessedDocument:
    """Result of processing a document through the semantic graph pipeline"""
    document_id: str
    original_text: str
    summary: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    weaviate_id: Optional[str] = None
    graph_stored: bool = False
    category: str = "general"


@dataclass
class SemanticQueryResult:
    """Result of a semantic + graph query"""
    query: str
    semantic_matches: List[Dict[str, Any]]
    graph_paths: List[Dict[str, Any]]
    entities_involved: List[Dict[str, Any]]
    answer_summary: str = ""


class SemanticGraphPipeline:
    """
    End-to-end pipeline for:
    1. Document ingestion with entity extraction
    2. Semantic storage in Weaviate
    3. Knowledge graph storage in NebulaGraph
    4. Combined semantic + graph queries
    
    Example Use Cases:
    - "Why are shipments getting delayed?" → Finds port congestion, weather events
    - "Which companies affected by Mumbai port?" → Graph traversal to find indirect connections
    - "Find similar cyber security incidents" → Pattern matching across documents
    """
    
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        nebula_host: str = "127.0.0.1",
        nebula_port: int = 9669,
        llm_provider: str = "gemini"
    ):
        self.weaviate_url = weaviate_url
        self.nebula_host = nebula_host
        self.nebula_port = nebula_port
        self.llm_provider = llm_provider
        
        # Initialize clients
        self.weaviate = WeaviateClient(weaviate_url)
        self.nebula = NebulaGraphClient(nebula_host, nebula_port)
        
        # Create schemas
        self._initialize_schemas()
    
    def _initialize_schemas(self):
        """Initialize Weaviate and NebulaGraph schemas"""
        try:
            # Weaviate schema
            if self.weaviate.client:
                self.weaviate.create_schema()
                
                # Also create enhanced semantic schema
                from semantic_search.semantic_engine import SemanticSearchEngine
                semantic_engine = SemanticSearchEngine(self.weaviate_url)
                semantic_engine.create_semantic_schema()
                
            # NebulaGraph schema
            if self.nebula.pool:
                self.nebula.create_space()
                
        except Exception as e:
            print(f"⚠ Schema initialization: {e}")
    
    def ingest_document(
        self,
        text: str,
        document_id: str,
        source_type: str = "document",
        category: str = "general"
    ) -> ProcessedDocument:
        """
        Process a document through the complete pipeline.
        
        1. Extract entities using LLM
        2. Generate relationships
        3. Store in Weaviate (semantic search)
        4. Store in NebulaGraph (knowledge graph)
        
        Args:
            text: Document text
            document_id: Unique identifier
            source_type: pdf, email, report, etc.
            category: supply_chain, logistics, incident, contract, etc.
        
        Returns:
            ProcessedDocument with all extraction results
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING DOCUMENT: {document_id}")
        print(f"{'='*60}")
        
        # Step 1: Extract entities
        print("\n▶ Step 1: Extracting entities...")
        extraction_result = extract_from_text(text, self.llm_provider)
        
        entities = [
            {
                "type": e.type,
                "value": e.value,
                "confidence": e.confidence
            }
            for e in extraction_result.entities
        ]
        
        # Step 1b: Clean and filter entities (remove noise, blocklist, deduplicate)
        entities = self._clean_and_filter_entities(entities)
        
        # Enhance entities with supply chain specific types
        entities = self._enhance_entities_for_supply_chain(entities, text)
        
        print(f"   ✓ Extracted {len(entities)} entities")
        
        # Step 2: Generate relationships
        print("\n▶ Step 2: Generating relationships...")
        relationships = self._generate_relationships(entities, text, category)
        print(f"   ✓ Generated {len(relationships)} relationships")
        
        # Step 3: Store in Weaviate
        print("\n▶ Step 3: Storing in Weaviate (semantic search)...")
        weaviate_id = self._store_in_weaviate(
            text, document_id, entities, source_type, category
        )
        
        # Step 4: Store in NebulaGraph
        print("\n▶ Step 4: Storing in NebulaGraph (knowledge graph)...")
        graph_result = store_in_nebula(
            entities, 
            relationships,
            self.nebula_host,
            self.nebula_port
        )
        
        # Create summary
        summary = self._generate_summary(text, entities)
        
        return ProcessedDocument(
            document_id=document_id,
            original_text=text,
            summary=summary,
            entities=entities,
            relationships=relationships,
            weaviate_id=weaviate_id,
            graph_stored=graph_result.get("entities_added", 0) > 0,
            category=category
        )
    
    def _clean_and_filter_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Clean and filter entities using domain schema validation.
        
        - Cleans entity values (removes trailing noise words like "shall")
        - Blocks invalid entities (role names, date-related terms)
        - Corrects entity types
        - Deduplicates (merges "HDFC" with "HDFC Bank", etc.)
        """
        cleaned = []
        
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            original_value = entity.get("value", "")
            
            # Skip empty values
            if not original_value or not original_value.strip():
                continue
            
            # Clean the value (remove trailing words like "shall", "and", etc.)
            cleaned_value = clean_entity_value(original_value, entity_type)
            if not cleaned_value or len(cleaned_value) < 2:
                continue
            
            cleaned.append({
                "type": entity_type,
                "value": cleaned_value,
                "confidence": entity.get("confidence", 0.5)
            })
        
        # Deduplicate (handles honorifics, abbreviations, substrings)
        return deduplicate_entities(cleaned)
    
    def _enhance_entities_for_supply_chain(
        self,
        entities: List[Dict[str, Any]],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Detect and add supply chain specific entities from text.
        
        Looks for:
        - Ports, warehouses, routes
        - Delays, congestion, disruptions
        - Weather events
        - Shipment references
        """
        import re
        
        # Define supply chain patterns
        patterns = {
            # Ports
            "port": [
                r"(?i)([\w\s]+)\s+port",
                r"(?i)port\s+of\s+([\w\s]+)",
                r"(?i)([\w]+)\s+harbor",
            ],
            # Events
            "delay": [
                r"(?i)(shipment\s+delay[s]?)",
                r"(?i)(delivery\s+delay[s]?)",
                r"(?i)([\w\s]+)\s+delay[s]?\s+(?:due\s+to|caused\s+by)",
            ],
            "congestion": [
                r"(?i)(port\s+congestion)",
                r"(?i)(traffic\s+congestion)",
                r"(?i)([\w\s]+)\s+congestion",
                r"(?i)(logistics\s+bottleneck)",
            ],
            "weather_event": [
                r"(?i)(heavy\s+rain(?:fall)?)",
                r"(?i)(monsoon)",
                r"(?i)(flood(?:ing)?)",
                r"(?i)(storm)",
                r"(?i)(cyclone)",
                r"(?i)(hurricane)",
            ],
            # Shipment
            "shipment": [
                r"(?i)shipment\s+(?:id|number|#)?\s*([A-Z0-9-]+)",
                r"(?i)(cargo|freight)\s+shipment",
            ],
            # Incident
            "incident": [
                r"(?i)(security\s+breach)",
                r"(?i)(cyber\s+attack)",
                r"(?i)(data\s+breach)",
                r"(?i)(system\s+failure)",
                r"(?i)(supply\s+chain\s+disruption)",
            ]
        }
        
        enhanced = list(entities)
        existing_values = {e.get('value', '').lower() for e in entities}
        
        for entity_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                for match in re.finditer(pattern, text):
                    value = match.group(1) if match.groups() else match.group(0)
                    value = value.strip()
                    
                    if value.lower() not in existing_values and len(value) > 2:
                        enhanced.append({
                            "type": entity_type,
                            "value": value,
                            "confidence": 0.85
                        })
                        existing_values.add(value.lower())
        
        return enhanced
    
    def _generate_relationships(
        self,
        entities: List[Dict[str, Any]],
        text: str,
        category: str
    ) -> List[Dict[str, Any]]:
        """Generate relationships based on entity types and context"""
        relationships = []
        
        def make_id(entity_type: str, value: str) -> str:
            clean = value.replace('\n', ' ').strip().replace(' ', '_')
            while '__' in clean:
                clean = clean.replace('__', '_')
            return f"{entity_type.lower()}_{clean}"
        
        # Group entities by type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for e in entities:
            t = e.get('type', 'unknown').lower()
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        # Define relationship rules based on category
        relationship_rules = {
            # Supply chain relationships
            ("organization", "shipment"): "SENT",
            ("shipment", "organization"): "RECEIVED_BY",
            ("organization", "port"): "OPERATES_AT",
            ("shipment", "port"): "ROUTED_THROUGH",
            ("shipment", "delay"): "EXPERIENCED",
            ("delay", "weather_event"): "CAUSED_BY",
            ("delay", "congestion"): "CAUSED_BY",
            ("port", "congestion"): "HAS",
            ("organization", "location"): "LOCATED_IN",
            
            # Incident relationships
            ("incident", "organization"): "AFFECTED",
            ("organization", "incident"): "EXPERIENCED",
            ("weather_event", "location"): "OCCURRED_AT",
            
            # Business relationships
            ("person", "organization"): "WORKS_AT",
            ("organization", "project"): "MANAGES",
            ("organization", "agreement"): "PARTY_TO",
            ("agreement", "amount"): "HAS_VALUE",
            ("invoice", "amount"): "HAS_AMOUNT",
        }
        
        # Generate relationships based on rules
        for (from_type, to_type), rel_type in relationship_rules.items():
            from_entities = by_type.get(from_type, [])
            to_entities = by_type.get(to_type, [])
            
            for from_e in from_entities:
                for to_e in to_entities:
                    # Check context - entities should be mentioned close together
                    from_val = from_e.get('value', '')
                    to_val = to_e.get('value', '')
                    
                    # Simple proximity check
                    if from_val in text and to_val in text:
                        from_pos = text.find(from_val)
                        to_pos = text.find(to_val)
                        
                        # If entities are within 500 chars, likely related
                        if abs(from_pos - to_pos) < 500:
                            relationships.append({
                                "from_id": make_id(from_type, from_val),
                                "to_id": make_id(to_type, to_val),
                                "from_type": from_type,
                                "to_type": to_type,
                                "type": rel_type,
                                "confidence": 0.8
                            })
        
        return relationships
    
    def _store_in_weaviate(
        self,
        text: str,
        document_id: str,
        entities: List[Dict[str, Any]],
        source_type: str,
        category: str
    ) -> Optional[str]:
        """Store document in Weaviate for semantic search"""
        try:
            # Store in legacy schema for compatibility
            doc = Document(
                id=document_id,
                content=text,
                entities=entities,
                metadata={"type": source_type, "category": category}
            )
            uuid = self.weaviate.store_document(doc)
            
            # Also store in enhanced semantic schema
            from semantic_search.semantic_engine import SemanticSearchEngine
            engine = SemanticSearchEngine(self.weaviate_url)
            engine.store_document(
                content=text,
                document_id=document_id,
                entities=entities,
                source_type=source_type,
                category=category
            )
            
            return uuid
            
        except Exception as e:
            print(f"⚠ Weaviate storage error: {e}")
            return None
    
    def _generate_summary(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> str:
        """Generate a brief summary of the document"""
        # Extract key entity values for summary
        key_entities = []
        for e in entities[:5]:  # Top 5 entities
            value = e.get('value', '')
            if value:
                key_entities.append(f"{e.get('type', 'unknown')}: {value}")
        
        summary = text[:200] + "..." if len(text) > 200 else text
        if key_entities:
            summary += f"\n\nKey entities: {', '.join(key_entities)}"
        
        return summary
    
    def semantic_query(
        self,
        query: str,
        include_graph_context: bool = True,
        limit: int = 10
    ) -> SemanticQueryResult:
        """
        Execute a semantic query that combines vector search with graph traversal.
        
        Example queries:
        - "Why are shipments getting delayed?"
        - "Which companies are affected by port congestion?"
        - "Find incidents similar to cyber attacks"
        
        Args:
            query: Natural language query
            include_graph_context: Whether to enrich with graph relationships
            limit: Maximum results
        
        Returns:
            SemanticQueryResult with combined insights
        """
        print(f"\n{'='*60}")
        print(f"SEMANTIC QUERY: {query}")
        print(f"{'='*60}")
        
        # Step 1: Semantic search in Weaviate
        print("\n▶ Step 1: Semantic search...")
        from semantic_search.semantic_engine import SemanticSearchEngine
        engine = SemanticSearchEngine(self.weaviate_url)
        semantic_results = engine.semantic_search(query, limit)
        
        matches = [
            {
                "content": r.content[:300] + "..." if len(r.content) > 300 else r.content,
                "score": r.score,
                "highlights": r.highlights,
                "source_type": r.source_type
            }
            for r in semantic_results
        ]
        
        print(f"   ✓ Found {len(matches)} semantic matches")
        
        # Step 2: Extract entities from results
        all_entities = []
        for r in semantic_results:
            all_entities.extend(r.entities)
        
        # Deduplicate
        unique_entities = {
            e.get('value', e.get('name', '')): e 
            for e in all_entities
        }
        
        # Step 3: Graph traversal for context
        graph_paths = []
        if include_graph_context and unique_entities:
            print("\n▶ Step 2: Graph traversal...")
            from semantic_search.graph_traversal import GraphTraversal
            graph = GraphTraversal(self.nebula_host, self.nebula_port)
            
            for entity_name, entity in list(unique_entities.items())[:5]:  # Top 5
                entity_type = entity.get('type', 'unknown')
                relationships = graph.get_entity_relationships(entity_name, entity_type)
                
                if relationships:
                    graph_paths.append({
                        "entity": entity_name,
                        "type": entity_type,
                        "relationships": relationships[:10]  # Limit relationships
                    })
            
            print(f"   ✓ Found {len(graph_paths)} graph contexts")
        
        # Step 4: Generate answer summary
        answer = self._generate_answer(query, matches, graph_paths)
        
        return SemanticQueryResult(
            query=query,
            semantic_matches=matches,
            graph_paths=graph_paths,
            entities_involved=list(unique_entities.values()),
            answer_summary=answer
        )
    
    def find_affected_companies(
        self,
        event: str,
        event_type: str = "Location"
    ) -> Dict[str, Any]:
        """
        Find companies affected by an event (e.g., port congestion).
        
        Example: "Which companies are indirectly affected by Mumbai port congestion?"
        
        Args:
            event: Event name (e.g., "Mumbai Port", "Weather Event")
            event_type: Type of event entity
        
        Returns:
            Dict with affected companies and their connection paths
        """
        print(f"\n▶ Finding companies affected by: {event}")
        
        from semantic_search.graph_traversal import GraphTraversal
        graph = GraphTraversal(self.nebula_host, self.nebula_port)
        
        # Get directly and indirectly affected companies
        companies = graph.find_companies_affected_by(event, event_type)
        
        # Also search semantically for related documents
        from semantic_search.semantic_engine import SemanticSearchEngine
        engine = SemanticSearchEngine(self.weaviate_url)
        
        docs = engine.semantic_search(f"companies affected by {event}", limit=5)
        
        # Extract additional companies from semantic results
        for doc in docs:
            for entity in doc.entities:
                if entity.get('type', '').lower() == 'organization':
                    company_name = entity.get('value', entity.get('name', ''))
                    if company_name and not any(c.get('name') == company_name for c in companies):
                        companies.append({
                            "name": company_name,
                            "connection_type": "semantic_match",
                            "affected_by": event
                        })
        
        return {
            "event": event,
            "event_type": event_type,
            "affected_companies": companies,
            "total_found": len(companies)
        }
    
    def find_similar_patterns(
        self,
        pattern_description: str
    ) -> Dict[str, Any]:
        """
        Find similar incidents/patterns in the knowledge base.
        
        Example: "Find incidents similar to cyber attacks on cloud vendors"
        
        Args:
            pattern_description: Description of the pattern to match
        
        Returns:
            Dict with similar patterns and their occurrences
        """
        print(f"\n▶ Finding patterns similar to: {pattern_description}")
        
        patterns = []
        
        try:
            # Semantic search for similar documents
            from semantic_search.semantic_engine import SemanticSearchEngine
            engine = SemanticSearchEngine(self.weaviate_url)
            
            semantic_matches = engine.semantic_search(pattern_description, limit=10)
            
            # From semantic search - ensure all required fields have defaults
            for match in semantic_matches:
                content = match.content if match.content else ""
                patterns.append({
                    "source": "semantic_search",
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "score": float(match.score) if match.score is not None else 0.5,
                    "category": match.metadata.get('category', 'general') if match.metadata else 'general',
                    "entities": match.entities if match.entities else []
                })
        except Exception as e:
            print(f"⚠ Semantic search error in find_similar_patterns: {e}")
        
        try:
            # Graph search for similar incidents
            from semantic_search.graph_traversal import GraphTraversal
            graph = GraphTraversal(self.nebula_host, self.nebula_port)
            
            incident_matches = graph.find_similar_incidents(pattern_description)
            
            # From graph - ensure all required fields have defaults
            for incident in incident_matches:
                name = incident.get('name') or ''
                patterns.append({
                    "source": "knowledge_graph",
                    "content": name,  # Use name as content
                    "name": name,
                    "score": 0.6,  # Default score for graph matches
                    "category": incident.get('category') or 'general',
                    "pattern_match": incident.get('pattern_match') or '',
                    "entities": []  # Graph matches don't have entities in same format
                })
        except Exception as e:
            print(f"⚠ Graph search error in find_similar_patterns: {e}")
        
        # Group by category - ensure we handle None categories
        categories: Dict[str, int] = {}
        for p in patterns:
            cat = p.get('category') or 'general'
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "pattern_query": pattern_description,
            "similar_patterns": patterns,
            "categories": categories,
            "total_found": len(patterns)
        }
    
    def _generate_answer(
        self,
        query: str,
        matches: List[Dict[str, Any]],
        graph_paths: List[Dict[str, Any]]
    ) -> str:
        """Generate a natural language answer from search results using LLM."""
        if not matches:
            return "No relevant information found."

        # Build context from matches
        context_parts = []
        for i, match in enumerate(matches[:3], 1):
            content = match.get('content', '')[:500]
            if content:
                context_parts.append(f"Document {i}: {content}")

        context = "\n\n".join(context_parts)

        # Add graph context
        graph_info = ""
        if graph_paths:
            graph_lines = []
            for path in graph_paths[:3]:
                entity = path.get('entity', '')
                for rel in path.get('relationships', [])[:2]:
                    conn = rel.get('connected_entity', '')
                    rel_type = rel.get('type', '')
                    graph_lines.append(f"- {entity} → {rel_type} → {conn}")
            if graph_lines:
                graph_info = "\n\nKnowledge Graph relationships:\n" + "\n".join(graph_lines)

        # Use LLM for answer generation
        try:
            import os
            from utils.ollama_handler import OllamaLLM
            llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3"))
            prompt = (
                f'Based on the following search results, answer this question:\n'
                f'Question: "{query}"\n\n'
                f'{context}{graph_info}\n\n'
                f'Provide a clear, concise answer (2-5 sentences) based ONLY on '
                f'the information above.'
            )
            answer = llm.chat(prompt)
            return answer.strip()
        except Exception as e:
            print(f"⚠ LLM answer generation failed: {e}")
            # Fallback
            answer_parts = []
            for match in matches[:3]:
                highlights = match.get('highlights', [])
                if highlights:
                    answer_parts.append(f"• {highlights[0]}")
                else:
                    content = match.get('content', '')[:150]
                    if content:
                        answer_parts.append(f"• {content}...")
            return "\n".join(answer_parts) if answer_parts else "Results found but no summary available."


# Convenience function for quick queries
def process_and_query(
    documents: List[Tuple[str, str]],  # (text, document_id)
    query: str,
    weaviate_url: str = "http://localhost:8080",
    nebula_host: str = "127.0.0.1"
) -> SemanticQueryResult:
    """
    Process documents and run a semantic query.
    
    Example:
        docs = [
            ("Company X faced shipment delays due to heavy rainfall near Mumbai port.", "doc1"),
            ("Port congestion affecting multiple logistics partners.", "doc2")
        ]
        result = process_and_query(docs, "Which companies affected by weather?")
    """
    pipeline = SemanticGraphPipeline(weaviate_url, nebula_host)
    
    # Process documents
    for text, doc_id in documents:
        pipeline.ingest_document(text, doc_id, category="supply_chain")
    
    # Run query
    return pipeline.semantic_query(query)

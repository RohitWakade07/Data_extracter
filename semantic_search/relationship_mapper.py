# Enhanced Relationship Mapper
# Provides reliable relationship extraction and mapping for semantic search answers
# Generates relationship-based answers with explanation charts

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re
import json


@dataclass
class MappedRelationship:
    """A mapped relationship with context"""
    source: str
    source_type: str
    target: str
    target_type: str
    relationship_type: str
    confidence: float
    context_snippet: str = ""
    mapping_reason: str = ""


@dataclass
class RelationshipAnswer:
    """Answer with relationship mappings and explanations"""
    query: str
    direct_answer: str
    entities_found: List[Dict[str, Any]]
    relationships: List[MappedRelationship]
    relationship_table: List[Dict[str, str]]  # For chart display
    relationship_explanation: str
    graph_summary: Dict[str, Any]


class EnhancedRelationshipMapper:
    """
    Provides reliable relationship mapping with multiple strategies:
    1. Rule-based mapping (entity type pairs)
    2. Context-based mapping (proximity and co-occurrence)
    3. Pattern-based mapping (regex patterns for specific relationships)
    4. LLM-enhanced mapping (when available)
    """
    
    # Extended relationship rules with bidirectional support
    RELATIONSHIP_RULES = {
        # Employment relationships
        ("person", "organization"): ("WORKS_AT", "employs"),
        ("organization", "person"): ("EMPLOYS", "has employee"),
        
        # Location relationships
        ("person", "location"): ("LOCATED_IN", "is based in"),
        ("organization", "location"): ("LOCATED_IN", "is headquartered in"),
        ("location", "organization"): ("HOSTS", "is home to"),
        
        # Project relationships
        ("person", "project"): ("WORKS_ON", "is assigned to"),
        ("organization", "project"): ("MANAGES", "owns"),
        ("project", "organization"): ("OWNED_BY", "is managed by"),
        
        # Agreement/Contract relationships
        ("person", "agreement"): ("PARTY_TO", "signed"),
        ("organization", "agreement"): ("PARTY_TO", "is party to"),
        ("agreement", "organization"): ("INVOLVES", "involves"),
        ("agreement", "amount"): ("HAS_VALUE", "is worth"),
        ("agreement", "date"): ("SIGNED_ON", "was signed on"),
        
        # Invoice relationships
        ("organization", "invoice"): ("ISSUED", "issued"),
        ("invoice", "organization"): ("BILLED_TO", "is billed to"),
        ("invoice", "amount"): ("HAS_AMOUNT", "is for"),
        ("invoice", "date"): ("DUE_ON", "is due on"),
        
        # Supply chain relationships
        ("organization", "shipment"): ("SENT", "shipped"),
        ("shipment", "organization"): ("RECEIVED_BY", "was received by"),
        ("shipment", "location"): ("ROUTED_THROUGH", "passes through"),
        ("shipment", "port"): ("ROUTED_THROUGH", "goes via"),
        ("organization", "port"): ("OPERATES_AT", "operates at"),
        ("organization", "supplier"): ("SUPPLIED_BY", "is supplied by"),
        ("supplier", "organization"): ("SUPPLIES_TO", "supplies to"),
        
        # Event/Incident relationships
        ("shipment", "delay"): ("EXPERIENCED", "was delayed by"),
        ("delay", "weather_event"): ("CAUSED_BY", "was caused by"),
        ("delay", "congestion"): ("CAUSED_BY", "resulted from"),
        ("port", "congestion"): ("HAS", "experienced"),
        ("incident", "organization"): ("AFFECTED", "impacted"),
        ("organization", "incident"): ("EXPERIENCED", "was affected by"),
        ("weather_event", "location"): ("OCCURRED_AT", "happened at"),
        
        # Generic fallback
        ("entity", "entity"): ("RELATED_TO", "is related to"),
    }
    
    # Patterns to detect relationship context in text
    RELATIONSHIP_PATTERNS = {
        "WORKS_AT": [
            r"(\w+(?:\s+\w+)?)\s+(?:works?\s+(?:at|for|with)|employed\s+(?:at|by)|is\s+(?:an?\s+)?employee\s+(?:of|at))\s+(\w+(?:\s+\w+)?)",
            r"(\w+(?:\s+\w+)?)\s+(?:is|was)\s+(?:the\s+)?(?:CEO|CTO|CFO|manager|director|lead|head|founder)\s+(?:of|at)\s+(\w+(?:\s+\w+)?)",
            r"(\w+(?:\s+\w+)?),?\s+(?:of|from)\s+(\w+(?:\s+\w+)?)",
        ],
        "LOCATED_IN": [
            r"(\w+(?:\s+\w+)?)\s+(?:is\s+)?(?:located|based|headquartered|situated)\s+(?:in|at)\s+(\w+(?:\s+\w+)?)",
            r"(\w+(?:\s+\w+)?)\s+(?:in|from)\s+(\w+(?:\s+\w+)?)",
        ],
        "PARTY_TO": [
            r"(\w+(?:\s+\w+)?)\s+(?:signed|entered|agreed\s+to)\s+(?:the\s+)?(?:agreement|contract|deal)\s+(?:with\s+)?(\w+(?:\s+\w+)?)?",
        ],
        "MANAGES": [
            r"(\w+(?:\s+\w+)?)\s+(?:manages?|leads?|heads?|runs?|oversees?)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        ],
        "SUPPLIED_BY": [
            r"(\w+(?:\s+\w+)?)\s+(?:is\s+)?(?:supplied|provided|delivered)\s+by\s+(\w+(?:\s+\w+)?)",
        ],
    }
    
    # Question type to expected relationship mappings
    QUESTION_RELATIONSHIP_MAP = {
        "who works": ["WORKS_AT", "EMPLOYED_BY", "EMPLOYS"],
        "who is employed": ["WORKS_AT", "EMPLOYED_BY"],
        "employees at": ["WORKS_AT", "EMPLOYS"],
        "employees of": ["WORKS_AT", "EMPLOYS"],
        "works at": ["WORKS_AT"],
        "works for": ["WORKS_AT"],
        "located in": ["LOCATED_IN", "BASED_IN"],
        "where is": ["LOCATED_IN", "BASED_IN"],
        "which companies": ["WORKS_AT", "LOCATED_IN", "PARTY_TO"],
        "affected by": ["AFFECTED", "IMPACTED", "EXPERIENCED"],
        "caused by": ["CAUSED_BY"],
        "supplies": ["SUPPLIED_BY", "SUPPLIES_TO"],
        "manages": ["MANAGES", "OWNED_BY"],
        "agreement": ["PARTY_TO", "SIGNED_ON"],
        "contract": ["PARTY_TO", "SIGNED_ON"],
    }

    def __init__(self, nebula_host: str = "127.0.0.1", nebula_port: int = 9669):
        self.nebula_host = nebula_host
        self.nebula_port = nebula_port
        self._graph_client = None
    
    @property
    def graph_client(self):
        """Lazy initialization of graph client"""
        if self._graph_client is None:
            try:
                from semantic_search.graph_traversal import GraphTraversal
                self._graph_client = GraphTraversal(self.nebula_host, self.nebula_port)
            except Exception as e:
                print(f"⚠ Graph client initialization failed: {e}")
        return self._graph_client
    
    def map_relationships_for_entities(
        self,
        entities: List[Dict[str, Any]],
        text: str,
        query: str = ""
    ) -> List[MappedRelationship]:
        """
        Map relationships between entities using multiple strategies.
        
        Args:
            entities: List of extracted entities
            text: Original document text
            query: Optional query for context-aware mapping
        
        Returns:
            List of mapped relationships with confidence scores
        """
        relationships = []
        
        # Strategy 1: Rule-based mapping
        rule_based = self._rule_based_mapping(entities, text)
        relationships.extend(rule_based)
        
        # Strategy 2: Pattern-based mapping from text
        pattern_based = self._pattern_based_mapping(entities, text)
        relationships.extend(pattern_based)
        
        # Strategy 3: Graph database lookup for existing relationships
        graph_based = self._graph_based_mapping(entities)
        relationships.extend(graph_based)
        
        # Strategy 4: Context-aware mapping based on query
        if query:
            context_based = self._context_aware_mapping(entities, text, query)
            relationships.extend(context_based)
        
        # Deduplicate and merge confidence scores
        return self._deduplicate_relationships(relationships)
    
    def _rule_based_mapping(
        self,
        entities: List[Dict[str, Any]],
        text: str
    ) -> List[MappedRelationship]:
        """Map relationships using predefined rules with context validation"""
        relationships = []
        
        # First deduplicate similar entities
        entities = self._deduplicate_similar_entities(entities)
        
        # Group entities by type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for e in entities:
            t = e.get('type', 'unknown').lower()
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        # Apply rules for each type pair
        for (from_type, to_type), (rel_type, description) in self.RELATIONSHIP_RULES.items():
            from_entities = by_type.get(from_type, [])
            to_entities = by_type.get(to_type, [])
            
            for from_e in from_entities:
                for to_e in to_entities:
                    from_val = from_e.get('value', from_e.get('name', ''))
                    to_val = to_e.get('value', to_e.get('name', ''))
                    
                    if from_val == to_val:
                        continue  # Skip self-relationships
                    
                    # Validate relationship in context - THIS IS THE KEY FIX
                    is_valid, context, confidence = self._validate_relationship_in_context(
                        from_val, to_val, rel_type, text
                    )
                    
                    if is_valid and confidence > 0.5:  # Stricter threshold
                        relationships.append(MappedRelationship(
                            source=from_val,
                            source_type=from_type,
                            target=to_val,
                            target_type=to_type,
                            relationship_type=rel_type,
                            confidence=confidence,
                            context_snippet=context,
                            mapping_reason=f"Context-validated: {from_type} → {to_type} ({description})"
                        ))
        
        return relationships
    
    def _deduplicate_similar_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge similar entities (e.g., 'Apple' and 'Apple Inc')"""
        if not entities:
            return entities
        
        # Group by normalized name
        name_groups: Dict[str, List[Dict[str, Any]]] = {}
        
        for entity in entities:
            name = entity.get('value', entity.get('name', '')).strip()
            normalized = self._normalize_entity_name(name)
            
            if normalized not in name_groups:
                name_groups[normalized] = []
            name_groups[normalized].append(entity)
        
        # Keep the most complete version of each entity
        deduplicated = []
        for group in name_groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Prefer longer names (more specific)
                best = max(group, key=lambda e: len(e.get('value', e.get('name', ''))))
                deduplicated.append(best)
        
        return deduplicated
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication"""
        name_lower = name.lower().strip()
        
        # Remove common company suffixes
        suffixes = [' inc', ' inc.', ' corp', ' corp.', ' co', ' co.', 
                   ' ltd', ' ltd.', ' llc', ' plc', ' gmbh']
        for suffix in suffixes:
            if name_lower.endswith(suffix):
                name_lower = name_lower[:-len(suffix)]
        
        # Remove extra whitespace
        name_lower = re.sub(r'\s+', ' ', name_lower).strip()
        
        return name_lower
    
    def _validate_relationship_in_context(
        self,
        entity1: str,
        entity2: str,
        rel_type: str,
        text: str
    ) -> Tuple[bool, str, float]:
        """
        Validate if a relationship actually exists between two entities in the text.
        Returns (is_valid, context_snippet, confidence)
        """
        text_lower = text.lower()
        e1_lower = entity1.lower().strip()
        e2_lower = entity2.lower().strip()
        
        # Clean entity names (remove trailing "will", "the", etc.)
        e1_clean = re.sub(r'\s+(will|the|a|an|is|was|has|have)$', '', e1_lower).strip()
        e2_clean = re.sub(r'\s+(will|the|a|an|is|was|has|have)$', '', e2_lower).strip()
        
        if e1_clean not in text_lower or e2_clean not in text_lower:
            return False, "", 0.0
        
        # Pragmatic sentence splitting that handles abbreviations
        # Replace common abbreviations with placeholders (but keep the period!)
        text_protected = text
        abbrev_map = {}
        for i, abbr in enumerate(['Inc', 'Corp', 'Ltd', 'Co', 'Jr', 'Sr', 'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'vs']):
            # Replace "Inc." with "<ABB0>." so the period is still there for sentence splitting
            placeholder = f'<ABB{i}>.'
            abbrev_map[abbr] = placeholder
            text_protected = text_protected.replace(abbr + '.', placeholder)
        
        # Now split on sentence boundaries (. ! ? followed by whitespace)
        sentences = re.split(r'[.!?]\s+', text_protected)
        
        # Restore abbreviations
        for abbr, placeholder in abbrev_map.items():
            sentences = [s.replace(placeholder, abbr + '.') for s in sentences]
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Relationship indicators for different types
        relationship_indicators = {
            "WORKS_AT": [
                "works at", "works for", "employed at", "employed by",
                "ceo of", "cto of", "cfo of", "founder of", "director at",
                "lead at", "manager at", "engineer at", "of",
                "joined", "heads", "leads", "runs"
            ],
            "EMPLOYS": [
                "employs", "hired", "employee", "staff", "worker"
            ],
            "LOCATED_IN": [
                "located in", "based in", "headquartered in",
                "from", "office in", "branch in"
            ],
            "WORKS_ON": [
                "works on", "working on", "develops", "developing",
                "leads", "manages", "part of"
            ],
            "PRODUCES": [
                "produces", "makes", "manufactures", "creates",
                "develops", "launched"
            ],
            "PARTY_TO": [
                "signed", "agreed", "entered", "party to",
                "contract with", "agreement with"
            ],
            "MANAGES": ["manages", "leads", "oversees", "runs", "directs"],
            "SUPPLIED_BY": ["supplied by", "provides", "delivered by"],
            "HOSTS": ["hosts", "home to", "houses"],
            "AFFECTED": ["affected", "impacted", "experienced"],
            "CAUSED_BY": ["caused by", "due to", "resulted from"],
            "HAS_VALUE": ["worth", "valued at", "amount"],
            "SIGNED_ON": ["signed on", "dated", "effective"],
        }
        
        indicators = relationship_indicators.get(rel_type, ["of", "at", "in", "for", "with"])
        best_confidence = 0.0
        best_context = ""
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if both entities are in this sentence
            if e1_clean in sentence_lower and e2_clean in sentence_lower:
                # Calculate distance within sentence
                pos1 = sentence_lower.find(e1_clean)
                pos2 = sentence_lower.find(e2_clean)
                distance = abs(pos1 - pos2)
                
                # Check for relationship indicators BETWEEN the entities
                text_between = sentence_lower[min(pos1, pos2):max(pos1 + len(e1_clean), pos2 + len(e2_clean))]
                
                # Also check the immediate context around entities (for patterns like "CEO Tim Cook")
                context_start = max(0, min(pos1, pos2) - 20)
                context_end = min(len(sentence_lower), max(pos1 + len(e1_clean), pos2 + len(e2_clean)) + 20)
                extended_context = sentence_lower[context_start:context_end]
                
                has_indicator = any(ind in text_between or ind in extended_context for ind in indicators)
                
                # Also check for title patterns like "CEO [name]" or "[title] at [org]"
                title_patterns = [
                    r'ceo\s+' + re.escape(e1_clean),
                    r'cto\s+' + re.escape(e1_clean),
                    r'cfo\s+' + re.escape(e1_clean),
                    r'vp\s+' + re.escape(e1_clean),
                    r'director\s+' + re.escape(e1_clean),
                    r'president\s+' + re.escape(e1_clean),
                    r'founder\s+' + re.escape(e1_clean),
                    re.escape(e1_clean) + r"'s\s+(ceo|cto|vp|director|new|head)",
                    re.escape(e2_clean) + r"'s\s+(ceo|cto|vp|director|new|head).*" + re.escape(e1_clean),
                ]
                has_title_pattern = any(re.search(p, sentence_lower) for p in title_patterns)
                
                if has_title_pattern:
                    has_indicator = True
                
                # Negative indicators - these mean entities are NOT in this relationship
                negative_indicators = [
                    "met with", "meeting with", "spoke to", "called",
                    "visited", "discussed with", "talked to", "partnered with",
                    "competed with", "versus", "vs", "against"
                ]
                has_negative = any(neg in text_between for neg in negative_indicators)
                
                if has_negative:
                    continue  # Skip - this is a meeting/interaction, not employment
                
                # Calculate confidence based on indicators and distance
                if has_indicator:
                    if distance < 50:
                        confidence = 0.95
                    elif distance < 100:
                        confidence = 0.85
                    elif distance < 150:
                        confidence = 0.7
                    else:
                        confidence = 0.55
                else:
                    # No explicit indicator - much lower confidence
                    if distance < 30:
                        confidence = 0.4  # Very close but no indicator
                    else:
                        confidence = 0.0  # Too far apart without indicator
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_context = sentence.strip()[:200]
        
        return best_confidence > 0.5, best_context, best_confidence
    
    def _pattern_based_mapping(
        self,
        entities: List[Dict[str, Any]],
        text: str
    ) -> List[MappedRelationship]:
        """Extract relationships using regex patterns"""
        relationships = []
        text_lower = text.lower()
        
        # Create entity value lookup
        entity_values = {
            (e.get('value', e.get('name', ''))).lower(): e 
            for e in entities
        }
        
        for rel_type, patterns in self.RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    if match.groups():
                        source = match.group(1).strip()
                        target = match.group(2).strip() if len(match.groups()) > 1 else ""
                        
                        # Find matching entities
                        source_entity = self._find_matching_entity(source, entity_values)
                        target_entity = self._find_matching_entity(target, entity_values) if target else None
                        
                        if source_entity and target_entity:
                            # Extract context around the match
                            start = max(0, match.start() - 50)
                            end = min(len(text), match.end() + 50)
                            context = "..." + text[start:end] + "..."
                            
                            relationships.append(MappedRelationship(
                                source=source_entity.get('value', source_entity.get('name', '')),
                                source_type=source_entity.get('type', 'unknown'),
                                target=target_entity.get('value', target_entity.get('name', '')),
                                target_type=target_entity.get('type', 'unknown'),
                                relationship_type=rel_type,
                                confidence=0.85,
                                context_snippet=context,
                                mapping_reason=f"Pattern match: '{pattern}'"
                            ))
        
        return relationships
    
    def _graph_based_mapping(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[MappedRelationship]:
        """Look up existing relationships from graph database"""
        relationships = []
        
        if not self.graph_client:
            return relationships
        
        for entity in entities:
            entity_name = entity.get('value', entity.get('name', ''))
            entity_type = entity.get('type', 'unknown')
            
            try:
                rels = self.graph_client.get_entity_relationships(entity_name, entity_type)
                
                for rel in (rels or []):
                    connected = rel.get('connected_entity', '')
                    rel_type = rel.get('type', 'RELATED_TO')
                    direction = rel.get('direction', 'outgoing')
                    
                    if direction == 'outgoing':
                        relationships.append(MappedRelationship(
                            source=entity_name,
                            source_type=entity_type,
                            target=self._clean_entity_id(connected),
                            target_type=self._extract_type_from_id(connected),
                            relationship_type=rel_type,
                            confidence=0.95,  # High confidence from graph
                            context_snippet="",
                            mapping_reason="Graph database lookup"
                        ))
                    else:
                        relationships.append(MappedRelationship(
                            source=self._clean_entity_id(connected),
                            source_type=self._extract_type_from_id(connected),
                            target=entity_name,
                            target_type=entity_type,
                            relationship_type=rel_type,
                            confidence=0.95,
                            context_snippet="",
                            mapping_reason="Graph database lookup (incoming)"
                        ))
            except Exception as e:
                print(f"⚠ Graph lookup error for {entity_name}: {e}")
        
        return relationships
    
    def _context_aware_mapping(
        self,
        entities: List[Dict[str, Any]],
        text: str,
        query: str
    ) -> List[MappedRelationship]:
        """Map relationships based on query context"""
        relationships = []
        query_lower = query.lower()
        
        # Determine expected relationship types from query
        expected_types = []
        for pattern, rel_types in self.QUESTION_RELATIONSHIP_MAP.items():
            if pattern in query_lower:
                expected_types.extend(rel_types)
        
        if not expected_types:
            return relationships
        
        # Prioritize relationships matching expected types
        all_rels = self._rule_based_mapping(entities, text)
        
        for rel in all_rels:
            if rel.relationship_type in expected_types:
                rel.confidence = min(rel.confidence + 0.15, 1.0)  # Boost confidence
                rel.mapping_reason += f" (Query context boost: matches '{query}')"
                relationships.append(rel)
        
        return relationships
    
    def _calculate_proximity_confidence(
        self,
        entity1: str,
        entity2: str,
        text: str
    ) -> Tuple[float, str]:
        """Calculate confidence based on text proximity"""
        text_lower = text.lower()
        e1_lower = entity1.lower()
        e2_lower = entity2.lower()
        
        if e1_lower not in text_lower or e2_lower not in text_lower:
            return 0.0, ""
        
        pos1 = text_lower.find(e1_lower)
        pos2 = text_lower.find(e2_lower)
        
        distance = abs(pos1 - pos2)
        
        # Extract context
        start = min(pos1, pos2)
        end = max(pos1 + len(entity1), pos2 + len(entity2))
        context = text[max(0, start-30):min(len(text), end+30)]
        
        # Calculate confidence based on distance
        if distance < 50:
            return 0.9, context
        elif distance < 100:
            return 0.8, context
        elif distance < 200:
            return 0.7, context
        elif distance < 500:
            return 0.5, context
        else:
            return 0.3, context
    
    def _find_matching_entity(
        self,
        text_value: str,
        entity_lookup: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find entity matching the extracted text value"""
        text_lower = text_value.lower().strip()
        
        # Exact match
        if text_lower in entity_lookup:
            return entity_lookup[text_lower]
        
        # Partial match
        for key, entity in entity_lookup.items():
            if text_lower in key or key in text_lower:
                return entity
        
        return None
    
    def _clean_entity_id(self, entity_id: str) -> str:
        """Clean entity ID to get readable name"""
        # Remove type prefix (e.g., "person_John_Doe" -> "John Doe")
        parts = entity_id.split('_', 1)
        if len(parts) > 1:
            return parts[1].replace('_', ' ')
        return entity_id.replace('_', ' ')
    
    def _extract_type_from_id(self, entity_id: str) -> str:
        """Extract entity type from ID"""
        parts = entity_id.split('_', 1)
        return parts[0] if parts else 'unknown'
    
    def _deduplicate_relationships(
        self,
        relationships: List[MappedRelationship]
    ) -> List[MappedRelationship]:
        """Remove duplicate relationships, keeping highest confidence"""
        seen: Dict[str, MappedRelationship] = {}
        
        for rel in relationships:
            # Normalize source and target for comparison
            source_norm = self._normalize_entity_name(rel.source)
            target_norm = self._normalize_entity_name(rel.target)
            key = f"{source_norm}|{rel.relationship_type}|{target_norm}"
            
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel
        
        return list(seen.values())
    
    def generate_relationship_answer(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        text: str,
        semantic_matches: Optional[List[Dict[str, Any]]] = None
    ) -> RelationshipAnswer:
        """
        Generate a comprehensive answer with relationship mappings.
        
        Returns an answer that includes:
        - Direct answer to the question
        - All relevant entities
        - Mapped relationships with explanations
        - A table format for easy visualization
        """
        # Map relationships
        relationships = self.map_relationships_for_entities(entities, text, query)
        
        # Generate relationship table for chart display
        relationship_table = []
        for rel in relationships:
            relationship_table.append({
                "entity": rel.source,
                "entity_type": rel.source_type.upper(),
                "relationship": rel.relationship_type,
                "related_to": rel.target,
                "related_type": rel.target_type.upper(),
                "confidence": f"{rel.confidence * 100:.0f}%",
                "mapping_method": rel.mapping_reason.split(':')[0] if ':' in rel.mapping_reason else rel.mapping_reason
            })
        
        # Generate direct answer based on query type
        direct_answer = self._generate_direct_answer(query, entities, relationships)
        
        # Generate relationship explanation
        explanation = self._generate_relationship_explanation(query, relationships)
        
        # Build graph summary
        graph_summary = {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "entity_types": self._count_by_key(entities, 'type'),
            "relationship_types": self._count_relationships_by_type(relationships),
            "confidence_distribution": {
                "high": len([r for r in relationships if r.confidence >= 0.8]),
                "medium": len([r for r in relationships if 0.5 <= r.confidence < 0.8]),
                "low": len([r for r in relationships if r.confidence < 0.5]),
            }
        }
        
        return RelationshipAnswer(
            query=query,
            direct_answer=direct_answer,
            entities_found=[{
                "name": e.get('value', e.get('name', '')),
                "type": e.get('type', 'unknown'),
                "confidence": e.get('confidence', 0.85)
            } for e in entities],
            relationships=relationships,
            relationship_table=relationship_table,
            relationship_explanation=explanation,
            graph_summary=graph_summary
        )
    
    def _generate_direct_answer(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[MappedRelationship]
    ) -> str:
        """Generate a direct answer using LLM — works for any query type."""
        # Build context from entities and relationships
        entity_list = "\n".join([
            f"- {e.get('value', e.get('name', ''))} ({e.get('type', 'unknown')})"
            for e in entities[:20]
        ])
        rel_list = "\n".join([
            f"- {r.source} --[{r.relationship_type}]--> {r.target}"
            for r in relationships[:15]
        ])

        try:
            import os
            from utils.ollama_handler import OllamaLLM
            llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3"))

            prompt = (
                f'Based on the following extracted information, answer this question concisely:\n\n'
                f'Question: "{query}"\n\n'
                f'Entities found:\n{entity_list}\n\n'
                f'Relationships found:\n{rel_list if rel_list else "(none found)"}\n\n'
                f'Provide a brief, direct answer (1-3 sentences). '
                f'Focus on specifically answering what was asked.'
            )
            answer = llm.chat(prompt)
            return answer.strip()
        except Exception as e:
            print(f"⚠ LLM direct answer failed: {e}")
            # Fallback to entity summary
            entity_summary = ", ".join([
                e.get('value', e.get('name', ''))[:30] for e in entities[:5]
            ])
            return f"Found {len(entities)} entities: {entity_summary}" + ("..." if len(entities) > 5 else "")
    
    def _generate_relationship_explanation(
        self,
        query: str,
        relationships: List[MappedRelationship]
    ) -> str:
        """Generate explanation of how relationships were mapped"""
        if not relationships:
            return "No relationships could be mapped from the available data."
        
        explanations = []
        explanations.append(f"## Relationship Mapping for: \"{query}\"\n")
        explanations.append(f"Found **{len(relationships)}** relationships:\n")
        
        # Group by mapping method
        by_method: Dict[str, List[MappedRelationship]] = {}
        for rel in relationships:
            method = rel.mapping_reason.split(':')[0] if ':' in rel.mapping_reason else rel.mapping_reason
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(rel)
        
        for method, rels in by_method.items():
            explanations.append(f"\n### {method} ({len(rels)} relationships)")
            for rel in rels[:5]:  # Show up to 5 per method
                explanations.append(
                    f"- **{rel.source}** →[{rel.relationship_type}]→ **{rel.target}** "
                    f"(confidence: {rel.confidence*100:.0f}%)"
                )
                if rel.context_snippet:
                    explanations.append(f"  > Context: \"{rel.context_snippet[:100]}...\"")
        
        return "\n".join(explanations)
    
    def _count_by_key(self, items: List[Dict], key: str) -> Dict[str, int]:
        """Count items by a specific key"""
        counts: Dict[str, int] = {}
        for item in items:
            val = item.get(key, 'unknown')
            counts[val] = counts.get(val, 0) + 1
        return counts
    
    def _count_relationships_by_type(
        self,
        relationships: List[MappedRelationship]
    ) -> Dict[str, int]:
        """Count relationships by type"""
        counts: Dict[str, int] = {}
        for rel in relationships:
            counts[rel.relationship_type] = counts.get(rel.relationship_type, 0) + 1
        return counts


# Convenience function
def map_and_explain_relationships(
    entities: List[Dict[str, Any]],
    text: str,
    query: str
) -> Dict[str, Any]:
    """
    Quick function to map relationships and generate explanation.
    
    Returns dict with:
    - answer: Direct answer
    - relationships: List of relationship dicts
    - table: Tabular format for display
    - explanation: Markdown explanation
    """
    mapper = EnhancedRelationshipMapper()
    result = mapper.generate_relationship_answer(query, entities, text)
    
    return {
        "answer": result.direct_answer,
        "relationships": [
            {
                "source": r.source,
                "source_type": r.source_type,
                "target": r.target,
                "target_type": r.target_type,
                "type": r.relationship_type,
                "confidence": r.confidence,
                "context": r.context_snippet,
                "reason": r.mapping_reason
            }
            for r in result.relationships
        ],
        "table": result.relationship_table,
        "explanation": result.relationship_explanation,
        "graph_summary": result.graph_summary
    }

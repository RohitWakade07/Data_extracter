"""
Central Relationship Authority
===============================

Single source of truth for all relationship extraction and mapping.
All pipelines must use this exclusively.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from core.service_interfaces import RelationshipMapperInterface, Entity, Relationship
from core.logging_config import Logger, log_performance
import re
import json


logger = Logger(__name__)


@dataclass
class RelationshipAnswer:
    """Structured relationship-based answer"""
    query: str
    direct_answer: str
    entities_found: List[Dict[str, Any]]
    relationships: List[Relationship]
    relationship_table: List[Dict[str, str]]
    relationship_explanation: str
    graph_summary: Dict[str, Any]


class CentralRelationshipMapper(RelationshipMapperInterface):
    """
    Canonical relationship extraction authority.
    Implements consistent relationship logic across the entire system.
    """

    # Authoritative relationship rules
    RELATIONSHIP_ONTOLOGY = {
        # Employment relationships
        ("person", "organization"): {
            "forward": "WORKS_AT",
            "backward": "EMPLOYS",
            "confidence_base": 0.8,
        },
        # Location relationships
        ("person", "location"): {
            "forward": "LOCATED_IN",
            "backward": "HOSTS",
            "confidence_base": 0.75,
        },
        ("organization", "location"): {
            "forward": "LOCATED_IN",
            "backward": "HOSTS",
            "confidence_base": 0.85,
        },
        # Project relationships
        ("person", "project"): {
            "forward": "WORKS_ON",
            "backward": "ASSIGNED_TO",
            "confidence_base": 0.8,
        },
        ("organization", "project"): {
            "forward": "MANAGES",
            "backward": "MANAGED_BY",
            "confidence_base": 0.85,
        },
        # Agreement/Contract relationships
        ("person", "agreement"): {
            "forward": "PARTY_TO",
            "backward": "INVOLVES",
            "confidence_base": 0.9,
        },
        ("organization", "agreement"): {
            "forward": "PARTY_TO",
            "backward": "INVOLVES",
            "confidence_base": 0.9,
        },
        ("agreement", "amount"): {
            "forward": "HAS_VALUE",
            "backward": "WORTH_OF",
            "confidence_base": 0.95,
        },
        ("agreement", "date"): {
            "forward": "SIGNED_ON",
            "backward": "SIGNATURE_DATE_OF",
            "confidence_base": 0.95,
        },
        # Invoice relationships
        ("organization", "invoice"): {
            "forward": "ISSUED",
            "backward": "ISSUED_BY",
            "confidence_base": 0.9,
        },
        ("invoice", "organization"): {
            "forward": "BILLED_TO",
            "backward": "INVOICE_FOR",
            "confidence_base": 0.9,
        },
        ("invoice", "amount"): {
            "forward": "HAS_AMOUNT",
            "backward": "AMOUNT_OF",
            "confidence_base": 0.95,
        },
        ("invoice", "date"): {
            "forward": "DUE_ON",
            "backward": "DUE_DATE_OF",
            "confidence_base": 0.95,
        },
        # Supply chain relationships
        ("organization", "shipment"): {
            "forward": "SENT",
            "backward": "SENT_BY",
            "confidence_base": 0.85,
        },
        ("shipment", "organization"): {
            "forward": "RECEIVED_BY",
            "backward": "RECEIVED_FROM",
            "confidence_base": 0.85,
        },
        ("shipment", "location"): {
            "forward": "ROUTED_THROUGH",
            "backward": "RECEIVES_SHIPMENT",
            "confidence_base": 0.8,
        },
        # Financial relationships
        ("organization", "amount"): {
            "forward": "HAS_AMOUNT",
            "backward": "AMOUNT_FOR",
            "confidence_base": 0.85,
        },
        ("amount", "date"): {
            "forward": "EFFECTIVE_ON",
            "backward": "EFFECTIVE_DATE_FOR",
            "confidence_base": 0.8,
        },
        # Generic fallback
        ("entity", "entity"): {
            "forward": "RELATED_TO",
            "backward": "RELATED_TO",
            "confidence_base": 0.5,
        },
    }

    # Linguistic patterns for relationship detection
    RELATIONSHIP_PATTERNS = {
        "WORKS_AT": [
            r"(\w+(?:\s+\w+)?)\s+(?:works?\s+(?:at|for|with)|employed)",
            r"(\w+(?:\s+\w+)?)\s+(?:is|was)\s+(?:the\s+)?(?:CEO|CTO|CFO|manager|director|lead)",
        ],
        "LOCATED_IN": [
            r"(\w+(?:\s+\w+)?)\s+(?:is\s+)?(?:based\s+)?in\s+(\w+(?:\s+\w+)?)",
            r"(\w+(?:\s+\w+)?)\s+(?:office|headquarters?)\s+in\s+(\w+(?:\s+\w+)?)",
        ],
        "MANAGES": [
            r"(\w+(?:\s+\w+)?)\s+(?:manages?|oversees?)\s+(\w+(?:\s+\w+)?)",
            r"(\w+(?:\s+\w+)?)\s+project\s+(.+)(?:led|managed|run)\s+by",
        ],
        "SENT": [
            r"(\w+(?:\s+\w+)?)\s+sent\s+(\w+(?:\s+\w+)?)",
            r"($\w+(?:\s+\w+)?)\s+shipment\s+(?:to|from)\s+(\w+(?:\s+\w+)?)",
        ],
    }

    def __init__(self):
        """Initialize relationship mapper"""
        logger.info("Central Relationship Mapper initialized")

    @log_performance
    def map_relationships(
        self,
        entities: List[Entity],
        text: str
    ) -> List[Relationship]:
        """
        Extract relationships between entities.
        Uses multi-strategy approach:
        1. Ontology-based (entity type pairs)
        2. Pattern-based (linguistic patterns)
        3. Context-based (proximity)
        """
        relationships = []

        try:
            # Strategy 1: Ontology-based mapping
            ontology_rels = self._map_by_ontology(entities)
            relationships.extend(ontology_rels)

            # Strategy 2: Pattern-based mapping
            pattern_rels = self._map_by_patterns(entities, text)
            relationships.extend(pattern_rels)

            # Strategy 3: Context-based validation
            validated_rels = []
            for rel in relationships:
                confidence = self.validate_relationship(rel, text)
                rel.confidence = confidence
                if confidence > 0.5:  # Threshold
                    validated_rels.append(rel)

            # Deduplicate
            unique_rels = self._deduplicate_relationships(validated_rels)

            logger.info(f"Mapped {len(unique_rels)} relationships from {len(entities)} entities")
            return unique_rels

        except Exception as e:
            logger.error(f"Relationship mapping failed: {e}", exc_info=True)
            return []

    def _map_by_ontology(self, entities: List[Entity]) -> List[Relationship]:
        """Map relationships using ontology rules"""
        relationships = []

        for i, source_entity in enumerate(entities):
            for target_entity in entities[i + 1 :]:
                # Try both directions
                key = (source_entity.type.lower(), target_entity.type.lower())
                reverse_key = (target_entity.type.lower(), source_entity.type.lower())
                reverse_defined = False

                if key in self.RELATIONSHIP_ONTOLOGY:
                    rule = self.RELATIONSHIP_ONTOLOGY[key]
                    rel_type = rule["forward"]
                    confidence = rule["confidence_base"]

                    relationship = Relationship(
                        source=source_entity.value,
                        source_type=source_entity.type,
                        target=target_entity.value,
                        target_type=target_entity.type,
                        relationship_type=rel_type,
                        confidence=confidence,
                    )
                    relationships.append(relationship)

                    # Add reverse if defined
                    if reverse_key in self.RELATIONSHIP_ONTOLOGY:
                        reverse_rule = self.RELATIONSHIP_ONTOLOGY[reverse_key]
                        reverse_rel_type = reverse_rule["forward"]
                        reverse_confidence = reverse_rule["confidence_base"]

                        reverse_relationship = Relationship(
                            source=target_entity.value,
                            source_type=target_entity.type,
                            target=source_entity.value,
                            target_type=source_entity.type,
                            relationship_type=reverse_rel_type,
                            confidence=reverse_confidence,
                        )
                        relationships.append(reverse_relationship)
                        reverse_defined = True

                # If not defined bidirectionally, try reverse key
                if not reverse_defined and reverse_key in self.RELATIONSHIP_ONTOLOGY:
                    rule = self.RELATIONSHIP_ONTOLOGY[reverse_key]
                    rel_type = rule["forward"]
                    confidence = rule["confidence_base"]

                    relationship = Relationship(
                        source=target_entity.value,
                        source_type=target_entity.type,
                        target=source_entity.value,
                        target_type=source_entity.type,
                        relationship_type=rel_type,
                        confidence=confidence,
                    )
                    relationships.append(relationship)

        return relationships

    def _map_by_patterns(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Map relationships using linguistic patterns"""
        relationships = []

        try:
            entity_names = {e.value.lower(): e for e in entities}

            for rel_type, patterns in self.RELATIONSHIP_PATTERNS.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        try:
                            groups = match.groups()
                            if len(groups) >= 2:
                                source_name = groups[0].lower()
                                target_name = groups[1].lower()

                                source_entity = entity_names.get(source_name)
                                target_entity = entity_names.get(target_name)

                                if source_entity and target_entity:
                                    relationship = Relationship(
                                        source=source_entity.value,
                                        source_type=source_entity.type,
                                        target=target_entity.value,
                                        target_type=target_entity.type,
                                        relationship_type=rel_type,
                                        confidence=0.75,
                                        context=match.group(0),
                                    )
                                    relationships.append(relationship)
                        except Exception:
                            continue

        except Exception as e:
            logger.debug(f"Pattern matching error: {e}")

        return relationships

    @log_performance
    def validate_relationship(
        self,
        relationship: Relationship,
        context: str
    ) -> float:
        """
        Validate relationship confidence in context.
        Returns adjusted confidence score 0.0 to 1.0.
        """
        confidence = relationship.confidence

        try:
            # Check if source and target appear together in context
            source_in_context = relationship.source in context
            target_in_context = relationship.target in context

            if source_in_context and target_in_context:
                confidence *= 1.1  # Boost if both present
            elif source_in_context or target_in_context:
                confidence *= 0.8  # Reduce if only one present
            else:
                confidence *= 0.5  # Significantly reduce if none present

            # Proximity check
            if source_in_context and target_in_context:
                source_pos = context.find(relationship.source)
                target_pos = context.find(relationship.target)
                distance = abs(source_pos - target_pos)

                if distance < 100:  # Close together
                    confidence *= 1.05
                elif distance > 500:  # Far apart
                    confidence *= 0.9

        except Exception as e:
            logger.debug(f"Validation error: {e}")

        # Clamp to [0.0, 1.0]
        return min(1.0, max(0.0, confidence))

    def _deduplicate_relationships(
        self,
        relationships: List[Relationship]
    ) -> List[Relationship]:
        """Remove duplicate relationships keeping highest confidence"""
        seen = {}

        for rel in sorted(relationships, key=lambda r: r.confidence, reverse=True):
            key = (
                rel.source.lower(),
                rel.target.lower(),
                rel.relationship_type.lower(),
            )

            if key not in seen:
                seen[key] = rel

        return list(seen.values())

    @log_performance
    def generate_relationship_answer(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        text: str,
    ) -> RelationshipAnswer:
        """
        Generate structured answer with relationship mappings and explanations.
        """
        try:
            # Convert entities if needed
            entity_objs = []
            for e in entities:
                if isinstance(e, dict):
                    entity_objs.append(Entity(**e))
                else:
                    entity_objs.append(e)

            # Map relationships
            relationships = self.map_relationships(entity_objs, text)

            # Build relationship table for visualization
            relationship_table = []
            for rel in relationships:
                relationship_table.append({
                    "from": rel.source,
                    "to": rel.target,
                    "type": rel.relationship_type,
                    "confidence": f"{rel.confidence:.2%}",
                })

            # Generate explanations
            explanation = self._generate_explanation(query, relationships)

            # Generate summary
            graph_summary = {
                "total_entities": len(entity_objs),
                "total_relationships": len(relationships),
                "avg_confidence": (
                    sum(r.confidence for r in relationships) / len(relationships)
                    if relationships
                    else 0
                ),
            }

            # Craft direct answer
            direct_answer = self._craft_direct_answer(query, relationships, entities)

            return RelationshipAnswer(
                query=query,
                direct_answer=direct_answer,
                entities_found=[e.__dict__ if hasattr(e, "__dict__") else e for e in entity_objs],
                relationships=relationships,
                relationship_table=relationship_table,
                relationship_explanation=explanation,
                graph_summary=graph_summary,
            )

        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            return RelationshipAnswer(
                query=query,
                direct_answer="Unable to generate answer",
                entities_found=[],
                relationships=[],
                relationship_table=[],
                relationship_explanation=str(e),
                graph_summary={},
            )

    def _generate_explanation(self, query: str, relationships: List[Relationship]) -> str:
        """Generate natural language explanation of relationships"""
        if not relationships:
            return "No relationships found."

        explanations = []
        for rel in relationships[:5]:  # Top 5
            explanation = (
                f"{rel.source} ({rel.source_type}) {rel.relationship_type.lower()} "
                f"{rel.target} ({rel.target_type}) [confidence: {rel.confidence:.0%}]"
            )
            explanations.append(explanation)

        return "Key relationships: " + "; ".join(explanations)

    def _craft_direct_answer(
        self,
        query: str,
        relationships: List[Relationship],
        entities: List[Dict[str, Any]],
    ) -> str:
        """Craft a direct answer based on relationships"""
        if not relationships:
            return f"No direct relationships found for query: {query}"

        # Group by relationship type
        rel_groups = {}
        for rel in relationships:
            if rel.relationship_type not in rel_groups:
                rel_groups[rel.relationship_type] = []
            rel_groups[rel.relationship_type].append(rel)

        # Build answer from strongest relationships
        answer_parts = []
        for rel_type, rels in sorted(rel_groups.items(), key=lambda x: len(x[1]), reverse=True):
            top_rel = rels[0]
            answer_parts.append(
                f"{top_rel.source} {rel_type.lower()} {top_rel.target}"
            )

        return "Based on relationships: " + "; ".join(answer_parts[:3])

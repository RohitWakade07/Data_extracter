"""
Production-Grade Relationship Validation Module

Implements the mitigations from the Entity Relationship Extraction Issues analysis:

1. NO_RELATION class enforcement
2. Sentence-level grounding
3. Verb/keyword gating
4. Directional constraints
5. Deterministic confidence scoring
6. Graph gatekeeper

This module acts as a validation layer BEFORE NebulaGraph ingestion.
"""

import re
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# 1. REDUCED HIGH-PRECISION RELATION SET (start small, expand later)
# ═══════════════════════════════════════════════════════════════════════════════

CORE_RELATIONS: Set[str] = {
    # Person → Organization (directional)
    "WORKS_AT",
    "LEADS",
    
    # Organization ↔ Organization (symmetric or directional)
    "PARTNERS_WITH",
    "CONTRACTED_BY",
    "AUDITED_BY",
    
    # Organization → Location
    "LOCATED_IN",
    
    # Organization → Amount
    "HAS_CONTRACT_VALUE",
    
    # Fallback (explicit "no relationship found")
    "NO_RELATION",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DIRECTIONAL CONSTRAINTS
#    Only these (source_type, relation, target_type) are valid
# ═══════════════════════════════════════════════════════════════════════════════

DIRECTIONAL_CONSTRAINTS: Dict[str, Tuple[str, str]] = {
    # Relation → (allowed_source_type, allowed_target_type)
    "WORKS_AT":           ("person", "organization"),
    "LEADS":              ("person", "organization"),
    "LOCATED_IN":         ("organization", "location"),
    "CONTRACTED_BY":      ("organization", "organization"),  # vendor contracted BY client
    "AUDITED_BY":         ("organization", "organization"),  # company audited BY auditor
    "HAS_CONTRACT_VALUE": ("organization", "amount"),
}

# Symmetric relations (can go either direction between same types)
SYMMETRIC_RELATIONS: Set[str] = {
    "PARTNERS_WITH",  # Org ↔ Org
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VERB/KEYWORD GATING
#    Relation requires explicit evidence words in the text
# ═══════════════════════════════════════════════════════════════════════════════

RELATION_EVIDENCE_KEYWORDS: Dict[str, List[str]] = {
    "WORKS_AT": [
        r"\bworks?\s+at\b", r"\bemployed\s+(by|at)\b", r"\banalyst\s+at\b",
        r"\bengineer\s+at\b", r"\bmanager\s+at\b", r"\bemployee\s+of\b",
        r"\bworking\s+(for|with)\b", r"\bjoined\b", r"\bhired\s+by\b",
        r"\bstaff\s+of\b", r"\bteam\s+at\b",
    ],
    "LEADS": [
        r"\bCEO\b", r"\bCTO\b", r"\bCFO\b", r"\bCOO\b",
        r"\bchief\s+\w+\s+officer\b", r"\bpresident\s+of\b",
        r"\bfounder\b", r"\bco-?founder\b",
        r"\bleads?\b", r"\bheads?\b", r"\bdirects?\b",
        r"\bchairman\b", r"\bchairperson\b", r"\bVP\b",
        r"\bvice\s+president\b", r"\bmanaging\s+director\b",
    ],
    "PARTNERS_WITH": [
        r"\bpartner(s|ship|ed|ing)?\b", r"\bcollaborat(e|ion|ing)\b",
        r"\bjoint\s+venture\b", r"\balliance\b", r"\bteamed\s+up\b",
        r"\bworking\s+together\b", r"\bstrategic\s+partner\b",
    ],
    "CONTRACTED_BY": [
        r"\bcontract(s|ed|ing)?\b", r"\bagreement\b", r"\bengag(e|ed|ement)\b",
        r"\bcommission(ed)?\b", r"\bappoint(ed|ment)\b", r"\bhir(e|ed)\b",
        r"\bawarded?\s+to\b", r"\bselected\s+by\b",
    ],
    "AUDITED_BY": [
        r"\baudit(s|ed|ing|or)?\b", r"\breviewed\s+by\b",
        r"\binspect(ed|ion)\b", r"\bexamin(e|ed|ation)\b",
        r"\bcertifi(ed|cation)\b", r"\bverifi(ed|cation)\b",
    ],
    "LOCATED_IN": [
        r"\blocated\s+(in|at)\b", r"\bbased\s+(in|at)\b",
        r"\bheadquarter(s|ed)\s+(in|at)\b", r"\bhq\s+in\b",
        r"\boffice\s+(in|at)\b", r"\bregistered\s+(in|at)\b",
        r"\bincorporated\s+in\b", r"\boperates?\s+(in|from)\b",
    ],
    "HAS_CONTRACT_VALUE": [
        r"\bworth\b", r"\bvalue[ds]?\s+(at|of)\b", r"\bamount\b",
        r"\bcontract\s+value\b", r"\bdeal\s+worth\b", r"\bestimated\s+at\b",
        r"\btotal(s|ing)?\b", r"\bbudget\b", r"\bcost\b",
    ],
}

# Compile regex patterns for efficiency
_COMPILED_EVIDENCE: Dict[str, List[re.Pattern]] = {
    rel: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for rel, patterns in RELATION_EVIDENCE_KEYWORDS.items()
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SENTENCE-LEVEL GROUNDING
# ═══════════════════════════════════════════════════════════════════════════════

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for grounding checks."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def entities_in_same_sentence(
    entity1: str, entity2: str, text: str
) -> Tuple[bool, Optional[str]]:
    """
    Check if two entities appear in the same sentence.
    
    Returns (is_same_sentence, sentence_text).
    """
    sentences = split_into_sentences(text)
    e1_lower = entity1.lower()
    e2_lower = entity2.lower()
    
    for sent in sentences:
        sent_lower = sent.lower()
        if e1_lower in sent_lower and e2_lower in sent_lower:
            return True, sent
    
    return False, None


def entities_in_same_paragraph(
    entity1: str, entity2: str, text: str
) -> Tuple[bool, Optional[str]]:
    """
    Check if two entities appear in the same paragraph.
    
    Returns (is_same_paragraph, paragraph_text).
    """
    paragraphs = text.split('\n\n')
    e1_lower = entity1.lower()
    e2_lower = entity2.lower()
    
    for para in paragraphs:
        para_lower = para.lower()
        if e1_lower in para_lower and e2_lower in para_lower:
            return True, para
    
    return False, None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVIDENCE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_evidence_for_relation(
    relation: str, context: str
) -> Tuple[bool, List[str]]:
    """
    Check if the context contains evidence keywords for the relation.
    
    Returns (has_evidence, list_of_matched_keywords).
    """
    patterns = _COMPILED_EVIDENCE.get(relation, [])
    if not patterns:
        return False, []
    
    matches = []
    for pat in patterns:
        m = pat.search(context)
        if m:
            matches.append(m.group())
    
    return len(matches) > 0, matches


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DETERMINISTIC CONFIDENCE SCORING
#    Replace LLM confidence with evidence-based scoring
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfidenceFactors:
    """Evidence factors for deterministic confidence calculation."""
    same_sentence: bool = False
    same_paragraph: bool = False
    explicit_verb: bool = False
    keyword_match: bool = False
    ontology_valid: bool = False
    direction_valid: bool = False


def compute_deterministic_confidence(factors: ConfidenceFactors) -> float:
    """
    Compute confidence score from deterministic evidence signals.
    
    Weighting:
      - Same sentence:    +0.30
      - Explicit verb:    +0.35
      - Ontology valid:   +0.15
      - Direction valid:  +0.10
      - Keyword match:    +0.10
      - Same paragraph:   +0.10 (only if not same sentence)
    
    Max score: 1.0
    """
    score = 0.0
    
    if factors.same_sentence:
        score += 0.30
    elif factors.same_paragraph:
        score += 0.10
    
    if factors.explicit_verb:
        score += 0.35
    
    if factors.ontology_valid:
        score += 0.15
    
    if factors.direction_valid:
        score += 0.10
    
    if factors.keyword_match:
        score += 0.10
    
    return min(1.0, score)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. VALIDATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result of relationship validation."""
    is_valid: bool
    relation: str
    source: str
    source_type: str
    target: str
    target_type: str
    confidence: float
    factors: ConfidenceFactors
    rejection_reason: Optional[str] = None
    evidence_matches: List[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GRAPH GATEKEEPER — Main Validation Function
# ═══════════════════════════════════════════════════════════════════════════════

class RelationshipValidator:
    """
    Graph Gatekeeper: validates relationships before NebulaGraph ingestion.
    
    Implements:
      - NO_RELATION enforcement
      - Sentence-level grounding
      - Directional constraints
      - Verb/keyword gating
      - Deterministic confidence scoring
    """
    
    def __init__(
        self,
        min_confidence: float = 0.45,
        require_same_sentence: bool = False,  # Set True for strictest mode
        require_evidence: bool = True,
    ):
        self.min_confidence = min_confidence
        self.require_same_sentence = require_same_sentence
        self.require_evidence = require_evidence
    
    def validate(
        self,
        source: str,
        source_type: str,
        target: str,
        target_type: str,
        proposed_relation: str,
        full_text: str,
    ) -> ValidationResult:
        """
        Validate a proposed relationship.
        
        Returns ValidationResult with is_valid=True only if all checks pass.
        """
        factors = ConfidenceFactors()
        evidence_matches = []
        
        src_type = source_type.lower()
        tgt_type = target_type.lower()
        relation = proposed_relation.upper()
        
        # ── Check 1: Is this a core relation? ──
        if relation not in CORE_RELATIONS:
            return ValidationResult(
                is_valid=False, relation=relation,
                source=source, source_type=src_type,
                target=target, target_type=tgt_type,
                confidence=0.0, factors=factors,
                rejection_reason=f"Relation '{relation}' not in allowed set",
            )
        
        # ── Check 2: NO_RELATION explicit rejection ──
        if relation == "NO_RELATION":
            return ValidationResult(
                is_valid=False, relation=relation,
                source=source, source_type=src_type,
                target=target, target_type=tgt_type,
                confidence=0.0, factors=factors,
                rejection_reason="Explicit NO_RELATION",
            )
        
        # ── Check 3: Directional constraint ──
        if relation in DIRECTIONAL_CONSTRAINTS:
            allowed_src, allowed_tgt = DIRECTIONAL_CONSTRAINTS[relation]
            if src_type != allowed_src or tgt_type != allowed_tgt:
                # Maybe it's reversed?
                if src_type == allowed_tgt and tgt_type == allowed_src:
                    # Swap source and target
                    source, target = target, source
                    src_type, tgt_type = tgt_type, src_type
                else:
                    return ValidationResult(
                        is_valid=False, relation=relation,
                        source=source, source_type=src_type,
                        target=target, target_type=tgt_type,
                        confidence=0.0, factors=factors,
                        rejection_reason=f"Invalid direction: {src_type}->{tgt_type} not allowed for {relation}",
                    )
            factors.direction_valid = True
        elif relation in SYMMETRIC_RELATIONS:
            # Symmetric relations just need same-type endpoints
            factors.direction_valid = True
        
        factors.ontology_valid = True  # Passed type checks
        
        # ── Check 4: Sentence-level grounding ──
        same_sent, sentence = entities_in_same_sentence(source, target, full_text)
        same_para, paragraph = entities_in_same_paragraph(source, target, full_text)
        
        factors.same_sentence = same_sent
        factors.same_paragraph = same_para
        
        if self.require_same_sentence and not same_sent:
            return ValidationResult(
                is_valid=False, relation=relation,
                source=source, source_type=src_type,
                target=target, target_type=tgt_type,
                confidence=0.0, factors=factors,
                rejection_reason="Entities not in same sentence (strict mode)",
            )
        
        if not same_sent and not same_para:
            return ValidationResult(
                is_valid=False, relation=relation,
                source=source, source_type=src_type,
                target=target, target_type=tgt_type,
                confidence=0.0, factors=factors,
                rejection_reason="Entities not co-located in text",
            )
        
        # ── Check 5: Verb/keyword evidence ──
        context = sentence if same_sent else (paragraph if same_para else full_text)
        has_evidence, matches = find_evidence_for_relation(relation, context or "")
        
        factors.explicit_verb = has_evidence
        factors.keyword_match = has_evidence
        evidence_matches = matches
        
        if self.require_evidence and not has_evidence:
            return ValidationResult(
                is_valid=False, relation=relation,
                source=source, source_type=src_type,
                target=target, target_type=tgt_type,
                confidence=0.0, factors=factors,
                rejection_reason=f"No evidence keywords for {relation}",
                evidence_matches=[],
            )
        
        # ── Check 6: Compute deterministic confidence ──
        confidence = compute_deterministic_confidence(factors)
        
        if confidence < self.min_confidence:
            return ValidationResult(
                is_valid=False, relation=relation,
                source=source, source_type=src_type,
                target=target, target_type=tgt_type,
                confidence=confidence, factors=factors,
                rejection_reason=f"Confidence {confidence:.2f} below threshold {self.min_confidence}",
                evidence_matches=evidence_matches,
            )
        
        # ── All checks passed ──
        return ValidationResult(
            is_valid=True, relation=relation,
            source=source, source_type=src_type,
            target=target, target_type=tgt_type,
            confidence=confidence, factors=factors,
            evidence_matches=evidence_matches,
        )
    
    def validate_batch(
        self,
        relationships: List[Dict[str, Any]],
        full_text: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate a batch of relationships.
        
        Returns (valid_relationships, rejected_relationships).
        """
        valid = []
        rejected = []
        
        for rel in relationships:
            result = self.validate(
                source=rel.get("source", rel.get("from_id", "")),
                source_type=rel.get("source_type", rel.get("from_type", "unknown")),
                target=rel.get("target", rel.get("to_id", "")),
                target_type=rel.get("target_type", rel.get("to_type", "unknown")),
                proposed_relation=rel.get("type", rel.get("relationship", "RELATED_TO")),
                full_text=full_text,
            )
            
            if result.is_valid:
                # Update the relationship with validated data
                validated_rel = rel.copy()
                validated_rel["confidence"] = result.confidence
                validated_rel["source"] = result.source
                validated_rel["target"] = result.target
                validated_rel["type"] = result.relation
                validated_rel["evidence"] = result.evidence_matches
                valid.append(validated_rel)
            else:
                rejected_rel = rel.copy()
                rejected_rel["rejection_reason"] = result.rejection_reason
                rejected.append(rejected_rel)
        
        return valid, rejected


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Global validator instance with production defaults
_default_validator = RelationshipValidator(
    min_confidence=0.45,
    require_same_sentence=False,
    require_evidence=True,
)


def validate_relationship(
    source: str,
    source_type: str,
    target: str,
    target_type: str,
    relation: str,
    text: str,
) -> ValidationResult:
    """Convenience function for single relationship validation."""
    return _default_validator.validate(
        source, source_type, target, target_type, relation, text
    )


def validate_relationships_batch(
    relationships: List[Dict[str, Any]],
    text: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience function for batch validation."""
    return _default_validator.validate_batch(relationships, text)


def create_strict_validator() -> RelationshipValidator:
    """Create a validator with stricter settings."""
    return RelationshipValidator(
        min_confidence=0.55,
        require_same_sentence=True,
        require_evidence=True,
    )


def create_lenient_validator() -> RelationshipValidator:
    """Create a validator with more lenient settings."""
    return RelationshipValidator(
        min_confidence=0.35,
        require_same_sentence=False,
        require_evidence=False,
    )

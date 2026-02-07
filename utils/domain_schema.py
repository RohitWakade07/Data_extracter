"""
Domain Schema – The single source of truth for entity types, relationship types,
and valid (source_type, relationship, target_type) triples.

Layer 4 of the extraction pipeline: Domain Schema Filter.

Any relationship that does NOT match a valid triple is rejected.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
import re


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENTITY TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

ENTITY_TYPES: Set[str] = {
    "person",
    "organization",
    "location",
    "date",
    "amount",
    "project",
    "invoice",
    "agreement",
    "role",        # job titles: CEO, Project Lead, Advocate
}

# Human-readable labels
ENTITY_TYPE_LABELS: Dict[str, str] = {
    "person":       "Person",
    "organization": "Organization",
    "location":     "Location / Place",
    "date":         "Date / Time",
    "amount":       "Monetary Amount",
    "project":      "Project / Initiative",
    "invoice":      "Invoice / Bill",
    "agreement":    "Agreement / Contract",
    "role":         "Job Role / Title",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RELATIONSHIP TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

RELATIONSHIP_TYPES: Set[str] = {
    # Person ↔ Organization
    "WORKS_AT",         # person works at / is employed by org
    "REPRESENTS",       # person represents org (legal/official context)
    "LEADS",            # person leads org / is CEO of org

    # Person ↔ Project
    "WORKS_ON",         # person works on a project

    # Person ↔ Location
    "BASED_IN",         # person is based in location

    # Person ↔ Role
    "HAS_ROLE",         # person has job title

    # Organization ↔ Organization
    "PARTNERS_WITH",    # org partners with another org
    "AUDITED_BY",       # org is audited by another org
    "CONTRACTED_BY",    # org is contracted by another org (client→vendor)
    "ENGAGED",          # org has engaged another org

    # Organization ↔ Location
    "LOCATED_IN",       # org is headquartered / based in location
    "OPERATES_IN",      # org operates in a location / zone

    # Organization ↔ Project
    "MANAGES",          # org manages / owns a project
    "AWARDED",          # org was awarded a project/contract

    # Organization ↔ Agreement
    "PARTY_TO",         # org is party to agreement / contract
    "SIGNED",           # org signed agreement

    # Organization ↔ Invoice
    "ISSUED",           # org issued an invoice
    "RECEIVED_INVOICE", # org received an invoice

    # Organization ↔ Amount
    "HAS_CONTRACT_VALUE",  # org has contract worth X

    # Organization ↔ Role
    "APPOINTED_AS",     # org is appointed as a role (e.g. escrow agent)

    # Project ↔ Location
    "SPANS",            # project spans locations / zones

    # Project ↔ Amount
    "HAS_COST",         # project has estimated cost

    # Project ↔ Date
    "STARTS_ON",        # project starts on date
    "ENDS_ON",          # project target completion date
    "MILESTONE_ON",     # project has milestone on date

    # Invoice ↔ Amount
    "HAS_AMOUNT",       # invoice has amount

    # Invoice ↔ Date
    "DUE_ON",           # invoice due date
    "DATED",            # invoice date

    # Agreement ↔ Amount
    "HAS_VALUE",        # agreement / proposal has monetary value

    # Agreement ↔ Date
    "SIGNED_ON",        # agreement signed on date
    "EFFECTIVE_FROM",   # agreement effective from date
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VALID RELATIONSHIP TRIPLES
#    (source_type, relationship, target_type) — the "schema edges"
# ═══════════════════════════════════════════════════════════════════════════════

VALID_TRIPLES: Set[Tuple[str, str, str]] = {
    # ---- Person ↔ Organization ----
    ("person",       "WORKS_AT",         "organization"),
    ("person",       "REPRESENTS",       "organization"),
    ("person",       "LEADS",            "organization"),

    # ---- Person ↔ Project ----
    ("person",       "WORKS_ON",         "project"),

    # ---- Person ↔ Location ----
    ("person",       "BASED_IN",         "location"),

    # ---- Person ↔ Role ----
    ("person",       "HAS_ROLE",         "role"),

    # ---- Organization ↔ Organization ----
    ("organization", "PARTNERS_WITH",    "organization"),
    ("organization", "AUDITED_BY",       "organization"),
    ("organization", "CONTRACTED_BY",    "organization"),
    ("organization", "ENGAGED",          "organization"),

    # ---- Organization ↔ Location ----
    ("organization", "LOCATED_IN",       "location"),
    ("organization", "OPERATES_IN",      "location"),

    # ---- Organization ↔ Project ----
    ("organization", "MANAGES",          "project"),
    ("organization", "AWARDED",          "project"),

    # ---- Organization ↔ Agreement ----
    ("organization", "PARTY_TO",         "agreement"),
    ("organization", "SIGNED",           "agreement"),

    # ---- Organization ↔ Invoice ----
    ("organization", "ISSUED",           "invoice"),
    ("organization", "RECEIVED_INVOICE", "invoice"),

    # ---- Organization ↔ Amount ----
    ("organization", "HAS_CONTRACT_VALUE", "amount"),

    # ---- Organization ↔ Role ----
    ("organization", "APPOINTED_AS",     "role"),

    # ---- Project ↔ Location ----
    ("project",      "SPANS",            "location"),

    # ---- Project ↔ Amount ----
    ("project",      "HAS_COST",         "amount"),

    # ---- Project ↔ Date ----
    ("project",      "STARTS_ON",        "date"),
    ("project",      "ENDS_ON",          "date"),
    ("project",      "MILESTONE_ON",     "date"),

    # ---- Invoice ↔ Amount ----
    ("invoice",      "HAS_AMOUNT",       "amount"),

    # ---- Invoice ↔ Date ----
    ("invoice",      "DUE_ON",           "date"),
    ("invoice",      "DATED",            "date"),

    # ---- Agreement ↔ Amount ----
    ("agreement",    "HAS_VALUE",        "amount"),

    # ---- Agreement ↔ Date ----
    ("agreement",    "SIGNED_ON",        "date"),
    ("agreement",    "EFFECTIVE_FROM",   "date"),
}

# Build a fast lookup: source_type → { (rel_type, target_type), … }
_SCHEMA_INDEX: Dict[str, Set[Tuple[str, str]]] = {}
for _src, _rel, _tgt in VALID_TRIPLES:
    _SCHEMA_INDEX.setdefault(_src, set()).add((_rel, _tgt))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENTITY TYPE IDENTIFICATION HELPERS
#    Heuristics to reclassify misidentified entities
# ═══════════════════════════════════════════════════════════════════════════════

# Words that strongly indicate an ORGANIZATION, NOT a person
ORG_INDICATORS = re.compile(
    r"(?i)\b("
    r"pvt|ltd|llp|inc|corp|corporation|company|co\.|group|"
    r"bank|insurance|finance|capital|ventures|"
    r"foundation|trust|university|institute|college|"
    r"services|solutions|engineering|technologies|tech|systems|infrastructure|"
    r"associates|partners|consulting|consultants|advisory|"
    r"agency|department|ministry|commission|authority|board|council|"
    r"municipal\s+corporation|civic|"
    r"industries|enterprises|holdings|media|labs|studio|"
    r"llc|plc|gmbh|sa|"
    r"limited"
    r")\b"
)

# Words that strongly indicate a LOCATION, NOT a person/org
LOCATION_INDICATORS = re.compile(
    r"(?i)\b("
    r"park|center|centre|hall|building|tower|plaza|square|"
    r"street|avenue|boulevard|road|lane|drive|court|"
    r"city|town|village|district|state|province|country|"
    r"headquarters|hq|office|campus|"
    r"zone|corridor|sector|block|ward|"
    r"station|terminal|airport|port|"
    r"garden|maidan|ground|stadium"
    r")\b"
)

# Words that indicate a ROLE / JOB TITLE
ROLE_INDICATORS = re.compile(
    r"(?i)\b("
    r"ceo|cto|cfo|coo|cmo|cio|cpo|"
    r"director|manager|head|lead|chief|president|"
    r"vice\s+president|vp|svp|evp|"
    r"founder|co-founder|partner|senior\s+partner|"
    r"officer|coordinator|administrator|supervisor|specialist|"
    r"advocate|lawyer|attorney|counsel|solicitor|"
    r"engineer|architect|analyst|consultant|developer|designer|scientist|"
    r"project\s+lead|operations\s+manager|team\s+lead|"
    r"escrow\s+agent|agent|auditor|reviewer"
    r")\b"
)


def identify_entity_type(value: str, current_type: str) -> str:
    """
    Re-classify an entity if its value strongly suggests a different type.

    This is the Type Identification layer (Step 2).
    Returns the corrected entity type.
    """
    val = value.strip()

    # Rule 1: If tagged as "person" but contains org indicators → reclassify
    if current_type == "person":
        if ORG_INDICATORS.search(val):
            return "organization"
        if LOCATION_INDICATORS.search(val):
            return "location"

    # Rule 2: If tagged as "organization" but looks like a location
    if current_type == "organization":
        # Pure location names that got org tag (e.g., "Pune Civic Center")
        # If it has a location indicator but NO org indicator → location
        if LOCATION_INDICATORS.search(val) and not ORG_INDICATORS.search(val):
            return "location"

    # Rule 3: Role extraction — "CFO", "Project Lead", etc.
    # Standalone roles should be typed as "role" not "person"
    if current_type == "person":
        words = val.split()
        # If the entire value is a role phrase (1-3 words, all role indicators)
        if len(words) <= 3 and ROLE_INDICATORS.search(val):
            # But make sure it's not "Rahul Deshmukh" (person name with no role words)
            # Only reclassify if ALL words match role patterns
            non_role_words = [w for w in words if not ROLE_INDICATORS.search(w)]
            if not non_role_words:
                return "role"

    return current_type


def validate_relationship_triple(
    source_type: str, rel_type: str, target_type: str
) -> bool:
    """
    Check if a (source_type, rel_type, target_type) triple is allowed by the schema.

    This is the Domain Schema Filter (Step 4).
    """
    return (source_type.lower(), rel_type.upper(), target_type.lower()) in VALID_TRIPLES


def get_allowed_relationships(source_type: str, target_type: str) -> List[str]:
    """
    Get all relationship types allowed between two entity types.

    Returns a list of valid rel_types, or empty list if no edge is allowed.
    """
    src = source_type.lower()
    tgt = target_type.lower()
    return [
        rel for (s, rel, t) in VALID_TRIPLES
        if s == src and t == tgt
    ]


def get_allowed_targets(source_type: str) -> List[Tuple[str, str]]:
    """
    Get all (rel_type, target_type) pairs allowed from a given source_type.
    """
    return list(_SCHEMA_INDEX.get(source_type.lower(), set()))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENTITY DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """Normalize an entity name for dedup comparison."""
    n = name.lower().strip()
    # Remove common suffixes
    for suffix in [
        " pvt. ltd.", " pvt.ltd.", " pvt ltd", " private limited",
        " ltd.", " ltd", " limited",
        " llp", " inc.", " inc", " corp.", " corp", " corporation",
        " co.", " co", " company",
    ]:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    # Collapse whitespace
    n = re.sub(r"\s+", " ", n)
    return n


def deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge entities that refer to the same real-world object.

    Example: "Meridian Infrastructure Solutions Pvt. Ltd." and "Meridian"
    are merged, keeping the longer (more specific) form.

    This is the Deduplication layer (Step 7).
    """
    if not entities:
        return entities

    # Group by type first
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for e in entities:
        t = e.get("type", "unknown").lower()
        by_type.setdefault(t, []).append(e)

    result: List[Dict[str, Any]] = []

    for entity_type, group in by_type.items():
        # Build normalized → [entity, …]
        norm_groups: Dict[str, List[Dict[str, Any]]] = {}
        for e in group:
            norm = _normalize_name(e.get("value", ""))
            norm_groups.setdefault(norm, []).append(e)

        # Also check for substring containment: "Meridian" is substring of
        # "Meridian Infrastructure Solutions"
        keys = list(norm_groups.keys())
        merged_into: Dict[str, str] = {}  # short_key → long_key

        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                if i == j:
                    continue
                # k1 is a substring of k2 (and not too short)
                if len(k1) >= 3 and k1 in k2 and k1 != k2:
                    merged_into[k1] = k2

        # Merge short-form groups into long-form groups
        for short_key, long_key in merged_into.items():
            if short_key in norm_groups:
                norm_groups.setdefault(long_key, []).extend(norm_groups.pop(short_key))

        # Pick the best (longest value, highest confidence) from each group
        for norm_key, ents in norm_groups.items():
            best = max(
                ents,
                key=lambda e: (
                    len(e.get("value", "")),
                    e.get("confidence", 0),
                ),
            )
            # Boost confidence if multiple mentions
            if len(ents) > 1:
                best["confidence"] = min(0.98, best.get("confidence", 0.5) + 0.05)
            result.append(best)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONFIDENCE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

# Entity confidence thresholds
ENTITY_ACCEPT_THRESHOLD = 0.60      # accept entity
ENTITY_DISCARD_THRESHOLD = 0.30     # discard entity

# Relationship confidence thresholds
REL_ACCEPT_THRESHOLD = 0.50         # accept relationship
REL_DISCARD_THRESHOLD = 0.25        # discard relationship


def filter_by_confidence(
    items: List[Dict[str, Any]],
    accept_threshold: float,
    discard_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split items into accepted and review-needed lists based on confidence.

    Returns (accepted, needs_review).
    Items below discard_threshold are dropped entirely.
    """
    accepted = []
    needs_review = []
    for item in items:
        conf = item.get("confidence", 0.5)
        if conf >= accept_threshold:
            accepted.append(item)
        elif conf >= discard_threshold:
            needs_review.append(item)
        # else: discarded
    return accepted, needs_review

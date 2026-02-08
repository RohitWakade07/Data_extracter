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

# Words that indicate a ROLE / JOB TITLE (NOT a person)
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
    r"escrow\s+agent|agent|auditor|reviewer|"
    r"contractor|vendor|supplier|client|party|"
    r"independent\s+\w+\s+auditor|technical\s+auditor"
    r")\b"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.1 NON-ENTITY BLOCKLIST — Phrases that should NEVER be entities
# ═══════════════════════════════════════════════════════════════════════════════

# These are commonly misclassified as persons but are actually concepts/roles/dates
NON_PERSON_BLOCKLIST = {
    # Date-related
    "effective date", "commencement date", "completion date", "due date",
    "start date", "end date", "termination date", "execution date",
    # Role phrases (not actual people)
    "independent technical auditor", "technical auditor", "escrow agent",
    "project manager", "site engineer", "authorized representative",
    "managing director", "chief executive", "legal counsel",
    "contractor", "vendor", "client", "party", "witness", "witnesses",
    "project director", "project lead", "team lead", "site manager",
    "chief financial officer", "chief executive officer",
    "authorized signatory", "legal representative",
    "arbitrator", "arbitrators", "the arbitrator", "mediator", "adjudicator",
    # Legal/document terms - these are NOT people!
    "first party", "second party", "third party",
    "first partner", "second partner", "third partner",
    "recitals", "whereas", "agreement", "contract",
    "appendix", "schedule", "annexure", "exhibit",
    "this agreement", "the agreement", "this contract", "the contract",
    "this deed", "the deed", "this memorandum", "the memorandum",
    "deed this", "partnership deed", "deed of partnership",
    # Generic references in legal docs - NOT specific people
    "the partner", "the partners", "any partner", "all partners",
    "all the partners", "each partner", "every partner",
    "the party", "the parties", "any party", "all parties",
    "all the parties", "each party", "every party",
    "minor", "minors", "a minor",
    "heir", "heirs", "legal heir", "legal heirs",
    "deceased", "deceased partner", "deceased party",
    "survivor", "surviving partner",
    "nominee", "nominees", "beneficiary", "beneficiaries",
    "guardian", "guardians",
    "legal representative", "legal heir",
    # Generic terms
    "the company", "the corporation", "the bank", "the contractor",
    "the authority", "the client", "the vendor",
    # Currency/amount fragments misclassified as person
    "rupees", "rupees ten", "rupees only", "thousand only",
}

# These are commonly misclassified but are not organizations
NON_ORG_BLOCKLIST = {
    # Document terms
    "agreement", "contract", "the agreement", "the contract",
    "this agreement", "this contract", "the deed", "this deed",
    "effective date", "the date", "high court", "supreme court",
    # Generic references - NOT specific organizations
    "the firm", "the company", "the partnership", "the business",
    "the partnership firm", "the partnership business",
    "firm", "company", "partnership", "business",
    "partnership firm", "partnership business",
    "court", "the court", "arbitration",
    # Legal bodies (too generic without specific name)
    "registrar", "the registrar",
}

# Patterns that should NOT be extracted as locations (postal codes, generic terms)
NON_LOCATION_BLOCKLIST = {
    # Generic address fragments
    "india",  # Too generic unless part of full address
    # Generic place references
    "the principal place of business", "principal place of business",
    "place of business", "registered office", "head office",
    "such other place", "other place", "places",
}

# Postal codes and pure numbers should not be locations
POSTAL_CODE_PATTERN = re.compile(r"^\\d{5,6}$")  # Indian PIN codes are 6 digits

# Non-date terms that get misclassified as dates
NON_DATE_BLOCKLIST = {
    # Duration/period terms (not specific dates)
    "time period", "the period", "period",
    "six monthly", "monthly", "yearly", "annually",
    # Clause numbers misclassified as dates
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "1.", "2.", "3.", "4.", "5.",
    "6.", "7.", "8.", "9.", "10.", "11.", "12.", "13.", "14.", "15.",
}

# Non-project terms
NON_PROJECT_BLOCKLIST = {
    "business", "the business", "affairs", "the affairs",
    "1", "2", "3", "clause", "section",
}

# Fragment patterns — partial entity names that should be filtered
# These are substrings that indicate an incomplete entity extraction
FRAGMENT_PATTERNS = [
    # Bank fragments
    (r"^(state\\s+)?bank$", "organization"),           # "Bank", "State Bank" without full name
    (r"^bank\\s+of\\s+\\w+$", "organization"),           # "Bank of India" without "State"
    (r"^(hdfc|icici|axis|kotak)$", "organization"),   # Bank abbreviation without "Bank"
    # LLP/company fragments  
    (r"^young\s+llp$", "organization"),               # "Young LLP" without "Ernst and"
    (r"^and\s+young", "organization"),                # "and Young" fragment
    (r"^(kpmg|deloitte|pwc|ey)$", "organization"),    # Abbreviations OK, but check context
    # Name fragments
    (r"^(mr|mrs|ms|dr)\.?$", "person"),               # Just honorific
    (r"^(kumar|sharma|singh|deshmukh|patel)$", "person"),  # Just surname
]


def is_blocked_entity(value: str, entity_type: str) -> bool:
    """
    Check if an entity value should be blocked (not extracted).
    
    Returns True if the entity is in a blocklist or matches a fragment pattern.
    """
    val_lower = value.lower().strip()
    
    # Check type-specific blocklists
    if entity_type == "person" and val_lower in NON_PERSON_BLOCKLIST:
        return True
    if entity_type == "organization" and val_lower in NON_ORG_BLOCKLIST:
        return True
    if entity_type == "location":
        # Block postal codes
        if POSTAL_CODE_PATTERN.match(val_lower):
            return True
        if val_lower in NON_LOCATION_BLOCKLIST:
            return True
    if entity_type == "date" and val_lower in NON_DATE_BLOCKLIST:
        return True
    if entity_type == "project" and val_lower in NON_PROJECT_BLOCKLIST:
        return True
    
    # Block pure numbers for most entity types (except amount)
    if entity_type not in ("amount", "percentage") and re.match(r"^\d+\.?$", val_lower):
        return True
    
    # Block "Witness 1", "Witnesss 1", numbered generic terms
    if entity_type == "person" and re.match(r"^(witness|witnesss?)\s*\d*$", val_lower, re.IGNORECASE):
        return True
    
    # Block "Deed This", "Agreement This" patterns (document title fragments)
    if entity_type == "person" and re.match(r"^(deed|agreement|contract|memorandum)\s+(this|of)$", val_lower, re.IGNORECASE):
        return True
    
    # Block duration patterns like "5 years", "6 monthly"
    if entity_type == "date" and re.match(r"^\d+\s*(years?|months?|days?|weeks?|monthly|yearly|annually)$", val_lower, re.IGNORECASE):
        return True
    
    # Check fragment patterns
    for pattern, frag_type in FRAGMENT_PATTERNS:
        if entity_type == frag_type and re.match(pattern, val_lower, re.IGNORECASE):
            return True
    
    # Block very short entities (likely fragments)
    if len(val_lower) < 3:
        return True
    
    # Block entities that are ALL CAPS and look like acronyms/headers
    if value.isupper() and len(value.split()) == 1 and entity_type not in ("amount", "date"):
        # Allow known acronyms like "HDFC", "SBI" only if followed by descriptive text
        if len(value) <= 5:  # Short acronyms might be OK
            pass  # Let it through for now, dedup will handle
        else:
            return True  # Long all-caps words like "AGREEMENT" blocked
    
    return False


# Common trailing words that get incorrectly attached to entity names
TRAILING_NOISE_WORDS = re.compile(
    r"\s+(shall|will|would|should|must|may|can|has|have|had|is|are|was|were|"
    r"and|or|the|a|an|to|for|of|in|on|at|by|with)$",
    re.IGNORECASE
)


def clean_entity_value(value: str, entity_type: str) -> str:
    """
    Clean up an entity value by removing trailing noise words.
    
    Examples:
      - "Anil Deshmukh shall" → "Anil Deshmukh"
      - "HDFC Bank and" → "HDFC Bank"
    """
    cleaned = value.strip()
    
    # Remove trailing noise words (iterate until no more matches)
    for _ in range(3):  # Max 3 iterations to handle "X and the"
        match = TRAILING_NOISE_WORDS.search(cleaned)
        if match:
            cleaned = cleaned[:match.start()].strip()
        else:
            break
    
    # Remove leading articles for orgs
    if entity_type == "organization":
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    
    return cleaned


def identify_entity_type(value: str, current_type: str) -> str:
    """
    Re-classify an entity if its value strongly suggests a different type.

    This is the Type Identification layer (Step 2).
    Returns the corrected entity type, or "BLOCKED" if entity should be discarded.
    """
    val = value.strip()
    val_lower = val.lower()

    # Rule 0: Check blocklist first
    if is_blocked_entity(val, current_type):
        return "BLOCKED"

    # Rule 1: If tagged as "person" but contains org indicators → reclassify
    if current_type == "person":
        # Strong org indicators - these should ALWAYS be org, never person
        strong_org_patterns = [
            r"(?i)\b(corporation|municipal|ltd\.?|pvt\.?|inc\.?|llc|llp|co\.?)\b",
            r"(?i)\b(bank|company|limited|private|public)\b",
            r"(?i)\b(foundation|trust|association|institute|university)\b",
        ]
        for pattern in strong_org_patterns:
            if re.search(pattern, val):
                return "organization"
        
        if ORG_INDICATORS.search(val):
            return "organization"
        if LOCATION_INDICATORS.search(val):
            return "location"
        # Check if it's actually a role/title phrase
        if val_lower in NON_PERSON_BLOCKLIST:
            return "BLOCKED"

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
        # If the entire value is a role phrase (1-4 words, matches role pattern)
        if len(words) <= 4 and ROLE_INDICATORS.search(val):
            # Check if it looks like "Independent Technical Auditor" (all role words)
            # vs "Mr. Rajesh Kumar" (has a proper name)
            has_proper_name = False
            for w in words:
                # Proper names: capitalized, not a role word, not an honorific
                w_clean = w.rstrip('.,')
                if (w_clean[0:1].isupper() and 
                    not ROLE_INDICATORS.search(w_clean) and
                    w_clean.lower() not in ('mr', 'mrs', 'ms', 'dr', 'prof', 'the', 'a', 'an')):
                    # Could be a name like "Kumar" or "Sharma"
                    # Check if it's a common surname (those are OK as part of names)
                    if len(w_clean) > 2:
                        has_proper_name = True
                        break
            
            if not has_proper_name:
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
# 5. ENTITY DEDUPLICATION (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

# Common honorifics/titles to strip for matching
HONORIFICS = re.compile(r"^(mr\.?|mrs\.?|ms\.?|dr\.?|prof\.?|shri\.?|smt\.?)\s+", re.IGNORECASE)

# Common organization suffixes to normalize
ORG_SUFFIXES = [
    " pvt. ltd.", " pvt.ltd.", " pvt ltd", " private limited",
    " ltd.", " ltd", " limited",
    " llp", " inc.", " inc", " corp.", " corp", " corporation",
    " co.", " co", " company",
    " & co", " and co",
]


def _normalize_name(name: str) -> str:
    """Normalize an entity name for dedup comparison."""
    n = name.lower().strip()
    
    # Remove honorifics for person matching
    n = HONORIFICS.sub("", n).strip()
    
    # Remove common org suffixes
    for suffix in ORG_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    
    # Collapse whitespace
    n = re.sub(r"\s+", " ", n)
    return n


def _is_substring_match(short: str, long: str, min_len: int = 3) -> bool:
    """
    Check if 'short' is a meaningful substring of 'long'.
    
    Examples:
      - "HDFC" in "HDFC Bank" → True
      - "Priya Sharma" in "Dr. Priya Sharma" → True
      - "of" in "Bank of India" → False (too short/common)
    """
    if len(short) < min_len:
        return False
    if short == long:
        return False
    
    # Direct substring
    if short in long:
        return True
    
    # Check if short is contained after normalizing both
    short_norm = _normalize_name(short)
    long_norm = _normalize_name(long)
    
    if short_norm in long_norm:
        return True
    
    # Check word-level containment: all words in short appear in long
    short_words = set(short_norm.split())
    long_words = set(long_norm.split())
    if short_words and short_words.issubset(long_words):
        return True
    
    return False


def _is_fragment(value: str, entity_type: str, all_values: List[str]) -> bool:
    """
    Check if an entity value is likely a fragment of a longer entity.
    
    Examples:
      - "Young LLP" is a fragment if "Ernst and Young LLP" exists
      - "State Bank" is a fragment if "State Bank of India" exists
      - "Bank of India" is a fragment if "State Bank of India" exists
    """
    val_norm = _normalize_name(value)
    val_words = val_norm.split()
    
    for other in all_values:
        if other == value:
            continue
        other_norm = _normalize_name(other)
        
        # If this value is a suffix of another (e.g., "Young LLP" in "Ernst and Young LLP")
        if other_norm.endswith(val_norm) and len(other_norm) > len(val_norm):
            return True
        
        # If this value is a prefix of another (e.g., "State Bank" in "State Bank of India")
        if other_norm.startswith(val_norm) and len(other_norm) > len(val_norm):
            return True
        
        # If all words in this value appear in another longer value
        other_words = other_norm.split()
        if len(val_words) < len(other_words) and set(val_words).issubset(set(other_words)):
            return True
    
    return False


def deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge entities that refer to the same real-world object.

    Enhanced to handle:
      1. Honorific variations: "Dr. Priya Sharma" ≈ "Priya Sharma"
      2. Abbreviations: "HDFC" ≈ "HDFC Bank"
      3. Fragments: "Young LLP" is dropped if "Ernst and Young LLP" exists
      4. Suffix variations: "Meridian Pvt. Ltd." ≈ "Meridian Pvt Ltd"

    This is the Deduplication layer (Step 7).
    """
    if not entities:
        return entities

    # First pass: filter out blocked entities
    filtered = []
    for e in entities:
        etype = identify_entity_type(e.get("value", ""), e.get("type", "unknown"))
        if etype != "BLOCKED":
            e["type"] = etype  # Update with corrected type
            filtered.append(e)
    
    entities = filtered

    # Group by type first
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for e in entities:
        t = e.get("type", "unknown").lower()
        by_type.setdefault(t, []).append(e)

    result: List[Dict[str, Any]] = []

    for entity_type, group in by_type.items():
        # Get all values for fragment detection
        all_values = [e.get("value", "") for e in group]
        
        # Filter out fragments first
        non_fragments = []
        for e in group:
            val = e.get("value", "")
            if not _is_fragment(val, entity_type, all_values):
                non_fragments.append(e)
        
        if not non_fragments:
            # If all were fragments, keep the longest one
            non_fragments = [max(group, key=lambda x: len(x.get("value", "")))]
        
        group = non_fragments
        
        # Build normalized → [entity, …]
        norm_groups: Dict[str, List[Dict[str, Any]]] = {}
        for e in group:
            norm = _normalize_name(e.get("value", ""))
            norm_groups.setdefault(norm, []).append(e)

        # Check for substring containment and merge
        keys = list(norm_groups.keys())
        merged_into: Dict[str, str] = {}  # short_key → long_key

        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                if i == j:
                    continue
                # k1 is a substring of k2
                if _is_substring_match(k1, k2):
                    merged_into[k1] = k2

        # Merge short-form groups into long-form groups
        for short_key, long_key in merged_into.items():
            if short_key in norm_groups and long_key in norm_groups:
                norm_groups[long_key].extend(norm_groups.pop(short_key))

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

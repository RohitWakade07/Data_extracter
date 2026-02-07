"""
Test: Semantic Extraction Pipeline vs Traditional Pipeline
==========================================================
Compares the embedding-first (semantic) approach with the existing
7-layer pipeline on the same business text.

Metrics compared:
  - Entity coverage (persons, organizations, locations)
  - Relationship coverage (key relationships)
  - Schema compliance (no invalid triples)
  - Semantic features (cross-chunk discoveries, embedding dedup)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', '.env'))

from utils.domain_schema import validate_relationship_triple

# ═══════════════════════════════════════════════════════════════════
# TEST DATA (same as test_pipeline_quality.py)
# ═══════════════════════════════════════════════════════════════════

BUSINESS_TEXT = """
Meridian Infrastructure Solutions Pvt. Ltd., headquartered in Pune, Maharashtra, has been awarded a 
major contract worth INR 45 crore by the Pune Municipal Corporation for the Smart City Infrastructure 
Planning initiative. The agreement was signed on January 15, 2025, at the Pune Civic Center, with 
project lead Rahul Deshmukh representing Meridian and advocate Anjali Patil overseeing the legal 
formalities on behalf of the municipal body.

The project, titled Urban Road Development Initiative, spans multiple zones across Pune including 
Hinjewadi IT Park, Shivajinagar, and Kothrud. Under the terms of the service agreement, Meridian 
will deliver a comprehensive infrastructure assessment report, environmental impact study, and a 
phased execution plan by March 31, 2025. The total project cost is estimated at INR 120 crore, with 
the first milestone payment of INR 15 crore due by February 28, 2025.

Vikram Joshi, Chief Financial Officer at Meridian Infrastructure Solutions Pvt. Ltd., confirmed that 
the company has allocated dedicated resources for this engagement. Invoice SHK-INF-0824-01, dated 
January 20, 2025, has already been submitted to the Pune Municipal Corporation for the initial 
consultancy fee of INR 8.5 crore. The invoice references the master service agreement signed between 
both parties.

Additionally, Meridian has partnered with GreenTech Engineering Services LLP, based in Bangalore, 
Karnataka, for environmental compliance and sustainability consulting. Neha Kulkarni, operations 
manager at GreenTech, will coordinate with Meridian's Pune team for site assessments scheduled 
between February and April 2025. GreenTech has issued a separate proposal valued at INR 3.2 crore 
for their scope of work, which includes soil testing, water resource mapping, and air quality 
monitoring across all project zones.

The Pune Municipal Corporation has also engaged Sharma Legal Associates, a law firm based in Mumbai, 
Maharashtra, to review all contractual obligations and ensure regulatory compliance. Senior partner 
Arun Sharma will lead the legal review team. The legal review is expected to conclude by mid-February 
2025, after which the formal project kickoff will commence.

For financial oversight, HDFC Bank has been appointed as the escrow agent for all project-related 
fund disbursements. The escrow account, opened on January 10, 2025, will hold the total committed 
funds of INR 45 crore, releasing payments upon verified milestone completion. All transactions will 
be audited quarterly by Deloitte India, whose engagement letter was countersigned on January 12, 2025.

This multi-stakeholder initiative represents one of the largest urban infrastructure investments in 
Pune's recent history and is expected to significantly improve transportation connectivity in the 
western corridor of the city. The project completion target is set for December 2026, with interim 
reviews scheduled every quarter at the Pune Municipal Corporation headquarters.
"""

# ═══════════════════════════════════════════════════════════════════
# Expected entities & relationships (ground truth)
# ═══════════════════════════════════════════════════════════════════

EXPECTED_PERSONS = {"Rahul Deshmukh", "Anjali Patil", "Vikram Joshi", "Neha Kulkarni", "Arun Sharma"}
EXPECTED_ORGS = {
    "Meridian Infrastructure Solutions Pvt. Ltd.", "Pune Municipal Corporation",
    "GreenTech Engineering Services LLP", "Sharma Legal Associates",
    "HDFC Bank", "Deloitte",
}
EXPECTED_LOCATIONS = {"Pune", "Mumbai", "Bangalore", "Maharashtra", "Karnataka", "Hinjewadi"}

EXPECTED_KEY_RELATIONSHIPS = [
    ("Rahul Deshmukh",  "WORKS_AT",       "Meridian"),
    ("Vikram Joshi",    "WORKS_AT",       "Meridian"),
    ("Neha Kulkarni",   "WORKS_AT",       "GreenTech"),
    ("Arun Sharma",     "WORKS_AT",       "Sharma Legal"),
    ("Meridian",        "MANAGES",        None),          # any target
    ("Meridian",        "PARTY_TO",       None),
    ("GreenTech",       "PARTNERS_WITH",  "Meridian"),
]


# ═══════════════════════════════════════════════════════════════════
# Scoring helpers
# ═══════════════════════════════════════════════════════════════════

def match_entity(found_value: str, expected_set: set) -> bool:
    fv = found_value.lower().strip()
    for exp in expected_set:
        if exp.lower() in fv or fv in exp.lower():
            return True
    return False

def score_entities(entities, expected_persons, expected_orgs, expected_locs):
    """Count how many expected entities are covered."""
    found_persons = {e for exp in expected_persons for e in [exp] if any(
        match_entity(ent_val, {exp}) for ent_val in
        [x.get("value", "") if isinstance(x, dict) else getattr(x, "value", "") for x in entities
         if (x.get("type", "") if isinstance(x, dict) else getattr(x, "type", "")).lower() == "person"]
    )}
    found_orgs = {e for exp in expected_orgs for e in [exp] if any(
        match_entity(ent_val, {exp}) for ent_val in
        [x.get("value", "") if isinstance(x, dict) else getattr(x, "value", "") for x in entities
         if (x.get("type", "") if isinstance(x, dict) else getattr(x, "type", "")).lower() == "organization"]
    )}
    found_locs = {e for exp in expected_locs for e in [exp] if any(
        match_entity(ent_val, {exp}) for ent_val in
        [x.get("value", "") if isinstance(x, dict) else getattr(x, "value", "") for x in entities
         if (x.get("type", "") if isinstance(x, dict) else getattr(x, "type", "")).lower() == "location"]
    )}
    return found_persons, found_orgs, found_locs

def score_relationships(relationships, expected_rels):
    """Count how many expected key relationships are found."""
    matched = 0
    for src_exp, rel_exp, tgt_exp in expected_rels:
        for r in relationships:
            r_src  = r.get("from_id", "") if isinstance(r, dict) else getattr(r, "source", "")
            r_type = r.get("type", "")    if isinstance(r, dict) else getattr(r, "relationship", "")
            r_tgt  = r.get("to_id", "")   if isinstance(r, dict) else getattr(r, "target", "")

            src_match = src_exp.lower() in r_src.lower()
            rel_match = rel_exp.upper() == r_type.upper()
            tgt_match = tgt_exp is None or tgt_exp.lower() in r_tgt.lower()

            if src_match and rel_match and tgt_match:
                matched += 1
                break
    return matched, len(expected_rels)


# ═══════════════════════════════════════════════════════════════════
# RUN BOTH PIPELINES
# ═══════════════════════════════════════════════════════════════════

def run_semantic_pipeline():
    """Run the new embedding-first semantic extraction."""
    from semantic_extraction.semantic_extractor import SemanticExtractor

    extractor = SemanticExtractor(
        window_size=3,
        overlap=1,
        similarity_threshold=0.75,
        store_in_weaviate=True,
    )

    t0 = time.time()
    result = extractor.extract(BUSINESS_TEXT, doc_id="test_business")
    elapsed = time.time() - t0

    # Convert to dicts for scoring
    entities = [{"type": e.type, "value": e.value, "confidence": e.confidence}
                for e in result.entities]
    relationships = result.relationships

    return entities, relationships, result, elapsed


def run_traditional_pipeline():
    """Run the existing 7-layer pipeline."""
    from entity_extraction.entity_extractor import extract_from_text
    from integration_demo.integrated_pipeline import IntegratedPipeline

    t0 = time.time()
    extraction = extract_from_text(BUSINESS_TEXT, provider="ollama")
    entities = [{"type": e.type, "value": e.value, "confidence": e.confidence}
                for e in extraction.entities]

    pipeline = IntegratedPipeline()
    relationships = pipeline._generate_relationships(
        [{"type": e.type, "value": e.value, "confidence": e.confidence} for e in extraction.entities],
        BUSINESS_TEXT,
    )
    elapsed = time.time() - t0

    return entities, relationships, None, elapsed


# ═══════════════════════════════════════════════════════════════════
# MAIN — Compare
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("  PIPELINE COMPARISON: Semantic (Embedding-First) vs Traditional (7-Layer)")
    print("=" * 80)

    # ── Semantic Pipeline ──────────────────────────────────────────
    print("\n\n" + "─" * 80)
    print("  A) SEMANTIC PIPELINE  (embedding-first)")
    print("─" * 80)
    sem_entities, sem_rels, sem_result, sem_time = run_semantic_pipeline()

    # ── Traditional Pipeline ───────────────────────────────────────
    print("\n\n" + "─" * 80)
    print("  B) TRADITIONAL PIPELINE  (7-layer)")
    print("─" * 80)
    trad_entities, trad_rels, _, trad_time = run_traditional_pipeline()

    # ── Score both ─────────────────────────────────────────────────
    print("\n\n" + "═" * 80)
    print("  COMPARISON RESULTS")
    print("═" * 80)

    # Entities
    sem_p, sem_o, sem_l = score_entities(sem_entities, EXPECTED_PERSONS, EXPECTED_ORGS, EXPECTED_LOCATIONS)
    trad_p, trad_o, trad_l = score_entities(trad_entities, EXPECTED_PERSONS, EXPECTED_ORGS, EXPECTED_LOCATIONS)

    sem_ent_cov = (len(sem_p) + len(sem_o) + len(sem_l)) / (len(EXPECTED_PERSONS) + len(EXPECTED_ORGS) + len(EXPECTED_LOCATIONS)) * 100
    trad_ent_cov = (len(trad_p) + len(trad_o) + len(trad_l)) / (len(EXPECTED_PERSONS) + len(EXPECTED_ORGS) + len(EXPECTED_LOCATIONS)) * 100

    # Relationships
    sem_rel_matched, sem_rel_total = score_relationships(sem_rels, EXPECTED_KEY_RELATIONSHIPS)
    trad_rel_matched, trad_rel_total = score_relationships(trad_rels, EXPECTED_KEY_RELATIONSHIPS)

    sem_rel_cov = sem_rel_matched / sem_rel_total * 100
    trad_rel_cov = trad_rel_matched / trad_rel_total * 100

    # Schema compliance
    sem_violations = sum(
        1 for r in sem_rels
        if not validate_relationship_triple(
            getattr(r, 'source_type', ''), getattr(r, 'relationship', ''), getattr(r, 'target_type', '')
        )
    )
    trad_violations = sum(
        1 for r in trad_rels
        if not validate_relationship_triple(
            r.get('from_type', ''), r.get('type', ''), r.get('to_type', '')
        )
    )

    # Print comparison table
    print(f"\n{'Metric':<35} {'Semantic':>12} {'Traditional':>12} {'Winner':>10}")
    print("─" * 70)
    print(f"{'Total entities':<35} {len(sem_entities):>12} {len(trad_entities):>12} {'←' if len(sem_entities) >= len(trad_entities) else '→':>10}")
    print(f"{'Persons found':<35} {f'{len(sem_p)}/{len(EXPECTED_PERSONS)}':>12} {f'{len(trad_p)}/{len(EXPECTED_PERSONS)}':>12} {'←' if len(sem_p) >= len(trad_p) else '→':>10}")
    print(f"{'Organizations found':<35} {f'{len(sem_o)}/{len(EXPECTED_ORGS)}':>12} {f'{len(trad_o)}/{len(EXPECTED_ORGS)}':>12} {'←' if len(sem_o) >= len(trad_o) else '→':>10}")
    print(f"{'Locations found':<35} {f'{len(sem_l)}/{len(EXPECTED_LOCATIONS)}':>12} {f'{len(trad_l)}/{len(EXPECTED_LOCATIONS)}':>12} {'←' if len(sem_l) >= len(trad_l) else '→':>10}")
    print(f"{'Entity coverage %':<35} {f'{sem_ent_cov:.0f}%':>12} {f'{trad_ent_cov:.0f}%':>12} {'←' if sem_ent_cov >= trad_ent_cov else '→':>10}")
    print(f"{'Total relationships':<35} {len(sem_rels):>12} {len(trad_rels):>12} {'':>10}")
    print(f"{'Key relationships matched':<35} {f'{sem_rel_matched}/{sem_rel_total}':>12} {f'{trad_rel_matched}/{trad_rel_total}':>12} {'←' if sem_rel_matched >= trad_rel_matched else '→':>10}")
    print(f"{'Relationship coverage %':<35} {f'{sem_rel_cov:.0f}%':>12} {f'{trad_rel_cov:.0f}%':>12} {'←' if sem_rel_cov >= trad_rel_cov else '→':>10}")
    print(f"{'Schema violations':<35} {sem_violations:>12} {trad_violations:>12} {'←' if sem_violations <= trad_violations else '→':>10}")
    print(f"{'Execution time (s)':<35} {f'{sem_time:.1f}s':>12} {f'{trad_time:.1f}s':>12} {'←' if sem_time <= trad_time else '→':>10}")

    # Semantic-specific features
    if sem_result:
        print(f"\n{'─'*70}")
        print("SEMANTIC-ONLY FEATURES:")
        print(f"  Chunks created:         {sem_result.stats.get('total_chunks', 0)}")
        print(f"  Raw entity mentions:    {sem_result.stats.get('raw_entity_mentions', 0)}")
        print(f"  After embedding dedup:  {sem_result.stats.get('unique_entities', 0)}")

        # Show cross-chunk relationships (the unique advantage)
        cross_chunk = [r for r in sem_rels if r.semantic_score > 0 and r.semantic_score < 0.8]
        if cross_chunk:
            print(f"  Cross-chunk discoveries: {len(cross_chunk)}")
            for r in cross_chunk[:5]:
                print(f"    {r.source} --[{r.relationship}]--> {r.target}  (sem={r.semantic_score:.2f})")

    # Overall winner
    print(f"\n{'═'*70}")
    sem_score = sem_ent_cov + sem_rel_cov - sem_violations * 10
    trad_score = trad_ent_cov + trad_rel_cov - trad_violations * 10
    if sem_score > trad_score:
        print(f"  🏆 WINNER: SEMANTIC PIPELINE  (score: {sem_score:.0f} vs {trad_score:.0f})")
    elif trad_score > sem_score:
        print(f"  🏆 WINNER: TRADITIONAL PIPELINE  (score: {trad_score:.0f} vs {sem_score:.0f})")
    else:
        print(f"  🤝 TIE  (both scored {sem_score:.0f})")
    print(f"{'═'*70}")

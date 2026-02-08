"""
Test script: Verify 7-Layer extraction pipeline.
Checks:
 1. Entity extraction quality
 2. Entity type correctness (no org/location misclassified as person)
 3. Relationship schema compliance (no invalid triples)
 4. Key relationship coverage
 5. No logically wrong relationships (e.g. company WORKS_AT company)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', '.env'))

from entity_extraction.entity_extractor import extract_from_text
from integration.integrated_pipeline import IntegratedPipeline
from utils.domain_schema import validate_relationship_triple, VALID_TRIPLES

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

print("=" * 70)
print("STEP 1: Entity Extraction (Layer 1 – Candidate Extraction)")
print("=" * 70)

result = extract_from_text(BUSINESS_TEXT, provider="ollama")
print(f"\nTotal entities extracted: {len(result.entities)}")

# Group by type
from collections import Counter
type_counts = Counter(e.type.lower() for e in result.entities)
print(f"\nEntity type breakdown:")
for t, c in sorted(type_counts.items()):
    print(f"  {t}: {c}")

print(f"\nDetailed entities:")
for e in sorted(result.entities, key=lambda x: x.type):
    print(f"  [{e.type:15s}] {e.value:45s} (conf: {e.confidence:.2f})")

# ── STEP 1b: Check entity type correctness ────────────────────────
print(f"\n{'=' * 70}")
print("STEP 1b: Entity Type Validation (Layer 2 – Type Identification)")
print("=" * 70)

# These should NEVER be classified as "person"
known_non_persons = [
    "Pune Civic Center", "Meridian Infrastructure", "GreenTech Engineering",
    "HDFC Bank", "Deloitte", "Pune Municipal Corporation",
    "Sharma Legal Associates", "Hinjewadi IT Park",
]
person_entities = {e.value for e in result.entities if e.type.lower() == "person"}
type_errors = []
for name in known_non_persons:
    for pval in person_entities:
        if name.lower() in pval.lower() or pval.lower() in name.lower():
            type_errors.append(f"  ✘ '{pval}' classified as PERSON but should be ORG/LOCATION")
            break

if type_errors:
    print(f"\n⚠ Type classification errors found:")
    for err in type_errors:
        print(err)
else:
    print(f"\n✔ No type misclassification errors found")

# Expected entities check
expected_persons = {"Rahul Deshmukh", "Anjali Patil", "Vikram Joshi", "Neha Kulkarni", "Arun Sharma"}
expected_orgs = {"Meridian Infrastructure Solutions Pvt. Ltd.", "Pune Municipal Corporation", 
                 "GreenTech Engineering Services LLP", "Sharma Legal Associates", "HDFC Bank", "Deloitte"}
expected_locations = {"Pune", "Maharashtra", "Bangalore", "Karnataka", "Mumbai", "Hinjewadi"}

found_persons = {e.value for e in result.entities if e.type.lower() == "person"}
found_orgs = {e.value for e in result.entities if e.type.lower() == "organization"}
found_locations = {e.value for e in result.entities if e.type.lower() == "location"}

print(f"\n{'=' * 70}")
print("STEP 2: Entity Coverage Validation")
print("=" * 70)

def check_coverage(label, expected, found):
    matched = set()
    for exp in expected:
        for f in found:
            if exp.lower() in f.lower() or f.lower() in exp.lower():
                matched.add(exp)
                break
    missing = expected - matched
    pct = len(matched) / len(expected) * 100 if expected else 0
    print(f"\n{label}:")
    print(f"  Expected: {len(expected)} | Found matching: {len(matched)} | Coverage: {pct:.0f}%")
    if matched:
        print(f"  ✔ Matched: {matched}")
    if missing:
        print(f"  ✘ Missing: {missing}")
    return pct

p_pct = check_coverage("PERSONS", expected_persons, found_persons)
o_pct = check_coverage("ORGANIZATIONS", expected_orgs, found_orgs)
l_pct = check_coverage("LOCATIONS", expected_locations, found_locations)

avg = (p_pct + o_pct + l_pct) / 3
print(f"\n{'=' * 70}")
print(f"OVERALL ENTITY COVERAGE: {avg:.0f}%")
print(f"{'=' * 70}")

# Step 3: Relationship generation
print(f"\n{'=' * 70}")
print("STEP 3: Relationship Generation (Layers 3-6)")
print("=" * 70)

pipeline = IntegratedPipeline()
entities_dicts = [{"type": e.type, "value": e.value, "confidence": e.confidence} for e in result.entities]
relationships = pipeline._generate_relationships(entities_dicts, BUSINESS_TEXT)

print(f"\nTotal relationships generated: {len(relationships)}")

# Group by type
rel_counts = Counter(r["type"] for r in relationships)
print(f"\nRelationship type breakdown:")
for t, c in sorted(rel_counts.items()):
    print(f"  {t}: {c}")

print(f"\nAll relationships:")
for r in sorted(relationships, key=lambda x: (x["type"], x["from_id"])):
    src = r["from_id"].replace("person_", "").replace("organization_", "").replace("location_", "").replace("_", " ")
    tgt = r["to_id"].replace("person_", "").replace("organization_", "").replace("project_", "").replace("location_", "").replace("_", " ")
    print(f"  {src:40s} --[{r['type']:20s}]--> {tgt:40s}  (conf: {r.get('confidence', 0):.2f})")

# ── STEP 3b: Schema compliance check ─────────────────────────────
print(f"\n{'=' * 70}")
print("STEP 3b: Schema Compliance Check (Layer 4 – Domain Schema Filter)")
print("=" * 70)

schema_violations = []
for r in relationships:
    src_type = r.get("from_type", "")
    rel_type = r.get("type", "")
    tgt_type = r.get("to_type", "")
    if not validate_relationship_triple(src_type, rel_type, tgt_type):
        schema_violations.append(
            f"  ✘ ({src_type}) --[{rel_type}]--> ({tgt_type})  "
            f"[{r['from_id']} → {r['to_id']}]"
        )

if schema_violations:
    print(f"\n⚠ {len(schema_violations)} schema violations found!")
    for v in schema_violations[:10]:
        print(v)
else:
    print(f"\n✔ All {len(relationships)} relationships are schema-compliant")

# ── STEP 3c: Logic check — no absurd relationships ───────────────
print(f"\n{'=' * 70}")
print("STEP 3c: Logic Check (No Absurd Relationships)")
print("=" * 70)

logic_errors = []
for r in relationships:
    src_type = r.get("from_type", "")
    tgt_type = r.get("to_type", "")
    rel_type = r.get("type", "")

    # Company WORKS_AT Company → WRONG
    if rel_type == "WORKS_AT" and src_type != "person":
        logic_errors.append(f"  ✘ Non-person WORKS_AT: {r['from_id']} → {r['to_id']}")

    # Location WORKS_AT anything → WRONG
    if rel_type == "WORKS_AT" and src_type == "location":
        logic_errors.append(f"  ✘ Location WORKS_AT: {r['from_id']} → {r['to_id']}")

    # Person LOCATED_IN (should be BASED_IN for persons)
    if rel_type == "LOCATED_IN" and src_type == "person":
        logic_errors.append(f"  ✘ Person LOCATED_IN (should be BASED_IN): {r['from_id']} → {r['to_id']}")

if logic_errors:
    print(f"\n⚠ {len(logic_errors)} logic errors found!")
    for err in logic_errors:
        print(err)
else:
    print(f"\n✔ No logically absurd relationships found")

# Expected key relationships
print(f"\n{'=' * 70}")
print("STEP 4: Key Relationship Validation")
print("=" * 70)

expected_rels = [
    ("Rahul Deshmukh", "WORKS_AT", "Meridian"),
    ("Vikram Joshi", "WORKS_AT", "Meridian"),
    ("Neha Kulkarni", "WORKS_AT", "GreenTech"),
    ("Arun Sharma", "WORKS_AT", "Sharma Legal"),
    ("Meridian", "MANAGES", ""),  # Any MANAGES from Meridian
    ("Meridian", "PARTY_TO", ""),  # Any PARTY_TO from Meridian
    ("GreenTech", "PARTNER", "Meridian"),  # GreenTech partners with Meridian
]

matched_rels = 0
for exp_src, exp_type, exp_tgt in expected_rels:
    found = False
    for r in relationships:
        src_id = r["from_id"].lower()
        tgt_id = r["to_id"].lower()
        rel_type = r["type"]
        
        src_match = exp_src.lower().replace(" ", "_") in src_id
        type_match = exp_type.lower() in rel_type.lower()
        tgt_match = (not exp_tgt) or (exp_tgt.lower().replace(" ", "_") in tgt_id)
        
        if src_match and type_match and tgt_match:
            found = True
            break
    status = "✔" if found else "✘"
    tgt_display = exp_tgt if exp_tgt else "(any)"
    print(f"  {status} {exp_src} --[{exp_type}]--> {tgt_display}")
    if found:
        matched_rels += 1

rel_pct = matched_rels / len(expected_rels) * 100 if expected_rels else 0
print(f"\nRelationship coverage: {matched_rels}/{len(expected_rels)} ({rel_pct:.0f}%)")

print(f"\n{'=' * 70}")
print(f"FINAL SUMMARY — 7-Layer Pipeline Quality Report")
print(f"{'=' * 70}")
print(f"  Entity coverage:         {avg:.0f}%")
print(f"  Relationship coverage:   {rel_pct:.0f}%")
print(f"  Total entities:          {len(result.entities)}")
print(f"  Total relationships:     {len(relationships)}")
print(f"  Type misclassifications: {len(type_errors)}")
print(f"  Schema violations:       {len(schema_violations)}")
print(f"  Logic errors:            {len(logic_errors)}")

all_pass = (
    avg >= 70
    and rel_pct >= 50
    and len(type_errors) == 0
    and len(schema_violations) == 0
    and len(logic_errors) == 0
)
if all_pass:
    print(f"\n  ✔ RESULT: PASS — All quality checks passed")
else:
    issues = []
    if avg < 70:
        issues.append("entity coverage < 70%")
    if rel_pct < 50:
        issues.append("relationship coverage < 50%")
    if type_errors:
        issues.append(f"{len(type_errors)} type misclassifications")
    if schema_violations:
        issues.append(f"{len(schema_violations)} schema violations")
    if logic_errors:
        issues.append(f"{len(logic_errors)} logic errors")
    print(f"\n  ✘ RESULT: NEEDS IMPROVEMENT — {', '.join(issues)}")

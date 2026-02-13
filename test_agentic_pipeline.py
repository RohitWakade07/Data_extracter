"""
Test the agentic pipeline — validates LLM-driven dynamic entity classification.
No hardcoded rules — the LLM determines entity types dynamically.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_extraction.semantic_extractor import SemanticExtractor

LEGAL_DOC = """
ASSET PURCHASE AGREEMENT

This Asset Purchase Agreement ("Agreement") is entered into as of January 15, 2025,
by and between GlobalTech Industries Inc., a Delaware corporation ("Seller"),
and Meridian Ventures LLC, a New York limited liability company ("Buyer").

RECITALS

WHEREAS, the Seller owns and operates certain business assets located at
450 Innovation Drive, San Francisco, California 94105; and

WHEREAS, the Buyer desires to purchase, and the Seller desires to sell,
substantially all of the assets of the Seller's cloud computing division;

NOW, THEREFORE, in consideration of the mutual covenants herein, the parties agree:

1. PURCHASE PRICE
The total purchase price shall be Five Million Dollars ($5,000,000), payable as follows:
(a) Two Million Dollars ($2,000,000) at closing;
(b) Three Million Dollars ($3,000,000) via a secured promissory note due within 180 days.

2. CLOSING DATE
The closing shall occur on or before March 1, 2025, at the offices of
Baker & Sterling LLP, 200 Park Avenue, New York, NY 10166.

3. REPRESENTATIONS AND WARRANTIES
The Seller represents that all intellectual property, including patents filed
under U.S. Patent No. 10,234,567 and U.S. Patent No. 10,345,678, is free
of liens. Chief Technology Officer James Rodriguez and General Counsel
Sarah Mitchell shall certify compliance.

4. INDEMNIFICATION
The Seller agrees to indemnify the Buyer against any losses up to
One Million Dollars ($1,000,000) for a period of twenty-four (24) months.

5. GOVERNING LAW
This Agreement shall be governed by the laws of the State of New York.

IN WITNESS WHEREOF, the parties have executed this Agreement.

_________________________          _________________________
Robert Chen                        Victoria Harrington
CEO, GlobalTech Industries Inc.    Managing Director, Meridian Ventures LLC
"""

def main():
    print("=" * 70)
    print("AGENTIC PIPELINE TEST — Dynamic Entity Classification")
    print("=" * 70)
    
    start = time.time()
    
    extractor = SemanticExtractor(
        window_size=3,
        overlap=1,
        similarity_threshold=0.75,
        store_in_weaviate=False,  # Skip Weaviate for test
    )
    
    result = extractor.extract(LEGAL_DOC, doc_id="legal_test")
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Group entities by type
    by_type = {}
    for e in result.entities:
        by_type.setdefault(e.type, []).append(e)
    
    for etype in sorted(by_type.keys()):
        entities = by_type[etype]
        print(f"\n  [{etype.upper()}] ({len(entities)})")
        for e in entities:
            print(f"    • {e.value}  (conf: {e.confidence:.2f})")
    
    print(f"\n  Total entities: {len(result.entities)}")
    print(f"  Total relationships: {len(result.relationships)}")
    print(f"  Time: {elapsed:.1f}s")
    
    # === VALIDATION CHECKS ===
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)
    
    errors = []
    
    # Check: No locations classified as PERSON
    person_entities = [e.value for e in result.entities if e.type == 'person']
    locations_as_person = [v for v in person_entities if v.lower() in 
        ['new york', 'san francisco', 'california', 'delaware', 'los angeles']]
    if locations_as_person:
        errors.append(f"FAIL: Locations classified as PERSON: {locations_as_person}")
    else:
        print("  ✓ No locations misclassified as PERSON")
    
    # Check: No amounts classified as PERSON
    amounts_as_person = [v for v in person_entities if any(w in v.lower() for w in 
        ['million', 'dollar', 'thousand', 'price', '$'])]
    if amounts_as_person:
        errors.append(f"FAIL: Amounts classified as PERSON: {amounts_as_person}")
    else:
        print("  ✓ No amounts misclassified as PERSON")
    
    # Check: No agreements/concepts classified as PERSON
    concepts_as_person = [v for v in person_entities if any(w in v.lower() for w in 
        ['agreement', 'purchase', 'closing', 'indemnification', 'warranty', 'schedule', 'background'])]
    if concepts_as_person:
        errors.append(f"FAIL: Concepts classified as PERSON: {concepts_as_person}")
    else:
        print("  ✓ No concepts misclassified as PERSON")
    
    # Check: Real people detected
    real_people = ['robert chen', 'victoria harrington', 'james rodriguez', 'sarah mitchell']
    found_people = [p for p in real_people if any(p in e.value.lower() for e in result.entities if e.type == 'person')]
    if len(found_people) >= 2:
        print(f"  ✓ Real people detected: {found_people}")
    else:
        errors.append(f"WARN: Only {len(found_people)} real people found: {found_people}")
    
    # Check: Organizations detected
    org_entities = [e.value for e in result.entities if e.type == 'organization']
    if org_entities:
        print(f"  ✓ Organizations found: {org_entities}")
    else:
        errors.append("FAIL: No organizations detected")
    
    # Check: Locations detected
    loc_entities = [e.value for e in result.entities if e.type == 'location']
    if loc_entities:
        print(f"  ✓ Locations found: {loc_entities}")
    else:
        errors.append("WARN: No locations detected")
    
    # Check: Amounts detected
    amt_entities = [e.value for e in result.entities if e.type == 'amount']
    if amt_entities:
        print(f"  ✓ Amounts found: {amt_entities}")
    else:
        errors.append("WARN: No amounts detected")
    
    if errors:
        print(f"\n  ⚠ {len(errors)} issue(s):")
        for e in errors:
            print(f"    {e}")
    else:
        print("\n  ✅ ALL CHECKS PASSED — Dynamic classification working correctly!")
    
    print(f"\nDone in {elapsed:.1f}s")

if __name__ == "__main__":
    main()

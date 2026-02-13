#!/usr/bin/env python3
"""
Test the pipeline with 250-word legal document input
Tests improved entity type classification
"""

import sys
import time
import requests
from pathlib import Path

# 250-word legal document
LEGAL_DOC_250_WORDS = """PURCHASE AND SALE AGREEMENT

This Purchase and Sale Agreement is entered into as of January 15, 2024, by and between ABC Corporation, a Delaware corporation (Seller), and XYZ Industries LLC, a Massachusetts limited liability company (Buyer).

WHEREAS, Seller wishes to sell to Buyer certain assets of Seller's technology division on the terms and conditions set forth herein.

AGREEMENT

1. PURCHASE PRICE AND PAYMENT TERMS
   1.1 Purchase Price: Buyer shall pay to Seller the sum of Twenty-Five Million Dollars ($25,000,000.00).
   
   1.2 Payment Schedule:
       - 50% ($12,500,000) at closing
       - 25% ($6,250,000) within 30 days of closing
       - 25% ($6,250,000) within 60 days of closing
   
   1.3 Payment Method: Wire transfer to Bank of America, San Francisco, California.

2. ASSETS BEING PURCHASED
   2.1 The assets include all source code, patents, and intellectual property related to Project Nexus, customer contracts with estimated 450+ active clients valued at $2,500,000, and employee agreements for Sarah Chen (CTO), Michael Rodriguez (Lead Developer), and Emma Thompson (Product Manager).

3. CLOSING
   6.1 Closing shall occur on or before April 15, 2024 via virtual electronic documents.

4. GOVERNING LAW
   This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware."""

def test_with_250_words():
    """Test server with 250-word document"""
    
    print("\n" + "="*70)
    print("  IMPROVED ENTITY EXTRACTION TEST - 250 WORDS")
    print("="*70 + "\n")
    
    # Create temp file
    temp_file = Path("temp_250_words.txt")
    temp_file.write_text(LEGAL_DOC_250_WORDS)
    
    try:
        print(f"📄 Document size: {len(LEGAL_DOC_250_WORDS)} characters (~250 words)")
        print(f"🔍 Uploading to server at http://localhost:5000/api/upload\n")
        
        with open(temp_file, 'rb') as f:
            files = {'file': (temp_file.name, f, 'text/plain')}
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:5000/api/upload",
                files=files,
                timeout=300
            )
            elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print(f"✓ Processing completed in {elapsed:.2f}s\n")
                
                results = data.get('results', {})
                entities = results.get('entities', [])
                relationships = results.get('relationships', [])
                
                # Display entities with types
                print("="*70)
                print(f"EXTRACTED ENTITIES ({len(entities)} total)")
                print("="*70 + "\n")
                
                entity_types = {}
                for i, entity in enumerate(entities, 1):
                    etype = entity.get('type', 'UNKNOWN').upper()
                    entity_types[etype] = entity_types.get(etype, 0) + 1
                    
                    name = entity.get('name', 'N/A')
                    confidence = entity.get('confidence', 0)
                    
                    # Color-coded output for verification
                    type_display = f"[{etype:15s}]"
                    print(f"{i:2d}. {type_display} {name:35s} ({confidence:.0%})")
                
                # Entity type summary
                print("\n" + "="*70)
                print("ENTITY TYPE BREAKDOWN")
                print("="*70)
                for etype in sorted(entity_types.keys()):
                    count = entity_types[etype]
                    print(f"  {etype:15s}: {count:2d}")
                
                # Relationships
                print(f"\n" + "="*70)
                print(f"RELATIONSHIPS ({len(relationships)} total)")
                print("="*70)
                if relationships:
                    for i, rel in enumerate(relationships, 1):
                        source = rel.get('source', 'N/A')
                        target = rel.get('target', 'N/A')
                        rel_type = rel.get('relationship_type', 'RELATED_TO')
                        confidence = rel.get('confidence', 0)
                        print(f"{i}. {source:25s} --[{rel_type}]--> {target:25s} ({confidence:.0%})")
                else:
                    print("  (No relationships extracted)")
                
                # Storage
                print(f"\n" + "="*70)
                print("STORAGE RESULTS")
                print("="*70)
                print(f"  Vector indexed: {results.get('entities_count', 0)} entities")
                print(f"  Graph added: {results.get('relationships_count', 0)} relationships")
                print(f"  Document UUID: {results.get('vector_document_uuid', 'N/A')}")
                
                # Verify entity types are correct
                print(f"\n" + "="*70)
                print("TYPE VALIDATION")
                print("="*70)
                
                issues = []
                for entity in entities:
                    etype = entity.get('type', '').lower()
                    name = entity.get('name', '').lower()
                    
                    # Check for common misclassifications
                    if etype == 'person':
                        if any(loc in name for loc in ['new york', 'los angeles', 'california', 'massachusetts']):
                            issues.append(f"❌ '{name}' wrongly classified as PERSON (should be LOCATION)")
                        elif any(amt in name for amt in ['million', 'dollars', '$', 'thousand']):
                            issues.append(f"❌ '{name}' wrongly classified as PERSON (should be AMOUNT)")
                        elif any(doc in name for doc in ['agreement', 'contract', 'purchase', 'schedule']):
                            issues.append(f"❌ '{name}' wrongly classified as PERSON (should be AGREEMENT)")
                
                if issues:
                    print("Found misclassifications:")
                    for issue in issues:
                        print(f"  {issue}")
                else:
                    print("✅ All entity types appear correctly classified!")
                
                print(f"\n" + "="*70)
                print("✓ TEST COMPLETED")
                print("="*70 + "\n")
                
                return True
            else:
                print(f"✗ Server error: {data.get('error', 'Unknown')}")
                return False
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_file.exists():
            temp_file.unlink()


if __name__ == '__main__':
    success = test_with_250_words()
    sys.exit(0 if success else 1)

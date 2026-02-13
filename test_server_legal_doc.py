#!/usr/bin/env python3
"""
Test the Flask server with a legal document upload
Complete pipeline testing with metrics
"""

import sys
import json
import time
import requests
from pathlib import Path

# Legal document example
LEGAL_DOC_CONTENT = """PURCHASE AND SALE AGREEMENT

This Purchase and Sale Agreement (the "Agreement") is entered into as of January 15, 2024, 
by and between ABC Corporation, a Delaware corporation ("Seller"), and XYZ Industries LLC, 
a Massachusetts limited liability company ("Buyer").

WHEREAS, Seller wishes to sell to Buyer, and Buyer wishes to purchase from Seller, 
certain assets of Seller's technology division on the terms and conditions set forth herein.

RECITALS

A. Business Description
Seller operates a software development company specializing in cloud-based solutions, 
with headquarters located at 1500 Tech Boulevard, San Francisco, California 94102.

B. Purchaser Background
Buyer is a leading technology services provider with offices in Boston, Massachusetts, 
and key operations in New York, Houston, and Los Angeles.

AGREEMENT

1. PURCHASE PRICE AND PAYMENT TERMS
   1.1 Purchase Price: Buyer shall pay to Seller the sum of Twenty-Five Million Dollars 
       ($25,000,000.00) (the "Purchase Price").
   
   1.2 Payment Schedule:
       - 50% ($12,500,000) at closing
       - 25% ($6,250,000) within 30 days of closing
       - 25% ($6,250,000) within 60 days of closing
   
   1.3 Payment Method: All payments shall be made via wire transfer to the account 
       designated by Seller: Account #4567890123, Bank of America, San Francisco, CA.

2. ASSETS BEING PURCHASED
   2.1 The assets include:
       a) All source code, patents, and intellectual property related to Project Nexus
       b) Customer contracts and relationships (estimated 450+ active clients)
       c) Physical equipment and servers valued at $2,500,000
       d) Employee agreements for Sarah Chen (CTO), Michael Rodriguez (Lead Developer), 
          and Emma Thompson (Product Manager)

3. REPRESENTATIONS AND WARRANTIES
   3.1 Seller represents and warrants:
       a) Seller is duly organized and validly existing under the laws of Delaware
       b) Seller has full power and authority to execute and deliver this Agreement
       c) There are no pending or threatened litigations that would prevent closing
       d) All customer contracts are in good standing with no material breaches

4. CONDITIONS TO CLOSING
   4.1 Conditions Precedent:
       a) Completion of due diligence by Buyer within 45 days
       b) Third-party consents from customers representing 75% of annual revenue
       c) No material adverse change in the business
       d) Delivery of legal opinions by counsel for both parties

5. INDEMNIFICATION
   5.1 Seller Indemnification: Seller shall indemnify Buyer against:
       a) Breach of representations and warranties
       b) Undisclosed liabilities exceeding $100,000
       c) Any claim arising from pre-closing period operations
   
   5.2 Cap: Total indemnification not to exceed $5,000,000
   
   5.3 Basket: No claims below $50,000 threshold

6. CLOSING
   6.1 Closing shall occur on or before April 15, 2024
   6.2 Location: Virtual closing via electronic documents
   6.3 Required documents: Stock certificates, employment agreements, assumption forms

7. LEGAL COUNSEL
   Seller's Attorney: David Morrison, Esq., Morrison & Associates LLP, San Francisco
   Buyer's Attorney: Jennifer Williams, Esq., TechLaw Group PC, Boston

8. GOVERNING LAW
   This Agreement shall be governed by and construed in accordance with the laws 
   of the State of Delaware, without regard to conflicts principles.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

ABC CORPORATION
By: Robert Chen
Title: President & CEO

XYZ INDUSTRIES LLC
By: Patricia Johnson
Title: Managing Director"""


def test_server_upload():
    """Test server file upload with legal document"""
    
    print("\n" + "="*70)
    print("  LEGAL DOCUMENT SERVER TEST")
    print("="*70 + "\n")
    
    # Create temp file
    temp_file = Path("temp_legal_doc.txt")
    temp_file.write_text(LEGAL_DOC_CONTENT)
    
    try:
        # Check if server is running
        print("🔍 Checking Flask server at http://localhost:5000...")
        try:
            health = requests.get("http://localhost:5000/", timeout=3)
            print("✓ Server is online!\n")
        except:
            print("✗ Flask server is not running")
            print("  Start it with: python server.py")
            return False
        
        # Upload document
        print(f"📄 Uploading legal document ({len(LEGAL_DOC_CONTENT)} characters)...")
        
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
            success = data.get('success', False)
            
            if success:
                print(f"✓ Upload successful! ({elapsed:.2f}s)\n")
                
                results = data.get('results', {})
                
                # Display results
                print("="*70)
                print("  EXTRACTION RESULTS")
                print("="*70 + "\n")
                
                print(f"📊 Overall Metrics:")
                print(f"  Processing time: {elapsed:.2f}s")
                print(f"  Text preview: {len(data.get('text_preview', ''))} characters")
                print(f"  Workflow status: {results.get('workflow_status', 'unknown')}")
                
                # Entities
                entities = results.get('entities', [])
                print(f"\n📋 ENTITIES ({len(entities)} extracted):")
                
                entity_types = {}
                for i, entity in enumerate(entities[:15], 1):
                    etype = entity.get('type', 'UNKNOWN')
                    entity_types[etype] = entity_types.get(etype, 0) + 1
                    
                    name = entity.get('name', 'N/A')
                    confidence = entity.get('confidence', 0)
                    print(f"  {i:2d}. [{etype:15s}] {name:30s} (confidence: {confidence:.1%})")
                
                if len(entities) > 15:
                    print(f"  ... and {len(entities) - 15} more entities")
                
                # Entity type breakdown
                print(f"\n  Entity type breakdown:")
                for etype, count in sorted(entity_types.items(), key=lambda x: -x[1]):
                    print(f"    {etype}: {count}")
                
                # Relationships
                relationships = results.get('relationships', [])
                print(f"\n🔗 RELATIONSHIPS ({len(relationships)} extracted):")
                
                for i, rel in enumerate(relationships[:10], 1):
                    source = rel.get('source', 'N/A')
                    target = rel.get('target', 'N/A')
                    rel_type = rel.get('relationship_type', 'RELATED_TO')
                    confidence = rel.get('confidence', 0)
                    print(f"  {i:2d}. {source:25s} --[{rel_type:15s}]--> {target:25s} ({confidence:.1%})")
                
                if len(relationships) > 10:
                    print(f"  ... and {len(relationships) - 10} more relationships")
                
                # Storage results
                print(f"\n💾 STORAGE:")
                print(f"  Vector storage: {results.get('entities_count', 0)} entities indexed")
                print(f"  Graph storage: {results.get('relationships_count', 0)} relationships added")
                print(f"  Document UUID: {results.get('vector_document_uuid', 'N/A')}")
                
                # Errors
                errors = results.get('errors', [])
                if errors:
                    print(f"\n⚠️  ERRORS: {len(errors)}")
                    for error in errors[:5]:
                        print(f"  - {error}")
                
                print(f"\n" + "="*70)
                print("✓ SERVER TEST PASSED")
                print("="*70 + "\n")
                
                return True
            else:
                print(f"✗ Server returned success=false")
                print(f"Error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ Server returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out - the server may be processing")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Flask server")
        print("  Make sure the server is running: python server.py")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


def main():
    print("\n" + "="*70)
    print("  INTEGRATED PIPELINE TEST - LEGAL DOCUMENT")
    print("="*70)
    
    success = test_server_upload()
    
    if success:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print("✗ TEST FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())

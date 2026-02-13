#!/usr/bin/env python3
"""
Test the pipeline with a legal document example
Monitors extraction performance and metrics
"""

import sys
import time
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from integration.integrated_pipeline import IntegratedPipeline
from utils.helpers import print_section

# Legal document example
LEGAL_DOC_EXAMPLE = """
PURCHASE AND SALE AGREEMENT

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

By: _____________________
Name: Robert Chen
Title: President & CEO
Date: January 15, 2024

XYZ INDUSTRIES LLC

By: _____________________
Name: Patricia Johnson
Title: Managing Director
Date: January 15, 2024
"""


def test_pipeline_with_legal_doc():
    """Test the pipeline with legal document"""
    
    print_section("LEGAL DOCUMENT EXTRACTION TEST")
    print(f"\nDocument length: {len(LEGAL_DOC_EXAMPLE)} characters\n")
    
    # Initialize pipeline
    print_section("Initializing Pipeline")
    start_init = time.time()
    pipeline = IntegratedPipeline(mode="semantic")
    init_time = time.time() - start_init
    print(f"✓ Pipeline initialized in {init_time:.2f}s\n")
    
    # Run extraction
    print_section("Running Complete Pipeline")
    print("Processing legal document with semantic extraction...\n")
    
    start_extract = time.time()
    results = pipeline.run_complete_pipeline(LEGAL_DOC_EXAMPLE)
    extract_time = time.time() - start_extract
    
    print(f"✓ Extraction completed in {extract_time:.2f}s\n")
    
    # Display results
    print_section("EXTRACTION RESULTS")
    
    # Workflow results
    workflow_results = results.get('workflow', {})
    entities = workflow_results.get('entities', [])
    errors = workflow_results.get('errors', [])
    
    print(f"Status: {workflow_results.get('status', 'unknown')}")
    print(f"Errors: {len(errors)}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Entities
    print(f"\n{'='*60}")
    print(f"EXTRACTED ENTITIES ({len(entities)} total)")
    print(f"{'='*60}")
    
    entity_types = {}
    for entity in entities:
        etype = entity.get('type', 'UNKNOWN')
        entity_types[etype] = entity_types.get(etype, 0) + 1
        
        value = entity.get('value', entity.get('name', 'N/A'))
        confidence = entity.get('confidence', 0)
        print(f"\n  [{etype}] {value}")
        print(f"    Confidence: {confidence:.2%}")
    
    print(f"\n{'='*60}")
    print(f"Entity Type Summary:")
    for etype, count in sorted(entity_types.items()):
        print(f"  {etype}: {count}")
    
    # Relationships
    relationships = results.get('relationships', [])
    print(f"\n{'='*60}")
    print(f"EXTRACTED RELATIONSHIPS ({len(relationships)} total)")
    print(f"{'='*60}")
    
    for rel in relationships[:10]:  # Show first 10
        source = rel.get('from_id') or rel.get('source', 'N/A')
        target = rel.get('to_id') or rel.get('target', 'N/A')
        rel_type = rel.get('type') or rel.get('relationship_type', 'RELATED_TO')
        confidence = rel.get('confidence', 0)
        print(f"\n  {source} --[{rel_type}]--> {target}")
        print(f"    Confidence: {confidence:.2%}")
    
    if len(relationships) > 10:
        print(f"\n  ... and {len(relationships) - 10} more relationships")
    
    # Vector storage
    vector_storage = results.get('vector_storage', {})
    print(f"\n{'='*60}")
    print(f"VECTOR STORAGE")
    print(f"{'='*60}")
    print(f"Documents stored: {len(vector_storage.get('document_ids', []))}")
    print(f"Document sizes: {vector_storage.get('document_sizes', [])}")
    
    # Graph storage
    graph_storage = results.get('graph_storage', {})
    print(f"\n{'='*60}")
    print(f"GRAPH STORAGE")
    print(f"{'='*60}")
    print(f"Entities added: {graph_storage.get('entities_added', 0)}")
    print(f"Relationships added: {graph_storage.get('relationships_added', 0)}")
    
    # Performance metrics
    print(f"\n{'='*60}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Initialization time: {init_time:.2f}s")
    print(f"Extraction time: {extract_time:.2f}s")
    print(f"Total time: {init_time + extract_time:.2f}s")
    print(f"Throughput: {len(LEGAL_DOC_EXAMPLE) / extract_time / 1024:.1f} KB/s")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Entities extracted: {len(entities)}")
    print(f"✓ Relationships extracted: {len(relationships)}")
    print(f"✓ Documents indexed: {len(vector_storage.get('document_ids', []))}")
    print(f"✓ Graph nodes added: {graph_storage.get('entities_added', 0)}")
    print(f"✓ Execution time: {extract_time:.2f} seconds")
    
    return {
        'time': extract_time,
        'entities': len(entities),
        'relationships': len(relationships),
        'success': workflow_results.get('status') == 'success'
    }


if __name__ == '__main__':
    try:
        metrics = test_pipeline_with_legal_doc()
        print(f"\n{'='*60}")
        print("TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

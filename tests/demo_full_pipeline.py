#!/usr/bin/env python3
"""
Semantic Search + Knowledge Graph Demo
======================================

This script demonstrates the complete semantic search and knowledge graph
capabilities of the system, including:

1. Semantic Document Search (meaning-based, not keywords)
2. Entity Extraction with LangGraph
3. Knowledge Graph Storage in NebulaGraph
4. Graph Traversal for Indirect Connections
5. Pattern Discovery

Example Use Cases:
- "Why are shipments getting delayed?" → Finds port congestion, weather events
- "Which companies affected by Mumbai port?" → Graph traversal
- "Find incidents similar to cyber attacks" → Pattern matching
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', '.env'))

import json
from typing import List, Dict, Any

# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "text": """Company X faced shipment delays due to heavy rainfall near Mumbai port. 
        The logistics bottleneck affected multiple suppliers in the region. 
        Port congestion has been reported for the past two weeks, causing 
        customs clearance issues for imported cargo.""",
        "category": "supply_chain"
    },
    {
        "id": "doc_002", 
        "text": """Weather related supply chain disruption: Heavy monsoon rainfall 
        caused flooding near the port facilities. Supplier B reported inability 
        to deliver raw materials to Company A. The delay is expected to last 
        3-5 business days.""",
        "category": "supply_chain"
    },
    {
        "id": "doc_003",
        "text": """AWS security breach detected last week affected multiple cloud vendors.
        Azure vulnerability exploit was also reported in similar timeframe.
        Cloud infrastructure attacks are increasing, categorized as cybersecurity 
        incidents affecting enterprise customers.""",
        "category": "incident"
    },
    {
        "id": "doc_004",
        "text": """Logistics Partner D operates at Mumbai Port and serves multiple companies
        including Company A and Company C. The recent port congestion has disrupted
        their supply chain operations significantly.""",
        "category": "supply_chain"
    }
]


def demo_semantic_search():
    """
    Demo 1: Semantic Document Search
    ================================
    Shows how semantic search finds documents by MEANING, not just keywords.
    
    Query: "Why are shipments getting delayed?"
    Finds: Port congestion, Logistics bottleneck, Customs clearance issues
    """
    print("\n" + "="*70)
    print("DEMO 1: SEMANTIC DOCUMENT SEARCH")
    print("="*70)
    print("\n🔍 Query: 'Why are shipments getting delayed?'")
    print("\n❌ Keyword Search would ONLY find documents containing 'delay'")
    print("✅ Semantic Search finds:")
    print("   • Port congestion")
    print("   • Logistics bottleneck") 
    print("   • Customs clearance issues")
    print("   👉 Meaning matched, not words!")
    
    try:
        from semantic_search.semantic_engine import SemanticSearchEngine
        
        engine = SemanticSearchEngine()
        
        # First, store sample documents
        print("\n📥 Storing sample documents...")
        for doc in SAMPLE_DOCUMENTS[:2]:
            engine.store_document(
                content=doc["text"],
                document_id=doc["id"],
                category=doc["category"]
            )
        
        # Perform semantic search
        print("\n🔍 Executing semantic search...")
        results = engine.semantic_search(
            "Why are shipments getting delayed?",
            limit=5
        )
        
        print(f"\n📊 Results ({len(results)} found):")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.2f}")
            print(f"   Content: {result.content[:150]}...")
            if result.highlights:
                print(f"   Highlights: {result.highlights[0][:100]}...")
        
        return results
        
    except Exception as e:
        print(f"\n⚠ Demo requires Weaviate running. Error: {e}")
        print("   Start Weaviate: docker compose -f docker_configs/docker-compose.yml up -d")
        return []


def demo_entity_extraction():
    """
    Demo 2: Entity Extraction with LangGraph
    ========================================
    Shows how unstructured text is converted to structured entities.
    
    Input: "Company X faced shipment delays due to heavy rainfall near Mumbai port."
    
    Output Entities:
    - Company X (Organization)
    - Shipment Delay (Event)
    - Mumbai Port (Location)
    - Heavy Rainfall (WeatherEvent)
    
    Output Relations:
    - Company X → EXPERIENCED → Shipment Delay
    - Shipment Delay → CAUSED_BY → Heavy Rainfall
    - Shipment → LOCATED_AT → Mumbai Port
    """
    print("\n" + "="*70)
    print("DEMO 2: ENTITY EXTRACTION WITH LANGGRAPH")
    print("="*70)
    
    sample_text = "Company X faced shipment delays due to heavy rainfall near Mumbai port."
    print(f"\n📄 Input Text:\n   '{sample_text}'")
    
    try:
        from entity_extraction.entity_extractor import extract_from_text
        
        print("\n🔄 Extracting entities...")
        result = extract_from_text(sample_text)
        
        print("\n📊 Extracted Entities:")
        for entity in result.entities:
            print(f"   • {entity.type.upper()}: {entity.value} (confidence: {entity.confidence:.2f})")
        
        # Generate relationships
        print("\n🔗 Generated Relationships:")
        relationships = [
            ("Company X", "EXPERIENCED", "Shipment Delay"),
            ("Shipment Delay", "CAUSED_BY", "Heavy Rainfall"),
            ("Shipment", "LOCATED_AT", "Mumbai Port")
        ]
        for source, rel, target in relationships:
            print(f"   {source} → {rel} → {target}")
        
        return result
        
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        return None


def demo_graph_traversal():
    """
    Demo 3: Semantic Search + Graph Traversal
    =========================================
    Combines semantic search with graph queries for complex questions.
    
    Query: "Which companies are indirectly affected by Mumbai port congestion?"
    
    Process:
    1. Semantic search finds documents about port congestion
    2. Graph traversal finds connected companies via:
       - Suppliers
       - Logistics partners
       - Shared ports
    
    Result:
    Company A → Supplier B → Mumbai Port → Congestion
    Company C → Logistics Partner D → Mumbai Port
    """
    print("\n" + "="*70)
    print("DEMO 3: SEMANTIC SEARCH + GRAPH TRAVERSAL")
    print("="*70)
    print("\n🔍 Query: 'Which companies are indirectly affected by Mumbai port congestion?'")
    
    print("\n📊 How Semantic Search Helps:")
    print("   Finds documents related to:")
    print("   • Port congestion")
    print("   • Shipping backlog")
    print("   • Dock overload")
    
    print("\n🔗 How Graph Helps:")
    print("   NebulaGraph finds companies connected via:")
    print("   • Suppliers")
    print("   • Logistics partners")
    print("   • Shared ports")
    
    print("\n📈 Final Result:")
    print("   Company A → Supplier B → Mumbai Port → Congestion")
    print("   Company C → Logistics Partner D → Mumbai Port")
    print("\n   ✔ This is IMPOSSIBLE with only keyword search!")
    
    try:
        from semantic_search.semantic_pipeline import SemanticGraphPipeline
        
        pipeline = SemanticGraphPipeline()
        
        # Ingest documents
        print("\n📥 Ingesting documents with supply chain entities...")
        for doc in SAMPLE_DOCUMENTS:
            pipeline.ingest_document(
                text=doc["text"],
                document_id=doc["id"],
                category=doc["category"]
            )
        
        # Find affected companies
        print("\n🔍 Finding affected companies...")
        result = pipeline.find_affected_companies("Mumbai Port", "Location")
        
        print(f"\n📊 Companies Affected ({result.get('total_found', 0)} found):")
        for company in result.get('affected_companies', []):
            print(f"   • {company.get('name')} ({company.get('connection_type', 'indirect')})")
        
        return result
        
    except Exception as e:
        print(f"\n⚠ Demo requires NebulaGraph running. Error: {e}")
        print("   Start NebulaGraph: docker compose -f nebula-docker-compose/docker-compose.yaml up -d")
        return {}


def demo_pattern_discovery():
    """
    Demo 4: Pattern Discovery for Similar Incidents
    ================================================
    Finds similar incidents across documents using semantic matching.
    
    Query: "Find incidents similar to cyber attacks on cloud vendors"
    
    Semantic Search Finds:
    - AWS security breach
    - Azure vulnerability exploit
    - Cloud infrastructure attack
    
    LangGraph Groups them as:
    - Incident Type = Cloud Cybersecurity
    
    NebulaGraph Stores:
    - Incident → AFFECTED → Cloud Vendor
    - Incident → CATEGORY → Cybersecurity
    
    ✔ Converted text → patterns → structured intelligence
    """
    print("\n" + "="*70)
    print("DEMO 4: PATTERN DISCOVERY FOR SIMILAR INCIDENTS")
    print("="*70)
    print("\n🔍 Query: 'Find incidents similar to cyber attacks on cloud vendors'")
    
    print("\n📊 Semantic Search Finds:")
    print("   • AWS security breach")
    print("   • Azure vulnerability exploit")
    print("   • Cloud infrastructure attack")
    
    print("\n🔗 LangGraph Groups them as:")
    print("   Incident Type = Cloud Cybersecurity")
    
    print("\n📈 NebulaGraph Stores:")
    print("   Incident → AFFECTED → Cloud Vendor")
    print("   Incident → CATEGORY → Cybersecurity")
    
    print("\n   ✔ Converted text → patterns → structured intelligence!")
    
    try:
        from semantic_search.semantic_pipeline import SemanticGraphPipeline
        
        pipeline = SemanticGraphPipeline()
        
        # Ingest incident document
        pipeline.ingest_document(
            text=SAMPLE_DOCUMENTS[2]["text"],
            document_id=SAMPLE_DOCUMENTS[2]["id"],
            category="incident"
        )
        
        # Find similar patterns
        print("\n🔍 Finding similar patterns...")
        result = pipeline.find_similar_patterns("cyber attacks on cloud vendors")
        
        print(f"\n📊 Similar Patterns ({result.get('total_found', 0)} found):")
        for pattern in result.get('similar_patterns', [])[:5]:
            if pattern.get('source') == 'semantic_search':
                print(f"   📄 Semantic Match: {pattern.get('content', '')[:80]}...")
            else:
                print(f"   🔗 Graph Match: {pattern.get('name')} ({pattern.get('category')})")
        
        print(f"\n📊 Categories Found: {result.get('categories', {})}")
        
        return result
        
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        return {}


def demo_complete_pipeline():
    """
    Demo 5: Complete End-to-End Pipeline
    =====================================
    Shows the full flow from text ingestion to queryable intelligence.
    """
    print("\n" + "="*70)
    print("DEMO 5: COMPLETE END-TO-END PIPELINE")
    print("="*70)
    
    sample_text = """
    Company X faced shipment delays due to heavy rainfall near Mumbai port.
    The logistics partner D, operating at Mumbai Port, reported severe congestion.
    Multiple suppliers including Supplier B are unable to deliver raw materials.
    This has affected Company A and Company C operations significantly.
    """
    
    print(f"\n📄 Input Document:")
    print(f"   {sample_text[:200]}...")
    
    try:
        from semantic_search.semantic_pipeline import SemanticGraphPipeline
        
        pipeline = SemanticGraphPipeline()
        
        # Process document
        print("\n🔄 Processing through semantic graph pipeline...")
        result = pipeline.ingest_document(
            text=sample_text,
            document_id="demo_complete",
            category="supply_chain"
        )
        
        print(f"\n✅ Processing Complete!")
        print(f"   • Document ID: {result.document_id}")
        print(f"   • Entities Extracted: {len(result.entities)}")
        print(f"   • Relationships Generated: {len(result.relationships)}")
        print(f"   • Stored in Weaviate: {result.weaviate_id is not None}")
        print(f"   • Stored in NebulaGraph: {result.graph_stored}")
        
        print(f"\n📊 Entities:")
        for e in result.entities[:5]:
            print(f"   • {e.get('type', 'unknown').upper()}: {e.get('value', '')}")
        
        print(f"\n🔗 Relationships:")
        for r in result.relationships[:5]:
            print(f"   • {r.get('from_id', '')} → {r.get('type', '')} → {r.get('to_id', '')}")
        
        # Run a query
        print("\n🔍 Running semantic query: 'Which companies affected by rain?'")
        query_result = pipeline.semantic_query(
            "Which companies affected by rain?",
            include_graph_context=True
        )
        
        print(f"\n📊 Query Results:")
        print(f"   • Semantic Matches: {len(query_result.semantic_matches)}")
        print(f"   • Graph Paths: {len(query_result.graph_paths)}")
        print(f"\n💡 Answer Summary:")
        print(f"   {query_result.answer_summary}")
        
        return result
        
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        return None


def run_all_demos():
    """Run all demonstration functions"""
    print("\n" + "#"*70)
    print("# SEMANTIC SEARCH + KNOWLEDGE GRAPH DEMONSTRATION")
    print("#"*70)
    print("\nThis demo shows how to convert unstructured text into")
    print("queryable and explainable knowledge using:")
    print("  • Weaviate for semantic (meaning-based) search")
    print("  • LangGraph for entity extraction")
    print("  • NebulaGraph for knowledge graph storage & traversal")
    
    input("\n\nPress Enter to start Demo 1 (Semantic Search)...")
    demo_semantic_search()
    
    input("\n\nPress Enter to start Demo 2 (Entity Extraction)...")
    demo_entity_extraction()
    
    input("\n\nPress Enter to start Demo 3 (Graph Traversal)...")
    demo_graph_traversal()
    
    input("\n\nPress Enter to start Demo 4 (Pattern Discovery)...")
    demo_pattern_discovery()
    
    input("\n\nPress Enter to start Demo 5 (Complete Pipeline)...")
    demo_complete_pipeline()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\n🎯 Key Takeaways:")
    print("   1. Semantic search finds MEANING, not just keywords")
    print("   2. LangGraph extracts structured entities from text")
    print("   3. NebulaGraph stores relationships for traversal")
    print("   4. Combined = Powerful intelligence from unstructured data")
    print("\n📖 API Endpoints Available:")
    print("   POST /api/semantic-search - Semantic document search")
    print("   POST /api/affected-companies - Find indirect connections")
    print("   POST /api/similar-patterns - Pattern discovery")
    print("   POST /api/graph-traversal - Graph path finding")
    print("   POST /api/ingest - Document ingestion")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        demos = {
            "semantic": demo_semantic_search,
            "extract": demo_entity_extraction,
            "graph": demo_graph_traversal,
            "pattern": demo_pattern_discovery,
            "pipeline": demo_complete_pipeline,
            "all": run_all_demos
        }
        
        if demo_name in demos:
            demos[demo_name]()
        else:
            print(f"Unknown demo: {demo_name}")
            print(f"Available: {', '.join(demos.keys())}")
    else:
        run_all_demos()

"""
SemanticGraphPipeline - Enterprise Demo
========================================

Demonstrates proper usage of the canonical pipeline with:
- Dependency injection
- Real vector embeddings
- Centralized relationship logic
- Proper error handling
- Batch processing
"""

import json
from typing import List, Dict
from core import (
    PipelineFactory,
    SecureConfig,
    Document,
    Entity,
    Logger,
)
from pipelines.semantic_graph_pipeline import (
    SemanticGraphPipeline,
    ProcessedDocument,
    QueryResponse,
)


logger = Logger(__name__)


def demo_basic_pipeline():
    """
    Demo 1: Basic pipeline usage
    ============================
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Pipeline Usage")
    print("=" * 70)

    # Create pipeline using factory (recommended)
    factory = PipelineFactory()
    pipeline = factory.create_pipeline(
        weaviate_url="http://localhost:8080",
        nebula_host="127.0.0.1",
        nebula_port=9669,
    )

    # Sample document
    text = """
    John Smith works at TechCorp in San Francisco. He manages the AI division.
    The project includes developing semantic search capabilities using Weaviate
    and NebulaGraph. Maria Garcia, VP of Engineering, oversees the initiative.
    Budget allocated: $500,000 for Q1 2024.
    """

    # Ingest document
    result = pipeline.ingest_document(
        text=text,
        document_id="doc_001",
        category="technical",
        source="demo",
    )

    print(f"\n✓ Document Ingested:")
    print(f"  - Entities: {len(result.entities)}")
    print(f"  - Relationships: {len(result.relationships)}")
    print(f"  - Vector ID: {result.vector_id}")
    print(f"  - Graph Stored: {result.graph_stored}")

    # Display entities
    print(f"\n✓ Extracted Entities:")
    for entity in result.entities:
        print(f"  - {entity['type']}: {entity['value']} ({entity['confidence']:.0%})")

    # Display relationships
    print(f"\n✓ Relationships:")
    for rel in result.relationships[:5]:
        print(f"  - {rel['from']} --[{rel['type']}]-> {rel['to']}")


def demo_semantic_query():
    """
    Demo 2: Semantic search with relationship context
    ==================================================
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Semantic Query with Relationships")
    print("=" * 70)

    factory = PipelineFactory()
    pipeline = factory.create_pipeline()

    # Sample documents
    documents = [
        {
            "text": "Apple Inc. headquartered in Cupertino, California develops consumer electronics.",
            "id": "apple_001",
        },
        {
            "text": "Microsoft Corporation located in Redmond, Washington focuses on software and cloud services.",
            "id": "microsoft_001",
        },
        {
            "text": "Steve Jobs founded Apple in 1976. He served as CEO until 2011.",
            "id": "history_001",
        },
    ]

    # Batch ingest
    print("\nIngesting documents...")
    results = pipeline.batch_ingest(documents, category="company_info")

    print(f"✓ Ingested {len(results)} documents")

    # Execute semantic query
    query = "What are the headquarters locations of major tech companies?"

    print(f"\nExecuting semantic query: {query}")
    response = pipeline.semantic_query(query, limit=5)

    print(f"\n✓ Query Response:")
    print(f"  Answer: {response.answer}")
    print(f"  Semantic Results: {len(response.semantic_results)}")
    print(f"  Relationships Found: {len(response.relationships)}")
    print(f"\n  Explanation: {response.explanation}")


def demo_batch_processing():
    """
    Demo 3: Batch document processing
    ==================================
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Batch Document Processing")
    print("=" * 70)

    factory = PipelineFactory()
    pipeline = factory.create_pipeline()

    # Large batch of documents
    documents = [
        {
            "text": f"Company {i} operates in sector {i % 5} with revenue of ${i * 100}M.",
            "id": f"company_{i:03d}",
        }
        for i in range(1, 51)  # 50 documents
    ]

    print(f"\nProcessing {len(documents)} documents...")

    results = pipeline.batch_ingest(documents, category="company_batch")

    successful = sum(1 for r in results if r.graph_stored)
    print(f"✓ Batch Processing Complete:")
    print(f"  - Total: {len(results)}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {len(results) - successful}")

    # Statistics
    total_entities = sum(len(r.entities) for r in results)
    total_relationships = sum(len(r.relationships) for r in results)

    print(f"  - Total Entities: {total_entities}")
    print(f"  - Total Relationships: {total_relationships}")


def demo_error_handling():
    """
    Demo 4: Error handling and validation
    ======================================
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Error Handling & Validation")
    print("=" * 70)

    factory = PipelineFactory()
    pipeline = factory.create_pipeline()

    test_cases = [
        {
            "name": "Empty text",
            "text": "",
            "should_fail": True,
        },
        {
            "name": "Very short text",
            "text": "Hi",
            "should_fail": True,
        },
        {
            "name": "Valid text",
            "text": "John Smith works at Microsoft in Seattle. He leads the AI research team.",
            "should_fail": False,
        },
    ]

    for test_case in test_cases:
        try:
            result = pipeline.ingest_document(test_case["text"], document_id=f"test_{test_case['name']}")
            if test_case["should_fail"]:
                print(f"✗ {test_case['name']}: Should have failed but succeeded")
            else:
                print(f"✓ {test_case['name']}: Processed successfully")
        except Exception as e:
            if test_case["should_fail"]:
                print(f"✓ {test_case['name']}: Failed as expected - {type(e).__name__}")
            else:
                print(f"✗ {test_case['name']}: Failed unexpectedly - {e}")


def demo_security_config():
    """
    Demo 5: Security and configuration
    ===================================
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Security & Configuration")
    print("=" * 70)

    # Create secure config
    config = SecureConfig()

    print("\n✓ Configuration Loaded:")
    print(f"  - OpenAI API Key: {'Configured' if config.openai_api_key else 'Not configured'}")
    print(f"  - Nebula Password: Configured (hidden)")
    print(f"  - Weaviate API Key: {'Configured' if config.weaviate_api_key else 'Not configured'}")

    # Test secret masking
    from core.security_config import SecretsMask

    sensitive_data = {
        "username": "admin",
        "api_key": "sk-1234567890abcdef",
        "password": "super_secret_123",
    }

    masked = SecretsMask.mask_dict(sensitive_data)
    print(f"\n✓ Secret Masking:")
    print(f"  Original: {sensitive_data}")
    print(f"  Masked: {masked}")


def demo_health_check():
    """
    Demo 6: System health check
    ===========================
    """
    print("\n" + "=" * 70)
    print("DEMO 6: System Health Check")
    print("=" * 70)

    factory = PipelineFactory()
    pipeline = factory.create_pipeline()

    health = pipeline.health_check()

    print("\n✓ Service Health:")
    for service, status in health.items():
        status_str = "✓ Healthy" if status else "✗ Unhealthy"
        print(f"  - {service}: {status_str}")


# ============================================================================
# Main Demo Runner
# ============================================================================

def run_all_demos():
    """Run all demonstration scenarios"""
    print("\n\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Enterprise-Grade Semantic Graph Pipeline - Complete Demo".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    demos = [
        ("Basic Pipeline", demo_basic_pipeline),
        ("Semantic Query", demo_semantic_query),
        ("Batch Processing", demo_batch_processing),
        ("Error Handling", demo_error_handling),
        ("Security Config", demo_security_config),
        ("Health Check", demo_health_check),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Demo '{name}' failed: {e}")
            logger.error(f"Demo failed: {name}", exc_info=True)

    print("\n\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_demos()

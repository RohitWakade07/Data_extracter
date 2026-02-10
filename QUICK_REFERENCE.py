"""
QUICK REFERENCE - Semantic Graph Pipeline
===========================================

Copy-paste ready code snippets for common tasks.


1. BASIC SETUP
==============

from core import PipelineFactory

factory = PipelineFactory()
pipeline = factory.create_pipeline()
"""

# 2. INGEST A SINGLE DOCUMENT
# ============================

def example_single_document():
    from core import PipelineFactory
    
    factory = PipelineFactory()
    pipeline = factory.create_pipeline()
    
    result = pipeline.ingest_document(
        text="John Smith works at Microsoft in Seattle.",
        document_id="doc_001",
        category="company_info"
    )
    
    print(f"✓ Entities: {len(result.entities)}")
    print(f"✓ Relationships: {len(result.relationships)}")
    print(f"✓ Vector ID: {result.vector_id}")


# 3. BATCH INGEST DOCUMENTS
# ==========================

def example_batch_ingestion():
    from core import PipelineFactory
    
    factory = PipelineFactory()
    pipeline = factory.create_pipeline()
    
    documents = [
        {"text": "Document 1 content", "id": "doc_1"},
        {"text": "Document 2 content", "id": "doc_2"},
        {"text": "Document 3 content", "id": "doc_3"},
    ]
    
    results = pipeline.batch_ingest(documents, category="batch")
    
    print(f"✓ Processed: {len(results)} documents")
    for r in results:
        if r.graph_stored:
            print(f"  ✓ {r.document_id}: {len(r.entities)} entities")


# 4. SEMANTIC SEARCH
# ==================

def example_semantic_query():
    from core import PipelineFactory
    
    factory = PipelineFactory()
    pipeline = factory.create_pipeline()
    
    response = pipeline.semantic_query(
        query="What are the key companies mentioned?",
        limit=10
    )
    
    print(f"Answer: {response.answer}")
    print(f"Results: {len(response.semantic_results)}")
    print(f"Relationships: {response.relationships}")


# 5. CUSTOM CONFIGURATION
# ========================

def example_custom_config():
    from core import SecureConfig, PipelineFactory
    from core import WeaviateVectorStore, NebulaGraphStore
    
    # Load configuration
    config = SecureConfig()
    
    # Create custom services
    vector_store = WeaviateVectorStore(
        weaviate_url="http://custom-weaviate:8080"
    )
    
    graph_store = NebulaGraphStore(
        host="custom-nebula-host",
        port=9669
    )
    
    # Create pipeline with custom services
    factory = PipelineFactory(config)
    pipeline = factory.create_pipeline(
        vector_store=vector_store,
        graph_store=graph_store
    )


# 6. ERROR HANDLING
# =================

def example_error_handling():
    from core import PipelineFactory
    from core.validation_and_errors import (
        ValidationException,
        StorageException,
        ExtractionException,
    )
    
    factory = PipelineFactory()
    pipeline = factory.create_pipeline()
    
    try:
        result = pipeline.ingest_document(
            text="Your text here",
            document_id="doc_001"
        )
    except ValidationException as e:
        print(f"Validation error: {e}")
    except StorageException as e:
        print(f"Storage error (will retry): {e}")
    except ExtractionException as e:
        print(f"Extraction error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# 7. ENTITY EXTRACTION ONLY
# ==========================

def example_entity_extraction():
    from core import CentralRelationshipMapper
    from core.service_interfaces import Entity as CoreEntity
    from entity_extraction.entity_extractor import extract_from_text
    
    # Extract entities
    extraction = extract_from_text(
        "John Smith works at Microsoft",
        provider="openai"
    )
    
    # Convert extracted entities to core entity format
    entities = [
        CoreEntity(
            type=e.type,
            value=e.value,
            confidence=e.confidence
        )
        for e in extraction.entities
    ]
    
    # Map relationships
    mapper = CentralRelationshipMapper()
    relationships = mapper.map_relationships(
        entities=entities,
        text="John Smith works at Microsoft"
    )
    
    print(f"Entities: {[e.value for e in entities]}")
    print(f"Relationships: {[(r.source, r.relationship_type, r.target) for r in relationships]}")


# 8. VECTOR SEARCH ONLY
# ======================

def example_vector_search():
    from core import WeaviateVectorStore, HybridEmbedding
    
    # Create vector store
    embedding_service = HybridEmbedding()
    vector_store = WeaviateVectorStore(
        embedding_service=embedding_service
    )
    vector_store.initialize()
    
    # Search
    results = vector_store.search(
        query="semantic search query",
        limit=10
    )
    
    for result in results:
        print(f"Score: {result.score:.2%}")
        print(f"Content: {result.content[:100]}...")


# 9. HEALTH CHECK
# ===============

def example_health_check():
    from core import PipelineFactory
    
    factory = PipelineFactory()
    pipeline = factory.create_pipeline()
    
    health = pipeline.health_check()
    
    for service, status in health.items():
        status_str = "✓" if status else "✗"
        print(f"{status_str} {service}")


# 10. DATA VALIDATION
# ===================

def example_validation():
    from core import DataValidator, Document, Entity
    
    validator = DataValidator()
    
    # Validate entities
    entities = [
        Entity(type="person", value="John Smith", confidence=0.95),
        Entity(type="organization", value="Microsoft", confidence=0.98),
    ]
    
    try:
        validator.validate_entities(entities)
        print("✓ Entities valid")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    
    # Validate document
    doc = Document(
        id="doc_001",
        content="Document content here",
        entities=[],
        metadata={"category": "test"}
    )
    
    try:
        validator.validate_document(doc)
        print("✓ Document valid")
    except Exception as e:
        print(f"✗ Validation error: {e}")


# 11. LOGGING & MONITORING
# ========================

def example_logging():
    from core import Logger, log_performance
    
    logger = Logger("MyModule")
    
    # Basic logging
    logger.info("Processing started", extra_field="value")
    logger.debug("Debug information")
    logger.warning("Something might be wrong")
    logger.error("An error occurred", exc_info=True)
    
    # Performance tracking
    @log_performance
    def my_function():
        import time
        time.sleep(0.1)
        return "result"
    
    result = my_function()  # Logged automatically


# 12. SECURITY & SECRETS
# ======================

def example_security():
    from core import SecureConfig, SecretsMask
    
    # Load configuration securely
    config = SecureConfig(env_file=".env")
    
    # Access secrets safely
    api_key = config.openai_api_key
    password = config.nebula_password
    
    # Mask sensitive data in logs
    sensitive_data = {
        "username": "admin",
        "api_key": "sk-123456789",
        "password": "secret123"
    }
    
    masked = SecretsMask.mask_dict(sensitive_data)
    print(masked)  # {"username": "admin", "api_key": "***MASKED***", ...}


# 13. TEXT PROCESSING
# ===================

def example_text_processing():
    from core import TextProcessor
    
    processor = TextProcessor(
        default_chunk_size=512,
        default_overlap=50
    )
    
    text = "Your long document text here..."
    
    # Chunk text
    chunks = processor.chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Normalize entity names
    normalized = processor.normalize_entity_name("Tech Mahindra Ltd.")
    print(f"Normalized: {normalized}")


# 14. RELATIONSHIP MAPPING
# ========================

def example_relationship_mapping():
    from core import CentralRelationshipMapper, Entity
    
    mapper = CentralRelationshipMapper()
    
    # Create entities
    entities = [
        Entity(type="person", value="Alice Johnson", confidence=0.95),
        Entity(type="organization", value="Google", confidence=0.98),
        Entity(type="location", value="Mountain View", confidence=0.99),
    ]
    
    text = "Alice Johnson works at Google in Mountain View"
    
    # Map relationships
    relationships = mapper.map_relationships(entities, text)
    
    for rel in relationships:
        print(f"{rel.source} --[{rel.relationship_type}]--> {rel.target}")


# 15. FACTORY PATTERN
# ===================

def example_factory_pattern():
    from core import PipelineFactory, WeaviateVectorStore, NebulaGraphStore
    
    # Factory creates everything
    factory = PipelineFactory()
    
    # Create pipeline
    pipeline = factory.create_pipeline()
    
    # Or create with custom services
    custom_vector = WeaviateVectorStore()
    custom_graph = NebulaGraphStore()
    
    pipeline = factory.create_pipeline(
        vector_store=custom_vector,
        graph_store=custom_graph
    )


"""
═══════════════════════════════════════════════════════════════════════════════

COMMON PATTERNS
===============

Pattern: Process + Query
───────────────────────
factory = PipelineFactory()
pipeline = factory.create_pipeline()
results = pipeline.batch_ingest(docs)
response = pipeline.semantic_query("Find companies")

Pattern: Error Handling
──────────────────────
try:
    result = pipeline.ingest_document(text)
except ValidationException:
    handle_validation_error()
except StorageException:
    retry_storage()

Pattern: Custom Services
────────────────────────
factory = PipelineFactory()
custom_store = MyCustomVectorStore()
pipeline = factory.create_pipeline(vector_store=custom_store)

Pattern: Batch Processing
─────────────────────────
docs = []
results = pipeline.batch_ingest(docs)
for r in results:
    if r.graph_stored:
        process(r)


═══════════════════════════════════════════════════════════════════════════════

ENVIRONMENT SETUP (.env)
========================

OPENAI_API_KEY=sk-...
WEAVIATE_API_KEY=your-api-key
NEBULA_PASSWORD=nebula_password


═══════════════════════════════════════════════════════════════════════════════

MOST COMMON OPERATIONS
======================

1. Process one document
   → pipeline.ingest_document(text)

2. Process many documents
   → pipeline.batch_ingest([docs])

3. Search for information
   → pipeline.semantic_query("query")

4. Check system health
   → pipeline.health_check()

5. Create custom pipeline
   → factory.create_pipeline(vector_store=custom)


═══════════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING
===============

Issue: "Weaviate not ready"
Solution: Ensure Weaviate is running
$ docker-compose up weaviate

Issue: "OpenAI API key not found"
Solution: Set environment variable
$ export OPENAI_API_KEY="sk-..."

Issue: "Nebula connection failed"
Solution: Ensure Nebula is running
$ docker-compose up nebula

Issue: "Low confidence scores"
Solution: Use more descriptive text


═══════════════════════════════════════════════════════════════════════════════

For more info, see:
- IMPLEMENTATION_GUIDE.py
- ENTERPRISE_REFACTORING.md
- demo_enterprise_pipeline.py

"""

if __name__ == "__main__":
    print(__doc__)
    print("\nAll examples are ready to use!")
    print("Copy-paste any example function into your code.")

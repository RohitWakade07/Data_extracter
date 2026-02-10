"""
Enterprise-Grade Semantic Graph Pipeline
=========================================

Canonical orchestration layer for:
1. Entity Extraction
2. Vector Semantic Search (with real embeddings)
3. Knowledge Graph Storage
4. Explainable Relationship Mapping
5. Hybrid Query Answering

Uses dependency injection for loose coupling and testability.
"""

import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from core.service_interfaces import (
    VectorStoreInterface,
    GraphStoreInterface,
    EntityExtractorInterface,
    RelationshipMapperInterface,
    Document,
    Entity,
    Relationship,
    SearchResult,
)
from core.logging_config import Logger, log_performance
from core.validation_and_errors import (
    DataValidator,
    ValidationException,
    StorageException,
    ExtractionException,
    with_retry,
)


logger = Logger(__name__)


@dataclass
class ProcessedDocument:
    """Result of document processing"""
    document_id: str
    summary: str
    entities: List[Entity]
    relationships: List[Relationship]
    vector_id: Optional[str] = None
    graph_stored: bool = False
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ingestion_version: int = 1


@dataclass
class QueryResponse:
    """Structured query response"""
    query: str
    answer: str
    semantic_results: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    explanation: str
    query_version: int = 1


class SemanticGraphPipeline:
    """
    Canonical enterprise pipeline orchestrator.
    
    All other pipelines should wrap or extend this class.
    Uses dependency injection for all services.
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        graph_store: GraphStoreInterface,
        relationship_mapper: RelationshipMapperInterface,
        entity_extractor: Optional[EntityExtractorInterface] = None,
        ingestion_version: int = 1,
    ):
        """
        Initialize pipeline with injected services.

        Args:
            vector_store: Service for vector database operations
            graph_store: Service for graph database operations
            relationship_mapper: Service for relationship extraction
            entity_extractor: Optional service for entity extraction
            ingestion_version: Track schema/logic version
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.relationship_mapper = relationship_mapper
        self.entity_extractor = entity_extractor
        self.ingestion_version = ingestion_version
        self.validator = DataValidator()

        self._initialize()

    def _initialize(self):
        """Initialize all services"""
        try:
            if not self.vector_store.initialize():
                logger.error("Vector store initialization failed")
                raise RuntimeError("Vector store not ready")

            if not self.graph_store.initialize():
                logger.error("Graph store initialization failed")
                raise RuntimeError("Graph store not ready")

            logger.info("✓ Pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            raise

    @log_performance
    @with_retry(max_attempts=3)
    def ingest_document(
        self,
        text: str,
        document_id: Optional[str] = None,
        category: str = "general",
        **metadata,
    ) -> ProcessedDocument:
        """
        Ingest a document end-to-end.

        Args:
            text: Document content
            document_id: Optional custom ID
            category: Document category
            **metadata: Additional metadata

        Returns:
            ProcessedDocument with extraction and storage results

        Raises:
            ExtractionException: If entity extraction fails
            StorageException: If storage fails
        """
        document_id = document_id or str(uuid.uuid4())

        try:
            logger.info(f"Ingesting document: {document_id}")

            # ----------- PHASE 1: Validation -----------
            sanitized_text = DataValidator.sanitize_text(text)
            document = Document(
                id=document_id,
                content=sanitized_text,
                entities=[],
                metadata={
                    "category": category,
                    "source_type": metadata.get("source_type", "document"),
                    "ingestion_version": self.ingestion_version,
                    **metadata,
                },
            )

            try:
                self.validator.validate_document(document)
            except ValidationException as e:
                raise ExtractionException(f"Document validation failed: {e}")

            # ----------- PHASE 2: Entity Extraction -----------
            entities = self._extract_entities(sanitized_text)
            document.entities = [
                {
                    "type": e.type,
                    "value": e.value,
                    "confidence": e.confidence,
                }
                for e in entities
            ]

            logger.info(f"Extracted {len(entities)} entities")

            # Validate entities
            try:
                self.validator.validate_entities(entities)
            except ValidationException as e:
                logger.warning(f"Entity validation warning: {e}")

            # ----------- PHASE 3: Relationship Mapping -----------
            relationships = self.relationship_mapper.map_relationships(
                entities, sanitized_text
            )

            logger.info(f"Mapped {len(relationships)} relationships")

            # Validate relationships
            try:
                self.validator.validate_relationships(relationships)
            except ValidationException as e:
                logger.warning(f"Relationship validation warning: {e}")

            # ----------- PHASE 4: Vector Storage -----------
            vector_id = self._store_vector(document)

            # ----------- PHASE 5: Graph Storage -----------
            graph_stored = self._store_graph(entities, relationships)

            # ----------- PHASE 6: Summary Generation -----------
            summary = self._generate_summary(sanitized_text, entities)

            logger.info(
                f"Document ingestion complete: {len(entities)} entities, "
                f"{len(relationships)} relationships, vector_id={vector_id}"
            )

            return ProcessedDocument(
                document_id=document_id,
                summary=summary,
                entities=entities,
                relationships=relationships,
                vector_id=vector_id,
                graph_stored=graph_stored,
                ingestion_version=self.ingestion_version,
            )

        except ExtractionException:
            raise
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)
            raise ExtractionException(f"Ingestion failed: {e}")

    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        if not self.entity_extractor:
            logger.warning("Entity extractor not configured, returning empty list")
            return []

        try:
            entities = self.entity_extractor.extract(text)
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            raise ExtractionException(f"Entity extraction failed: {e}")

    @with_retry(max_attempts=3, exceptions=(StorageException,))
    def _store_vector(self, document: Document) -> Optional[str]:
        """Store document in vector store"""
        try:
            vector_id = self.vector_store.store_document(
                document, generate_embeddings=True
            )

            if not vector_id:
                raise StorageException("Vector store returned no ID")

            return vector_id

        except Exception as e:
            logger.error(f"Vector storage failed: {e}", exc_info=True)
            raise StorageException(f"Vector storage failed: {e}")

    @with_retry(max_attempts=3, exceptions=(StorageException,))
    def _store_graph(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> bool:
        """Store entities and relationships in graph"""
        try:
            # Convert entities to dict format
            entity_dicts = [
                {"type": e.type, "value": e.value, "confidence": e.confidence}
                for e in entities
            ]

            result = self.graph_store.store_graph(entity_dicts, relationships)

            success = result.get("entities_added", 0) > 0 or result.get("relationships_added", 0) > 0

            logger.debug(
                f"Graph storage: {result.get('entities_added', 0)} entities, "
                f"{result.get('relationships_added', 0)} relationships"
            )

            return success

        except Exception as e:
            logger.error(f"Graph storage failed: {e}", exc_info=True)
            raise StorageException(f"Graph storage failed: {e}")

    @staticmethod
    def _generate_summary(text: str, entities: List[Entity]) -> str:
        """Generate document summary"""
        snippet = text[:200]
        entity_names = [e.value for e in entities[:5]]
        entities_str = ", ".join(entity_names) if entity_names else "none"
        return f"{snippet}...\nKey entities: {entities_str}"

    @log_performance
    def semantic_query(self, query: str, limit: int = 10) -> QueryResponse:
        """
        Execute semantic query with relationship context.

        Args:
            query: Natural language query
            limit: Maximum results

        Returns:
            QueryResponse with semantic and relationship context
        """
        try:
            logger.info(f"Executing query: {query}")

            # ----------- PHASE 1: Vector Semantic Search -----------
            semantic_results = self.vector_store.search(query, limit=limit)

            logger.debug(f"Semantic search returned {len(semantic_results)} results")

            # ----------- PHASE 2: Extract Entities from Results -----------
            all_entities = []
            for result in semantic_results:
                all_entities.extend(result.entities)

            # Deduplicate entities
            unique_entities = {}
            for entity in all_entities:
                key = (entity.type.lower(), entity.value.lower())
                if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                    unique_entities[key] = entity

            entity_list = list(unique_entities.values())

            logger.debug(f"Found {len(entity_list)} unique entities")

            # ----------- PHASE 3: Graph Context -----------
            graph_relationships = []

            for entity in entity_list:
                try:
                    rels = self.graph_store.query_relationships(
                        entity.value, entity_type=entity.type
                    )
                    graph_relationships.extend(rels)
                except Exception as e:
                    logger.debug(f"Graph query failed for {entity.value}: {e}")

            logger.debug(f"Found {len(graph_relationships)} graph relationships")

            # ----------- PHASE 4: Generate Relationship Answer -----------
            answer_package = self.relationship_mapper.generate_relationship_answer(
                query=query,
                entities=[e.__dict__ for e in entity_list],
                text=" ".join([r.content for r in semantic_results]),
            )

            # ----------- PHASE 5: Build Final Response -----------
            semantic_result_dicts = [
                {
                    "content": r.content[:250],
                    "score": r.score,
                    "document_id": r.id,
                }
                for r in semantic_results
            ]

            relationship_dicts = [
                {
                    "from": r.source,
                    "to": r.target,
                    "type": r.relationship_type,
                    "confidence": f"{r.confidence:.0%}",
                }
                for r in graph_relationships[:10]
            ]

            response = QueryResponse(
                query=query,
                answer=answer_package.direct_answer,
                semantic_results=semantic_result_dicts,
                relationships=relationship_dicts,
                explanation=answer_package.relationship_explanation,
                query_version=self.ingestion_version,
            )

            logger.info(f"Query complete: {len(semantic_result_dicts)} documents, "
                       f"{len(relationship_dicts)} relationships")

            return response

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return QueryResponse(
                query=query,
                answer="Query execution failed",
                semantic_results=[],
                relationships=[],
                explanation=str(e),
            )

    @log_performance
    def batch_ingest(
        self,
        documents: List[Dict[str, str]],
        category: str = "general",
    ) -> List[ProcessedDocument]:
        """
        Batch ingest multiple documents.

        Args:
            documents: List of {'text': ..., 'id': ...} dicts
            category: Category for all documents

        Returns:
            List of ProcessedDocument results
        """
        results = []

        logger.info(f"Starting batch ingestion of {len(documents)} documents")

        for i, doc in enumerate(documents):
            try:
                text = doc.get("text", "")
                doc_id = doc.get("id", f"batch_{i}")

                result = self.ingest_document(
                    text, document_id=doc_id, category=category
                )
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Batch progress: {i + 1}/{len(documents)}")

            except Exception as e:
                logger.error(f"Failed to ingest document {i}: {e}")
                results.append(
                    ProcessedDocument(
                        document_id=doc.get("id", f"batch_{i}"),
                        summary="Failed to process",
                        entities=[],
                        relationships=[],
                        graph_stored=False,
                    )
                )

        logger.info(f"Batch ingestion complete: {len(results)} documents processed")
        return results

    def health_check(self) -> Dict[str, bool]:
        """Check health of all services"""
        return {
            "vector_store": self.vector_store.initialize() if hasattr(self.vector_store, 'initialized') else True,
            "graph_store": self.graph_store.initialize() if hasattr(self.graph_store, 'initialized') else True,
            "relationship_mapper": True,
        }

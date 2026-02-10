"""
Pipeline Factory
================

Factory pattern for creating properly configured pipeline instances.
Handles service composition and dependency injection.
"""

from typing import Optional, Dict, Any
from core.service_interfaces import (
    VectorStoreInterface,
    GraphStoreInterface,
    EntityExtractorInterface,
    RelationshipMapperInterface,
)
from core.embedding_service import HybridEmbedding
from core.vector_store import WeaviateVectorStore
from core.graph_store import NebulaGraphStore
from core.relationship_mapper import CentralRelationshipMapper
from core.security_config import SecureConfig
from core.logging_config import Logger
from pipelines.semantic_graph_pipeline import SemanticGraphPipeline


logger = Logger(__name__)


class PipelineFactory:
    """
    Factory for creating properly configured pipeline instances.
    Handles service composition, configuration, and initialization.
    """

    def __init__(self, config: Optional[SecureConfig] = None):
        """
        Initialize factory with optional config.

        Args:
            config: Optional SecureConfig instance. If not provided,
                   creates default configuration.
        """
        self.config = config or SecureConfig()
        self.logger = Logger(__name__)

    def create_pipeline(
        self,
        vector_store: Optional[VectorStoreInterface] = None,
        graph_store: Optional[GraphStoreInterface] = None,
        relationship_mapper: Optional[RelationshipMapperInterface] = None,
        entity_extractor: Optional[EntityExtractorInterface] = None,
        ingestion_version: int = 1,
        **kwargs,
    ) -> SemanticGraphPipeline:
        """
        Create a fully configured pipeline.

        Args:
            vector_store: Vector store implementation
            graph_store: Graph store implementation
            relationship_mapper: Relationship mapper implementation
            entity_extractor: Entity extractor implementation
            ingestion_version: Schema/logic version
            **kwargs: Additional configuration

        Returns:
            Configured SemanticGraphPipeline instance
        """
        try:
            # Use provided implementations or create defaults
            vector_store = vector_store or self._create_vector_store(**kwargs)
            graph_store = graph_store or self._create_graph_store(**kwargs)
            relationship_mapper = relationship_mapper or self._create_relationship_mapper()

            # Create and initialize pipeline
            pipeline = SemanticGraphPipeline(
                vector_store=vector_store,
                graph_store=graph_store,
                relationship_mapper=relationship_mapper,
                entity_extractor=entity_extractor,
                ingestion_version=ingestion_version,
            )

            self.logger.info("✓ Pipeline created successfully")
            return pipeline

        except Exception as e:
            self.logger.error(f"Pipeline creation failed: {e}", exc_info=True)
            raise

    def _create_vector_store(
        self,
        weaviate_url: str = "http://localhost:8080",
        **kwargs,
    ) -> WeaviateVectorStore:
        """Create vector store with default settings"""
        try:
            embedding_service = HybridEmbedding()

            store = WeaviateVectorStore(
                weaviate_url=weaviate_url,
                api_key=self.config.weaviate_api_key,
                embedding_service=embedding_service,
            )

            self.logger.info("✓ Vector store created")
            return store

        except Exception as e:
            self.logger.error(f"Vector store creation failed: {e}", exc_info=True)
            raise

    def _create_graph_store(
        self,
        nebula_host: str = "127.0.0.1",
        nebula_port: int = 9669,
        nebula_user: str = "root",
        **kwargs,
    ) -> NebulaGraphStore:
        """Create graph store with default settings"""
        try:
            store = NebulaGraphStore(
                host=nebula_host,
                port=nebula_port,
                user=nebula_user,
                password=self.config.nebula_password,
            )

            self.logger.info("✓ Graph store created")
            return store

        except Exception as e:
            self.logger.error(f"Graph store creation failed: {e}", exc_info=True)
            raise

    def _create_relationship_mapper(
        self,
    ) -> RelationshipMapperInterface:
        """Create relationship mapper"""
        try:
            mapper = CentralRelationshipMapper()
            self.logger.info("✓ Relationship mapper created")
            return mapper
        except Exception as e:
            self.logger.error(f"Relationship mapper creation failed: {e}", exc_info=True)
            raise

    @staticmethod
    def create_simple_processing_pipeline(
        weaviate_url: str = "http://localhost:8080",
        nebula_host: str = "127.0.0.1",
        nebula_port: int = 9669,
    ) -> SemanticGraphPipeline:
        """
        Create a simple processing pipeline with sensible defaults.
        Useful for quick setup and testing.
        """
        factory = PipelineFactory()
        return factory.create_pipeline(
            weaviate_url=weaviate_url,
            nebula_host=nebula_host,
            nebula_port=nebula_port,
        )

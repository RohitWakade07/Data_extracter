"""Core module - service interfaces and configurations"""

from .service_interfaces import (
    VectorStoreInterface,
    GraphStoreInterface,
    EntityExtractorInterface,
    RelationshipMapperInterface,
    EmbeddingInterface,
    TextProcessingInterface,
    ValidationServiceInterface,
    Document,
    Entity,
    Relationship,
    SearchResult,
)
from .logging_config import Logger, setup_logger, log_performance
from .embedding_service import HybridEmbedding, OpenAIEmbedding, LocalTransformerEmbedding
from .vector_store import WeaviateVectorStore
from .graph_store import NebulaGraphStore
from .relationship_mapper import CentralRelationshipMapper, RelationshipAnswer
from .text_processor import TextProcessor
from .validation_and_errors import (
    DataValidator,
    RetryStrategy,
    CircuitBreaker,
    ValidationException,
    StorageException,
    ExtractionException,
)
from .security_config import SecureConfig, Credentials, SecretsMask
from .pipeline_factory import PipelineFactory

__all__ = [
    # Interfaces
    "VectorStoreInterface",
    "GraphStoreInterface",
    "EntityExtractorInterface",
    "RelationshipMapperInterface",
    "EmbeddingInterface",
    "TextProcessingInterface",
    "ValidationServiceInterface",
    # Data models
    "Document",
    "Entity",
    "Relationship",
    "SearchResult",
    # Logging
    "Logger",
    "setup_logger",
    "log_performance",
    # Embeddings
    "HybridEmbedding",
    "OpenAIEmbedding",
    "LocalTransformerEmbedding",
    # Implementations
    "WeaviateVectorStore",
    "NebulaGraphStore",
    "CentralRelationshipMapper",
    "RelationshipAnswer",
    "TextProcessor",
    # Validation & Errors
    "DataValidator",
    "RetryStrategy",
    "CircuitBreaker",
    "ValidationException",
    "StorageException",
    "ExtractionException",
    # Security & Config
    "SecureConfig",
    "Credentials",
    "SecretsMask",
    # Factory
    "PipelineFactory",
]

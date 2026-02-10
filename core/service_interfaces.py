"""
Enterprise Service Interfaces
=============================

Defines the core abstractions for dependency injection.
All services implement these interfaces for loose coupling.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Unified document representation"""
    id: str
    content: str
    entities: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    vector_id: Optional[str] = None


@dataclass
class Entity:
    """Unified entity representation"""
    type: str
    value: str
    confidence: float
    position: Optional[int] = None


@dataclass
class Relationship:
    """Unified relationship representation"""
    source: str
    source_type: str
    target: str
    target_type: str
    relationship_type: str
    confidence: float
    context: Optional[str] = None


@dataclass
class SearchResult:
    """Unified search result"""
    id: str
    content: str
    score: float
    entities: List[Entity]
    metadata: Dict[str, Any]


class VectorStoreInterface(ABC):
    """
    Abstract interface for vector database operations.
    Implementations: WeaviateVectorStore, PineconeVectorStore, etc.
    """

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize connection and create schema if needed"""
        pass

    @abstractmethod
    def store_document(
        self,
        document: Document,
        generate_embeddings: bool = True
    ) -> Optional[str]:
        """Store document and return vector ID"""
        pass

    @abstractmethod
    def store_batch(
        self,
        documents: List[Document],
        generate_embeddings: bool = True
    ) -> Dict[str, str]:
        """Batch store documents. Returns mapping of doc_id to vector_id"""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Semantic search with optional filtering"""
        pass

    @abstractmethod
    def get_document(self, vector_id: str) -> Optional[Document]:
        """Retrieve document by vector ID"""
        pass

    @abstractmethod
    def delete_document(self, vector_id: str) -> bool:
        """Delete document from vector store"""
        pass


class GraphStoreInterface(ABC):
    """
    Abstract interface for graph database operations.
    Implementations: NebulaGraphStore, Neo4jGraphStore, etc.
    """

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize connection and create schema"""
        pass

    @abstractmethod
    def store_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Store entities. Returns mapping of type to count"""
        pass

    @abstractmethod
    def store_relationships(
        self,
        relationships: List[Relationship]
    ) -> int:
        """Store relationships. Returns count stored"""
        pass

    @abstractmethod
    def store_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Relationship]
    ) -> Dict[str, Any]:
        """Store both entities and relationships atomically"""
        pass

    @abstractmethod
    def query_relationships(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None
    ) -> List[Relationship]:
        """Query relationships for an entity"""
        pass

    @abstractmethod
    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 3
    ) -> List[List[Relationship]]:
        """Find paths between entities"""
        pass


class EntityExtractorInterface(ABC):
    """
    Abstract interface for entity extraction.
    Implementations: LLMExtractor, RuleBasedExtractor, HybridExtractor
    """

    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        pass


class RelationshipMapperInterface(ABC):
    """
    Abstract interface for relationship extraction.
    Central authority for all relationship logic.
    """

    @abstractmethod
    def map_relationships(
        self,
        entities: List[Entity],
        text: str
    ) -> List[Relationship]:
        """Map relationships between entities"""
        pass

    @abstractmethod
    def validate_relationship(
        self,
        relationship: Relationship,
        context: str
    ) -> float:
        """Validate relationship confidence in context"""
        pass

    @abstractmethod
    def generate_relationship_answer(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        text: str,
    ) -> Any:
        """
        Generate structured answer with relationship mappings and explanations.
        
        Returns:
            RelationshipAnswer or similar structured response
        """
        pass


class EmbeddingInterface(ABC):
    """
    Abstract interface for text embeddings.
    Implementations: OpenAIEmbedding, LocalTransformerEmbedding, etc.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass


class TextProcessingInterface(ABC):
    """
    Abstract interface for text preprocessing.
    Handles chunking, normalization, etc.
    """

    @abstractmethod
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """Split text into semantic chunks"""
        pass

    @abstractmethod
    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity names for deduplication"""
        pass


class ValidationServiceInterface(ABC):
    """
    Abstract interface for data validation.
    """

    @abstractmethod
    def validate_document(self, document: Document) -> bool:
        """Validate document structure and content"""
        pass

    @abstractmethod
    def validate_entities(self, entities: List[Entity]) -> bool:
        """Validate entity list"""
        pass

    @abstractmethod
    def validate_relationships(self, relationships: List[Relationship]) -> bool:
        """Validate relationship list"""
        pass

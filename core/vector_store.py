"""
Unified Weaviate Vector Store
==============================

Implements VectorStoreInterface with:
- Real semantic embeddings
- Unified schema (no duplication)
- Batch operations
- Error resilience
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from core.service_interfaces import (
    VectorStoreInterface,
    EmbeddingInterface,
    Document,
    SearchResult,
    Entity,
)
from core.logging_config import Logger, log_performance


logger = Logger(__name__)


class WeaviateVectorStore(VectorStoreInterface):
    """
    Modern Weaviate integration with:
    - Real embeddings (text2vec-openai)
    - Unified document schema
    - Batch operations
    - Proper error handling
    """

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        embedding_service: Optional[EmbeddingInterface] = None,
        class_name: str = "UnifiedDocument"
    ):
        self.url = weaviate_url.rstrip("/")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")
        self.class_name = class_name
        self._embedding_service = embedding_service
        self.client: Any = None
        self.initialized = False
        self._is_v4 = False

    @property
    def embedding_service(self) -> EmbeddingInterface:
        """Lazy load embedding service"""
        if self._embedding_service is None:
            from core.embedding_service import HybridEmbedding
            self._embedding_service = HybridEmbedding()
        return self._embedding_service

    @log_performance
    def initialize(self) -> bool:
        """Initialize Weaviate connection and create schema"""
        try:
            import weaviate

            # Parse host from URL
            host = self.url.replace("http://", "").replace("https://", "")
            if ":" in host:
                host = host.split(":")[0]

            # Try v4 client first, fallback to v3
            try:
                self.client = weaviate.connect_to_local(
                    host=host,
                    port=8080,
                )
            except AttributeError:
                # Fallback for older weaviate-client versions (v3 API)
                self.client = weaviate.Client(self.url)  # type: ignore[arg-type]

            # Check readiness
            is_ready = False
            if hasattr(self.client, 'is_ready'):
                is_ready = self.client.is_ready()
            elif hasattr(self.client, 'is_live'):
                is_ready = self.client.is_live()
            else:
                is_ready = True  # Assume ready if we got here

            if is_ready:
                logger.info("✓ Weaviate connection established")
                self.initialized = True
                self._create_unified_schema()
                return True
            else:
                logger.error("Weaviate not ready")
                return False

        except ImportError:
            logger.error("weaviate-client package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            return False

    def _create_unified_schema(self) -> bool:
        """Create or update the unified document schema"""
        if not self.client:
            return False

        try:
            # Try v4 API first
            if hasattr(self.client, 'collections'):
                return self._create_schema_v4()
            else:
                # Fallback to v3 API
                return self._create_schema_v3()

        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            return False

    def _create_schema_v4(self) -> bool:
        """Create schema using Weaviate v4 API"""
        try:
            from weaviate.classes.config import Configure, Property, DataType

            if self.client.collections.exists(self.class_name):
                logger.info(f"✓ Schema '{self.class_name}' already exists")
                return True

            properties = [
                Property(name="content", data_type=DataType.TEXT),
                Property(name="summary", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="source_type", data_type=DataType.TEXT),
                Property(name="entities_json", data_type=DataType.TEXT),
                Property(name="metadata_json", data_type=DataType.TEXT),
                Property(name="ingestion_version", data_type=DataType.INT),
                Property(name="confidence_score", data_type=DataType.NUMBER),
            ]

            # Create without vectorizer (we provide our own embeddings)
            self.client.collections.create(
                name=self.class_name,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),
            )

            logger.info(f"✓ Created unified schema '{self.class_name}'")
            return True

        except Exception as e:
            logger.error(f"Schema v4 creation failed: {e}")
            return False

    def _create_schema_v3(self) -> bool:
        """Create schema using Weaviate v3 API"""
        try:
            # Check if class exists
            schema = self.client.schema.get()
            class_names = [c["class"] for c in schema.get("classes", [])]

            if self.class_name in class_names:
                logger.info(f"✓ Schema '{self.class_name}' already exists")
                return True

            # Create class schema
            class_schema = {
                "class": self.class_name,
                "vectorizer": "none",  # We provide our own embeddings
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "summary", "dataType": ["text"]},
                    {"name": "document_id", "dataType": ["text"]},
                    {"name": "category", "dataType": ["text"]},
                    {"name": "source_type", "dataType": ["text"]},
                    {"name": "entities_json", "dataType": ["text"]},
                    {"name": "metadata_json", "dataType": ["text"]},
                    {"name": "ingestion_version", "dataType": ["int"]},
                    {"name": "confidence_score", "dataType": ["number"]},
                ],
            }

            self.client.schema.create_class(class_schema)
            logger.info(f"✓ Created unified schema '{self.class_name}'")
            return True

        except Exception as e:
            logger.error(f"Schema v3 creation failed: {e}")
            return False

    @log_performance
    def store_document(
        self,
        document: Document,
        generate_embeddings: bool = True
    ) -> Optional[str]:
        """Store single document with real embeddings"""
        if not self.initialized:
            logger.error("Vector store not initialized")
            return None

        try:
            # Prepare object
            obj = {
                "content": document.content,
                "document_id": document.id,
                "category": document.metadata.get("category", "general"),
                "source_type": document.metadata.get("source_type", "document"),
                "entities_json": json.dumps(document.entities),
                "metadata_json": json.dumps(document.metadata),
                "ingestion_version": document.metadata.get("ingestion_version", 1),
                "confidence_score": document.metadata.get("confidence", 1.0),
            }

            # Generate embedding if needed
            vector = None
            if generate_embeddings:
                try:
                    vector = self.embedding_service.embed_text(document.content)
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}")

            # Store in Weaviate (v4 or v3)
            if hasattr(self.client, 'collections'):
                return self._store_document_v4(obj, vector, document)
            else:
                return self._store_document_v3(obj, vector, document)

        except Exception as e:
            logger.error(f"Failed to store document: {e}", exc_info=True)
            return None

    def _store_document_v4(
        self, obj: Dict[str, Any], vector: Optional[List[float]], document: Document
    ) -> Optional[str]:
        """Store document using Weaviate v4 API"""
        collection = self.client.collections.get(self.class_name)
        result = collection.data.insert(
            properties=obj,
            vector=vector
        )
        vector_id = str(result)
        document.vector_id = vector_id
        logger.debug(f"Stored document {document.id} with vector ID {vector_id}")
        return vector_id

    def _store_document_v3(
        self, obj: Dict[str, Any], vector: Optional[List[float]], document: Document
    ) -> Optional[str]:
        """Store document using Weaviate v3 API"""
        data_object = {
            "class": self.class_name,
            "properties": obj,
        }
        if vector:
            data_object["vector"] = vector

        result = self.client.data_object.create(**data_object)
        vector_id = str(result)
        document.vector_id = vector_id
        logger.debug(f"Stored document {document.id} with vector ID {vector_id}")
        return vector_id

    @log_performance
    def store_batch(
        self,
        documents: List[Document],
        generate_embeddings: bool = True
    ) -> Dict[str, str]:
        """Batch store documents efficiently"""
        if not self.initialized:
            logger.error("Vector store not initialized")
            return {}

        result_mapping: Dict[str, str] = {}
        failed_count = 0

        try:
            import json
            from typing import cast

            # Prepare batch
            batch_objects = []
            embeddings: List[Optional[List[float]]] = []

            if generate_embeddings and self.embedding_service:
                try:
                    texts = [d.content for d in documents]
                    raw_embeddings = self.embedding_service.embed_batch(texts)
                    embeddings = cast(List[Optional[List[float]]], raw_embeddings)
                except Exception as e:
                    logger.warning(f"Batch embedding failed: {e}")
                    embeddings = [None] * len(documents)

            for i, doc in enumerate(documents):
                obj = {
                    "content": doc.content,
                    "document_id": doc.id,
                    "category": doc.metadata.get("category", "general"),
                    "source_type": doc.metadata.get("source_type", "document"),
                    "entities_json": json.dumps(doc.entities),
                    "metadata_json": json.dumps(doc.metadata),
                    "ingestion_version": doc.metadata.get("ingestion_version", 1),
                    "confidence_score": doc.metadata.get("confidence", 1.0),
                }

                vector = None
                if embeddings and i < len(embeddings) and embeddings[i]:
                    vector = embeddings[i]

                batch_objects.append({"obj": obj, "vector": vector, "doc_id": doc.id})

            # Insert batch using appropriate API version
            if self._is_v4:
                result_mapping, failed_count = self._store_batch_v4(batch_objects)
            else:
                result_mapping, failed_count = self._store_batch_v3(batch_objects)

            logger.info(
                f"Batch storage complete: {len(result_mapping)} stored, {failed_count} failed"
            )
            return result_mapping

        except Exception as e:
            logger.error(f"Batch operation failed: {e}", exc_info=True)
            return result_mapping

    def _store_batch_v4(
        self, batch_objects: List[Dict[str, Any]]
    ) -> tuple[Dict[str, str], int]:
        """Store batch using Weaviate v4 API"""
        result_mapping: Dict[str, str] = {}
        failed_count = 0
        collection = self.client.collections.get(self.class_name)
        
        for item in batch_objects:
            try:
                vector_id = collection.data.insert(
                    properties=item["obj"],
                    vector=item["vector"]
                )
                result_mapping[item["doc_id"]] = str(vector_id)
            except Exception as e:
                logger.error(f"Failed to store {item['doc_id']}: {e}")
                failed_count += 1
        
        return result_mapping, failed_count

    def _store_batch_v3(
        self, batch_objects: List[Dict[str, Any]]
    ) -> tuple[Dict[str, str], int]:
        """Store batch using Weaviate v3 API"""
        result_mapping: Dict[str, str] = {}
        failed_count = 0
        
        for item in batch_objects:
            try:
                data_object: Dict[str, Any] = {
                    "class": self.class_name,
                    "properties": item["obj"],
                }
                if item["vector"]:
                    data_object["vector"] = item["vector"]
                
                result = self.client.data_object.create(**data_object)
                result_mapping[item["doc_id"]] = str(result)
            except Exception as e:
                logger.error(f"Failed to store {item['doc_id']}: {e}")
                failed_count += 1
        
        return result_mapping, failed_count

    @log_performance
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Semantic search using embeddings"""
        if not self.initialized:
            logger.error("Vector store not initialized")
            return []

        try:
            import json

            # Generate query embedding
            query_vector = self.embedding_service.embed_text(query)

            if self._is_v4:
                return self._search_v4(query_vector, limit, filters)
            else:
                return self._search_v3(query_vector, limit, filters)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    def _search_v4(
        self,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Execute search using Weaviate v4 API"""
        import json
        from weaviate.classes.query import MetadataQuery, Filter
        
        collection = self.client.collections.get(self.class_name)
        
        # Build query arguments
        query_args: Dict[str, Any] = {
            "near_vector": query_vector,
            "limit": limit,
            "return_metadata": MetadataQuery(distance=True)
        }
        
        results = collection.query.near_vector(**query_args)
        
        return self._parse_search_results(results.objects, is_v4=True)

    def _search_v3(
        self,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Execute search using Weaviate v3 API"""
        import json
        
        query = (
            self.client.query
            .get(self.class_name, ["content", "document_id", "entities_json", "metadata_json"])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .with_additional(["id", "distance"])
        )
        
        result = query.do()
        
        search_results: List[SearchResult] = []
        data = result.get("data", {}).get("Get", {}).get(self.class_name, [])
        
        for item in data:
            entities: List[Entity] = []
            try:
                entities_data = json.loads(item.get("entities_json", "[]"))
                entities = [Entity(**e) for e in entities_data]
            except Exception:
                pass

            metadata: Dict[str, Any] = {}
            try:
                metadata = json.loads(item.get("metadata_json", "{}"))
            except Exception:
                pass

            additional = item.get("_additional", {})
            distance = additional.get("distance", 1.0)
            score = 1.0 - distance  # Convert distance to similarity score

            search_result = SearchResult(
                id=additional.get("id", ""),
                content=item.get("content", ""),
                score=score,
                entities=entities,
                metadata=metadata,
            )
            search_results.append(search_result)
        
        logger.debug(f"Search returned {len(search_results)} results")
        return search_results

    def _parse_search_results(self, objects: Any, is_v4: bool) -> List[SearchResult]:
        """Parse search results from Weaviate response"""
        import json
        
        search_results: List[SearchResult] = []
        for item in objects:
            entities: List[Entity] = []
            try:
                entities_data = json.loads(item.properties.get("entities_json", "[]"))
                entities = [Entity(**e) for e in entities_data]
            except Exception:
                pass

            metadata: Dict[str, Any] = {}
            try:
                metadata = json.loads(item.properties.get("metadata_json", "{}"))
            except Exception:
                pass

            # Calculate score from distance
            distance = getattr(item.metadata, 'distance', 1.0) if item.metadata else 1.0
            score = 1.0 - (distance if distance else 1.0)

            result = SearchResult(
                id=str(item.uuid),
                content=str(item.properties.get("content", "")),
                score=score,
                entities=entities,
                metadata=metadata,
            )
            search_results.append(result)

        logger.debug(f"Search returned {len(search_results)} results")
        return search_results

    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Weaviate WHERE filter from dict"""
        # Simplified filter building - extend as needed
        if not filters:
            return None

        # This is a placeholder for more complex filter logic
        return None

    @log_performance
    def get_document(self, vector_id: str) -> Optional[Document]:
        """Retrieve document by vector ID"""
        if not self.initialized:
            return None

        try:
            import json

            if self._is_v4:
                return self._get_document_v4(vector_id)
            else:
                return self._get_document_v3(vector_id)

        except Exception as e:
            logger.error(f"Failed to retrieve document: {e}")
            return None

    def _get_document_v4(self, vector_id: str) -> Optional[Document]:
        """Get document using Weaviate v4 API"""
        import json
        
        collection = self.client.collections.get(self.class_name)
        result = collection.query.fetch_object_by_id(vector_id)

        if not result:
            return None

        props = result.properties
        return Document(
            id=str(props.get("document_id", "")),
            content=str(props.get("content", "")),
            entities=json.loads(str(props.get("entities_json", "[]"))),
            metadata=json.loads(str(props.get("metadata_json", "{}"))),
            vector_id=str(result.uuid),
        )

    def _get_document_v3(self, vector_id: str) -> Optional[Document]:
        """Get document using Weaviate v3 API"""
        import json
        
        result = self.client.data_object.get_by_id(
            vector_id,
            class_name=self.class_name
        )
        
        if not result:
            return None
        
        props = result.get("properties", {})
        return Document(
            id=props.get("document_id", ""),
            content=props.get("content", ""),
            entities=json.loads(props.get("entities_json", "[]")),
            metadata=json.loads(props.get("metadata_json", "{}")),
            vector_id=result.get("id", vector_id),
        )

    @log_performance
    def delete_document(self, vector_id: str) -> bool:
        """Delete document from vector store"""
        if not self.initialized:
            return False

        try:
            if self._is_v4:
                return self._delete_document_v4(vector_id)
            else:
                return self._delete_document_v3(vector_id)
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def _delete_document_v4(self, vector_id: str) -> bool:
        """Delete document using Weaviate v4 API"""
        collection = self.client.collections.get(self.class_name)
        collection.data.delete_by_id(vector_id)
        logger.info(f"Deleted document {vector_id}")
        return True

    def _delete_document_v3(self, vector_id: str) -> bool:
        """Delete document using Weaviate v3 API"""
        self.client.data_object.delete(
            uuid=vector_id,
            class_name=self.class_name
        )
        logger.info(f"Deleted document {vector_id}")
        return True

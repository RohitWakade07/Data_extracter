# Phase 3 - Vector Database (Weaviate)
# Store documents and metadata for semantic search

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class Document:
    """Document for vector storage"""
    id: str
    content: str
    entities: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class WeaviateClient:
    """Client for Weaviate vector database"""
    
    def __init__(self, weaviate_url: str = "http://localhost:8080", api_key: Optional[str] = None):
        self.url = weaviate_url
        self.api_key = api_key
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Weaviate client"""
        try:
            import weaviate
            from weaviate.config import AdditionalConfig, Timeout
            
            # Parse URL to extract host and port
            url_clean = self.url.replace("http://", "").replace("https://", "")
            host_parts = url_clean.split(":")
            host = host_parts[0]
            http_port = int(host_parts[1]) if len(host_parts) > 1 else 8080
            grpc_port = http_port + 1
            
            # Connect to Weaviate instance with increased timeout
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=http_port,
                http_secure=self.url.startswith("https"),
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=self.url.startswith("https"),
                headers={} if not self.api_key else {"authorization": f"Bearer {self.api_key}"},
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=120, query=120)
                )
            )
            return client
        except Exception as e:
            print(f"Warning: Could not initialize Weaviate client: {str(e)}")
            print("Attempting fallback with skip_init_checks...")
            try:
                import weaviate
                url_clean = self.url.replace("http://", "").replace("https://", "")
                host_parts = url_clean.split(":")
                host = host_parts[0]
                http_port = int(host_parts[1]) if len(host_parts) > 1 else 8080
                grpc_port = http_port + 1
                
                client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=http_port,
                    http_secure=self.url.startswith("https"),
                    grpc_host=host,
                    grpc_port=grpc_port,
                    grpc_secure=self.url.startswith("https"),
                    headers={} if not self.api_key else {"authorization": f"Bearer {self.api_key}"},
                    skip_init_checks=True
                )
                print("Connected with skip_init_checks")
                return client
            except Exception as e2:
                print(f"Fallback also failed: {str(e2)}")
                return None
    
    def create_schema(self) -> bool:
        """Create Weaviate schema for data extraction"""
        if not self.client:
            print("Weaviate client not initialized")
            return False
        
        try:
            from weaviate.classes.config import Configure, Property, DataType
            
            # Define the class with proper schema
            if not self.client.collections.exists("ExtractedDocument"):
                self.client.collections.create(
                    name="ExtractedDocument",
                    properties=[
                        Property(name="content", data_type=DataType.TEXT, description="Original document content"),
                        Property(name="entities", data_type=DataType.TEXT, description="JSON string of extracted entities"),
                        Property(name="documentId", data_type=DataType.TEXT, description="Unique document identifier"),
                        Property(name="sourceType", data_type=DataType.TEXT, description="Type of source document"),
                    ]
                )
            print("Weaviate schema created successfully")
            return True
        except Exception as e:
            print(f"Schema creation error: {str(e)}")
            return False
    
    def store_document(self, document: Document) -> Optional[str]:
        """Store a document in Weaviate"""
        if not self.client:
            print("Weaviate client not initialized")
            return None
        
        try:
            collection = self.client.collections.get("ExtractedDocument")
            
            doc_obj = {
                "content": document.content,
                "entities": json.dumps(document.entities),
                "documentId": document.id,
                "sourceType": document.metadata.get("type", "unknown")
            }
            
            uuid = collection.data.insert(properties=doc_obj)
            
            print(f"Document stored with UUID: {uuid}")
            return str(uuid)
        except Exception as e:
            print(f"Error storing document: {str(e)}")
            return None
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        if not self.client:
            print("Weaviate client not initialized")
            return []
        
        try:
            collection = self.client.collections.get("ExtractedDocument")
            
            results = collection.query.near_text(
                query=query,
                limit=limit,
                return_properties=["content", "entities", "documentId"]
            )
            
            # Convert WeaviateProperties objects to dicts
            result_list: List[Dict[str, Any]] = []
            if results.objects:
                for obj in results.objects:
                    # Safely convert properties to dict
                    props_dict: Dict[str, Any] = {}
                    if isinstance(obj.properties, dict):
                        props_dict = obj.properties
                    else:
                        # Handle WeaviateProperties or other objects
                        try:
                            props_dict = dict(obj.properties)
                        except (TypeError, ValueError):
                            # Fallback: try to get as dict using vars()
                            props_dict = vars(obj.properties) if hasattr(obj.properties, '__dict__') else {}
                    result_list.append(props_dict)
            return result_list
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

def store_in_weaviate(documents: List[Document], weaviate_url: str = "http://localhost:8080") -> Dict[str, Any]:
    """Store extracted documents in Weaviate"""
    print("\n== STORING IN WEAVIATE (Vector Database)")
    print("-" * 50)
    
    client = WeaviateClient(weaviate_url)
    
    results = {
        "total_documents": len(documents),
        "stored_successfully": 0,
        "failed": 0,
        "document_ids": []
    }
    
    try:
        if not client.client:
            return results
        
        for doc in documents:
            uuid = client.store_document(doc)
            if uuid:
                results["stored_successfully"] += 1
                results["document_ids"].append(str(uuid))
            else:
                results["failed"] += 1
        
        print(f"Storage complete: {results['stored_successfully']}/{results['total_documents']} documents")
    finally:
        # 🔥 CRITICAL: Always close the connection
        if client.client:
            try:
                client.client.close()
                print("Weaviate connection closed properly")
            except Exception as e:
                print(f"Warning: Error closing Weaviate connection: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_doc = Document(
        id="doc_001",
        content="John Smith from Acme Corporation signed a contract on 2024-12-14 for $50,000.",
        entities=[
            {"type": "person", "value": "John Smith"},
            {"type": "organization", "value": "Acme Corporation"},
            {"type": "date", "value": "2024-12-14"},
            {"type": "amount", "value": "$50,000"}
        ],
        metadata={"type": "contract", "source": "email"}
    )
    
    result = store_in_weaviate([sample_doc])
    print("\nStorage Result:")
    print(json.dumps(result, indent=2))

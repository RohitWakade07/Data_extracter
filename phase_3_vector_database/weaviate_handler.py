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
            if self.api_key:
                client = weaviate.Client(
                    url=self.url,
                    auth_client_secret=weaviate.auth.AuthApiKey(api_key=self.api_key)
                )
            else:
                client = weaviate.Client(self.url)
            return client
        except Exception as e:
            print(f"Warning: Could not initialize Weaviate client: {str(e)}")
            return None
    
    def create_schema(self) -> bool:
        """Create Weaviate schema for data extraction"""
        if not self.client:
            print("Weaviate client not initialized")
            return False
        
        try:
            schema = {
                "classes": [
                    {
                        "class": "ExtractedDocument",
                        "properties": [
                            {
                                "name": "content",
                                "dataType": ["text"],
                                "description": "Original document content"
                            },
                            {
                                "name": "entities",
                                "dataType": ["text"],
                                "description": "JSON string of extracted entities"
                            },
                            {
                                "name": "documentId",
                                "dataType": ["string"],
                                "description": "Unique document identifier"
                            },
                            {
                                "name": "sourceType",
                                "dataType": ["string"],
                                "description": "Type of source document"
                            }
                        ]
                    }
                ]
            }
            
            self.client.schema.create(schema)
            print("✓ Weaviate schema created successfully")
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
            data = {
                "content": document.content,
                "entities": json.dumps(document.entities),
                "documentId": document.id,
                "sourceType": document.metadata.get("type", "unknown")
            }
            
            uuid = self.client.data_object.create(
                class_name="ExtractedDocument",
                data_object=data
            )
            
            print(f"✓ Document stored with UUID: {uuid}")
            return uuid
        except Exception as e:
            print(f"Error storing document: {str(e)}")
            return None
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        if not self.client:
            print("Weaviate client not initialized")
            return []
        
        try:
            result = self.client.query.get("ExtractedDocument", ["content", "entities", "documentId"]) \
                .with_near_text({"concepts": [query]}) \
                .with_limit(limit) \
                .do()
            
            return result.get("data", {}).get("Get", {}).get("ExtractedDocument", [])
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

def store_in_weaviate(documents: List[Document], weaviate_url: str = "http://localhost:8080") -> Dict[str, Any]:
    """Store extracted documents in Weaviate"""
    print("\n📊 STORING IN WEAVIATE (Vector Database)")
    print("-" * 50)
    
    client = WeaviateClient(weaviate_url)
    
    results = {
        "total_documents": len(documents),
        "stored_successfully": 0,
        "failed": 0,
        "document_ids": []
    }
    
    for doc in documents:
        uuid = client.store_document(doc)
        if uuid:
            results["stored_successfully"] += 1
            results["document_ids"].append(str(uuid))
        else:
            results["failed"] += 1
    
    print(f"✓ Storage complete: {results['stored_successfully']}/{results['total_documents']} documents")
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

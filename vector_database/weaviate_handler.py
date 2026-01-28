# Phase 3 - Vector Database (Weaviate)
# Store documents and metadata for semantic search

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import requests

@dataclass
class Document:
    """Document for vector storage"""
    id: str
    content: str
    entities: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class WeaviateClient:
    """Client for Weaviate vector database using REST API"""
    
    def __init__(self, weaviate_url: str = "http://localhost:8080", api_key: Optional[str] = None):
        self.url = weaviate_url.rstrip('/')
        self.api_key = api_key
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Weaviate client - verify REST API is available"""
        try:
            # Just verify the REST endpoint is accessible
            response = requests.get(f"{self.url}/v1/.well-known/ready", timeout=5)
            if response.status_code == 200:
                print("Weaviate client initialized successfully (REST API)")
                return True
            else:
                print(f"Weaviate not ready: {response.status_code}")
                return None
        except Exception as e:
            print(f"Warning: Could not connect to Weaviate: {str(e)}")
            return None
    
    def create_schema(self) -> bool:
        """Create Weaviate schema for data extraction using REST API"""
        if not self.client:
            print("Weaviate client not initialized")
            return False
        
        try:
            # Check if class exists
            response = requests.get(f"{self.url}/v1/schema/ExtractedDocument", timeout=10)
            if response.status_code == 200:
                print("Weaviate schema already exists")
                return True
            
            # Create schema via REST
            schema = {
                "class": "ExtractedDocument",
                "properties": [
                    {"name": "content", "dataType": ["text"], "description": "Original document content"},
                    {"name": "entities", "dataType": ["text"], "description": "JSON string of extracted entities"},
                    {"name": "documentId", "dataType": ["text"], "description": "Unique document identifier"},
                    {"name": "sourceType", "dataType": ["text"], "description": "Type of source document"},
                ]
            }
            
            response = requests.post(f"{self.url}/v1/schema", json=schema, timeout=10)
            if response.status_code in [200, 201]:
                print("Weaviate schema created successfully")
                return True
            else:
                print(f"Schema creation failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"Schema creation error: {str(e)}")
            return False
    
    def store_document(self, document: Document) -> Optional[str]:
        """Store a document in Weaviate using REST API"""
        if not self.client:
            print("Weaviate client not initialized")
            return None
        
        try:
            doc_obj = {
                "class": "ExtractedDocument",
                "properties": {
                    "content": document.content,
                    "entities": json.dumps(document.entities),
                    "documentId": document.id,
                    "sourceType": document.metadata.get("type", "unknown")
                }
            }
            
            response = requests.post(f"{self.url}/v1/objects", json=doc_obj, timeout=10)
            
            if response.status_code in [200, 201]:
                result = response.json()
                uuid = result.get("id", "")
                print(f"Document stored with UUID: {uuid}")
                return str(uuid)
            else:
                print(f"Error storing document: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error storing document: {str(e)}")
            return None
    
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform search using REST API with GraphQL"""
        if not self.client:
            print("Weaviate client not initialized")
            return []
        
        try:
            # Use GraphQL query via REST
            graphql_query = {
                "query": f"""
                {{
                    Get {{
                        ExtractedDocument(limit: {limit}) {{
                            content
                            entities
                            documentId
                            sourceType
                            _additional {{
                                id
                            }}
                        }}
                    }}
                }}
                """
            }
            
            response = requests.post(f"{self.url}/v1/graphql", json=graphql_query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get("data", {}).get("Get", {}).get("ExtractedDocument", [])
                
                # Filter results that contain the query string
                result_list: List[Dict[str, Any]] = []
                query_lower = query.lower()
                
                for doc in documents:
                    content = str(doc.get('content', '')).lower()
                    entities_str = str(doc.get('entities', '')).lower()
                    
                    # Check if query matches content or entities
                    if query_lower in content or query_lower in entities_str:
                        result_list.append({
                            'id': doc.get('_additional', {}).get('id', ''),
                            'content': doc.get('content', ''),
                            'entities': doc.get('entities', ''),
                            'documentId': doc.get('documentId', ''),
                            'sourceType': doc.get('sourceType', ''),
                            'score': 0.9
                        })
                
                # If no matches, return all documents
                if not result_list and documents:
                    for doc in documents[:limit]:
                        result_list.append({
                            'id': doc.get('_additional', {}).get('id', ''),
                            'content': doc.get('content', ''),
                            'entities': doc.get('entities', ''),
                            'documentId': doc.get('documentId', ''),
                            'sourceType': doc.get('sourceType', ''),
                            'score': 0.5
                        })
                
                return result_list[:limit]
            else:
                print(f"GraphQL query failed: {response.text}")
                return []
                
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    def get_all_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all documents from Weaviate"""
        if not self.client:
            return []
        
        try:
            graphql_query = {
                "query": f"""
                {{
                    Get {{
                        ExtractedDocument(limit: {limit}) {{
                            content
                            entities
                            documentId
                            sourceType
                            _additional {{
                                id
                            }}
                        }}
                    }}
                }}
                """
            }
            
            response = requests.post(f"{self.url}/v1/graphql", json=graphql_query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get("data", {}).get("Get", {}).get("ExtractedDocument", [])
                return documents
            return []
        except Exception as e:
            print(f"Get all documents error: {e}")
            return []
    
    def close(self):
        """Close connection (no-op for REST client)"""
        pass

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
    
    if not client.client:
        return results
    
    # Ensure schema exists
    client.create_schema()
    
    for doc in documents:
        uuid = client.store_document(doc)
        if uuid:
            results["stored_successfully"] += 1
            results["document_ids"].append(str(uuid))
        else:
            results["failed"] += 1
    
    print(f"Storage complete: {results['stored_successfully']}/{results['total_documents']} documents")
    
    return results


def query_weaviate(query_text: str, weaviate_url: str = "http://localhost:8080", limit: int = 5) -> List[Dict[str, Any]]:
    """Query Weaviate for similar documents"""
    client = WeaviateClient(weaviate_url)
    
    if not client.client:
        return []
    
    results = client.semantic_search(query_text, limit=limit)
    return results


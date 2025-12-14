# Phase 4 - Knowledge Graph (NebulaGraph)
# Store entities as nodes and relationships as edges

from typing import List, Dict, Any, Optional
import json

class NebulaGraphClient:
    """Client for NebulaGraph knowledge graph"""
    
    def __init__(self, host: str = "localhost", port: int = 3699, 
                 user: str = "root", password: str = "nebula", 
                 space: str = "extraction_db"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.space = space
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize NebulaGraph client"""
        try:
            from nebula3.Config import Config
            from nebula3.Client import Client
            
            config = Config()
            config.max_connection_pool_size = 50
            client = Client()
            client.connect(self.host, self.port)
            return client
        except Exception as e:
            print(f"Warning: Could not initialize NebulaGraph client: {str(e)}")
            return None
    
    def create_space(self) -> bool:
        """Create graph space and schema"""
        if not self.client:
            print("NebulaGraph client not initialized")
            return False
        
        try:
            # Create space
            self.client.execute(f"CREATE SPACE IF NOT EXISTS {self.space}")
            print(f"✓ Space '{self.space}' created/verified")
            
            # Use space
            self.client.execute(f"USE {self.space}")
            
            # Create tags (node types)
            self.client.execute("""
                CREATE TAG IF NOT EXISTS Person(
                    name string,
                    confidence float
                )
            """)
            
            self.client.execute("""
                CREATE TAG IF NOT EXISTS Organization(
                    name string,
                    confidence float
                )
            """)
            
            self.client.execute("""
                CREATE TAG IF NOT EXISTS Date(
                    value string,
                    confidence float
                )
            """)
            
            self.client.execute("""
                CREATE TAG IF NOT EXISTS Amount(
                    value string,
                    confidence float
                )
            """)
            
            self.client.execute("""
                CREATE TAG IF NOT EXISTS Location(
                    name string,
                    confidence float
                )
            """)
            
            # Create edges (relationships)
            self.client.execute("""
                CREATE EDGE IF NOT EXISTS WORKS_AT(
                    confidence float
                )
            """)
            
            self.client.execute("""
                CREATE EDGE IF NOT EXISTS LOCATED_IN(
                    confidence float
                )
            """)
            
            self.client.execute("""
                CREATE EDGE IF NOT EXISTS SIGNED_ON(
                    confidence float
                )
            """)
            
            print("✓ Graph schema created successfully")
            return True
            
        except Exception as e:
            print(f"Schema creation error: {str(e)}")
            return False
    
    def add_entity_node(self, entity_id: str, entity_type: str, 
                       entity_value: str, confidence: float = 1.0) -> bool:
        """Add entity node to graph"""
        if not self.client:
            print("NebulaGraph client not initialized")
            return False
        
        try:
            tag = entity_type.upper()
            self.client.execute(
                f'INSERT VERTEX {tag}(name, confidence) VALUES "{entity_id}":("{entity_value}", {confidence})'
            )
            print(f"✓ Added {tag} node: {entity_value}")
            return True
        except Exception as e:
            print(f"Error adding node: {str(e)}")
            return False
    
    def add_relationship(self, from_id: str, to_id: str, relationship_type: str, 
                        confidence: float = 1.0) -> bool:
        """Add edge (relationship) to graph"""
        if not self.client:
            print("NebulaGraph client not initialized")
            return False
        
        try:
            edge_type = relationship_type.upper()
            self.client.execute(
                f'INSERT EDGE {edge_type}(confidence) VALUES "{from_id}"->"{to_id}":({confidence})'
            )
            print(f"✓ Added relationship: {from_id} -{edge_type}-> {to_id}")
            return True
        except Exception as e:
            print(f"Error adding relationship: {str(e)}")
            return False
    
    def query_graph(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute a graph query"""
        if not self.client:
            print("NebulaGraph client not initialized")
            return None
        
        try:
            response = self.client.execute(query)
            if response.is_succeeded():
                return {
                    "success": True,
                    "result": response.as_primitive()
                }
            else:
                return {
                    "success": False,
                    "error": response.error_msg()
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def store_in_nebula(entities: List[Dict[str, Any]], relationships: List[Dict[str, str]],
                   host: str = "localhost", port: int = 3699) -> Dict[str, Any]:
    """Store entities and relationships in NebulaGraph"""
    print("\n🔗 STORING IN NEBULA GRAPH (Knowledge Graph)")
    print("-" * 50)
    
    client = NebulaGraphClient(host=host, port=port)
    
    results = {
        "space_created": client.create_space(),
        "entities_added": 0,
        "relationships_added": 0,
        "failed_operations": []
    }
    
    # Add entity nodes
    for entity in entities:
        entity_id = f"{entity.get('type')}_{entity.get('value').replace(' ', '_')}"
        if client.add_entity_node(
            entity_id,
            entity.get('type', 'unknown'),
            entity.get('value', ''),
            entity.get('confidence', 1.0)
        ):
            results["entities_added"] += 1
        else:
            results["failed_operations"].append(f"Failed to add entity: {entity}")
    
    # Add relationships
    for rel in relationships:
        if client.add_relationship(
            rel.get('from_id', ''),
            rel.get('to_id', ''),
            rel.get('type', 'RELATED_TO'),
            rel.get('confidence', 1.0)
        ):
            results["relationships_added"] += 1
        else:
            results["failed_operations"].append(f"Failed to add relationship: {rel}")
    
    print(f"✓ Graph storage complete: {results['entities_added']} entities, {results['relationships_added']} relationships")
    return results

if __name__ == "__main__":
    # Example usage
    sample_entities = [
        {"type": "person", "value": "John Smith", "confidence": 0.95},
        {"type": "organization", "value": "Acme Corporation", "confidence": 0.92},
        {"type": "date", "value": "2024-12-14", "confidence": 1.0},
        {"type": "amount", "value": "$50,000", "confidence": 0.88}
    ]
    
    sample_relationships = [
        {"from_id": "person_John_Smith", "to_id": "organization_Acme_Corporation", "type": "works_at"},
        {"from_id": "person_John_Smith", "to_id": "date_2024-12-14", "type": "signed_on"}
    ]
    
    result = store_in_nebula(sample_entities, sample_relationships)
    print("\nGraph Storage Result:")
    print(json.dumps(result, indent=2))

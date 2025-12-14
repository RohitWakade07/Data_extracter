# Phase 5 - Integration & Demo
# End-to-end pipeline demonstrating semantic and graph queries

import json
from typing import Dict, Any, List
from entity_extraction.entity_extractor import extract_from_text
from agentic_workflow.workflow import run_workflow
from vector_database.weaviate_handler import store_in_weaviate, Document
from knowledge_graph.nebula_handler import store_in_nebula

class IntegratedPipeline:
    """Complete end-to-end extraction pipeline"""
    
    def __init__(self):
        self.results = {}
    
    def run_complete_pipeline(self, unstructured_text: str) -> Dict[str, Any]:
        """
        Execute the complete integrated pipeline
        
        Phase 1: Extract entities
        Phase 2: Agentic workflow validation
        Phase 3: Store in vector database
        Phase 4: Store in knowledge graph
        Phase 5: Execute queries
        """
        
        print("\n" + "="*70)
        print("COMPLETE AUTOMATED DATA EXTRACTION PIPELINE")
        print("="*70)
        
        # Phase 2: Run agentic workflow
        print("\n== PHASE 2: AGENTIC WORKFLOW")
        workflow_result = run_workflow(unstructured_text)
        self.results["workflow"] = workflow_result
        
        entities = workflow_result.get("entities", [])
        
        if not entities:
            print("No entities extracted. Pipeline incomplete.")
            return self.results
        
        # Phase 3: Store in Weaviate
        print("\n== PHASE 3: VECTOR DATABASE (WEAVIATE)")
        doc = Document(
            id="doc_001",
            content=unstructured_text,
            entities=entities,
            metadata={"type": "document"}
        )
        vector_result = store_in_weaviate([doc])
        self.results["vector_storage"] = vector_result
        
        # Phase 4: Store in NebulaGraph
        print("\n== PHASE 4: KNOWLEDGE GRAPH (NEBULA)")
        relationships = self._generate_relationships(entities)
        self.results["relationships"] = relationships
        graph_result = store_in_nebula(entities, relationships)
        self.results["graph_storage"] = graph_result
        
        # Phase 5: Execute queries
        print("\n== PHASE 5: EXECUTE QUERIES")
        self._demonstrate_queries(entities)
        
        return self.results
    
    def _generate_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate relationships between entities with type information"""
        relationships = []
        
        # Helper function to create consistent entity IDs
        def make_entity_id(entity_type: str, entity_value: str) -> str:
            """Create consistent entity ID from type and value"""
            clean = entity_value.replace('\n', ' ').strip()  # Replace newlines with space, strip
            clean = clean.replace(' ', '_')  # Replace all spaces with underscore
            # Normalize multiple underscores to single underscore
            while '__' in clean:
                clean = clean.replace('__', '_')
            return f"{entity_type}_{clean}"
        
        # Group entities by type
        person_entities = [e for e in entities if e.get("type") == "person"]
        org_entities = [e for e in entities if e.get("type") == "organization"]
        date_entities = [e for e in entities if e.get("type") == "date"]
        amount_entities = [e for e in entities if e.get("type") == "amount"]
        location_entities = [e for e in entities if e.get("type") == "location"]
        
        # Person → Organization (WORKS_AT)
        for person in person_entities:
            for org in org_entities:
                person_id = make_entity_id("person", person.get('value', ''))
                org_id = make_entity_id("organization", org.get('value', ''))
                relationships.append({
                    "from_id": person_id,
                    "to_id": org_id,
                    "from_type": "person",
                    "to_type": "organization",
                    "type": "WORKS_AT",
                    "confidence": 0.9
                })
        
        # Person → Location (LOCATED_IN)
        for person in person_entities:
            for location in location_entities:
                person_id = make_entity_id("person", person.get('value', ''))
                loc_id = make_entity_id("location", location.get('value', ''))
                relationships.append({
                    "from_id": person_id,
                    "to_id": loc_id,
                    "from_type": "person",
                    "to_type": "location",
                    "type": "LOCATED_IN",
                    "confidence": 0.8
                })
        
        # Organization → Amount (HAS_AMOUNT)
        for org in org_entities:
            for amount in amount_entities:
                org_id = make_entity_id("organization", org.get('value', ''))
                amt_id = make_entity_id("amount", amount.get('value', ''))
                relationships.append({
                    "from_id": org_id,
                    "to_id": amt_id,
                    "from_type": "organization",
                    "to_type": "amount",
                    "type": "HAS_AMOUNT",
                    "confidence": 0.95
                })
        
        # Amount → Date (EFFECTIVE_ON)
        for amount in amount_entities:
            for date in date_entities:
                amt_id = make_entity_id("amount", amount.get('value', ''))
                date_id = make_entity_id("date", date.get('value', ''))
                relationships.append({
                    "from_id": amt_id,
                    "to_id": date_id,
                    "from_type": "amount",
                    "to_type": "date",
                    "type": "EFFECTIVE_ON",
                    "confidence": 0.9
                })
        
        return relationships
    
    def _demonstrate_queries(self, entities: List[Dict[str, Any]]):
        """Demonstrate semantic and graph queries"""
        
        # Semantic Query Example
        print("\n" + "-"*70)
        print("SEMANTIC QUERY (Weaviate)")
        print("-"*70)
        query = "contracts and agreements"
        print(f"Query: '{query}'")
        print("Expected: Return similar documents using vector similarity")
        print("(Actual execution requires Weaviate to be running)")
        
        # Graph Query Example
        print("\n" + "-"*70)
        print("GRAPH QUERY (NebulaGraph)")
        print("-"*70)
        
        persons = [e for e in entities if e.get("type") == "person"]
        if persons:
            person_name = persons[0].get("value", "")
            person_id = f"person_{person_name.replace(' ', '_')}"
            
            query = f'FETCH PROP ON Person "{person_id}";'
            print(f"Query: Fetch all properties of person: {person_name}")
            print("Expected: Return person node with all attributes")
            print(f"(Graph query would be: {query})")
        
        # Relationship Query
        print("\n" + "-"*70)
        print("RELATIONSHIP QUERY (NebulaGraph)")
        print("-"*70)
        print("Query: Find all organizations associated with extracted persons")
        print("Expected: Return graph paths showing person->works_at->organization")
        print("(Actual execution requires NebulaGraph to be running)")

def main():
    """Main demonstration function"""
    
    # Sample unstructured text
    unstructured_text = """
    John Smith from Acme Corporation signed a major contract on December 14, 2024, 
    valued at $5 million. The agreement was finalized in New York City and involves 
    collaboration with Microsoft on AI research initiatives. 
    
    Sarah Johnson, CEO of TechVision Inc., also participated in the negotiations.
    The project duration is 2 years and includes $1.2 million in Phase 1 funding.
    """
    
    # Run the complete pipeline
    pipeline = IntegratedPipeline()
    results = pipeline.run_complete_pipeline(unstructured_text)
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print("\nSummary:")
    print(f"- Workflow Status: {results.get('workflow', {}).get('status', 'unknown')}")
    print(f"- Entities Extracted: {len(results.get('workflow', {}).get('entities', []))}")
    print(f"- Vector Documents Stored: {results.get('vector_storage', {}).get('stored_successfully', 0)}")
    print(f"- Graph Entities Added: {results.get('graph_storage', {}).get('entities_added', 0)}")
    print(f"- Graph Relationships Added: {results.get('graph_storage', {}).get('relationships_added', 0)}")

if __name__ == "__main__":
    main()

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
        """Generate relationships between entities with context-based matching"""
        relationships = []
        
        # Helper function to create consistent entity IDs
        def make_entity_id(entity_type: str, entity_value: str) -> str:
            """Create consistent entity ID from type and value"""
            clean = entity_value.replace('\n', ' ').strip()
            clean = clean.replace(' ', '_')
            while '__' in clean:
                clean = clean.replace('__', '_')
            return f"{entity_type}_{clean}"
        
        # Group entities by type
        person_entities = [e for e in entities if e.get("type") == "person"]
        org_entities = [e for e in entities if e.get("type") == "organization"]
        location_entities = [e for e in entities if e.get("type") == "location"]
        amount_entities = [e for e in entities if e.get("type") == "amount"]
        date_entities = [e for e in entities if e.get("type") == "date"]
        project_entities = [e for e in entities if e.get("type") == "project"]
        invoice_entities = [e for e in entities if e.get("type") == "invoice"]
        agreement_entities = [e for e in entities if e.get("type") == "agreement"]
        
        # Debug: Print what was found
        print(f"\nRelationship Generation Debug:")
        print(f"  Persons: {len(person_entities)}")
        print(f"  Organizations: {len(org_entities)}")
        print(f"  Projects: {len(project_entities)}")
        print(f"  Invoices: {len(invoice_entities)}")
        print(f"  Agreements: {len(agreement_entities)}")
        
        # IMPROVED: Only match high-confidence entities
        # Threshold: persons with confidence >= 0.85, orgs with confidence >= 0.80
        high_conf_persons = [e for e in person_entities if e.get('confidence', 0) >= 0.85]
        high_conf_orgs = [e for e in org_entities if e.get('confidence', 0) >= 0.80]
        
        print(f"  High-confidence Persons (>=0.85): {len(high_conf_persons)}")
        print(f"  High-confidence Organizations (>=0.80): {len(high_conf_orgs)}")
        
        # Person → Organization (WORKS_AT) relationships
        # Only create relationships between high-confidence entities
        for person in high_conf_persons:
            for org in high_conf_orgs:
                person_id = make_entity_id("person", person.get('value', ''))
                org_id = make_entity_id("organization", org.get('value', ''))
                
                # Relationship confidence based on entity confidences
                relationship_confidence = min(0.95, (person.get('confidence', 0.5) + org.get('confidence', 0.5)) / 2.2)
                
                relationships.append({
                    "from_id": person_id,
                    "to_id": org_id,
                    "from_type": "person",
                    "to_type": "organization",
                    "type": "WORKS_AT",
                    "confidence": relationship_confidence
                })

        # Person → Project (WORKS_ON)
        for person in high_conf_persons:
            for proj in project_entities:
                relationships.append({
                    "from_id": make_entity_id("person", person.get('value', '')),
                    "to_id": make_entity_id("project", proj.get('value', '')),
                    "from_type": "person",
                    "to_type": "project",
                    "type": "WORKS_ON",
                    "confidence": 0.8
                })

        # Organization → Project (MANAGES)
        for org in high_conf_orgs:
            for proj in project_entities:
                relationships.append({
                    "from_id": make_entity_id("organization", org.get('value', '')),
                    "to_id": make_entity_id("project", proj.get('value', '')),
                    "from_type": "organization",
                    "to_type": "project",
                    "type": "MANAGES",
                    "confidence": 0.85
                })

        # Organization → Invoice (ISSUED)
        for org in high_conf_orgs:
            for inv in invoice_entities:
                relationships.append({
                    "from_id": make_entity_id("organization", org.get('value', '')),
                    "to_id": make_entity_id("invoice", inv.get('value', '')),
                    "from_type": "organization",
                    "to_type": "invoice",
                    "type": "ISSUED",
                    "confidence": 0.85
                })

        # Organization → Agreement (PARTY_TO)
        for org in high_conf_orgs:
            for agr in agreement_entities:
                relationships.append({
                    "from_id": make_entity_id("organization", org.get('value', '')),
                    "to_id": make_entity_id("agreement", agr.get('value', '')),
                    "from_type": "organization",
                    "to_type": "agreement",
                    "type": "PARTY_TO",
                    "confidence": 0.85
                })

        # Invoice → Amount (HAS_AMOUNT)
        for inv in invoice_entities:
            for amt in amount_entities:
                relationships.append({
                    "from_id": make_entity_id("invoice", inv.get('value', '')),
                    "to_id": make_entity_id("amount", amt.get('value', '')),
                    "from_type": "invoice",
                    "to_type": "amount",
                    "type": "HAS_AMOUNT",
                    "confidence": 0.9
                })

        # Invoice → Date (DUE_ON)
        for inv in invoice_entities:
            for date in date_entities:
                relationships.append({
                    "from_id": make_entity_id("invoice", inv.get('value', '')),
                    "to_id": make_entity_id("date", date.get('value', '')),
                    "from_type": "invoice",
                    "to_type": "date",
                    "type": "DUE_ON",
                    "confidence": 0.7
                })

        # Agreement → Amount (HAS_VALUE)
        for agr in agreement_entities:
            for amt in amount_entities:
                relationships.append({
                    "from_id": make_entity_id("agreement", agr.get('value', '')),
                    "to_id": make_entity_id("amount", amt.get('value', '')),
                    "from_type": "agreement",
                    "to_type": "amount",
                    "type": "HAS_VALUE",
                    "confidence": 0.8
                })

        # Agreement → Date (SIGNED_ON)
        for agr in agreement_entities:
            for date in date_entities:
                relationships.append({
                    "from_id": make_entity_id("agreement", agr.get('value', '')),
                    "to_id": make_entity_id("date", date.get('value', '')),
                    "from_type": "agreement",
                    "to_type": "date",
                    "type": "SIGNED_ON",
                    "confidence": 0.7
                })
        
        # Project → Location (LOCATED_IN)
        for proj in project_entities:
            for loc in location_entities:
                relationships.append({
                    "from_id": make_entity_id("project", proj.get('value', '')),
                    "to_id": make_entity_id("location", loc.get('value', '')),
                    "from_type": "project",
                    "to_type": "location",
                    "type": "LOCATED_IN",
                    "confidence": 0.8
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
        
        print(f"  Generated relationships: {len(relationships)}")
        return relationships
    
    def _demonstrate_queries(self, entities: List[Dict[str, Any]]):
        """Execute semantic and graph queries"""
        
        # Semantic Query - Weaviate
        print("\n" + "-"*70)
        print("SEMANTIC QUERY (Weaviate)")
        print("-"*70)
        query_text = "contracts and agreements"
        print(f"Query: '{query_text}'")
        print("Expected: Return similar documents using vector similarity")
        
        try:
            from vector_database.weaviate_handler import query_weaviate
            results = query_weaviate(query_text)
            if results:
                print(f"\n✓ Found {len(results)} similar documents:")
                for i, result in enumerate(results[:3], 1):  # Show top 3
                    print(f"  {i}. Score: {result.get('score', 'N/A')}")
            else:
                print("(No results or Weaviate not responding)")
        except Exception as e:
            print(f"(Query execution error: {str(e)[:50]}...)")
        
        # Graph Query - NebulaGraph
        print("\n" + "-"*70)
        print("GRAPH QUERY (NebulaGraph)")
        print("-"*70)
        
        persons = [e for e in entities if e.get("type") == "person"]
        if persons:
            try:
                from knowledge_graph.nebula_handler import execute_graph_query
                person_name = persons[0].get("value", "")
                person_id = f'"{person_name}"'
                
                query = f'FETCH PROP ON Person {person_id};'
                print(f"Query: Fetch all properties of person: {person_name}")
                print("Expected: Return person node with all attributes")
                
                result = execute_graph_query(query)
                if result:
                    print(f"✓ Query executed successfully")
                    print(f"Result: {result}")
                else:
                    print("(Query returned no results)")
            except Exception as e:
                print(f"(Query execution error: {str(e)[:50]}...)")
        else:
            print("No persons found to query")
        
        # Relationship Query - NebulaGraph
        print("\n" + "-"*70)
        print("RELATIONSHIP QUERY (NebulaGraph)")
        print("-"*70)
        print("Query: Find all organizations associated with extracted persons")
        print("Expected: Return graph paths showing person->works_at->organization")
        
        try:
            from knowledge_graph.nebula_handler import execute_graph_query
            
            # Find all person->works_at->organization relationships
            query = 'MATCH (p:Person)-[e:WORKS_AT]->(o:Organization) RETURN p.name, o.name LIMIT 10;'
            print(f"\n✓ Executing relationship query...")
            result = execute_graph_query(query)
            if result:
                print(f"Found relationships:")
                print(result)
            else:
                print("(No relationships found or NebulaGraph not responding)")
        except Exception as e:
            print(f"(Query execution error: {str(e)[:50]}...)")


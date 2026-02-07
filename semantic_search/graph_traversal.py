# Graph Traversal Module
# Advanced graph queries for NebulaGraph
# Supports multi-hop traversal, indirect connections, and pattern discovery

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TraversalResult:
    """Result from graph traversal query"""
    paths: List[List[Dict[str, Any]]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    query_executed: str = ""
    depth: int = 0


class GraphTraversal:
    """
    Advanced graph traversal for NebulaGraph.
    
    Enables queries like:
    - "Which companies are indirectly affected by Mumbai port congestion?"
    - "Find all suppliers connected to a specific vendor"
    - "Discover patterns of similar incidents"
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9669,
        user: str = "root",
        password: str = "nebula",
        space: str = "extraction_db"
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.space = space
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize NebulaGraph connection"""
        try:
            from nebula3.gclient.net import ConnectionPool
            from nebula3.Config import Config
            
            config = Config()
            config.max_connection_pool_size = 10
            config.timeout = 30000
            
            pool = ConnectionPool()
            ok = pool.init([(self.host, self.port)], config)
            
            if ok:
                print("✓ Graph traversal connected to NebulaGraph")
                return pool
            return None
            
        except Exception as e:
            print(f"⚠ NebulaGraph connection failed: {e}")
            return None
    
    def _get_session(self):
        """Get authenticated session"""
        if not self.client:
            return None
        return self.client.get_session(self.user, self.password)
    
    def _execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a graph query and return results"""
        session = self._get_session()
        if not session:
            return {"success": False, "error": "No session available"}
        
        try:
            session.execute(f"USE {self.space}")
            result = session.execute(query)
            
            if result.is_succeeded():
                return {
                    "success": True,
                    "data": result.as_primitive()
                }
            else:
                return {
                    "success": False,
                    "error": result.error_msg()
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            session.release()
    
    def find_indirect_connections(
        self,
        start_entity: str,
        start_type: str,
        target_type: str = "Organization",
        max_depth: int = 3
    ) -> TraversalResult:
        """
        Find entities indirectly connected to a start entity.
        
        Example:
        - Start: "Mumbai Port", Type: "Location"
        - Target: "Organization"
        - Returns: All companies connected via suppliers, logistics partners, etc.
        
        Args:
            start_entity: Name or ID of starting entity
            start_type: Type of starting entity (Location, Organization, etc.)
            target_type: Type of entities to find
            max_depth: Maximum hops to traverse
        
        Returns:
            TraversalResult with paths and connected entities
        """
        # Build entity ID
        entity_id = self._build_entity_id(start_entity, start_type)
        
        # Multi-hop traversal query using GO statement
        query = f'''
            GO 1 TO {max_depth} STEPS FROM "{entity_id}" 
            OVER * BIDIRECT
            YIELD 
                src(edge) as source,
                dst(edge) as target,
                type(edge) as relationship,
                properties($$).name as target_name
            | LIMIT 100
        '''
        
        result = self._execute_query(query)
        
        paths = []
        entities = []
        relationships = []
        
        if result.get("success"):
            data = result.get("data", [])
            
            seen_entities = set()
            for row in data:
                source = row.get("source", "")
                target = row.get("target", "")
                rel_type = row.get("relationship", "")
                target_name = row.get("target_name", "")
                
                # Filter by target type if specified
                if target_type and target_type.lower() in target.lower():
                    if target not in seen_entities:
                        entities.append({
                            "id": target,
                            "name": target_name or target,
                            "type": target_type
                        })
                        seen_entities.add(target)
                
                relationships.append({
                    "source": source,
                    "target": target,
                    "type": rel_type
                })
                
                paths.append([
                    {"entity": source},
                    {"relationship": rel_type},
                    {"entity": target}
                ])
        
        return TraversalResult(
            paths=paths,
            entities=entities,
            relationships=relationships,
            query_executed=query,
            depth=max_depth
        )
    
    def find_companies_affected_by(
        self,
        event_entity: str,
        event_type: str = "Location"
    ) -> List[Dict[str, Any]]:
        """
        Find companies affected by an event (e.g., port congestion, weather).
        
        Query: "Which companies are indirectly affected by Mumbai port congestion?"
        
        Returns:
            List of companies with their connection paths
        """
        entity_id = self._build_entity_id(event_entity, event_type)
        
        # Find organizations connected through any path
        query = f'''
            GO 1 TO 4 STEPS FROM "{entity_id}"
            OVER * BIDIRECT
            WHERE id($$) CONTAINS "organization"
            YIELD 
                $$.Organization.name as company_name,
                id($$) as company_id,
                type(edge) as via_relationship
            | DISTINCT company_name, company_id, via_relationship
        '''
        
        result = self._execute_query(query)
        
        companies = []
        if result.get("success"):
            data = result.get("data", [])
            
            seen_companies = set()
            for row in data:
                company_name = row.get("company_name", "")
                company_id = row.get("company_id", "")
                
                if company_name and company_name not in seen_companies:
                    companies.append({
                        "name": company_name,
                        "id": company_id,
                        "connection_type": row.get("via_relationship", ""),
                        "affected_by": event_entity
                    })
                    seen_companies.add(company_name)
        
        # If GO query fails, try MATCH
        if not companies:
            companies = self._find_companies_via_match(entity_id, event_entity)
        
        return companies
    
    def _find_companies_via_match(
        self,
        entity_id: str,
        event_name: str
    ) -> List[Dict[str, Any]]:
        """Fallback using MATCH query for company discovery"""
        query = f'''
            MATCH (start)-[*1..4]-(org:Organization)
            WHERE id(start) == "{entity_id}"
            RETURN org.Organization.name as company_name, id(org) as company_id
            LIMIT 50
        '''
        
        result = self._execute_query(query)
        
        companies = []
        if result.get("success"):
            for row in result.get("data", []):
                if row.get("company_name"):
                    companies.append({
                        "name": row.get("company_name"),
                        "id": row.get("company_id", ""),
                        "connection_type": "indirect",
                        "affected_by": event_name
                    })
        
        return companies
    
    def get_entity_relationships(
        self,
        entity_name: str,
        entity_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships for an entity with multiple fallback strategies.
        
        Args:
            entity_name: Entity name
            entity_type: Entity type (person, organization, location, etc.)
        
        Returns:
            List of relationships with connected entities
        """
        relationships = []
        
        # Strategy 1: Try exact entity ID match
        entity_id = self._build_entity_id(entity_name, entity_type)
        rels = self._get_relationships_by_id(entity_id)
        if rels:
            relationships.extend(rels)
        
        # Strategy 2: Try with different type variations
        if not relationships:
            type_variations = [
                entity_type.lower(),
                entity_type.capitalize(),
                entity_type.upper()
            ]
            for type_var in type_variations:
                alt_id = self._build_entity_id(entity_name, type_var)
                if alt_id != entity_id:
                    rels = self._get_relationships_by_id(alt_id)
                    if rels:
                        relationships.extend(rels)
                        break
        
        # Strategy 3: Try fuzzy match using LOOKUP
        if not relationships:
            rels = self._fuzzy_relationship_lookup(entity_name, entity_type)
            if rels:
                relationships.extend(rels)
        
        return relationships
    
    def _get_relationships_by_id(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get relationships for a specific entity ID"""
        relationships = []
        
        # Get outgoing relationships
        query_out = f'''
            GO FROM "{entity_id}"
            OVER * 
            YIELD 
                type(edge) as relationship,
                id($$) as target_id,
                properties($$) as target_props
            LIMIT 50
        '''
        
        # Get incoming relationships  
        query_in = f'''
            GO FROM "{entity_id}"
            OVER * REVERSELY
            YIELD 
                type(edge) as relationship,
                id($$) as source_id,
                properties($$) as source_props
            LIMIT 50
        '''
        
        # Outgoing
        result_out = self._execute_query(query_out)
        if result_out.get("success"):
            for row in result_out.get("data", []):
                rel_type = row.get("relationship", "")
                target_id = row.get("target_id", "")
                if rel_type and target_id:
                    relationships.append({
                        "direction": "outgoing",
                        "type": rel_type,
                        "connected_entity": target_id,
                        "target": self._clean_entity_id(target_id),
                        "related_entity": self._clean_entity_id(target_id),
                        "relationship": rel_type,
                        "properties": row.get("target_props", {})
                    })
        
        # Incoming
        result_in = self._execute_query(query_in)
        if result_in.get("success"):
            for row in result_in.get("data", []):
                rel_type = row.get("relationship", "")
                source_id = row.get("source_id", "")
                if rel_type and source_id:
                    relationships.append({
                        "direction": "incoming",
                        "type": rel_type,
                        "connected_entity": source_id,
                        "target": self._clean_entity_id(source_id),
                        "related_entity": self._clean_entity_id(source_id),
                        "relationship": rel_type,
                        "properties": row.get("source_props", {})
                    })
        
        return relationships
    
    def _fuzzy_relationship_lookup(
        self,
        entity_name: str,
        entity_type: str
    ) -> List[Dict[str, Any]]:
        """Try to find relationships using fuzzy matching on entity names"""
        relationships = []
        
        # Map entity type to tag
        tag_map = {
            'person': 'Person',
            'organization': 'Organization',
            'location': 'Location',
            'project': 'Project',
            'date': 'Date',
            'amount': 'Amount',
            'invoice': 'Invoice',
            'agreement': 'Agreement',
            'shipment': 'Shipment',
            'port': 'Port',
        }
        
        tag = tag_map.get(entity_type.lower(), entity_type.capitalize())
        
        # Try LOOKUP to find entities with similar names
        clean_name = entity_name.replace('"', '\\"')
        
        # Determine property name based on tag
        prop_name = "value" if tag in ['Date', 'Amount'] else "name"
        
        query = f'''
            LOOKUP ON `{tag}` 
            WHERE `{tag}`.{prop_name} CONTAINS "{clean_name}"
            YIELD id(vertex) as vid
            LIMIT 5
        '''
        
        result = self._execute_query(query)
        
        if result.get("success"):
            for row in result.get("data", []):
                vid = row.get("vid", "")
                if vid:
                    rels = self._get_relationships_by_id(vid)
                    relationships.extend(rels)
        
        return relationships
    
    def _clean_entity_id(self, entity_id: str) -> str:
        """Clean entity ID to get readable name"""
        if not entity_id:
            return ""
        # Remove type prefix (e.g., "person_John_Doe" -> "John Doe")
        parts = entity_id.split('_', 1)
        if len(parts) > 1:
            return parts[1].replace('_', ' ')
        return entity_id.replace('_', ' ')
    
    def find_similar_incidents(
        self,
        incident_type: str,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find incidents similar to a given type.
        
        Example: "Find incidents similar to cyber attacks on cloud vendors"
        
        Args:
            incident_type: Type of incident to search for
            category: Optional category filter
        
        Returns:
            List of similar incidents with their patterns
        """
        # Map incident types to related graph patterns
        incident_patterns = {
            "cyber attack": ["security", "breach", "hack", "vulnerability", "exploit"],
            "cloud": ["aws", "azure", "gcp", "infrastructure", "saas"],
            "supply chain": ["logistics", "shipment", "supplier", "vendor", "delay"],
            "weather": ["rain", "flood", "storm", "monsoon", "natural disaster"]
        }
        
        # Find related patterns
        related_terms = []
        incident_lower = incident_type.lower()
        for pattern_key, terms in incident_patterns.items():
            if pattern_key in incident_lower or any(t in incident_lower for t in terms):
                related_terms.extend(terms)
        
        incidents = []
        
        # Query for Incident entities
        query = '''
            LOOKUP ON Incident
            YIELD id(vertex) as id, properties(vertex).name as name, properties(vertex).category as category
        '''
        
        result = self._execute_query(query)
        
        if result.get("success"):
            for row in result.get("data", []):
                incident_name = row.get("name", "").lower()
                incident_category = row.get("category", "")
                
                # Check if incident matches pattern
                if any(term in incident_name for term in related_terms) or incident_type.lower() in incident_name:
                    incidents.append({
                        "id": row.get("id"),
                        "name": row.get("name"),
                        "category": incident_category,
                        "pattern_match": incident_type
                    })
        
        # Also look for events/issues that match the pattern
        event_query = '''
            LOOKUP ON Event
            YIELD id(vertex) as id, properties(vertex).name as name, properties(vertex).type as type
        '''
        
        event_result = self._execute_query(event_query)
        
        if event_result.get("success"):
            for row in event_result.get("data", []):
                event_name = row.get("name", "").lower()
                
                if any(term in event_name for term in related_terms):
                    incidents.append({
                        "id": row.get("id"),
                        "name": row.get("name"),
                        "category": row.get("type", "Event"),
                        "pattern_match": incident_type
                    })
        
        return incidents
    
    def trace_supply_chain_path(
        self,
        company: str,
        direction: str = "both"  # "upstream", "downstream", "both"
    ) -> Dict[str, Any]:
        """
        Trace supply chain connections for a company.
        
        Args:
            company: Company name
            direction: upstream (suppliers), downstream (customers), or both
        
        Returns:
            Supply chain map with all connected entities
        """
        company_id = self._build_entity_id(company, "organization")
        
        supply_chain = {
            "company": company,
            "suppliers": [],
            "customers": [],
            "logistics_partners": [],
            "locations": []
        }
        
        # Find suppliers (incoming SUPPLIES_TO edges)
        if direction in ["upstream", "both"]:
            supplier_query = f'''
                GO FROM "{company_id}"
                OVER SUPPLIES_TO REVERSELY
                YIELD $$.Organization.name as supplier_name, id($$) as supplier_id
            '''
            result = self._execute_query(supplier_query)
            if result.get("success"):
                for row in result.get("data", []):
                    if row.get("supplier_name"):
                        supply_chain["suppliers"].append({
                            "name": row.get("supplier_name"),
                            "id": row.get("supplier_id")
                        })
        
        # Find customers (outgoing SUPPLIES_TO edges)
        if direction in ["downstream", "both"]:
            customer_query = f'''
                GO FROM "{company_id}"
                OVER SUPPLIES_TO
                YIELD $$.Organization.name as customer_name, id($$) as customer_id
            '''
            result = self._execute_query(customer_query)
            if result.get("success"):
                for row in result.get("data", []):
                    if row.get("customer_name"):
                        supply_chain["customers"].append({
                            "name": row.get("customer_name"),
                            "id": row.get("customer_id")
                        })
        
        # Find logistics partners
        logistics_query = f'''
            GO FROM "{company_id}"
            OVER USES_LOGISTICS, PARTNERS_WITH
            YIELD $$.Organization.name as partner_name, id($$) as partner_id, type(edge) as relationship
        '''
        result = self._execute_query(logistics_query)
        if result.get("success"):
            for row in result.get("data", []):
                if row.get("partner_name"):
                    supply_chain["logistics_partners"].append({
                        "name": row.get("partner_name"),
                        "id": row.get("partner_id"),
                        "relationship": row.get("relationship")
                    })
        
        # Find locations
        location_query = f'''
            GO FROM "{company_id}"
            OVER LOCATED_IN, OPERATES_AT
            YIELD $$.Location.name as location_name, id($$) as location_id
        '''
        result = self._execute_query(location_query)
        if result.get("success"):
            for row in result.get("data", []):
                if row.get("location_name"):
                    supply_chain["locations"].append({
                        "name": row.get("location_name"),
                        "id": row.get("location_id")
                    })
        
        return supply_chain
    
    def _build_entity_id(self, entity_name: str, entity_type: str) -> str:
        """Build consistent entity ID from name and type"""
        clean_name = entity_name.replace('\n', ' ').strip()
        clean_name = clean_name.replace(' ', '_')
        while '__' in clean_name:
            clean_name = clean_name.replace('__', '_')
        return f"{entity_type.lower()}_{clean_name}"


# Convenience functions
def find_indirect_connections(
    start_entity: str,
    start_type: str,
    target_type: str = "Organization",
    host: str = "127.0.0.1",
    port: int = 9669
) -> TraversalResult:
    """Find entities indirectly connected to a start entity"""
    graph = GraphTraversal(host, port)
    return graph.find_indirect_connections(start_entity, start_type, target_type)


def find_similar_incidents(
    incident_type: str,
    host: str = "127.0.0.1",
    port: int = 9669
) -> List[Dict[str, Any]]:
    """Find incidents similar to a given type"""
    graph = GraphTraversal(host, port)
    return graph.find_similar_incidents(incident_type)

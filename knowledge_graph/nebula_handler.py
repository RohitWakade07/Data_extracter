

from typing import List, Dict, Any, Optional
import json
import time

# Entity type mapping for consistent tagging
ENTITY_TAG_MAP = {
    "person": "Person",
    "organization": "Organization",
    "amount": "Amount",
    "date": "Date",
    "location": "Location",
    "project": "Project",
    "invoice": "Invoice",
    "agreement": "Agreement"
}

# Relationship mapping rules
RELATIONSHIP_MAP = {
    ("Person", "Organization"): "WORKS_AT",
    ("Person", "Location"): "LOCATED_IN",
    ("Person", "Date"): "SIGNED_ON",
    ("Organization", "Location"): "LOCATED_IN",
    ("Organization", "Amount"): "HAS_AMOUNT",
    ("Amount", "Date"): "EFFECTIVE_ON",
    ("Person", "Project"): "WORKS_ON",
    ("Organization", "Project"): "MANAGES",
    ("Organization", "Invoice"): "ISSUED",
    ("Invoice", "Organization"): "BILLED_TO",
    ("Organization", "Agreement"): "PARTY_TO",
    ("Agreement", "Date"): "SIGNED_ON",
    ("Agreement", "Amount"): "HAS_VALUE",
    ("Invoice", "Amount"): "HAS_AMOUNT",
    ("Invoice", "Date"): "DUE_ON",
    ("Project", "Location"): "LOCATED_IN",
}


class NebulaGraphClient:
    """Client for NebulaGraph knowledge graph"""

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

        self.pool = self._initialize_pool()

   
    def _initialize_pool(self):
        """Initialize NebulaGraph connection pool"""
        try:
            from nebula3.gclient.net import ConnectionPool
            from nebula3.Config import Config

            config = Config()
            config.max_connection_pool_size = 10
            config.timeout = 30000

            pool = ConnectionPool()
            ok = pool.init([(self.host, self.port)], config)

            if not ok:
                raise RuntimeError("Failed to initialize NebulaGraph connection pool")

            print("NebulaGraph connection pool initialized")
            return pool

        except Exception as e:
            print(f"NebulaGraph init failed: {e}")
            return None

    def _get_session(self):
        """Create authenticated session"""
        if not self.pool:
            return None
        return self.pool.get_session(self.user, self.password)

    
    def create_space(self) -> bool:
        """Create graph space and schema"""
        session = self._get_session()
        if not session:
            print("NebulaGraph session not available")
            return False

        try:
            # Create space with VID type
            session.execute(f"""
                CREATE SPACE IF NOT EXISTS {self.space}(
                    vid_type=FIXED_STRING(64)
                )
            """)

            # Wait for space to be ready
            time.sleep(2)

            session.execute(f"USE {self.space}")
            print(f"Space '{self.space}' ready")

            # ----------------- TAGS -----------------
            session.execute("""
                CREATE TAG IF NOT EXISTS Person(
                    name string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Organization(
                    name string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS `Date`(
                    value string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Amount(
                    value string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Location(
                    name string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Project(
                    name string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Invoice(
                    name string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Agreement(
                    name string,
                    confidence float
                )
            """)

            # ----------------- EDGES -----------------
            session.execute("""
                CREATE EDGE IF NOT EXISTS WORKS_AT(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS LOCATED_IN(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS SIGNED_ON(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS HAS_AMOUNT(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS EFFECTIVE_ON(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS WORKS_ON(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS MANAGES(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS ISSUED(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS BILLED_TO(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS PARTY_TO(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS HAS_VALUE(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS DUE_ON(
                    confidence float
                )
            """)

            print("Graph schema created successfully. Creating indexes for MATCH queries...")
            time.sleep(3)  # Wait for schema to propagate
            
            # Create tag indexes for LOOKUP queries to work
            tag_types = ['Person', 'Organization', 'Location', 'Project', 'Invoice', 'Agreement', 'Amount', 'Date']
            for tag in tag_types:
                try:
                    session.execute(f"CREATE TAG INDEX IF NOT EXISTS idx_{tag.lower()} ON `{tag}`()")
                except Exception as idx_err:
                    print(f"Index creation for {tag} skipped: {idx_err}")
            
            # Create edge indexes for ALL edge types
            edge_types = ['WORKS_AT', 'LOCATED_IN', 'WORKS_ON', 'MANAGES', 'PARTY_TO', 
                          'SIGNED_ON', 'HAS_AMOUNT', 'EFFECTIVE_ON', 'HAS_VALUE', 'ISSUED', 'DUE_ON', 'BILLED_TO']
            for edge in edge_types:
                try:
                    session.execute(f"CREATE EDGE INDEX IF NOT EXISTS idx_{edge.lower()} ON `{edge}`()")
                except Exception as idx_err:
                    print(f"Edge index creation for {edge} skipped: {idx_err}")
            
            print("Indexes created. Rebuilding indexes...")
            time.sleep(2)
            
            # Rebuild indexes to make existing data queryable
            for tag in tag_types:
                try:
                    session.execute(f"REBUILD TAG INDEX idx_{tag.lower()}")
                except:
                    pass
            
            for edge in edge_types:
                try:
                    session.execute(f"REBUILD EDGE INDEX idx_{edge.lower()}")
                except:
                    pass
            
            print("Schema and indexes ready.")
            time.sleep(3)  # Wait for index rebuild
            return True

        except Exception as e:
            print(f"Schema creation error: {e}")
            return False

        finally:
            session.release()

    # ------------------------------------------------------------------
    # Data Insertion
    # ------------------------------------------------------------------
    def add_entity_node(
        self,
        entity_id: str,
        entity_type: str,
        entity_value: str,
        confidence: float = 1.0
    ) -> bool:
        """Add entity node with standardized tagging"""
        session = self._get_session()
        if not session:
            return False

        try:
            session.execute(f"USE {self.space}")

            
            tag = ENTITY_TAG_MAP.get(entity_type.lower(), entity_type.capitalize())
            
            # Clean entity value: remove newlines and strip whitespace
            clean_value = entity_value.replace("\n", " ").strip()

            # Handle different tag schemas
            if tag in ["Date", "Amount"]:
                query = (
                    f'INSERT VERTEX `{tag}`(value, confidence) '
                    f'VALUES "{entity_id}":("{clean_value}", {confidence})'
                )
            else:
                query = (
                    f'INSERT VERTEX `{tag}`(name, confidence) '
                    f'VALUES "{entity_id}":("{clean_value}", {confidence})'
                )

            session.execute(query)
            print(f"Added {tag} node - {entity_value}")
            return True

        except Exception as e:
            print(f"Node insert error: {e}")
            return False

        finally:
            session.release()

    def add_relationship(
        self,
        from_id: str,
        to_id: str,
        from_type: str,
        to_type: str,
        relationship_type: Optional[str] = None,
        confidence: float = 1.0
    ) -> bool:
        """Add relationship edge with intelligent type mapping"""
        session = self._get_session()
        if not session:
            return False

        try:
            session.execute(f"USE {self.space}")

            # Standardize entity types
            from_tag = ENTITY_TAG_MAP.get(from_type.lower(), from_type.capitalize())
            to_tag = ENTITY_TAG_MAP.get(to_type.lower(), to_type.capitalize())
            
            # Determine relationship type if not provided
            if not relationship_type:
                relationship_key = (from_tag, to_tag)
                relationship_type = RELATIONSHIP_MAP.get(
                    relationship_key,
                    "RELATED_TO"  # Default relationship
                )
            
            edge = relationship_type.upper()

            query = (
                f'INSERT EDGE {edge}(confidence) '
                f'VALUES "{from_id}"->"{to_id}":({confidence})'
            )
            
            result = session.execute(query)
            
            if not result.is_succeeded():
                print(f"Edge insert error: {from_id} -[{edge}]-> {to_id}")
                print(f"   Error: {result.error_msg()}")
                return False

            print(f"Added relationship {from_id} -[{edge}]-> {to_id}")
            return True

        except Exception as e:
            print(f"Edge insert error for {from_id} -[{relationship_type}]-> {to_id}: {e}")
            return False

        finally:
            session.release()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def query_graph(self, query: str) -> Dict[str, Any]:
        """Execute graph query"""
        session = self._get_session()
        if not session:
            return {"success": False, "error": "Session unavailable"}

        try:
            session.execute(f"USE {self.space}")
            result = session.execute(query)

            if result.is_succeeded():
                return {
                    "success": True,
                    "result": result.as_primitive()
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

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all entities from the graph using LOOKUP with tag indexes"""
        session = self._get_session()
        if not session:
            return []

        try:
            session.execute(f"USE {self.space}")
            
            entities = []
            # Query each entity type tag using LOOKUP
            tag_types = ['Person', 'Organization', 'Location', 'Project', 'Invoice', 'Agreement', 'Amount', 'Date']
            
            for tag_type in tag_types:
                try:
                    # LOOKUP with proper YIELD
                    if tag_type in ['Date', 'Amount']:
                        result = session.execute(f"LOOKUP ON `{tag_type}` YIELD id(vertex) as vid, properties(vertex).value as name")
                    else:
                        result = session.execute(f"LOOKUP ON `{tag_type}` YIELD id(vertex) as vid, properties(vertex).name as name")
                    
                    if result.is_succeeded():
                        # Use as_primitive() which returns a list of dicts
                        data = result.as_primitive()
                        for row in data:
                            name = row.get('name', '')
                            if name:
                                entities.append({
                                    "id": row.get('vid', ''),
                                    "name": name,
                                    "type": tag_type.lower(),
                                    "confidence": 0.8
                                })
                    else:
                        print(f"LOOKUP for {tag_type}: {result.error_msg()}")
                except Exception as tag_err:
                    print(f"Query for {tag_type} failed: {tag_err}")
                    continue
            
            return entities

        except Exception as e:
            print(f"Get entities error: {e}")
            return []

        finally:
            session.release()

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships from the graph using LOOKUP"""
        session = self._get_session()
        if not session:
            return []

        try:
            session.execute(f"USE {self.space}")
            
            relationships = []
            # Query all edge types using LOOKUP
            edge_types = ['WORKS_AT', 'LOCATED_IN', 'WORKS_ON', 'MANAGES', 'PARTY_TO', 
                          'SIGNED_ON', 'HAS_AMOUNT', 'EFFECTIVE_ON', 'HAS_VALUE', 'ISSUED', 'DUE_ON']
            
            for edge_type in edge_types:
                try:
                    result = session.execute(f"LOOKUP ON `{edge_type}` YIELD src(edge) as src, dst(edge) as dst")
                    
                    if result.is_succeeded():
                        data = result.as_primitive()
                        for row in data:
                            src = row.get('src', '')
                            dst = row.get('dst', '')
                            if src and dst:
                                relationships.append({
                                    "source": src,
                                    "type": edge_type,
                                    "target": dst
                                })
                    else:
                        print(f"LOOKUP for edge {edge_type}: {result.error_msg()}")
                except Exception as edge_err:
                    print(f"Query for {edge_type} failed: {edge_err}")
                    continue
            
            return relationships

        except Exception as e:
            print(f"Get relationships error: {e}")
            return []

        finally:
            session.release()

    def get_graph_stats(self) -> Dict[str, int]:
        """Get graph statistics using SHOW STATS"""
        session = self._get_session()
        if not session:
            return {"entities": 0, "relationships": 0}

        try:
            session.execute(f"USE {self.space}")
            
            # Use SHOW STATS which returns pre-computed statistics
            result = session.execute("SHOW STATS")
            
            if result.is_succeeded():
                entity_count = 0
                rel_count = 0
                
                # Parse the stats result
                data = result.as_primitive()
                for item in data:
                    if item.get('Type') == 'Space' and item.get('Name') == 'vertices':
                        entity_count = item.get('Count', 0)
                    elif item.get('Type') == 'Space' and item.get('Name') == 'edges':
                        rel_count = item.get('Count', 0)
                
                return {"entities": entity_count, "relationships": rel_count}
            else:
                print(f"SHOW STATS error: {result.error_msg()}")
                return {"entities": 0, "relationships": 0}

        except Exception as e:
            print(f"Get stats error: {e}")
            return {"entities": 0, "relationships": 0}

        finally:
            session.release()


# ------------------------------------------------------------------
# Helper Function for Pipeline Integration
# ------------------------------------------------------------------
def store_in_nebula(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    host: str = "127.0.0.1",
    port: int = 9669
) -> Dict[str, Any]:

    print("\n== STORING DATA IN NEBULAGRAPH")
    print("-" * 50)

    client = NebulaGraphClient(host=host, port=port)

    results = {
        "space_created": client.create_space(),
        "entities_added": 0,
        "relationships_added": 0,
        "failed_operations": []
    }

    # Insert entities
    for e in entities:
        # Create consistent entity IDs by cleaning newlines and spaces
        clean_value = e['value'].replace('\n', ' ').strip()
        clean_value = clean_value.replace(' ', '_')
        # Normalize multiple underscores to single underscore
        while '__' in clean_value:
            clean_value = clean_value.replace('__', '_')
        eid = f"{e['type']}_{clean_value}"
        ok = client.add_entity_node(
            eid,
            e["type"],
            e["value"],
            e.get("confidence", 1.0)
        )

        if ok:
            results["entities_added"] += 1
        else:
            results["failed_operations"].append({"entity": e})

    # Insert relationships
    for r in relationships:
        ok = client.add_relationship(
            r["from_id"],
            r["to_id"],
            r.get("from_type", ""),
            r.get("to_type", ""),
            r.get("type"),
            r.get("confidence", 1.0)
        )

        if ok:
            results["relationships_added"] += 1
        else:
            results["failed_operations"].append({"relationship": r})

    print("NebulaGraph storage completed")
    return results


def execute_graph_query(query: str, host: str = "127.0.0.1", port: int = 9669) -> Any:
    """Execute a raw query against NebulaGraph"""
    client = NebulaGraphClient(host=host, port=port)
    result = client.query_graph(query)
    if result["success"]:
        return result["result"]
    else:
        print(f"Query failed: {result.get('error')}")
        return None


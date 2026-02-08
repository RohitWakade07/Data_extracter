

from typing import List, Dict, Any, Optional
import json
import time

# Entity type mapping for consistent tagging
# Extended with supply chain, incident, and event types
ENTITY_TAG_MAP = {
    # Core entities
    "person": "Person",
    "organization": "Organization",
    "amount": "Amount",
    "date": "Date",
    "location": "Location",
    "project": "Project",
    "invoice": "Invoice",
    "agreement": "Agreement",
    
    # Supply chain entities
    "shipment": "Shipment",
    "port": "Port",
    "supplier": "Supplier",
    "customer": "Customer",
    "warehouse": "Warehouse",
    "logistics_partner": "LogisticsPartner",
    "cargo": "Cargo",
    
    # Event & Incident entities
    "event": "Event",
    "incident": "Incident",
    "delay": "Delay",
    "congestion": "Congestion",
    "weather_event": "WeatherEvent",
    "disruption": "Disruption",
    
    # Additional entities
    "product": "Product",
    "category": "Category",
    "route": "Route"
}

# Relationship mapping rules - Extended for supply chain intelligence
RELATIONSHIP_MAP = {
    # Existing relationships
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
    
    # Supply chain relationships
    ("Organization", "Shipment"): "SENT",
    ("Shipment", "Organization"): "RECEIVED_BY",
    ("Shipment", "Location"): "LOCATED_AT",
    ("Shipment", "Port"): "ROUTED_THROUGH",
    ("Organization", "Supplier"): "SUPPLIED_BY",
    ("Organization", "Customer"): "SELLS_TO",
    ("Organization", "LogisticsPartner"): "USES_LOGISTICS",
    ("Supplier", "Organization"): "SUPPLIES_TO",
    ("Shipment", "Delay"): "EXPERIENCED",
    ("Delay", "WeatherEvent"): "CAUSED_BY",
    ("Delay", "Congestion"): "CAUSED_BY",
    ("Port", "Congestion"): "HAS",
    ("Organization", "Port"): "OPERATES_AT",
    ("Warehouse", "Location"): "LOCATED_AT",
    ("Cargo", "Shipment"): "PART_OF",
    ("Route", "Port"): "PASSES_THROUGH",
    
    # Incident relationships
    ("Incident", "Organization"): "AFFECTED",
    ("Incident", "Category"): "BELONGS_TO",
    ("Event", "Location"): "OCCURRED_AT",
    ("Event", "Organization"): "IMPACTED",
    ("Disruption", "Organization"): "DISRUPTED",
    ("WeatherEvent", "Location"): "OCCURRED_AT",
    ("Congestion", "Port"): "AFFECTED",
    
    # Generic fallback
    ("Entity", "Entity"): "RELATED_TO"
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

        # Track dynamically-created schema elements (per instance)
        self._dynamic_tags: set = set()
        self._dynamic_edges: set = set()
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

            # ----------------- SUPPLY CHAIN TAGS -----------------
            session.execute("""
                CREATE TAG IF NOT EXISTS Shipment(
                    name string,
                    status string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Port(
                    name string,
                    country string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Supplier(
                    name string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Customer(
                    name string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS LogisticsPartner(
                    name string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Warehouse(
                    name string,
                    capacity string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Cargo(
                    name string,
                    type string,
                    confidence float
                )
            """)

            # ----------------- EVENT/INCIDENT TAGS -----------------
            session.execute("""
                CREATE TAG IF NOT EXISTS Event(
                    name string,
                    type string,
                    severity string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Incident(
                    name string,
                    category string,
                    severity string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Delay(
                    name string,
                    duration string,
                    cause string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Congestion(
                    name string,
                    severity string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS WeatherEvent(
                    name string,
                    type string,
                    severity string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Disruption(
                    name string,
                    type string,
                    impact string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Category(
                    name string,
                    confidence float
                )
            """)

            session.execute("""
                CREATE TAG IF NOT EXISTS Route(
                    name string,
                    origin string,
                    destination string,
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
                CREATE EDGE IF NOT EXISTS LOCATED_AT(
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

            # ----------------- SUPPLY CHAIN EDGES -----------------
            session.execute("""
                CREATE EDGE IF NOT EXISTS SENT(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS RECEIVED_BY(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS ROUTED_THROUGH(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS SUPPLIED_BY(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS SUPPLIES_TO(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS SELLS_TO(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS USES_LOGISTICS(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS OPERATES_AT(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS PASSES_THROUGH(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS PART_OF(
                    confidence float
                )
            """)

            # ----------------- EVENT/INCIDENT EDGES -----------------
            session.execute("""
                CREATE EDGE IF NOT EXISTS EXPERIENCED(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS CAUSED_BY(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS HAS(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS AFFECTED(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS IMPACTED(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS DISRUPTED(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS BELONGS_TO(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS OCCURRED_AT(
                    confidence float
                )
            """)

            session.execute("""
                CREATE EDGE IF NOT EXISTS RELATED_TO(
                    confidence float
                )
            """)

            print("Graph schema created successfully. Creating indexes for MATCH queries...")
            time.sleep(3)  # Wait for schema to propagate
            
            # Create tag indexes for LOOKUP queries to work - Extended list
            tag_types = [
                'Person', 'Organization', 'Location', 'Project', 'Invoice', 'Agreement', 'Amount', 'Date',
                'Shipment', 'Port', 'Supplier', 'Customer', 'LogisticsPartner', 'Warehouse', 'Cargo',
                'Event', 'Incident', 'Delay', 'Congestion', 'WeatherEvent', 'Disruption', 'Category', 'Route'
            ]
            for tag in tag_types:
                try:
                    session.execute(f"CREATE TAG INDEX IF NOT EXISTS idx_{tag.lower()} ON `{tag}`()")
                except Exception as idx_err:
                    print(f"Index creation for {tag} skipped: {idx_err}")
            
            # Create edge indexes for ALL edge types - Extended list
            edge_types = [
                'WORKS_AT', 'LOCATED_IN', 'LOCATED_AT', 'WORKS_ON', 'MANAGES', 'PARTY_TO', 
                'SIGNED_ON', 'HAS_AMOUNT', 'EFFECTIVE_ON', 'HAS_VALUE', 'ISSUED', 'DUE_ON', 'BILLED_TO',
                'SENT', 'RECEIVED_BY', 'ROUTED_THROUGH', 'SUPPLIED_BY', 'SUPPLIES_TO', 'SELLS_TO',
                'USES_LOGISTICS', 'OPERATES_AT', 'PASSES_THROUGH', 'PART_OF',
                'EXPERIENCED', 'CAUSED_BY', 'HAS', 'AFFECTED', 'IMPACTED', 'DISRUPTED', 'BELONGS_TO',
                'OCCURRED_AT', 'RELATED_TO'
            ]
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

    def _ensure_tag_exists(self, session, tag: str) -> None:
        """Dynamically create a tag if it doesn't exist yet."""
        if tag in self._dynamic_tags:
            return
        if tag in ["Date", "Amount"]:
            session.execute(
                f'CREATE TAG IF NOT EXISTS `{tag}`(value string, confidence float)'
            )
        else:
            session.execute(
                f'CREATE TAG IF NOT EXISTS `{tag}`(name string, confidence float)'
            )
        # Index so LOOKUP works
        idx = f"idx_{tag.lower()}"
        session.execute(f"CREATE TAG INDEX IF NOT EXISTS `{idx}` ON `{tag}`()")
        self._dynamic_tags.add(tag)

    def _ensure_edge_exists(self, session, edge: str) -> None:
        """Dynamically create an edge type if it doesn't exist yet."""
        if edge in self._dynamic_edges:
            return
        session.execute(
            f'CREATE EDGE IF NOT EXISTS `{edge}`(confidence float)'
        )
        idx = f"idx_{edge.lower()}"
        session.execute(f"CREATE EDGE INDEX IF NOT EXISTS `{idx}` ON `{edge}`()")
        self._dynamic_edges.add(edge)

    def add_entity_node(
        self,
        entity_id: str,
        entity_type: str,
        entity_value: str,
        confidence: float = 1.0,
        extra_props: Dict[str, Any] = None
    ) -> bool:
        """Add entity node with standardized tagging – auto-creates missing tags"""
        session = self._get_session()
        if not session:
            return False

        try:
            session.execute(f"USE {self.space}")

            tag = ENTITY_TAG_MAP.get(entity_type.lower(), entity_type.capitalize())

            # Clean entity value: remove newlines and strip whitespace
            clean_value = entity_value.replace("\n", " ").strip()
            # Escape quotes for nGQL
            clean_value = clean_value.replace('"', '\\"')

            # Handle different tag schemas based on tag type
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

            result = session.execute(query)

            # If the tag doesn't exist, create it dynamically and retry
            if not result.is_succeeded() and "TagNotFound" in result.error_msg():
                print(f"   Auto-creating tag `{tag}` …")
                self._ensure_tag_exists(session, tag)
                time.sleep(1)  # let schema propagate
                session.execute(f"REBUILD TAG INDEX idx_{tag.lower()}")
                time.sleep(1)
                result = session.execute(query)

            if result.is_succeeded():
                print(f"Added {tag} node - {entity_value}")
                return True
            else:
                print(f"Node insert error: {result.error_msg()}")
                return False

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
        """Add relationship edge with intelligent type mapping and retry logic"""
        import time
        
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
                f'INSERT EDGE `{edge}`(confidence) '
                f'VALUES "{from_id}"->"{to_id}":({confidence})'
            )

            # Retry logic with dynamic edge creation
            max_retries = 3
            for attempt in range(max_retries):
                result = session.execute(query)

                if result.is_succeeded():
                    print(f"Added relationship {from_id} -[{edge}]-> {to_id}")
                    return True

                error_msg = result.error_msg()

                # Edge type doesn't exist → create it on the fly and retry
                if "EdgeNotFound" in error_msg:
                    print(f"   Auto-creating edge type `{edge}` …")
                    self._ensure_edge_exists(session, edge)
                    time.sleep(1)  # let schema propagate
                    try:
                        session.execute(f"REBUILD EDGE INDEX idx_{edge.lower()}")
                    except Exception:
                        pass
                    time.sleep(1)
                    # retry immediately
                    result = session.execute(query)
                    if result.is_succeeded():
                        print(f"Added relationship {from_id} -[{edge}]-> {to_id}")
                        return True
                    # fall through to next attempt

                # Atomic operation failure (race condition) → backoff & retry
                if "Atomic operation failed" in error_msg:
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue

                # Other errors or final attempt failed
                print(f"Edge insert error: {from_id} -[{edge}]-> {to_id}")
                print(f"   Error: {error_msg}")
                return False

            return False

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

    # Tags whose value property is called 'value' instead of 'name'
    _VALUE_TAGS = {'Date', 'Amount'}

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Dynamically discover all tags via SHOW TAGS and LOOKUP every one."""
        session = self._get_session()
        if not session:
            return []

        try:
            session.execute(f"USE {self.space}")

            # ── Discover every tag that exists in the space ──
            tag_result = session.execute("SHOW TAGS")
            tag_types: List[str] = []
            if tag_result.is_succeeded():
                for row in tag_result.as_primitive():
                    tag_name = row.get('Name') or row.get('name') or ''
                    if tag_name:
                        tag_types.append(tag_name)

            if not tag_types:
                print("No tags found in graph")
                return []

            entities = []
            for tag_type in tag_types:
                try:
                    # Choose the right property name
                    if tag_type in self._VALUE_TAGS:
                        q = f'LOOKUP ON `{tag_type}` YIELD id(vertex) as vid, properties(vertex).value as name'
                    else:
                        q = f'LOOKUP ON `{tag_type}` YIELD id(vertex) as vid, properties(vertex).name as name'

                    result = session.execute(q)

                    if result.is_succeeded():
                        for row in result.as_primitive():
                            name = row.get('name', '')
                            if name:
                                entities.append({
                                    "id": row.get('vid', ''),
                                    "name": name,
                                    "type": tag_type.lower(),
                                    "confidence": 0.8
                                })
                    # silently skip tags with no index
                except Exception as tag_err:
                    print(f"Query for {tag_type} failed: {tag_err}")
                    continue

            return entities

        except Exception as e:
            print(f"Get entities error: {e}")
            return []

        finally:
            session.release()

    @staticmethod
    def _vid_to_name(vid: str) -> str:
        """Convert vertex ID to readable name:  person_Rahul_Deshmukh → Rahul Deshmukh"""
        parts = vid.split('_', 1)
        name = parts[1] if len(parts) > 1 else vid
        return name.replace('_', ' ')

    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Dynamically discover all edge types via SHOW EDGES and LOOKUP every one."""
        session = self._get_session()
        if not session:
            return []

        try:
            session.execute(f"USE {self.space}")

            # ── Discover every edge type that exists in the space ──
            edge_result = session.execute("SHOW EDGES")
            edge_types: List[str] = []
            if edge_result.is_succeeded():
                for row in edge_result.as_primitive():
                    edge_name = row.get('Name') or row.get('name') or ''
                    if edge_name:
                        edge_types.append(edge_name)

            if not edge_types:
                print("No edge types found in graph")
                return []

            relationships = []
            for edge_type in edge_types:
                try:
                    result = session.execute(
                        f"LOOKUP ON `{edge_type}` YIELD src(edge) as src, dst(edge) as dst"
                    )

                    if result.is_succeeded():
                        for row in result.as_primitive():
                            src = row.get('src', '')
                            dst = row.get('dst', '')
                            if src and dst:
                                relationships.append({
                                    "source": self._vid_to_name(src),
                                    "target": self._vid_to_name(dst),
                                    "type": edge_type,
                                    "relationship_type": edge_type,
                                    "source_id": src,
                                    "target_id": dst,
                                })
                    # silently skip edges with no index
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
    import time

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

    # Small delay after entities to let indexes settle
    time.sleep(0.5)

    # Insert relationships in smaller batches with delays to avoid atomic conflicts
    batch_size = 10
    for i, r in enumerate(relationships):
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
        
        # Small delay every batch to reduce atomic conflicts
        if (i + 1) % batch_size == 0:
            time.sleep(0.05)

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


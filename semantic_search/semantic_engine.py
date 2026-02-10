# Semantic Search Engine
# Vector-based semantic search using Weaviate with embeddings
# Supports meaning-based queries, not just keyword matching

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import requests
import os

@dataclass
class SearchResult:
    """Semantic search result with metadata"""
    id: str
    content: str
    score: float
    entities: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "document"
    highlights: List[str] = field(default_factory=list)


class SemanticSearchEngine:
    """
    Advanced semantic search engine using Weaviate vector database.
    
    Features:
    - True vector similarity search using embeddings
    - Semantic understanding of queries
    - Entity-aware search
    - Multi-modal document support
    """
    
    def __init__(
        self, 
        weaviate_url: str = "http://localhost:8080",
        openai_api_key: Optional[str] = None
    ):
        self.url = weaviate_url.rstrip('/')
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._vector_store = None
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> bool:
        """Initialize and verify Weaviate connection"""
        try:
            response = requests.get(f"{self.url}/v1/.well-known/ready", timeout=5)
            if response.status_code == 200:
                print("✓ Semantic search engine connected to Weaviate")
                return True
            return False
        except Exception as e:
            print(f"⚠ Weaviate connection failed: {e}")
            return False
    
    def create_semantic_schema(self) -> bool:
        """
        Create schema for semantic search.
        Vector search is handled via core/vector_store.py with embeddings.
        This schema is for document storage and metadata filtering.
        """
        if not self.client:
            return False
            
        try:
            # Check if schema exists
            response = requests.get(f"{self.url}/v1/schema/SemanticDocument", timeout=10)
            if response.status_code == 200:
                print("✓ Semantic schema already exists")
                return True
            
            # Schema for document storage (vectors handled separately)
            schema = {
                "class": "SemanticDocument",
                "description": "Document for semantic search with vector embeddings",
                "vectorizer": "none",  # Vectors provided externally via embedding service
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Full document content for semantic search",
                        "indexSearchable": True,
                        "tokenization": "word"
                    },
                    {
                        "name": "summary",
                        "dataType": ["text"],
                        "description": "Document summary",
                        "indexSearchable": True
                    },
                    {
                        "name": "entities_json",
                        "dataType": ["text"],
                        "description": "JSON string of extracted entities"
                    },
                    {
                        "name": "document_id",
                        "dataType": ["text"],
                        "description": "Unique document identifier"
                    },
                    {
                        "name": "source_type",
                        "dataType": ["text"],
                        "description": "Type: pdf, email, report, invoice, etc."
                    },
                    {
                        "name": "category",
                        "dataType": ["text"],
                        "description": "Category: supply_chain, logistics, incident, contract, etc.",
                        "indexFilterable": True
                    },
                    {
                        "name": "keywords",
                        "dataType": ["text[]"],
                        "description": "Extracted keywords for search",
                        "indexSearchable": True
                    },
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "Document creation timestamp"
                    }
                ]
            }
            
            # Create schema
            response = requests.post(f"{self.url}/v1/schema", json=schema, timeout=15)
            
            if response.status_code in [200, 201]:
                print("✓ Semantic schema created successfully")
                return True
            else:
                print(f"⚠ Schema creation failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"⚠ Schema creation error: {e}")
            return False
    
    def store_document(
        self,
        content: str,
        document_id: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        source_type: str = "document",
        category: str = "general",
        keywords: Optional[List[str]] = None,
        summary: Optional[str] = None
    ) -> Optional[str]:
        """
        Store document for semantic search.
        
        Args:
            content: Full document text
            document_id: Unique identifier
            entities: Extracted entities list
            source_type: pdf, email, report, etc.
            category: supply_chain, logistics, incident, contract, etc.
            keywords: Optional keywords for hybrid search
            summary: Optional document summary
        
        Returns:
            UUID of stored document or None
        """
        if not self.client:
            return None
            
        try:
            # Auto-generate summary if not provided
            if not summary:
                summary = content[:500] + "..." if len(content) > 500 else content
            
            # Auto-extract keywords if not provided
            if not keywords:
                keywords = self._extract_keywords(content, entities)
            
            doc_object = {
                "class": "SemanticDocument",
                "properties": {
                    "content": content,
                    "summary": summary,
                    "entities_json": json.dumps(entities or []),
                    "document_id": document_id,
                    "source_type": source_type,
                    "category": category,
                    "keywords": keywords
                }
            }
            
            response = requests.post(
                f"{self.url}/v1/objects",
                json=doc_object,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                uuid = result.get("id", "")
                print(f"✓ Document stored: {document_id} (UUID: {uuid[:8]}...)")
                return uuid
            else:
                print(f"⚠ Store failed: {response.text[:100]}")
                return None
                
        except Exception as e:
            print(f"⚠ Store error: {e}")
            return None
    
    def _extract_keywords(
        self, 
        content: str, 
        entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Extract keywords from content and entities for hybrid search"""
        keywords = set()
        
        # Add entity values as keywords
        if entities:
            for entity in entities:
                value = entity.get('value') or entity.get('name', '')
                if value:
                    keywords.add(value.lower())
        
        # Add important domain terms from content
        supply_chain_terms = [
            'shipment', 'delay', 'congestion', 'logistics', 'supplier',
            'port', 'customs', 'warehouse', 'delivery', 'freight',
            'cargo', 'inventory', 'procurement', 'vendor', 'transport',
            'distribution', 'bottleneck', 'disruption', 'backlog'
        ]
        
        content_lower = content.lower()
        for term in supply_chain_terms:
            if term in content_lower:
                keywords.add(term)
        
        return list(keywords)[:20]  # Limit to 20 keywords
    
    @property
    def vector_store(self):
        """Lazy-load vector store for semantic search"""
        if self._vector_store is None:
            try:
                from core.vector_store import WeaviateVectorStore
                from core.embedding_service import HybridEmbedding
                
                embedding_service = HybridEmbedding()
                self._vector_store = WeaviateVectorStore(
                    weaviate_url=self.url,
                    embedding_service=embedding_service
                )
                self._vector_store.initialize()
            except Exception as e:
                print(f"⚠ Vector store initialization failed: {e}")
                self._vector_store = None
        return self._vector_store
    
    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        certainty: float = 0.7,
        category_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search using vector similarity.
        
        This finds documents by MEANING using embeddings, not keywords.
        
        Example:
        - Query: "Why are shipments getting delayed?"
        - Finds: "Port congestion", "Logistics bottleneck", "Customs clearance issues"
        
        Args:
            query: Natural language query
            limit: Maximum results to return
            certainty: Minimum similarity threshold (0-1)
            category_filter: Optional category filter
        
        Returns:
            List of SearchResult objects ranked by semantic similarity
        """
        if not self.client:
            return []
        
        # Use vector similarity search
        return self._vector_semantic_search(query, limit, category_filter)
    
    def _vector_semantic_search(
        self,
        query: str,
        limit: int = 10,
        category_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Semantic search using real vector embeddings.
        
        Uses OpenAI embeddings for true semantic similarity matching.
        """
        try:
            if self.vector_store is None:
                print("⚠ Vector store not available, returning empty results")
                return []
            
            # Build filters
            filters = {}
            if category_filter:
                filters["category"] = category_filter
            
            # Perform vector search
            core_results = self.vector_store.search(
                query=query,
                limit=limit,
                filters=filters if filters else None
            )
            
            # Convert to SearchResult format
            results = []
            for r in core_results:
                # Convert entities from Entity objects to dicts
                entities = []
                for e in r.entities:
                    if hasattr(e, '__dict__'):
                        entities.append({
                            'type': getattr(e, 'type', 'unknown'),
                            'value': getattr(e, 'value', ''),
                            'confidence': getattr(e, 'confidence', 1.0)
                        })
                    elif isinstance(e, dict):
                        entities.append(e)
                
                results.append(SearchResult(
                    id=r.id,
                    content=r.content,
                    score=r.score,
                    entities=entities,
                    metadata=r.metadata,
                    source_type=r.metadata.get('source_type', 'document'),
                    highlights=[]
                ))
            
            print(f"✓ Vector search returned {len(results)} results for: {query[:50]}...")
            return results
            
        except Exception as e:
            print(f"⚠ Vector search error: {e}")
            return []

    def _expand_query_semantically(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms for semantic-like matching.
        
        This is the key to achieving semantic search without vector embeddings.
        """
        # Semantic relationships (synonyms and related concepts)
        semantic_expansions = {
            # Shipment/Delay concepts
            'delay': ['delayed', 'congestion', 'bottleneck', 'backlog', 'hold', 'stuck', 'waiting', 'late', 'overdue', 'slow'],
            'delayed': ['delay', 'congestion', 'bottleneck', 'held', 'stuck'],
            'shipment': ['cargo', 'freight', 'delivery', 'package', 'consignment', 'goods', 'transport', 'shipping'],
            'shipping': ['shipment', 'cargo', 'freight', 'delivery', 'transport', 'logistics'],
            'congestion': ['overcrowding', 'traffic', 'blockage', 'jam', 'queue', 'pile-up', 'bottleneck', 'delay'],
            'bottleneck': ['congestion', 'constraint', 'delay', 'blockage', 'slowdown'],
            
            # Weather/Natural events
            'weather': ['rain', 'rainfall', 'storm', 'flood', 'monsoon', 'cyclone', 'hurricane', 'climate', 'natural'],
            'rain': ['rainfall', 'precipitation', 'monsoon', 'downpour', 'shower', 'wet', 'flooding'],
            'rainfall': ['rain', 'precipitation', 'monsoon', 'weather'],
            'flood': ['flooding', 'inundation', 'rain', 'monsoon', 'water'],
            
            # Supply chain
            'supply': ['supplier', 'procurement', 'sourcing', 'vendor', 'provision', 'stock'],
            'chain': ['supply chain', 'logistics', 'network', 'pipeline'],
            'logistics': ['transport', 'shipping', 'freight', 'warehouse', 'distribution', 'delivery', 'supply chain'],
            'supplier': ['vendor', 'provider', 'source', 'manufacturer', 'partner'],
            
            # Location concepts  
            'port': ['harbor', 'dock', 'terminal', 'wharf', 'pier', 'seaport', 'maritime'],
            'customs': ['clearance', 'inspection', 'duty', 'tariff', 'import', 'export', 'border'],
            
            # Incident/Security concepts
            'cyber': ['security', 'breach', 'hack', 'attack', 'vulnerability', 'malware', 'digital'],
            'attack': ['breach', 'intrusion', 'exploit', 'compromise', 'incident', 'hack'],
            'security': ['cyber', 'breach', 'protection', 'vulnerability', 'safety'],
            'breach': ['attack', 'hack', 'intrusion', 'compromise', 'leak', 'security'],
            'cloud': ['aws', 'azure', 'gcp', 'infrastructure', 'server', 'saas', 'iaas', 'hosting'],
            
            # Business/Impact concepts
            'company': ['organization', 'firm', 'corporation', 'enterprise', 'business', 'vendor'],
            'companies': ['organizations', 'firms', 'corporations', 'enterprises', 'businesses'],
            'affected': ['impacted', 'influenced', 'disrupted', 'hit', 'suffered', 'experienced'],
            'impact': ['affect', 'influence', 'disrupt', 'effect', 'consequence'],
            'disruption': ['disrupted', 'interruption', 'breakdown', 'failure', 'stoppage'],
            
            # BUSINESS DOCUMENT EXPANSIONS
            # People/Employment
            'works': ['employed', 'working', 'employee', 'staff', 'team', 'member', 'personnel'],
            'employee': ['staff', 'worker', 'personnel', 'team member', 'associate', 'employed'],
            'staff': ['employee', 'worker', 'personnel', 'team', 'workforce'],
            'manager': ['lead', 'head', 'director', 'supervisor', 'chief', 'executive'],
            'director': ['manager', 'head', 'chief', 'executive', 'lead', 'officer'],
            'ceo': ['chief executive', 'president', 'head', 'director', 'leader', 'founder'],
            'founder': ['creator', 'owner', 'entrepreneur', 'ceo', 'established'],
            
            # Company/Organization
            'mahindra': ['tech mahindra', 'mahindra group', 'mahindra & mahindra', 'm&m'],
            'tata': ['tata group', 'tcs', 'tata consultancy', 'tata motors', 'tata steel'],
            'reliance': ['reliance industries', 'jio', 'reliance retail', 'ril'],
            'infosys': ['infy', 'infosys technologies', 'infosys limited'],
            'wipro': ['wipro technologies', 'wipro limited'],
            
            # Contract/Legal terms
            'contract': ['agreement', 'deal', 'terms', 'engagement', 'arrangement', 'pact'],
            'agreement': ['contract', 'deal', 'terms', 'understanding', 'arrangement', 'mou'],
            'invoice': ['bill', 'receipt', 'payment', 'billing', 'charge', 'amount due'],
            'payment': ['pay', 'remittance', 'transaction', 'settlement', 'amount', 'fee'],
            'terms': ['conditions', 'clauses', 'provisions', 'stipulations', 'requirements'],
            'legal': ['law', 'contract', 'compliance', 'regulation', 'statutory', 'binding'],
            
            # Project/Work
            'project': ['initiative', 'program', 'engagement', 'assignment', 'work', 'task'],
            'initiative': ['project', 'program', 'effort', 'drive', 'campaign'],
            'deliverable': ['output', 'result', 'milestone', 'product', 'completion'],
            'deadline': ['due date', 'timeline', 'schedule', 'target date', 'completion date'],
            
            # Financial terms
            'revenue': ['income', 'sales', 'earnings', 'turnover', 'receipts'],
            'cost': ['expense', 'expenditure', 'spending', 'outlay', 'price'],
            'profit': ['gain', 'earnings', 'margin', 'return', 'surplus'],
            'budget': ['allocation', 'funds', 'funding', 'financial plan', 'appropriation'],
            'investment': ['funding', 'capital', 'stake', 'financing', 'injection'],
            
            # Roles/Positions
            'lead': ['head', 'manager', 'chief', 'principal', 'senior'],
            'senior': ['lead', 'principal', 'chief', 'head', 'experienced'],
            'consultant': ['advisor', 'specialist', 'expert', 'professional'],
            'analyst': ['researcher', 'specialist', 'examiner', 'evaluator'],
            'engineer': ['developer', 'technical', 'specialist', 'programmer'],
            
            # Question words - expand to related concepts
            'who': ['person', 'employee', 'staff', 'individual', 'member', 'name'],
            'works': ['employed', 'working', 'employee', 'job', 'position', 'role'],
            'work': ['employed', 'job', 'position', 'role', 'occupation'],
            'why': ['cause', 'reason', 'because', 'due'],
            'which': ['what', 'who', 'that'],
            'how': ['way', 'method', 'process'],
            'what': ['which', 'details', 'information', 'specifics'],
            'where': ['location', 'place', 'site', 'address', 'office'],
            'when': ['date', 'time', 'period', 'timeline', 'schedule'],
        }
        
        # Start with original query terms - these are MOST important
        query_lower = query.lower()
        original_terms = query_lower.split()
        
        # Use a priority system: original terms get boosted
        expanded_terms = set()
        
        # Add original terms with high priority (will appear first in search)
        for term in original_terms:
            if len(term) > 2:  # Skip very short words like 'in', 'at'
                expanded_terms.add(term)
        
        # Add ONLY direct semantic expansions (no partial matching)
        # This prevents "company" from pulling in weather-related terms
        for term in original_terms:
            if term in semantic_expansions:
                # Only add most relevant expansions (limit to 3)
                relevant_expansions = semantic_expansions[term][:3]
                expanded_terms.update(relevant_expansions)
        
        # Filter out common words that don't help search
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                      'ought', 'used', 'to', 'of', 'for', 'with', 'about', 'against',
                      'between', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
                      'further', 'then', 'once', 'here', 'there', 'all', 'each', 'few',
                      'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
                      'than', 'too', 'very', 'just', 'also'}
        
        expanded_terms = {t for t in expanded_terms if t not in stop_words}
        
        return list(expanded_terms)
    
    def _fallback_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Fallback to basic text search when vector search unavailable"""
        try:
            graphql_query = {
                "query": f"""
                {{
                    Get {{
                        SemanticDocument(limit: {limit}) {{
                            content
                            summary
                            entities_json
                            document_id
                            source_type
                            category
                            keywords
                            _additional {{
                                id
                            }}
                        }}
                    }}
                }}
                """
            }
            
            response = requests.post(
                f"{self.url}/v1/graphql",
                json=graphql_query,
                timeout=30
            )
            
            if response.status_code != 200:
                # Try with ExtractedDocument class (legacy)
                return self._legacy_search(query, limit)
            
            data = response.json()
            documents = data.get("data", {}).get("Get", {}).get("SemanticDocument", [])
            
            # Score documents based on keyword matching
            results = []
            query_terms = set(query.lower().split())
            
            for doc in documents:
                content = doc.get("content", "").lower()
                
                # Calculate simple relevance score
                matches = sum(1 for term in query_terms if term in content)
                score = matches / max(len(query_terms), 1)
                
                if score > 0 or not query_terms:
                    entities = []
                    try:
                        entities = json.loads(doc.get("entities_json", "[]"))
                    except:
                        pass
                    
                    results.append(SearchResult(
                        id=doc.get("_additional", {}).get("id", ""),
                        content=doc.get("content", ""),
                        score=min(score + 0.3, 1.0),  # Boost base score
                        entities=entities,
                        metadata={
                            "summary": doc.get("summary", ""),
                            "category": doc.get("category", "")
                        },
                        source_type=doc.get("source_type", "document"),
                        highlights=self._extract_highlights(doc.get("content", ""), query)
                    ))
            
            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"⚠ Fallback search error: {e}")
            return self._legacy_search(query, limit)
    
    def _legacy_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search using legacy ExtractedDocument schema"""
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
            
            response = requests.post(
                f"{self.url}/v1/graphql",
                json=graphql_query,
                timeout=30
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            documents = data.get("data", {}).get("Get", {}).get("ExtractedDocument", [])
            
            results = []
            query_lower = query.lower()
            
            for doc in documents:
                content = doc.get("content", "").lower()
                entities_str = doc.get("entities", "").lower()
                
                # Check for semantic relevance using synonyms/related terms
                if self._is_semantically_related(query_lower, content, entities_str):
                    entities = []
                    try:
                        entities = json.loads(doc.get("entities", "[]"))
                    except:
                        pass
                    
                    results.append(SearchResult(
                        id=doc.get("_additional", {}).get("id", ""),
                        content=doc.get("content", ""),
                        score=0.75,
                        entities=entities,
                        source_type=doc.get("sourceType", "document"),
                        highlights=self._extract_highlights(doc.get("content", ""), query)
                    ))
            
            return results
            
        except Exception as e:
            print(f"⚠ Legacy search error: {e}")
            return []
    
    def _is_semantically_related(
        self, 
        query: str, 
        content: str, 
        entities: str
    ) -> bool:
        """Check if content is semantically related to query using synonym mapping"""
        
        # Define semantic relationships (synonyms and related concepts)
        semantic_map = {
            # Shipment/Delay concepts
            'delay': ['congestion', 'bottleneck', 'backlog', 'hold', 'stuck', 'waiting', 'late', 'overdue'],
            'shipment': ['cargo', 'freight', 'delivery', 'package', 'consignment', 'goods', 'transport'],
            'congestion': ['overcrowding', 'traffic', 'blockage', 'jam', 'queue', 'pile-up'],
            
            # Weather/Natural events
            'weather': ['rain', 'rainfall', 'storm', 'flood', 'monsoon', 'cyclone', 'hurricane', 'climate'],
            'rain': ['rainfall', 'precipitation', 'monsoon', 'downpour', 'shower', 'wet'],
            
            # Supply chain
            'supply chain': ['logistics', 'procurement', 'sourcing', 'vendor', 'supplier', 'distribution'],
            'logistics': ['transport', 'shipping', 'freight', 'warehouse', 'distribution', 'delivery'],
            
            # Location concepts  
            'port': ['harbor', 'dock', 'terminal', 'wharf', 'pier', 'seaport'],
            'customs': ['clearance', 'inspection', 'duty', 'tariff', 'import', 'export'],
            
            # Incident/Security concepts
            'cyber': ['security', 'breach', 'hack', 'attack', 'vulnerability', 'malware'],
            'attack': ['breach', 'intrusion', 'exploit', 'compromise', 'incident'],
            'cloud': ['aws', 'azure', 'gcp', 'infrastructure', 'server', 'saas', 'iaas'],
            
            # Business concepts
            'company': ['organization', 'firm', 'corporation', 'enterprise', 'business'],
            'affected': ['impacted', 'influenced', 'disrupted', 'hit', 'suffered'],
        }
        
        combined_text = f"{content} {entities}"
        query_terms = query.split()
        
        for term in query_terms:
            # Direct match
            if term in combined_text:
                return True
            
            # Synonym match
            related_terms = semantic_map.get(term, [])
            for related in related_terms:
                if related in combined_text:
                    return True
        
        return False
    
    def _extract_highlights(self, content: str, query: str) -> List[str]:
        """Extract relevant snippets from content based on query"""
        highlights = []
        query_terms = query.lower().split()
        
        # Split content into sentences
        sentences = content.replace('\n', '. ').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_lower = sentence.lower()
            
            # Check if sentence contains any query terms or related concepts
            for term in query_terms:
                if term in sentence_lower:
                    highlights.append(sentence)
                    break
        
        # Return top 3 most relevant highlights
        return highlights[:3]
    
    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.75
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity and keyword matching.
        
        Args:
            query: Search query
            limit: Maximum results
            alpha: Balance between vector (1.0) and keyword (0.0) search
        
        Returns:
            Combined search results
        """
        if not self.client:
            return []
        
        try:
            # Use Weaviate's hybrid search
            graphql_query = {
                "query": f"""
                {{
                    Get {{
                        SemanticDocument(
                            hybrid: {{
                                query: "{query}"
                                alpha: {alpha}
                            }}
                            limit: {limit}
                        ) {{
                            content
                            summary
                            entities_json
                            document_id
                            source_type
                            category
                            _additional {{
                                id
                                score
                            }}
                        }}
                    }}
                }}
                """
            }
            
            response = requests.post(
                f"{self.url}/v1/graphql",
                json=graphql_query,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "errors" in data:
                    return self.semantic_search(query, limit)
                
                documents = data.get("data", {}).get("Get", {}).get("SemanticDocument", [])
                
                results = []
                for doc in documents:
                    additional = doc.get("_additional", {})
                    
                    entities = []
                    try:
                        entities = json.loads(doc.get("entities_json", "[]"))
                    except:
                        pass
                    
                    results.append(SearchResult(
                        id=additional.get("id", ""),
                        content=doc.get("content", ""),
                        score=additional.get("score", 0.5),
                        entities=entities,
                        metadata={
                            "summary": doc.get("summary", ""),
                            "category": doc.get("category", "")
                        },
                        source_type=doc.get("source_type", "document"),
                        highlights=self._extract_highlights(doc.get("content", ""), query)
                    ))
                
                return results
            
            return self.semantic_search(query, limit)
            
        except Exception as e:
            print(f"⚠ Hybrid search error: {e}")
            return self.semantic_search(query, limit)


# Convenience functions
def semantic_search(
    query: str,
    weaviate_url: str = "http://localhost:8080",
    limit: int = 10,
    category: Optional[str] = None
) -> List[SearchResult]:
    """
    Perform semantic search on documents.
    
    Example:
        results = semantic_search("Why are shipments getting delayed?")
        # Returns documents about port congestion, logistics bottleneck, etc.
    """
    engine = SemanticSearchEngine(weaviate_url)
    return engine.semantic_search(query, limit, category_filter=category)


def search_with_graph_context(
    query: str,
    weaviate_url: str = "http://localhost:8080",
    nebula_host: str = "127.0.0.1",
    nebula_port: int = 9669,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Combined semantic search + graph context for comprehensive results.
    
    First finds relevant documents semantically, then enriches with graph relationships.
    """
    from .graph_traversal import GraphTraversal
    
    # Step 1: Semantic search
    engine = SemanticSearchEngine(weaviate_url)
    search_results = engine.semantic_search(query, limit)
    
    # Step 2: Extract entities from search results
    all_entities = []
    for result in search_results:
        all_entities.extend(result.entities)
    
    # Step 3: Get graph context for entities
    graph = GraphTraversal(nebula_host, nebula_port)
    
    graph_context = []
    unique_entities = {e.get('value', e.get('name', '')): e for e in all_entities}
    
    for entity_name, entity in unique_entities.items():
        entity_type = entity.get('type', 'unknown')
        relationships = graph.get_entity_relationships(entity_name, entity_type)
        if relationships:
            graph_context.append({
                "entity": entity_name,
                "type": entity_type,
                "relationships": relationships
            })
    
    return {
        "query": query,
        "semantic_results": [
            {
                "content": r.content,
                "score": r.score,
                "highlights": r.highlights,
                "source_type": r.source_type
            }
            for r in search_results
        ],
        "graph_context": graph_context,
        "entities_found": list(unique_entities.keys())
    }

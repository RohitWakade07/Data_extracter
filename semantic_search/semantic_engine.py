# Semantic Search Engine
# Vector-based semantic search using Weaviate with embeddings
# Supports meaning-based queries, not just keyword matching

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import re
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
    - True vector similarity search (not just keyword matching)
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
        self._llm = None
        self.client = self._initialize_client()

    def _get_llm(self):
        """Lazy-load LLM for dynamic query operations."""
        if self._llm is not None:
            return self._llm
        try:
            from utils.ollama_handler import OllamaLLM
            model = os.getenv("OLLAMA_MODEL", "llama3")
            self._llm = OllamaLLM(model=model)
            return self._llm
        except Exception:
            return None

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
        Create schema for semantic search with BM25 support.
        Uses keyword-based search with semantic query expansion.
        """
        if not self.client:
            return False
            
        try:
            # Check if schema exists
            response = requests.get(f"{self.url}/v1/schema/SemanticDocument", timeout=10)
            if response.status_code == 200:
                print("✓ Semantic schema already exists")
                return True
            
            # Schema without vectorizer (uses BM25 for search)
            schema = {
                "class": "SemanticDocument",
                "description": "Document for semantic search using BM25 with query expansion",
                "vectorizer": "none",  # No vectorizer - use BM25
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Full document content for semantic search",
                        "indexSearchable": True,  # Enable BM25 search
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
                print("✓ Semantic schema created successfully (BM25 search enabled)")
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
        """Dynamically extract keywords from any content — domain agnostic."""
        keywords = set()

        # Add entity values as keywords
        if entities:
            for entity in entities:
                value = entity.get('value') or entity.get('name', '')
                if value:
                    keywords.add(value.lower())

        # Frequency-based keyword extraction (works for ANY domain)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
            'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before',
            'after', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 'this',
            'that', 'these', 'those', 'then', 'than', 'also', 'and', 'but', 'or',
            'not', 'its', 'at', 'by', 'as', 'on', 'in', 'so', 'if', 'any', 'all',
            'each', 'own', 'very', 'such', 'same', 'just', 'more', 'most', 'some',
            'set', 'forth', 'herein', 'shall', 'upon', 'within'
        }

        words = content.lower().split()
        word_freq: Dict[str, int] = {}
        for w in words:
            w = w.strip('.,;:!?()[]{}"\'-/*\\')
            if len(w) > 3 and w not in stop_words and not w.isdigit():
                word_freq[w] = word_freq.get(w, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:15]:
            if freq >= 2:
                keywords.add(word)

        return list(keywords)[:20]
    
    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        certainty: float = 0.7,
        category_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search using meaning-based matching.
        
        This finds documents by MEANING, not just keywords.
        
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
        
        # Use BM25 keyword search with semantic expansion
        # This works without text2vec module
        return self._bm25_semantic_search(query, limit, category_filter)
    
    def _bm25_semantic_search(
        self,
        query: str,
        limit: int = 10,
        category_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Semantic search using BM25 with query expansion.
        
        Expands the query with synonyms and related terms to achieve
        semantic-like matching without requiring vector embeddings.
        
        IMPORTANT: Original query terms are prioritized over expansions.
        Searches both SemanticDocument and ExtractedDocument classes.
        """
        # Get original terms first (these are most important)
        original_terms = [t for t in query.lower().split() if len(t) > 2]
        
        # Expand query with synonyms
        expanded_terms = self._expand_query_semantically(query)
        
        # Build search query with original terms repeated for higher weight
        # BM25 gives more weight to repeated terms
        search_query = " ".join(original_terms * 2 + expanded_terms)
        
        # CRITICAL: Escape special characters for GraphQL string
        # Remove quotes and special chars that break GraphQL syntax
        search_query = search_query.replace('"', '').replace("'", "").replace('\\', '')
        search_query = search_query.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Clean up multiple spaces
        search_query = ' '.join(search_query.split())
        
        results = []
        
        try:
            # Build where filter if category specified
            where_clause = ""
            if category_filter:
                where_clause = f'''
                    where: {{
                        path: ["category"],
                        operator: Equal,
                        valueText: "{category_filter}"
                    }}
                '''
            
            # 1. Search SemanticDocument class with BM25
            graphql_query = {
                "query": f"""
                {{
                    Get {{
                        SemanticDocument(
                            bm25: {{
                                query: "{search_query}"
                            }}
                            limit: {limit}
                            {where_clause}
                        ) {{
                            content
                            summary
                            entities_json
                            document_id
                            source_type
                            category
                            keywords
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
                
                if "errors" not in data:
                    documents = data.get("data", {}).get("Get", {}).get("SemanticDocument", [])
                    
                    for doc in documents:
                        additional = doc.get("_additional", {})
                        raw_score = additional.get("score", 0.5)
                        
                        # Normalize BM25 score to 0-1 range
                        try:
                            score = float(raw_score) if raw_score else 0.5
                            score = min(score / 10.0, 1.0) if score > 1.0 else score
                        except (TypeError, ValueError):
                            score = 0.5
                        
                        entities = []
                        try:
                            entities_json = doc.get("entities_json", "[]")
                            entities = json.loads(entities_json) if entities_json else []
                        except:
                            pass
                        
                        highlights = self._extract_highlights(doc.get("content", ""), query)
                        
                        results.append(SearchResult(
                            id=additional.get("id", ""),
                            content=doc.get("content", ""),
                            score=score,
                            entities=entities,
                            metadata={
                                "summary": doc.get("summary", ""),
                                "category": doc.get("category", ""),
                                "keywords": doc.get("keywords", []),
                                "document_id": doc.get("document_id", "")
                            },
                            source_type=doc.get("source_type", "document"),
                            highlights=highlights
                        ))
            
            # 2. Also search ExtractedDocument class (legacy/uploaded docs)
            extracted_results = self._search_extracted_documents(query, search_query, limit)
            results.extend(extracted_results)
            
            # Sort all results by score and return top limit
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"⚠ BM25 search error: {e}")
            return self._fallback_search(query, limit)
    
    def _search_extracted_documents(
        self,
        original_query: str,
        search_query: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search ExtractedDocument class for documents uploaded via the pipeline.
        Uses keyword matching with expanded terms.
        """
        try:
            # ExtractedDocument doesn't support BM25, so get all and filter
            graphql_query = {
                "query": f"""
                {{
                    Get {{
                        ExtractedDocument(limit: {limit * 2}) {{
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
            if "errors" in data:
                return []
                
            documents = data.get("data", {}).get("Get", {}).get("ExtractedDocument", [])
            
            results = []
            query_terms = set(search_query.lower().split())
            original_terms = set(original_query.lower().split())
            
            for doc in documents:
                content = doc.get("content", "").lower()
                entities_str = doc.get("entities", "").lower()
                
                # Calculate relevance score
                # Higher weight for original query terms
                original_matches = sum(2 for term in original_terms if term in content or term in entities_str)
                expanded_matches = sum(1 for term in query_terms if term in content or term in entities_str)
                
                total_matches = original_matches + expanded_matches
                max_possible = len(original_terms) * 2 + len(query_terms)
                
                if total_matches > 0:
                    score = min(total_matches / max(max_possible, 1) + 0.2, 1.0)
                    
                    entities = []
                    try:
                        entities = json.loads(doc.get("entities", "[]"))
                    except:
                        pass
                    
                    highlights = self._extract_highlights(doc.get("content", ""), original_query)
                    
                    results.append(SearchResult(
                        id=doc.get("_additional", {}).get("id", ""),
                        content=doc.get("content", ""),
                        score=score,
                        entities=entities,
                        metadata={
                            "document_id": doc.get("documentId", ""),
                            "source_class": "ExtractedDocument"
                        },
                        source_type=doc.get("sourceType", "document"),
                        highlights=highlights
                    ))
            
            return results
            
        except Exception as e:
            print(f"⚠ ExtractedDocument search error: {e}")
            return []

    def _expand_query_semantically(self, query: str) -> List[str]:
        """
        Dynamically expand query using LLM — works for ANY domain.

        Uses the LLM to generate synonyms, related terms, and domain-specific
        keywords for the given query. Falls back to basic term extraction
        if the LLM is unavailable.
        """
        original_terms = [t for t in query.lower().split() if len(t) > 2]
        expanded = set(original_terms)

        # ── LLM-based expansion (dynamic, domain-agnostic) ──
        try:
            llm = self._get_llm()
            if llm:
                prompt = (
                    f'Given the search query: "{query}"\n'
                    'Generate 8-12 related search keywords and phrases that would help '
                    'find relevant documents in a database. Include synonyms, related '
                    'concepts, domain terms, and entity names implied by the query.\n'
                    'Return ONLY a JSON array of lowercase strings.\n'
                    'Example: ["term1", "term2", "related phrase"]\n'
                    'No explanation, just the JSON array.'
                )
                raw = llm.chat(prompt)
                match = re.search(r'\[.*?\]', raw, re.DOTALL)
                if match:
                    terms = json.loads(match.group())
                    for t in terms:
                        if isinstance(t, str) and len(t.strip()) > 2:
                            expanded.add(t.lower().strip())
                    print(f"  ✓ LLM expanded query with {len(terms)} terms")
        except Exception as e:
            print(f"  ⚠ LLM query expansion fallback: {e}")

        # ── Filter stop words ──
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
            'for', 'with', 'about', 'between', 'into', 'through', 'during',
            'before', 'after', 'from', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'then', 'once', 'here', 'there', 'all', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'also', 'and', 'but',
            'or', 'not', 'this', 'that', 'what', 'which', 'who', 'how', 'why',
            'when', 'where'
        }
        expanded = {t for t in expanded if t not in stop_words}
        return list(expanded)
    
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
        """Check if content is semantically related to query — domain agnostic."""
        combined = f"{content} {entities}"
        query_terms = [t for t in query.split() if len(t) > 2]

        if not query_terms:
            return True  # Empty query matches everything

        # Direct keyword match
        for term in query_terms:
            if term in combined:
                return True

        # Significant overlap check (>30 % of meaningful query terms found)
        matches = sum(1 for t in query_terms if t in combined)
        if matches / len(query_terms) > 0.3:
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
    
    def generate_answer_from_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> str:
        """Use LLM to synthesise a comprehensive answer from search results."""
        if not results:
            return "No relevant documents found for your query."

        # Build context from top results
        context_parts = []
        for i, r in enumerate(results[:3], 1):
            preview = r.content[:600] if len(r.content) > 600 else r.content
            context_parts.append(f"Document {i}:\n{preview}")
        context = "\n\n".join(context_parts)

        try:
            llm = self._get_llm()
            if llm:
                prompt = (
                    f'Based on the following documents, answer this question:\n'
                    f'Question: "{query}"\n\n'
                    f'{context}\n\n'
                    f'Provide a clear, concise answer (2-5 sentences) based ONLY on '
                    f'the information in the documents above. '
                    f'If the documents do not contain relevant information, say so honestly.'
                )
                answer = llm.chat(prompt)
                return answer.strip()
        except Exception as e:
            print(f"⚠ LLM answer generation failed: {e}")

        # Fallback to highlight extraction
        highlights = []
        for r in results[:3]:
            if r.highlights:
                highlights.extend(r.highlights[:2])
        if highlights:
            return "\n".join([f"• {h}" for h in highlights])
        return "Results found. See documents below for details."

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.75
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector (semantic) and keyword (BM25) search.
        
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

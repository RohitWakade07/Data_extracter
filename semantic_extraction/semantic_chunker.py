# Semantic Chunker — Vector Embedding Layer
#
# Splits unstructured text into overlapping semantic chunks,
# embeds them with sentence-transformers, and stores the vectors
# in Weaviate for later retrieval.
#
# Key ideas:
#   - Overlapping windows preserve cross-sentence context
#   - Embeddings capture *meaning*, not just keywords
#   - Similarity search lets us pull the most relevant chunks
#     for any entity or relationship query

import re
import json
import hashlib
import numpy as np
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SemanticChunk:
    """A segment of text with its vector embedding."""
    chunk_id: str
    text: str
    embedding: List[float] = field(default_factory=list)
    sentence_indices: List[int] = field(default_factory=list)
    entity_hints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sentence splitter (abbreviation-aware)
# ---------------------------------------------------------------------------

_ABBREVS = re.compile(
    r'(?:Pvt|Ltd|Inc|Corp|LLP|Mr|Mrs|Ms|Dr|Jr|Sr|St|Ave|Blvd|No|vs|etc|approx|dept|govt|est)\.',
    re.IGNORECASE,
)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, respecting abbreviations."""
    # Collapse soft line-breaks inside paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    sentences: List[str] = []
    for para in paragraphs:
        para = ' '.join(para.split())
        if not para:
            continue
        # Protect abbreviation periods
        protected = _ABBREVS.sub(lambda m: m.group().replace('.', '<DOT>'), para)
        # Split on real sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+', protected)
        sentences.extend(p.replace('<DOT>', '.').strip() for p in parts if p.strip())
    return sentences


# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_EMBED_MODEL = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality


def _get_embed_model():
    """Lazy-load the sentence-transformer model (first call takes ~2 s)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"⏳ Loading embedding model '{_EMBED_MODEL_NAME}' …")
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
        print(f"✓ Embedding model ready  (dim={_EMBED_MODEL.get_sentence_embedding_dimension()})")
    return _EMBED_MODEL


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Return (N, dim) float32 array of embeddings for *texts*."""
    model = _get_embed_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """
    Chunk text into overlapping windows of sentences, embed each chunk.

    Parameters
    ----------
    window_size : int
        Number of sentences per chunk (default 3).
    overlap : int
        Number of overlapping sentences between consecutive chunks (default 1).
    """

    def __init__(self, window_size: int = 3, overlap: int = 1):
        self.window_size = window_size
        self.overlap = overlap

    def chunk_and_embed(self, text: str, doc_id: str = "doc") -> List[SemanticChunk]:
        """
        Split *text* → sentences → overlapping chunks → embed.

        Returns list of SemanticChunk with populated embeddings.
        """
        sentences = split_sentences(text)
        if not sentences:
            return []

        # Build overlapping windows
        step = max(1, self.window_size - self.overlap)
        windows: List[Tuple[List[int], str]] = []
        for start in range(0, len(sentences), step):
            end = min(start + self.window_size, len(sentences))
            indices = list(range(start, end))
            window_text = " ".join(sentences[i] for i in indices)
            windows.append((indices, window_text))
            if end >= len(sentences):
                break

        # Embed all chunks in one batch
        texts = [w[1] for w in windows]
        embeddings = embed_texts(texts)

        # Build SemanticChunk objects
        chunks: List[SemanticChunk] = []
        for i, ((indices, window_text), emb) in enumerate(zip(windows, embeddings)):
            chunk_hash = hashlib.md5(window_text.encode()).hexdigest()[:10]
            chunk = SemanticChunk(
                chunk_id=f"{doc_id}_chunk_{i}_{chunk_hash}",
                text=window_text,
                embedding=emb.tolist(),
                sentence_indices=indices,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "sentence_range": f"{indices[0]}-{indices[-1]}",
                    "char_len": len(window_text),
                },
            )
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # In-memory similarity search (no Weaviate needed for extraction)
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))

    def find_relevant_chunks(
        self,
        query: str,
        chunks: List[SemanticChunk],
        top_k: int = 5,
    ) -> List[Tuple[SemanticChunk, float]]:
        """
        Return the *top_k* chunks most semantically similar to *query*.
        """
        if not chunks:
            return []
        q_emb = embed_texts([query])[0]
        scored: List[Tuple[SemanticChunk, float]] = []
        for chunk in chunks:
            c_emb = np.array(chunk.embedding, dtype=np.float32)
            sim = self.cosine_similarity(q_emb, c_emb)
            scored.append((chunk, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def find_entity_context(
        self,
        entity_value: str,
        chunks: List[SemanticChunk],
        top_k: int = 3,
    ) -> str:
        """
        Build a focused context window for a specific entity by finding
        chunks that are semantically close to it.
        """
        relevant = self.find_relevant_chunks(entity_value, chunks, top_k)
        context_parts = [chunk.text for chunk, _ in relevant]
        return " … ".join(context_parts)


# ---------------------------------------------------------------------------
# Weaviate vector storage (optional — for persistent retrieval)
# ---------------------------------------------------------------------------

class WeaviateChunkStore:
    """
    Store semantic chunks in Weaviate with their embeddings for
    persistent vector search.  Falls back gracefully if Weaviate is
    not running.
    """

    SCHEMA_CLASS = "SemanticChunk"

    def __init__(self, weaviate_url: str = "http://localhost:8080"):
        self.url = weaviate_url.rstrip("/")
        self._ready = self._check_ready()
        if self._ready:
            self._ensure_schema()

    # ── connectivity ──────────────────────────────────────────────
    def _check_ready(self) -> bool:
        try:
            r = requests.get(f"{self.url}/v1/.well-known/ready", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ── schema ────────────────────────────────────────────────────
    def _ensure_schema(self):
        try:
            r = requests.get(f"{self.url}/v1/schema/{self.SCHEMA_CLASS}", timeout=5)
            if r.status_code == 200:
                return  # already exists
            schema = {
                "class": self.SCHEMA_CLASS,
                "description": "Semantic text chunk with embedding",
                "vectorizer": "none",
                "properties": [
                    {"name": "chunk_id",   "dataType": ["text"]},
                    {"name": "text",       "dataType": ["text"], "indexSearchable": True, "tokenization": "word"},
                    {"name": "doc_id",     "dataType": ["text"]},
                    {"name": "chunk_idx",  "dataType": ["int"]},
                    {"name": "sent_range", "dataType": ["text"]},
                ],
            }
            requests.post(f"{self.url}/v1/schema", json=schema, timeout=10)
        except Exception as e:
            print(f"⚠ Weaviate schema setup: {e}")

    # ── store ─────────────────────────────────────────────────────
    def store_chunks(self, chunks: List[SemanticChunk]) -> int:
        """Batch-store chunks with vectors. Returns count stored."""
        if not self._ready:
            return 0
        stored = 0
        for chunk in chunks:
            try:
                obj = {
                    "class": self.SCHEMA_CLASS,
                    "vector": chunk.embedding,
                    "properties": {
                        "chunk_id":   chunk.chunk_id,
                        "text":       chunk.text,
                        "doc_id":     chunk.metadata.get("doc_id", ""),
                        "chunk_idx":  chunk.metadata.get("chunk_index", 0),
                        "sent_range": chunk.metadata.get("sentence_range", ""),
                    },
                }
                r = requests.post(f"{self.url}/v1/objects", json=obj, timeout=10)
                if r.status_code in (200, 201):
                    stored += 1
            except Exception:
                pass
        return stored

    # ── vector search ─────────────────────────────────────────────
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Near-vector search in Weaviate."""
        if not self._ready:
            return []
        try:
            gql = {
                "query": f"""
                {{
                    Get {{
                        {self.SCHEMA_CLASS}(
                            nearVector: {{ vector: {json.dumps(query_embedding)}, certainty: 0.6 }}
                            limit: {top_k}
                        ) {{
                            chunk_id
                            text
                            doc_id
                            chunk_idx
                            sent_range
                            _additional {{ certainty distance }}
                        }}
                    }}
                }}"""
            }
            r = requests.post(f"{self.url}/v1/graphql", json=gql, timeout=15)
            if r.status_code == 200:
                data = r.json().get("data", {}).get("Get", {}).get(self.SCHEMA_CLASS, [])
                return data
        except Exception:
            pass
        return []

"""
Embedding Generation Service
=============================

Provides real vector embeddings using multiple backends.
Supports: OpenAI, LOCAL transformers, Hugging Face
"""

import os
from typing import List, Optional
from core.service_interfaces import EmbeddingInterface
from core.logging_config import Logger, log_performance


logger = Logger(__name__)


class OpenAIEmbedding(EmbeddingInterface):
    """OpenAI embedding service using text-embedding-3-small"""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            logger.warning("OpenAI API key not configured. Embeddings will fail.")

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI embedding service initialized with {model}")
        except ImportError:
            logger.error("openai package not installed")
            self.client = None

    @log_performance
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        response = self.client.embeddings.create(
            input=text.strip(),
            model=self.model
        )
        return response.data[0].embedding

    @log_performance
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        # Filter empty texts
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            raise ValueError("No non-empty texts to embed")

        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )

        # Sort by index to maintain order
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        for item in response.data:
            embeddings[item.index] = item.embedding

        return embeddings


class LocalTransformerEmbedding(EmbeddingInterface):
    """Local transformer-based embeddings using sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the transformer model"""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Local transformer embedding initialized with {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers package not installed")

    @log_performance
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        if not self.model:
            raise RuntimeError("Transformer model not initialized")

        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        embedding = self.model.encode(text.strip(), convert_to_tensor=False)
        return embedding.tolist()

    @log_performance
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        if not self.model:
            raise RuntimeError("Transformer model not initialized")

        # Filter empty texts
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            raise ValueError("No non-empty texts to embed")

        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [e.tolist() for e in embeddings]


class HybridEmbedding(EmbeddingInterface):
    """
    Hybrid embedding that falls back gracefully.
    Tries OpenAI first, then local transformers if API unavailable.
    """

    def __init__(
        self,
        primary: Optional[EmbeddingInterface] = None,
        fallback: Optional[EmbeddingInterface] = None,
    ):
        self.primary = primary or self._get_primary()
        self.fallback = fallback or self._get_fallback()

    def _get_primary(self) -> EmbeddingInterface:
        """Get primary embedding service"""
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbedding()
        return LocalTransformerEmbedding()

    def _get_fallback(self) -> EmbeddingInterface:
        """Get fallback embedding service"""
        return LocalTransformerEmbedding()

    @log_performance
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding with fallback"""
        try:
            return self.primary.embed_text(text)
        except Exception as e:
            logger.warning(f"Primary embedding failed: {e}. Using fallback.")
            try:
                return self.fallback.embed_text(text)
            except Exception as e2:
                logger.error(f"Fallback embedding also failed: {e2}")
                raise

    @log_performance
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings with fallback"""
        try:
            return self.primary.embed_batch(texts)
        except Exception as e:
            logger.warning(f"Primary batch embedding failed: {e}. Using fallback.")
            try:
                return self.fallback.embed_batch(texts)
            except Exception as e2:
                logger.error(f"Fallback batch embedding also failed: {e2}")
                raise

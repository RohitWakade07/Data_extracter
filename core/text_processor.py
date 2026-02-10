"""
Text Processing Service
=======================

Handles text chunking, normalization, and preprocessing.
"""

from typing import List, Optional
from core.service_interfaces import TextProcessingInterface
from core.validation_and_errors import DataValidator
from core.logging_config import Logger


logger = Logger(__name__)


class TextProcessor(TextProcessingInterface):
    """Text processing and chunking service"""

    def __init__(self, default_chunk_size: int = 512, default_overlap: int = 50):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Split text into semantic chunks with overlap.

        Args:
            text: Input text
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap

        if not text or len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                # Look for period, exclamation, or question mark
                sentence_end = max(
                    text.rfind(".", start, end),
                    text.rfind("!", start, end),
                    text.rfind("?", start, end),
                    text.rfind("\n", start, end),
                )

                if sentence_end > start:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap

        logger.debug(f"Split {len(text)} characters into {len(chunks)} chunks")
        return chunks

    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication"""
        return DataValidator.normalize_entity_name(name)

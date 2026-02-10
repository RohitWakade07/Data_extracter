"""
Data Validation and Error Handling
===================================

Provides robust validation, error handling, and resilience patterns.
"""

from typing import List, Dict, Any, Optional, Callable, Type
from dataclasses import dataclass
from functools import wraps
import time
from core.service_interfaces import (
    ValidationServiceInterface,
    Document,
    Entity,
    Relationship,
)
from core.logging_config import Logger


logger = Logger(__name__)


# Custom exceptions
class PipelineException(Exception):
    """Base exception for pipeline operations"""
    pass


class ValidationException(PipelineException):
    """Raised when validation fails"""
    pass


class StorageException(PipelineException):
    """Raised when storage operations fail"""
    pass


class ExtractionException(PipelineException):
    """Raised when extraction fails"""
    pass


class RetryStrategy:
    """Configurable retry strategy with exponential backoff"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        delay = self.initial_delay
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)

        logger.error(f"All {self.max_attempts} attempts failed")
        if last_exception is not None:
            raise last_exception
        raise RuntimeError(f"All {self.max_attempts} attempts failed without exception")


def with_retry(
    max_attempts: int = 3,
    exceptions: tuple = (Exception,),
):
    """Decorator for automatic retry on failure"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            strategy = RetryStrategy(
                max_attempts=max_attempts,
                exceptions=exceptions,
            )
            return strategy.execute(func, *args, **kwargs)

        return wrapper

    return decorator


def safe_operation(default_return: Any = None):
    """Decorator for safe operation with exception logging"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}", exc_info=True)
                return default_return

        return wrapper

    return decorator


class DataValidator(ValidationServiceInterface):
    """Comprehensive data validation"""

    # Validation rules
    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 100000
    MIN_ENTITY_VALUE_LENGTH = 1
    MAX_ENTITY_VALUE_LENGTH = 500
    VALID_CONFIDENCE_RANGE = (0.0, 1.0)

    def validate_document(self, document: Document) -> bool:
        """Validate document structure"""
        try:
            # Check required fields
            assert document.id, "Document ID is required"
            assert document.content, "Document content is required"
            assert isinstance(document.entities, list), "Entities must be a list"
            assert isinstance(document.metadata, dict), "Metadata must be a dict"

            # Check content length
            assert (
                self.MIN_TEXT_LENGTH <= len(document.content) <= self.MAX_TEXT_LENGTH
            ), f"Content length must be between {self.MIN_TEXT_LENGTH} and {self.MAX_TEXT_LENGTH}"

            # Validate entities
            for entity in document.entities:
                if isinstance(entity, dict):
                    assert "type" in entity, "Entity must have 'type' field"
                    assert "value" in entity, "Entity must have 'value' field"
                    assert (
                        self.MIN_ENTITY_VALUE_LENGTH
                        <= len(entity["value"])
                        <= self.MAX_ENTITY_VALUE_LENGTH
                    ), f"Entity value length must be between {self.MIN_ENTITY_VALUE_LENGTH} and {self.MAX_ENTITY_VALUE_LENGTH}"

                    if "confidence" in entity:
                        conf = entity["confidence"]
                        assert (
                            self.VALID_CONFIDENCE_RANGE[0]
                            <= conf
                            <= self.VALID_CONFIDENCE_RANGE[1]
                        ), "Confidence must be between 0.0 and 1.0"

            logger.debug(f"Document {document.id} validation passed")
            return True

        except AssertionError as e:
            logger.error(f"Document validation failed: {e}")
            raise ValidationException(str(e))
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}", exc_info=True)
            return False

    def validate_entities(self, entities: List[Entity]) -> bool:
        """Validate entity list"""
        try:
            assert isinstance(entities, list), "Entities must be a list"
            assert len(entities) > 0, "Entities list cannot be empty"

            for entity in entities:
                assert entity.type, "Entity type is required"
                assert entity.value, "Entity value is required"
                assert (
                    self.VALID_CONFIDENCE_RANGE[0]
                    <= entity.confidence
                    <= self.VALID_CONFIDENCE_RANGE[1]
                ), "Confidence must be between 0.0 and 1.0"

            logger.debug(f"Validated {len(entities)} entities")
            return True

        except AssertionError as e:
            logger.error(f"Entity validation failed: {e}")
            raise ValidationException(str(e))
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}", exc_info=True)
            return False

    def validate_relationships(self, relationships: List[Relationship]) -> bool:
        """Validate relationship list"""
        try:
            assert isinstance(relationships, list), "Relationships must be a list"

            for rel in relationships:
                assert rel.source, "Relationship source is required"
                assert rel.target, "Relationship target is required"
                assert rel.relationship_type, "Relationship type is required"
                assert (
                    self.VALID_CONFIDENCE_RANGE[0]
                    <= rel.confidence
                    <= self.VALID_CONFIDENCE_RANGE[1]
                ), "Confidence must be between 0.0 and 1.0"

            logger.debug(f"Validated {len(relationships)} relationships")
            return True

        except AssertionError as e:
            logger.error(f"Relationship validation failed: {e}")
            raise ValidationException(str(e))
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}", exc_info=True)
            return False

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove null bytes and invalid characters
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

        return text.strip()

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        """Normalize entity names for deduplication"""
        if not name:
            return ""

        # Convert to lowercase
        name = name.lower().strip()

        # Remove extra spaces
        name = " ".join(name.split())

        # Remove common suffixes
        for suffix in [" inc", " corp", " ltd", " llc", " co", " company"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]

        return name

    @staticmethod
    def is_duplicate_entity(
        entity1: Entity,
        entity2: Entity,
        threshold: float = 0.9,
    ) -> bool:
        """Check if two entities are likely duplicates using fuzzy matching"""
        if entity1.type != entity2.type:
            return False

        norm1 = DataValidator.normalize_entity_name(entity1.value)
        norm2 = DataValidator.normalize_entity_name(entity2.value)

        # Simple similarity check
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= threshold


class CircuitBreaker:
    """Circuit breaker pattern for handling cascading failures"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self.last_failure_time is not None and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            logger.info("Circuit breaker closed after successful call")

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")


# Import for deduplication
from difflib import SequenceMatcher

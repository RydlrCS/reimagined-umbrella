"""
Motion embedding service for vector database search using Gemini embeddings.

Generates dense vector embeddings from motion analysis data using:
- Google Gemini gemini-embedding-001 (768-dimensional by default)
- Supports task types: SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY
- Batch embedding support for efficiency
- Deterministic fallback for offline/development mode

Based on Gemini Embeddings API:
https://ai.google.dev/gemini-api/docs/embeddings
"""
import hashlib
import logging
import os
from typing import Any, List, Literal, Optional

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import for Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed. Using fallback embeddings.")

# Gemini embedding configuration
EMBED_MODEL_NAME = "gemini-embedding-001"
# Recommended dimensions: 768 (default), 1536, or 3072
# Using 768 for balance between quality and storage
VECTOR_DIMENSIONS = 768

# Task types for optimized embeddings
TaskType = Literal[
    "SEMANTIC_SIMILARITY",  # For similarity comparison
    "RETRIEVAL_DOCUMENT",   # For documents to be searched
    "RETRIEVAL_QUERY",      # For search queries
    "CLASSIFICATION",       # For text classification
    "CLUSTERING",           # For grouping similar texts
]


class EmbeddingService:
    """
    Production embedding service using Gemini embeddings API.
    
    Features:
    - Gemini gemini-embedding-001 model (768-dim default)
    - Task-specific optimization (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, etc.)
    - Batch embedding support for efficiency
    - Singleton pattern for client reuse
    - Deterministic fallback when Gemini unavailable
    - Automatic embedding normalization for accuracy
    """
    
    _instance: Optional['EmbeddingService'] = None
    _client: Optional[Any] = None
    _available: bool = False
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = EMBED_MODEL_NAME,
        output_dimensionality: int = VECTOR_DIMENSIONS,
    ):
        """
        Initialize embedding service (lazy - doesn't connect immediately).
        
        Args:
            api_key: Gemini API key (uses GEMINI_API_KEY env var if not provided)
            model_name: Gemini embedding model name
            output_dimensionality: Output dimension size (768, 1536, or 3072 recommended)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.dimensions = output_dimensionality
        
        logger.info(
            f"EmbeddingService initialized (lazy) - "
            f"model={model_name}, dims={output_dimensionality}"
        )
    
    @classmethod
    def get_instance(cls, **kwargs) -> 'EmbeddingService':
        """Singleton pattern for service instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    def _initialize_client(self) -> bool:
        """
        Lazy client initialization.
        
        Returns:
            True if client initialized successfully, False otherwise
        """
        if self._client is not None:
            return self._available
        
        if not GEMINI_AVAILABLE:
            logger.warning("google-genai not available - using fallback embeddings")
            self._available = False
            return False
        
        if not self.api_key:
            logger.warning(
                "GEMINI_API_KEY not configured - using fallback embeddings. "
                "Set environment variable to enable Gemini embeddings."
            )
            self._available = False
            return False
        
        try:
            logger.info(f"Initializing Gemini client for embeddings")
            
            self._client = genai.Client(api_key=self.api_key)
            
            self._available = True
            logger.info(
                f"✅ Gemini embedding client initialized: "
                f"{self.model_name} ({self.dimensions} dimensions)"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
            self._available = False
            return False
    
    def is_available(self) -> bool:
        """Check if Gemini embedding service is available."""
        if self._client is None:
            self._initialize_client()
        return self._available
    
    def embed_text(
        self,
        text: Optional[str],
        task_type: TaskType = "RETRIEVAL_DOCUMENT",
    ) -> List[float]:
        """
        Generate embedding vector from text using Gemini.
        
        Args:
            text: Input text (query descriptor, motion name, etc.)
            task_type: Optimization task type (see TaskType for options)
        
        Returns:
            List of floats representing the embedding (768 dimensions by default)
        """
        # Handle empty text
        if not text or not text.strip():
            logger.debug("Empty text - returning zero-like vector")
            rng = np.random.RandomState(0)
            return (rng.rand(self.dimensions) * 0.001).tolist()
        
        # Try Gemini-based embedding
        if self._initialize_client() and self._client is not None:
            try:
                result = self._client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimensions
                    )
                )
                
                [embedding_obj] = result.embeddings
                embedding = embedding_obj.values
                
                # Normalize for dimensions < 3072 (per Gemini docs)
                if self.dimensions < 3072:
                    embedding = self._normalize_embedding(embedding)
                
                logger.debug(
                    f"Generated Gemini embedding: '{text[:50]}...' "
                    f"({len(embedding)} dims, task={task_type})"
                )
                return embedding
                
            except Exception as e:
                logger.error(
                    f"Gemini embedding failed: {e}, falling back to pseudo-embedding"
                )
                # Fall through to fallback
        
        # Fallback: deterministic pseudo-embedding
        return self._pseudo_embedding(text)
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize embedding to unit length.
        
        Per Gemini docs: embeddings with output_dimensionality < 3072 need normalization
        for accurate semantic similarity (cosine distance).
        
        Args:
            embedding: Raw embedding vector
        
        Returns:
            Normalized embedding vector
        """
        embedding_np = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_np)
        
        if norm > 0:
            embedding_np = embedding_np / norm
        
        return embedding_np.tolist()
    
    def _pseudo_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic pseudo-embedding from text hash.
        
        Used as fallback when Gemini is unavailable.
        Ensures consistent embeddings for same text across runs.
        
        Args:
            text: Input text
        
        Returns:
            List of floats (768 dimensions) pseudo-embedding
        """
        # Generate deterministic seed from text hash
        text_hash = abs(hash(text)) % (2**32)
        rng = np.random.RandomState(text_hash)
        
        # Generate random vector and normalize
        vector = rng.rand(self.dimensions).astype(np.float32)
        
        # Normalize to unit length (similar to real embeddings)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        logger.debug(f"Generated pseudo-embedding for: '{text[:50]}...'")
        return vector.tolist()
    
    def embed_motion_descriptor(
        self,
        style_labels: Optional[List[str]] = None,
        npc_tags: Optional[List[str]] = None,
        summary: Optional[str] = None,
        source_motion: Optional[str] = None,
        target_motion: Optional[str] = None,
        task_type: TaskType = "RETRIEVAL_DOCUMENT",
    ) -> List[float]:
        """
        Generate embedding from Gemini motion analysis results.
        
        Constructs a query descriptor from analysis components and generates embedding.
        This descriptor is used for semantic search and similarity matching.
        
        Args:
            style_labels: Motion style labels from Gemini analysis
            npc_tags: NPC character tags from Gemini analysis
            summary: Gemini-generated motion summary
            source_motion: Source motion name
            target_motion: Target motion name
            task_type: Task optimization (RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for search)
        
        Returns:
            Embedding vector (768 dimensions by default)
        """
        # Build query descriptor from components
        descriptor_parts = []
        
        if style_labels:
            descriptor_parts.append(" ".join(style_labels))
        
        if npc_tags:
            descriptor_parts.append(" ".join(npc_tags))
        
        if summary:
            descriptor_parts.append(summary)
        
        if source_motion:
            descriptor_parts.append(f"from {source_motion}")
        
        if target_motion:
            descriptor_parts.append(f"to {target_motion}")
        
        # Combine into single descriptor
        query_descriptor = " | ".join(descriptor_parts) if descriptor_parts else "motion blend"
        
        logger.info(
            f"Building motion embedding from descriptor: "
            f"'{query_descriptor[:100]}...' "
            f"(labels={len(style_labels or [])}, tags={len(npc_tags or [])}, task={task_type})"
        )
        
        return self.embed_text(query_descriptor, task_type=task_type)
    
    def batch_embed_text(
        self,
        texts: List[str],
        task_type: TaskType = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """
        Batch embedding for multiple texts using Gemini batch API.
        
        More efficient than individual calls (50% cheaper, higher throughput).
        
        Args:
            texts: List of text strings to embed
            task_type: Task optimization for all texts
        
        Returns:
            List of embedding vectors (768 dimensions each by default)
        """
        if not texts:
            return []
        
        # Filter empty strings (replace with space to avoid errors)
        valid_texts = [t if t and t.strip() else " " for t in texts]
        
        if self._initialize_client() and self._client is not None:
            try:
                # Use Gemini batch embedding API
                results = self._client.models.batch_embed_contents(
                    model=self.model_name,
                    requests=[
                        types.EmbedContentRequest(
                            content=text,
                            config=types.EmbedContentConfig(
                                task_type=task_type,
                                output_dimensionality=self.dimensions
                            )
                        )
                        for text in valid_texts
                    ]
                )
                
                # Extract and normalize embeddings
                embeddings = []
                for result in results.embeddings:
                    embedding = result.values
                    
                    # Normalize if needed
                    if self.dimensions < 3072:
                        embedding = self._normalize_embedding(embedding)
                    
                    embeddings.append(embedding)
                
                logger.info(
                    f"Batch generated {len(texts)} Gemini embeddings "
                    f"({self.dimensions} dims, task={task_type})"
                )
                return embeddings
                
            except Exception as e:
                logger.error(
                    f"Gemini batch embedding failed: {e}, "
                    f"falling back to individual embeddings"
                )
                # Fall through to fallback
        
        # Fallback: embed individually
        return [self.embed_text(text, task_type=task_type) for text in texts]


# Global singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService.get_instance()
    return _embedding_service


def embed_text(text: Optional[str]) -> List[float]:
    """Convenience function for text embedding."""
    service = get_embedding_service()
    return service.embed_text(text)


def embed_motion_descriptor(**kwargs) -> List[float]:
    """Convenience function for motion descriptor embedding."""
    service = get_embedding_service()
    return service.embed_motion_descriptor(**kwargs)

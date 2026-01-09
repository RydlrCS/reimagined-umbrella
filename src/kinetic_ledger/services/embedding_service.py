"""
Motion embedding service for vector database search.

Generates dense vector embeddings from motion analysis data using:
- SentenceTransformers (all-MiniLM-L6-v2) for text embeddings
- Deterministic fallback for offline/development mode
- Query descriptor generation from Gemini analysis results

Based on MotionBlendAI patterns from:
- project/search_api/search_service.py (embed_text function)
- scripts/seed_motions.py (text_from_file, embed_text)
"""
import hashlib
import logging
import os
import struct
from typing import Any, List, Optional

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. Install with: "
        "pip install sentence-transformers"
    )

# Model configuration (384-dimensional embeddings)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIMENSIONS = 384


class EmbeddingService:
    """
    Production embedding service with model caching and fallback support.
    
    Features:
    - Lazy model loading (loads only when needed)
    - Singleton pattern for model reuse
    - Deterministic fallback when model unavailable
    - Comprehensive error handling and logging
    """
    
    _instance: Optional['EmbeddingService'] = None
    _model: Optional[Any] = None
    _available: bool = False
    
    def __init__(
        self,
        model_name: str = EMBED_MODEL_NAME,
        device: Optional[str] = None,
    ):
        """
        Initialize embedding service (lazy - doesn't load model immediately).
        
        Args:
            model_name: SentenceTransformer model name
            device: Device for inference ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device
        self.dimensions = VECTOR_DIMENSIONS
        
        logger.info(f"EmbeddingService initialized (lazy) - model={model_name}")
    
    @classmethod
    def get_instance(cls, **kwargs) -> 'EmbeddingService':
        """Singleton pattern for service instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    def _initialize_model(self) -> bool:
        """
        Lazy model initialization with caching.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model is not None:
            return self._available
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available - using fallback embeddings")
            self._available = False
            return False
        
        try:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            self._available = True
            logger.info(f"✅ Loaded model: {self.model_name} ({self.dimensions} dimensions)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
            self._available = False
            return False
    
    def is_available(self) -> bool:
        """Check if embedding model is available."""
        if self._model is None:
            self._initialize_model()
        return self._available
    
    def embed_text(self, text: Optional[str]) -> List[float]:
        """
        Generate embedding vector from text.
        
        Uses SentenceTransformer if available, otherwise falls back to
        deterministic pseudo-embedding based on text hash.
        
        Args:
            text: Input text (query descriptor, motion name, etc.)
        
        Returns:
            List of floats (384 dimensions) representing the embedding
        """
        # Handle empty text
        if not text or not text.strip():
            logger.debug("Empty text - returning zero-like vector")
            rng = np.random.RandomState(0)
            return (rng.rand(self.dimensions) * 0.001).tolist()
        
        # Try model-based embedding
        if self._initialize_model() and self._model is not None:
            try:
                vec = self._model.encode(text)
                embedding = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
                logger.debug(f"Generated embedding for text: '{text[:50]}...' ({len(embedding)} dims)")
                return embedding
                
            except Exception as e:
                logger.error(f"Model encoding failed: {e}, falling back to pseudo-embedding")
                # Fall through to fallback
        
        # Fallback: deterministic pseudo-embedding
        return self._pseudo_embedding(text)
    
    def _pseudo_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic pseudo-embedding from text hash.
        
        Used as fallback when SentenceTransformer is unavailable.
        Ensures consistent embeddings for same text across runs.
        
        Args:
            text: Input text
        
        Returns:
            List of floats (384 dimensions) pseudo-embedding
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
        
        Returns:
            Embedding vector (384 dimensions)
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
            f"(labels={len(style_labels or [])}, tags={len(npc_tags or [])})"
        )
        
        return self.embed_text(query_descriptor)
    
    def batch_embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Try model-based batch encoding
        if self._initialize_model() and self._model is not None:
            try:
                vectors = self._model.encode(texts, show_progress_bar=False)
                embeddings = [
                    vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
                    for vec in vectors
                ]
                logger.info(f"Generated {len(embeddings)} embeddings in batch")
                return embeddings
                
            except Exception as e:
                logger.error(f"Batch encoding failed: {e}, falling back to individual processing")
        
        # Fallback: process individually
        return [self.embed_text(text) for text in texts]


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

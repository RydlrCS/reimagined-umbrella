"""
Gemini File Search connector for motion analysis vector storage.

Replaces Elasticsearch with Gemini's native File Search API for:
- Automatic chunking and indexing
- Vector similarity search
- Natural language queries
- Free storage and query-time embeddings
"""
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json
import numpy as np

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from ..utils.logging import get_logger

logger = get_logger(__name__)


class FileSearchConnector:
    """
    Gemini File Search connector for motion analysis data.
    
    Provides vector search capabilities using Gemini's built-in File Search tool.
    Automatically handles embedding, chunking, and indexing without manual configuration.
    
    Features:
    - Automatic document chunking and indexing
    - Free storage and query-time embeddings
    - Natural language search queries
    - Corpus-based organization
    - Lazy initialization with graceful degradation
    """
    
    _instance: Optional['FileSearchConnector'] = None
    _client: Optional[Any] = None
    _corpus: Optional[Any] = None
    _available: bool = False
    _embedding_cache: Dict[str, np.ndarray] = {}
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        corpus_name: str = "kinetic-motion-analysis",
        corpus_description: str = "Motion capture analysis and blend recommendations"
    ):
        """
        Initialize File Search connector (lazy - doesn't connect immediately).
        
        Args:
            api_key: Gemini API key (uses GEMINI_API_KEY env var if not provided)
            corpus_name: Name for the corpus storing motion documents
            corpus_description: Description of the corpus contents
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.corpus_name = corpus_name
        self.corpus_description = corpus_description
        
        logger.info(
            f"FileSearchConnector initialized (lazy) - corpus={corpus_name}"
        )
    
    @classmethod
    def get_instance(cls, **kwargs) -> 'FileSearchConnector':
        """Singleton pattern for connector instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    def _initialize_client(self) -> bool:
        """
        Lazy client initialization with corpus setup.
        
        Returns:
            True if client initialized successfully, False otherwise
        """
        if self._client is not None:
            return self._available
        
        if not GENAI_AVAILABLE:
            logger.warning("google-genai not available - File Search disabled")
            self._available = False
            return False
        
        if not self.api_key:
            logger.warning(
                "GEMINI_API_KEY not configured - File Search disabled. "
                "Set environment variable to enable."
            )
            self._available = False
            return False
        
        try:
            logger.info("Initializing Gemini File Search client")
            
            self._client = genai.Client(api_key=self.api_key)
            
            # Get or create corpus
            self._corpus = self._get_or_create_corpus()
            
            self._available = True
            logger.info(
                f"✅ Gemini File Search initialized: {self.corpus_name}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize File Search client: {e}", exc_info=True)
            self._available = False
            return False
    
    def _get_or_create_corpus(self) -> Any:
        """
        Get existing corpus or create new one.
        
        Returns:
            Corpus object
        """
        if not self._client:
            return None
        
        try:
            # List existing corpora
            corpora = list(self._client.files.list_corpora())
            
            # Find matching corpus by name
            for corpus in corpora:
                if corpus.display_name == self.corpus_name:
                    logger.info(f"Using existing corpus: {self.corpus_name}")
                    return corpus
            
            # Create new corpus if not found
            logger.info(f"Creating new corpus: {self.corpus_name}")
            corpus = self._client.files.create_corpus(
                display_name=self.corpus_name
            )
            
            return corpus
            
        except Exception as e:
            logger.error(f"Error managing corpus: {e}", exc_info=True)
            return None
    
    def is_available(self) -> bool:
        """Check if File Search is available."""
        if self._client is None:
            self._initialize_client()
        return self._available
    
    def index_document(
        self,
        document: Dict[str, Any],
        document_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Index a single motion analysis document using File Search.
        
        Also stores the embedding in local cache for kNN/RkCNN similarity checks.
        
        Args:
            document: Motion analysis document with fields like:
                - analysis_id: Unique analysis identifier
                - query_descriptor: Searchable text description
                - style_labels: List of style tags
                - gemini_summary: Analysis summary
                - source_motion: Source motion filename
                - etc.
            document_id: Optional custom document ID
            embedding: Optional embedding vector for kNN/RkCNN (if available)
        
        Returns:
            True if indexing succeeded, False otherwise
        """
        # Create document metadata (always)
        doc_id = document_id or document.get("analysis_id", f"doc_{datetime.utcnow().timestamp()}")
        
        # Store embedding in cache (even if File Search is unavailable)
        if embedding is not None:
            self._embedding_cache[doc_id] = embedding
            logger.debug(f"Cached embedding for {doc_id}: {embedding.shape}")
        
        # Try to index in File Search if available
        if not self._initialize_client():
            logger.warning("File Search unavailable - embedding cached but not indexed")
            return embedding is not None  # Success if we cached the embedding
        
        try:
            # Convert document to searchable text format
            text_content = self._format_document_for_search(document)
            
            # Upload document to corpus
            file = self._client.files.create(
                corpus=self._corpus,
                display_name=doc_id,
                mime_type="text/plain"
            )
            
            # Write content
            file.write(text_content)
            
            logger.info(f"Indexed document: {doc_id} ({len(text_content)} chars)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}", exc_info=True)
            return False
    
    def _format_document_for_search(self, document: Dict[str, Any]) -> str:
        """
        Format motion analysis document as searchable text.
        
        Args:
            document: Motion analysis document
        
        Returns:
            Formatted text for File Search indexing
        """
        parts = []
        
        # Add structured metadata
        if "analysis_id" in document:
            parts.append(f"Analysis ID: {document['analysis_id']}")
        
        if "motion_id" in document:
            parts.append(f"Motion ID: {document['motion_id']}")
        
        # Add motion files
        if "source_motion" in document:
            parts.append(f"Source Motion: {document['source_motion']}")
        
        if "target_motion" in document:
            parts.append(f"Target Motion: {document['target_motion']}")
        
        # Add search descriptor (most important for matching)
        if "query_descriptor" in document:
            parts.append(f"\nQuery Descriptor:\n{document['query_descriptor']}")
        
        # Add Gemini analysis
        if "gemini_summary" in document:
            parts.append(f"\nGemini Summary:\n{document['gemini_summary']}")
        
        # Add style labels
        if "style_labels" in document and document["style_labels"]:
            parts.append(f"\nStyle Labels: {', '.join(document['style_labels'])}")
        
        # Add NPC tags
        if "npc_tags" in document and document["npc_tags"]:
            parts.append(f"\nNPC Tags: {', '.join(document['npc_tags'])}")
        
        # Add blend metadata
        if "blend_ratio" in document:
            parts.append(f"\nBlend Ratio: {document['blend_ratio']}")
        
        if "blend_method" in document:
            parts.append(f"Blend Method: {document['blend_method']}")
        
        # Add timestamps
        if "created_at" in document:
            parts.append(f"\nCreated: {document['created_at']}")
        
        # Add full document as JSON for completeness
        parts.append(f"\n\nFull Document JSON:\n{json.dumps(document, indent=2)}")
        
        return "\n".join(parts)
    
    def bulk_index(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Index multiple motion analysis documents.
        
        Args:
            documents: List of motion analysis documents
        
        Returns:
            Summary dict with success/failure counts
        """
        if not documents:
            return {"indexed": 0, "failed": 0}
        
        indexed = 0
        failed = 0
        
        for doc in documents:
            doc_id = doc.get("analysis_id")
            success = self.index_document(doc, document_id=doc_id)
            
            if success:
                indexed += 1
            else:
                failed += 1
        
        logger.info(
            f"Bulk index complete: {indexed} indexed, {failed} failed "
            f"(total {len(documents)})"
        )
        
        return {"indexed": indexed, "failed": failed, "total": len(documents)}
    
    def search(
        self,
        query: str,
        model: str = "gemini-2.0-flash-exp",
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar motions using natural language query.
        
        Args:
            query: Natural language search query
            model: Gemini model to use for search
            max_results: Maximum number of results to return
        
        Returns:
            List of search results with document data
        """
        if not self._initialize_client():
            logger.warning("File Search unavailable - returning empty results")
            return []
        
        try:
            logger.info(f"Searching File Search corpus: '{query[:100]}...'")
            
            # Use Gemini generate_content with File Search tool
            response = self._client.models.generate_content(
                model=model,
                contents=f"Search the motion analysis corpus for: {query}. Return the most relevant motion blends.",
                config=types.GenerateContentConfig(
                    tools=[types.Tool(file_search=self._corpus)],
                    temperature=0.1  # Low temperature for factual retrieval
                )
            )
            
            # Extract grounding metadata (file references)
            results = []
            
            if hasattr(response, 'grounding_metadata') and response.grounding_metadata:
                for chunk in response.grounding_metadata.grounding_chunks:
                    if hasattr(chunk, 'file'):
                        results.append({
                            "document_id": chunk.file.display_name,
                            "score": getattr(chunk, 'score', None),
                            "content": getattr(chunk, 'content', ''),
                            "text": response.text
                        })
            
            # Fallback: return text response if no grounding metadata
            if not results and response.text:
                results = [{
                    "document_id": "unknown",
                    "score": None,
                    "content": response.text,
                    "text": response.text
                }]
            
            logger.info(f"Found {len(results)} search results")
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"File Search query failed: {e}", exc_info=True)
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the corpus.
        
        Args:
            document_id: Document identifier to delete
        
        Returns:
            True if deletion succeeded, False otherwise
        """
        if not self._initialize_client():
            logger.warning("File Search unavailable - cannot delete document")
            return False
        
        try:
            # List files in corpus
            files = list(self._client.files.list(corpus=self._corpus))
            
            # Find and delete matching file
            for file in files:
                if file.display_name == document_id:
                    file.delete()
                    logger.info(f"Deleted document: {document_id}")
                    return True
            
            logger.warning(f"Document not found: {document_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}", exc_info=True)
            return False
    
    def clear_corpus(self) -> bool:
        """
        Clear all documents from the corpus.
        
        Returns:
            True if clearing succeeded, False otherwise
        """
        if not self._initialize_client():
            logger.warning("File Search unavailable - cannot clear corpus")
            return False
        
        try:
            # List all files
            files = list(self._client.files.list(corpus=self._corpus))
            
            # Delete each file
            deleted = 0
            for file in files:
                try:
                    file.delete()
                    deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete file {file.display_name}: {e}")
            
            # Clear embedding cache
            self._embedding_cache.clear()
            
            logger.info(f"Cleared corpus: {deleted} documents deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear corpus: {e}", exc_info=True)
            return False
    
    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """
        Get all cached embeddings for kNN/RkCNN similarity checks.
        
        Returns:
            List of (document_id, embedding) tuples
        """
        return list(self._embedding_cache.items())
    
    def get_embedding(self, document_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            Embedding vector if found, None otherwise
        """
        return self._embedding_cache.get(document_id)
    
    def cache_size(self) -> int:
        """
        Get number of cached embeddings.
        
        Returns:
            Number of embeddings in cache
        """
        return len(self._embedding_cache)


# Global singleton instance
_file_search_connector: Optional[FileSearchConnector] = None


def get_file_search_connector() -> FileSearchConnector:
    """Get global File Search connector instance."""
    global _file_search_connector
    if _file_search_connector is None:
        _file_search_connector = FileSearchConnector.get_instance()
    return _file_search_connector

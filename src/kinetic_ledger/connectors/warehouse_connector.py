"""
Elasticsearch vector database connector for Kinetic Ledger.

Provides production-grade integration with Elasticsearch Cloud for:
- Motion analysis vector search (k-NN with cosine similarity)
- Semantic text search using ELSER model
- Hybrid search combining vector + text relevance
- Bulk indexing with retry logic and connection pooling

Based on MotionBlendAI architecture with lazy initialization and fallback support.
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for optional dependencies
try:
    from elasticsearch import Elasticsearch, exceptions as es_exceptions
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    logger.warning("elasticsearch package not installed. Install with: pip install elasticsearch")

from ..schemas.elasticsearch_mappings import (
    MOTION_INDEX_NAME,
    VECTOR_DIMENSIONS,
    create_full_index_config,
)

# Legacy Fivetran schema definition
__MAX_RETRIES = 3
__BASE_DELAY_SECONDS = 1

SCHEMA_DEFINITION = [
    {"table": "blendanim_operations", "primary_key": ["operation_id"]},
    {"table": "skeleton_metadata", "primary_key": ["skeleton_id"]},
]


class ElasticsearchConnector:
    """
    Production Elasticsearch connector with lazy initialization and fallback support.
    
    Features:
    - Lazy client initialization (connects only when needed)
    - Automatic index creation with proper mappings
    - Connection pooling and keep-alive
    - Graceful degradation when ES unavailable
    - Comprehensive error handling and logging
    
    Based on MotionBlendAI patterns from:
    - project/elastic_search/app.py (lazy client, fallback)
    - project/search_api/search_service.py (embeddings, k-NN)
    """
    
    _instance: Optional['ElasticsearchConnector'] = None
    _client: Optional[Any] = None
    _available: bool = False
    
    def __init__(
        self,
        cloud_url: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = MOTION_INDEX_NAME,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize Elasticsearch connector (lazy - doesn't connect immediately).
        
        Args:
            cloud_url: Elasticsearch Cloud URL (e.g., https://my-deployment.es.us-central1.gcp.elastic.cloud:443)
            api_key: Elasticsearch API key for authentication
            index_name: Index name for motion analysis documents
            timeout: Request timeout in seconds
            max_retries: Maximum number of connection retries
        """
        self.cloud_url = cloud_url or os.getenv("ES_CLOUD_URL", "")
        self.api_key = api_key or os.getenv("ES_API_KEY", "")
        self.index_name = index_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(
            f"ElasticsearchConnector initialized (lazy) - "
            f"index={index_name}, timeout={timeout}s"
        )
    
    @classmethod
    def get_instance(cls, **kwargs) -> 'ElasticsearchConnector':
        """Singleton pattern for connector instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    def _initialize_client(self) -> Tuple[Optional[Any], bool]:
        """
        Lazy client initialization with connection pooling.
        
        Returns:
            Tuple of (client, available) where available indicates successful connection
        """
        if self._client is not None:
            return self._client, self._available
        
        if not ES_AVAILABLE:
            logger.warning("Elasticsearch package not available - using mock mode")
            self._available = False
            return None, False
        
        if not self.cloud_url or not self.api_key:
            logger.warning(
                "ES_CLOUD_URL or ES_API_KEY not configured - using mock mode. "
                "Set environment variables to enable Elasticsearch."
            )
            self._available = False
            return None, False
        
        try:
            # Initialize Elasticsearch client with connection pooling
            self._client = Elasticsearch(
                [self.cloud_url],
                api_key=self.api_key,
                request_timeout=self.timeout,
                max_retries=self.max_retries,
                retry_on_timeout=True,
            )
            
            # Test connection with ping
            if self._client.ping():
                logger.info(f"✅ Connected to Elasticsearch: {self.cloud_url}")
                self._available = True
                
                # Ensure index exists with proper mappings
                self._ensure_index()
                
                return self._client, True
            else:
                logger.warning(f"Elasticsearch ping failed: {self.cloud_url}")
                self._available = False
                return None, False
                
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {e}", exc_info=True)
            self._available = False
            return None, False
    
    def _ensure_index(self) -> bool:
        """
        Ensure motion analysis index exists with proper mappings.
        
        Creates index if it doesn't exist using schema from elasticsearch_mappings.py
        
        Returns:
            True if index exists/created successfully, False otherwise
        """
        client, available = self._initialize_client()
        if not available or client is None:
            return False
        
        try:
            if not client.indices.exists(index=self.index_name):
                logger.info(f"Creating index: {self.index_name}")
                
                # Get full index configuration with mappings and settings
                index_config = create_full_index_config()
                
                # Create index
                client.indices.create(
                    index=self.index_name,
                    body=index_config
                )
                
                logger.info(f"✅ Created index: {self.index_name} with {VECTOR_DIMENSIONS}-dim vectors")
                return True
            else:
                logger.debug(f"Index already exists: {self.index_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to ensure index {self.index_name}: {e}", exc_info=True)
            return False
    
    def is_available(self) -> bool:
        """Check if Elasticsearch is available."""
        if self._client is None:
            self._initialize_client()
        return self._available
    
    def index_document(
        self,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> bool:
        """
        Index a single motion analysis document.
        
        Args:
            document: Document data (must include motion_vector field)
            doc_id: Optional document ID (generates UUID if not provided)
        
        Returns:
            True if indexed successfully, False otherwise
        """
        client, available = self._initialize_client()
        if not available or client is None:
            logger.warning("Elasticsearch unavailable - document not indexed")
            return False
        
        try:
            # Add timestamps if not present
            if "created_at" not in document:
                document["created_at"] = datetime.utcnow().isoformat()
            if "updated_at" not in document:
                document["updated_at"] = datetime.utcnow().isoformat()
            
            # Index document
            response = client.index(
                index=self.index_name,
                id=doc_id,
                body=document,
                refresh="wait_for"  # Wait for refresh so document is searchable
            )
            
            logger.info(
                f"✅ Indexed document: {response['_id']} "
                f"(version={response.get('_version', 'N/A')})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}", exc_info=True)
            return False
    
    def bulk_index(
        self,
        documents: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Bulk index multiple motion analysis documents.
        
        Args:
            documents: List of document data dictionaries
        
        Returns:
            Tuple of (success_count, failure_count)
        """
        client, available = self._initialize_client()
        if not available or client is None:
            logger.warning(f"Elasticsearch unavailable - {len(documents)} documents not indexed")
            return 0, len(documents)
        
        try:
            from elasticsearch.helpers import bulk
            
            # Prepare bulk actions
            actions = []
            for doc in documents:
                # Add timestamps
                if "created_at" not in doc:
                    doc["created_at"] = datetime.utcnow().isoformat()
                if "updated_at" not in doc:
                    doc["updated_at"] = datetime.utcnow().isoformat()
                
                action = {
                    "_index": self.index_name,
                    "_source": doc,
                }
                
                # Use analysis_id as document ID if present
                if "analysis_id" in doc:
                    action["_id"] = doc["analysis_id"]
                
                actions.append(action)
            
            # Execute bulk operation
            success, failed = bulk(
                client,
                actions,
                refresh="wait_for",
                raise_on_error=False,
            )
            
            logger.info(
                f"✅ Bulk indexed {success}/{len(documents)} documents "
                f"({failed} failures)"
            )
            
            return success, failed
            
        except Exception as e:
            logger.error(f"Bulk index failed: {e}", exc_info=True)
            return 0, len(documents)
    
    def search_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform k-NN vector similarity search.
        
        Args:
            query_vector: Motion embedding vector (must be VECTOR_DIMENSIONS length)
            k: Number of results to return
            filters: Optional filter clauses (e.g., {"status": "completed"})
        
        Returns:
            List of matching documents with similarity scores
        """
        client, available = self._initialize_client()
        if not available or client is None:
            logger.warning("Elasticsearch unavailable - returning empty results")
            return []
        
        try:
            # Validate vector dimensions
            if len(query_vector) != VECTOR_DIMENSIONS:
                logger.error(
                    f"Invalid vector dimensions: {len(query_vector)} "
                    f"(expected {VECTOR_DIMENSIONS})"
                )
                return []
            
            # Build k-NN query
            query_body: Dict[str, Any] = {
                "size": k,
                "query": {
                    "knn": {
                        "field": "motion_vector",
                        "query_vector": query_vector,
                        "k": k,
                        "num_candidates": k * 2,
                    }
                },
                "_source": {
                    "excludes": ["motion_vector"]  # Exclude large vector from response
                }
            }
            
            # Add filters if provided
            if filters:
                query_body["query"] = {
                    "bool": {
                        "must": [query_body["query"]],
                        "filter": [
                            {"term": {key: value}} for key, value in filters.items()
                        ]
                    }
                }
            
            # Execute search
            response = client.search(
                index=self.index_name,
                body=query_body,
                timeout=f"{self.timeout}s"
            )
            
            # Parse results
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["_id"] = hit["_id"]
                doc["_score"] = hit["_score"]
                results.append(doc)
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return []
    
    def search_text(
        self,
        query_text: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-field text search with semantic support.
        
        Args:
            query_text: Search query text
            k: Number of results to return
            filters: Optional filter clauses
        
        Returns:
            List of matching documents with relevance scores
        """
        client, available = self._initialize_client()
        if not available or client is None:
            logger.warning("Elasticsearch unavailable - returning empty results")
            return []
        
        try:
            # Build text search query
            should_queries = [
                # Semantic search on query descriptor
                {
                    "semantic": {
                        "field": "query_descriptor.semantic",
                        "query": query_text
                    }
                },
                # Multi-match on key fields
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "style_labels^3",
                            "npc_tags^2",
                            "gemini_summary",
                            "source_motion",
                            "target_motion"
                        ],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                }
            ]
            
            query_body: Dict[str, Any] = {
                "size": k,
                "query": {
                    "bool": {
                        "should": should_queries,
                        "minimum_should_match": 1
                    }
                },
                "highlight": {
                    "fields": {
                        "query_descriptor": {},
                        "gemini_summary": {},
                        "style_labels": {},
                        "npc_tags": {}
                    }
                }
            }
            
            # Add filters if provided
            if filters:
                query_body["query"]["bool"]["filter"] = [
                    {"term": {key: value}} for key, value in filters.items()
                ]
            
            # Execute search
            response = client.search(
                index=self.index_name,
                body=query_body,
                timeout=f"{self.timeout}s"
            )
            
            # Parse results
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["_id"] = hit["_id"]
                doc["_score"] = hit["_score"]
                if "highlight" in hit:
                    doc["_highlight"] = hit["highlight"]
                results.append(doc)
            
            logger.info(f"Text search returned {len(results)} results for: '{query_text}'")
            return results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}", exc_info=True)
            return []
    
    def search_hybrid(
        self,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        vector_weight: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and text relevance.
        
        Args:
            query_vector: Optional motion embedding vector
            query_text: Optional search query text
            k: Number of results to return
            vector_weight: Weight for vector component (0.0-1.0), text = 1 - vector_weight
            filters: Optional filter clauses
        
        Returns:
            List of matching documents with hybrid scores
        """
        if not query_vector and not query_text:
            logger.error("At least one of query_vector or query_text must be provided")
            return []
        
        client, available = self._initialize_client()
        if not available or client is None:
            logger.warning("Elasticsearch unavailable - returning empty results")
            return []
        
        try:
            text_weight = 1.0 - vector_weight
            should_queries = []
            
            # Add vector similarity if provided
            if query_vector:
                if len(query_vector) != VECTOR_DIMENSIONS:
                    logger.error(f"Invalid vector dimensions: {len(query_vector)}")
                    return []
                
                should_queries.append({
                    "knn": {
                        "field": "motion_vector",
                        "query_vector": query_vector,
                        "k": k * 2,
                        "num_candidates": k * 4,
                        "boost": vector_weight
                    }
                })
            
            # Add text search if provided
            if query_text:
                should_queries.extend([
                    {
                        "semantic": {
                            "field": "query_descriptor.semantic",
                            "query": query_text,
                            "boost": text_weight * 2.0
                        }
                    },
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["style_labels^3", "npc_tags^2", "gemini_summary"],
                            "boost": text_weight
                        }
                    }
                ])
            
            query_body: Dict[str, Any] = {
                "size": k,
                "query": {
                    "bool": {
                        "should": should_queries,
                        "minimum_should_match": 1
                    }
                },
                "_source": {
                    "excludes": ["motion_vector"]
                }
            }
            
            # Add filters
            if filters:
                query_body["query"]["bool"]["filter"] = [
                    {"term": {key: value}} for key, value in filters.items()
                ]
            
            # Execute search
            response = client.search(
                index=self.index_name,
                body=query_body,
                timeout=f"{self.timeout}s"
            )
            
            # Parse results
            results = []
            for hit in response["hits"]["hits"]:
                doc = hit["_source"]
                doc["_id"] = hit["_id"]
                doc["_score"] = hit["_score"]
                results.append(doc)
            
            logger.info(
                f"Hybrid search returned {len(results)} results "
                f"(vector_weight={vector_weight:.2f})"
            )
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            return []


# Legacy Fivetran connector support
class Operations:
    @staticmethod
    def upsert(table: str, data: Dict):
        logger.info(f"UPSERT {table}: {data.get('operation_id') or data.get('skeleton_id')}")

    @staticmethod
    def checkpoint(state: Dict):
        logger.info(f"CHECKPOINT: {state}")


def schema(configuration: Dict) -> List[Dict]:
    """Legacy Fivetran schema definition."""
    return SCHEMA_DEFINITION


def update(configuration: Dict, state: Dict):
    """Legacy Fivetran update handler."""
    Operations.upsert(
        table="blendanim_operations",
        data={
            "operation_id": "op_demo_001",
            "source_animation": "capoeira.fbx",
            "target_animation": "breakdance.fbx",
            "blend_method": "single_shot_temporal_conditioning",
            "blend_ratio": 0.5,
            "frames_processed": 250,
            "status": "completed",
        },
    )
    Operations.checkpoint({"last_synced": "now"})

"""
Elasticsearch index mappings for Kinetic Ledger motion analysis data.

Follows MotionBlendAI patterns with dense_vector fields for similarity search,
semantic text fields for natural language queries, and metadata for filtering.

Optimized for Gemini embeddings (gemini-embedding-001, 768 dimensions).
"""
from typing import Any, Dict


# Index name for motion analysis documents
MOTION_INDEX_NAME = "kinetic-motion-analysis"

# Vector dimensions - aligned with Gemini embeddings (768-dim recommended)
VECTOR_DIMENSIONS = 768


def create_motion_mappings() -> Dict[str, Any]:
    """
    Define comprehensive field mappings for motion capture analysis data.
    
    Based on MotionBlendAI elasticsearch implementation with enhancements for:
    - Gemini analysis results (style_labels, npc_tags, transition_window)
    - Motion blend metadata (source animations, blend ratios)
    - Attestation data (validation scores, signatures)
    - Vector similarity search (dense_vector with cosine similarity)
    - Semantic text search (ELSER model integration)
    
    Returns:
        Dictionary containing Elasticsearch mapping configuration
    """
    return {
        "properties": {
            # Primary identification
            "analysis_id": {
                "type": "keyword"
            },
            "motion_id": {
                "type": "keyword"
            },
            
            # Motion blend source information
            "source_motion": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    },
                    "semantic": {
                        "type": "semantic_text"
                    }
                }
            },
            "target_motion": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    },
                    "semantic": {
                        "type": "semantic_text"
                    }
                }
            },
            
            # Gemini analysis descriptor for vector search
            "query_descriptor": {
                "type": "text",
                "fields": {
                    "semantic": {
                        "type": "semantic_text"
                    }
                }
            },
            
            # Motion embedding vector for similarity search
            "motion_vector": {
                "type": "dense_vector",
                "dims": VECTOR_DIMENSIONS,
                "index": True,
                "similarity": "cosine"
            },
            
            # Gemini analysis results
            "style_labels": {
                "type": "keyword"
            },
            "npc_tags": {
                "type": "keyword"
            },
            "safety_flags": {
                "type": "keyword"
            },
            "gemini_summary": {
                "type": "text",
                "fields": {
                    "semantic": {
                        "type": "semantic_text"
                    }
                }
            },
            
            # Blend metadata
            "blend_ratio": {
                "type": "float"
            },
            "blend_method": {
                "type": "keyword"
            },
            "transition_start": {
                "type": "integer"
            },
            "transition_end": {
                "type": "integer"
            },
            
            # Attestation metadata
            "validation_score": {
                "type": "float"
            },
            "novelty_score": {
                "type": "float"
            },
            "knn_distance": {
                "type": "float"
            },
            "rkcnn_ensemble_variance": {
                "type": "float"
            },
            "attestation_hash": {
                "type": "keyword"
            },
            "signature": {
                "type": "keyword"
            },
            
            # Motion quality metrics
            "frames_processed": {
                "type": "integer"
            },
            "duration_seconds": {
                "type": "float"
            },
            "quality_score": {
                "type": "float"
            },
            
            # Status and timestamps
            "status": {
                "type": "keyword"
            },
            "created_at": {
                "type": "date"
            },
            "updated_at": {
                "type": "date"
            },
            
            # Model information
            "model_provider": {
                "type": "keyword"
            },
            "model_name": {
                "type": "keyword"
            },
            "model_version": {
                "type": "keyword"
            }
        }
    }


def create_index_settings() -> Dict[str, Any]:
    """
    Define index settings for optimal performance.
    
    Single shard for development, can scale to multiple shards in production.
    No replicas for single-node setups, adjust for production HA.
    
    Returns:
        Dictionary containing Elasticsearch index settings
    """
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "1s",
            "max_result_window": 10000
        }
    }


def create_full_index_config() -> Dict[str, Any]:
    """
    Combine mappings and settings for complete index configuration.
    
    Returns:
        Full index configuration ready for es.indices.create()
    """
    return {
        "mappings": create_motion_mappings(),
        **create_index_settings()
    }


# Example search queries for documentation
EXAMPLE_QUERIES = {
    "vector_similarity": {
        "description": "Find similar motions using k-NN vector search",
        "query": {
            "size": 10,
            "query": {
                "knn": {
                    "field": "motion_vector",
                    "query_vector": [0.1] * VECTOR_DIMENSIONS,  # Replace with actual embedding
                    "k": 10,
                    "num_candidates": 20
                }
            }
        }
    },
    
    "semantic_text": {
        "description": "Natural language search using ELSER semantic model",
        "query": {
            "size": 10,
            "query": {
                "bool": {
                    "should": [
                        {
                            "semantic": {
                                "field": "query_descriptor.semantic",
                                "query": "explosive athletic jumping with high energy"
                            }
                        },
                        {
                            "multi_match": {
                                "query": "athletic jump",
                                "fields": ["style_labels^3", "npc_tags^2", "gemini_summary"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }
    },
    
    "hybrid_search": {
        "description": "Combine vector similarity and text relevance",
        "query": {
            "size": 10,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "field": "motion_vector",
                                "query_vector": [0.1] * VECTOR_DIMENSIONS,
                                "k": 5,
                                "boost": 0.6  # Vector weight
                            }
                        },
                        {
                            "multi_match": {
                                "query": "dance performance",
                                "fields": ["style_labels", "npc_tags", "query_descriptor"],
                                "boost": 0.4  # Text weight
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }
    },
    
    "filtered_search": {
        "description": "Filter by attestation quality and blend method",
        "query": {
            "size": 10,
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "validation_score": {
                                    "gte": 0.8
                                }
                            }
                        }
                    ],
                    "filter": [
                        {
                            "term": {
                                "blend_method": "single_shot_temporal_conditioning"
                            }
                        },
                        {
                            "term": {
                                "status": "completed"
                            }
                        }
                    ]
                }
            }
        }
    }
}

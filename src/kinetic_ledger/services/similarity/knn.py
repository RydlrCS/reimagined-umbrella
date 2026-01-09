"""
kNN (k-Nearest Neighbors) implementation for motion similarity.
"""
import numpy as np
from typing import List, Tuple


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute L2 (Euclidean) distance between two vectors.
    
    Note: In high dimensions, L2 distance can suffer from concentration.
    We use this as a baseline and rely on RkCNN ensembles for robustness.
    """
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - cosine similarity)."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0
    
    similarity = dot_product / (norm_a * norm_b)
    return float(1.0 - similarity)


def knn(
    query: np.ndarray,
    items: List[Tuple[str, np.ndarray]],
    k: int,
    distance_metric: str = "euclidean",
) -> List[Tuple[str, float]]:
    """
    Find k nearest neighbors.
    
    Args:
        query: Query vector
        items: List of (item_id, vector) tuples
        k: Number of neighbors
        distance_metric: "euclidean" or "cosine"
    
    Returns:
        List of (item_id, distance) tuples, sorted by distance
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    
    if not items:
        return []
    
    # Select distance function
    dist_fn = l2_distance if distance_metric == "euclidean" else cosine_distance
    
    # Compute distances
    distances = [(item_id, dist_fn(query, vec)) for (item_id, vec) in items]
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    return distances[:k]


def compute_separation_score(distances: List[float]) -> float:
    """
    Compute separation score from nearest neighbor distances.
    
    Separation score measures how distinct the query is from the neighborhood:
    separation = (d_k - d_1) / (d_k + ε)
    
    Higher score means:
    - Large gap between closest match and k-th neighbor
    - Query is on the boundary of the neighborhood
    - Higher likelihood of novelty
    
    Args:
        distances: List of distances (sorted)
    
    Returns:
        Separation score in [0, 1]
    """
    if not distances:
        return 1.0  # No neighbors = maximum separation
    
    if len(distances) == 1:
        return 0.5  # Single neighbor = moderate separation
    
    d1 = distances[0]
    dk = distances[-1]
    eps = 1e-9
    
    return float((dk - d1) / (dk + eps))

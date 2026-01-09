"""
RkCNN (Random k-Conditional Nearest Neighbors) ensemble implementation.

RkCNN provides robustness in high-dimensional spaces by:
1. Sampling random subspaces
2. Running kNN in each subspace
3. Aggregating results via voting or averaging

This addresses the "curse of dimensionality" where distances concentrate.
"""
import numpy as np
from typing import List, Tuple, Optional
import math

from .knn import knn as knn_search, compute_separation_score


def compute_subspace_dim(d: int) -> int:
    """
    Compute natural subspace dimension using mathematical heuristic.
    
    Formula: m = min(d, max(16, round(4*sqrt(d))))
    
    This balances:
    - Preserving enough dimensions for discrimination
    - Reducing dimensionality to avoid concentration
    
    Args:
        d: Full dimensionality
    
    Returns:
        Subspace dimension
    """
    m = max(16, round(4 * math.sqrt(d)))
    return min(d, m)


def compute_ensemble_size(d: int) -> int:
    """
    Compute natural ensemble size.
    
    Formula: E = max(32, min(128, 8*ceil(log2(d))))
    
    Args:
        d: Full dimensionality
    
    Returns:
        Ensemble size
    """
    if d <= 1:
        return 32
    
    e = 8 * math.ceil(math.log2(d))
    return max(32, min(128, e))


def random_subspace_indices(
    d: int,
    sub_d: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample random subspace indices.
    
    Args:
        d: Full dimensionality
        sub_d: Subspace dimensionality
        rng: Random number generator
    
    Returns:
        Array of dimension indices
    """
    if sub_d <= 0 or sub_d > d:
        raise ValueError(f"subspace_dim must be in (0, {d}], got {sub_d}")
    
    return rng.choice(d, size=sub_d, replace=False)


def compute_vote_margin(
    ensemble_results: List[List[Tuple[str, float]]],
    k: int,
) -> float:
    """
    Compute vote margin from ensemble results.
    
    Vote margin = (V_top - V_second) / E
    
    Where:
    - V_top: votes for most frequent top-1 neighbor
    - V_second: votes for second most frequent top-1 neighbor
    - E: ensemble size
    
    Higher margin indicates consensus across subspaces.
    
    Args:
        ensemble_results: List of kNN results from each subspace
        k: Number of neighbors
    
    Returns:
        Vote margin in [0, 1]
    """
    if not ensemble_results:
        return 0.0
    
    # Count votes for top-1 neighbor in each subspace
    top_votes: dict[str, int] = {}
    
    for result in ensemble_results:
        if result:
            top_id = result[0][0]
            top_votes[top_id] = top_votes.get(top_id, 0) + 1
    
    if not top_votes:
        return 0.0
    
    # Sort by votes
    sorted_votes = sorted(top_votes.values(), reverse=True)
    
    if len(sorted_votes) == 1:
        # Perfect consensus
        return 1.0
    
    v_top = sorted_votes[0]
    v_second = sorted_votes[1]
    e = len(ensemble_results)
    
    return float((v_top - v_second) / e)


def rkcnn(
    query: np.ndarray,
    items: List[Tuple[str, np.ndarray]],
    k: int,
    ensembles: Optional[int] = None,
    subspace_dim: Optional[int] = None,
    distance_metric: str = "euclidean",
    seed: int = 42,
) -> Tuple[float, float, List[List[Tuple[str, float]]]]:
    """
    Random k-Conditional Nearest Neighbors ensemble.
    
    Args:
        query: Query vector
        items: List of (item_id, vector) tuples
        k: Number of neighbors per subspace
        ensembles: Number of random subspaces (default: computed from dimensionality)
        subspace_dim: Subspace dimension (default: computed from dimensionality)
        distance_metric: "euclidean" or "cosine"
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (separation_score, vote_margin, ensemble_results)
    """
    d = query.shape[0]
    
    # Compute natural parameters if not provided
    if ensembles is None:
        ensembles = compute_ensemble_size(d)
    
    if subspace_dim is None:
        subspace_dim = compute_subspace_dim(d)
    
    rng = np.random.default_rng(seed)
    
    all_knn_results: List[List[Tuple[str, float]]] = []
    all_separations: List[float] = []
    
    for _ in range(ensembles):
        # Sample random subspace
        indices = random_subspace_indices(d, subspace_dim, rng)
        
        # Project query and items to subspace
        q_sub = query[indices]
        items_sub = [(item_id, vec[indices]) for (item_id, vec) in items]
        
        # Run kNN in subspace
        nn = knn_search(q_sub, items_sub, k=k, distance_metric=distance_metric)
        all_knn_results.append(nn)
        
        # Compute separation in this subspace
        distances = [dist for _, dist in nn]
        sep = compute_separation_score(distances)
        all_separations.append(sep)
    
    # Aggregate results
    avg_separation = float(np.mean(all_separations)) if all_separations else 0.0
    vote_margin = compute_vote_margin(all_knn_results, k)
    
    return avg_separation, vote_margin, all_knn_results

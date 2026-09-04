# Segmentation - RANSAC Plane Fitting and DBSCAN Clustering
# Point cloud processing for LiDAR data

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, List, Dict, Any
import random

def fit_plane_ransac(points: np.ndarray, threshold_m: float = 0.008, 
                    iterations: int = 2000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """RANSAC the dominant plane from point cloud.
    
    Returns (plane, inlier_mask) where plane is (4,) as [nx, ny, nz, d] with the
    normal unit length, so a point x on the plane satisfies n.x + d = 0.
    
    The normal is oriented so that the MAJORITY of non-inlier points have positive
    signed distance. That is the "above the table" direction. Deterministic for a seed.
    
    Args:
        points: (N,3) array of points in METRES
        threshold_m: Distance threshold for inlier classification
        iterations: Number of RANSAC iterations
        seed: Random seed for reproducibility
        
    Returns:
        plane: (4,) array [nx, ny, nz, d]
        inlier_mask: Boolean array indicating inliers
    """
    if len(points) < 3:
        raise ValueError("At least 3 points required for plane fitting")
    
    random.seed(seed)
    np.random.seed(seed)
    
    best_inliers = None
    best_plane = None
    
    for _ in range(iterations):
        # Randomly select 3 points, but prefer points with lower z-values (table points)
        # This helps when bodies are floating above the table
        z_values = points[:, 2]
        # Create weights that strongly favor lower z values
        z_weights = np.exp(-5 * z_values / (z_values.max() + 1e-10))
        z_weights = z_weights / z_weights.sum()
        
        # Sample with replacement using the weights
        sampled_indices = np.random.choice(len(points), size=10, p=z_weights, replace=False)
        # From the sampled points, pick 3 distinct ones
        if len(np.unique(sampled_indices)) >= 3:
            idx1, idx2, idx3 = sampled_indices[:3]
        else:
            # Fallback to random sampling if not enough unique points
            idx1, idx2, idx3 = random.sample(range(len(points)), 3)
        p1, p2, p3 = points[idx1], points[idx2], points[idx3]
        
        # Create two vectors in the plane
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Calculate normal vector using cross product
        normal = np.cross(v1, v2)
        
        # Normalize normal vector
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue  # Degenerate case
        
        normal = normal / norm
        
        # Calculate plane equation: normal · (x - p1) = 0 → normal · x + d = 0
        d = -np.dot(normal, p1)
        plane = np.array([normal[0], normal[1], normal[2], d])
        
        # Calculate distances from all points to plane
        distances = np.abs(np.dot(points, normal) + d)
        
        # Find inliers
        inliers = distances < threshold_m
        
        # Update best plane if this one has more inliers
        if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_plane = plane
    
    if best_plane is None:
        raise ValueError("Could not find a valid plane")
    
    # Orient normal: majority of non-inliers should have positive signed distance
    non_inliers = ~best_inliers
    if np.sum(non_inliers) > 0:
        non_inlier_points = points[non_inliers]
        signed_distances = np.dot(non_inlier_points, best_plane[:3]) + best_plane[3]
        
        # Count positive vs negative distances
        pos_count = np.sum(signed_distances > 0)
        neg_count = np.sum(signed_distances < 0)
        
        # Flip normal if majority are negative
        if neg_count > pos_count:
            best_plane[:3] = -best_plane[:3]
            best_plane[3] = -best_plane[3]
    
    return best_plane, best_inliers

def points_above_plane(points: np.ndarray, plane: np.ndarray, 
                      min_h_m: float = 0.010, max_h_m: float = 0.200) -> np.ndarray:
    """Boolean mask of points whose signed distance to the plane lies in the band.
    
    Args:
        points: (N,3) array of points in METRES
        plane: (4,) array [nx, ny, nz, d] defining plane
        min_h_m: Minimum height above plane
        max_h_m: Maximum height above plane
        
    Returns:
        Boolean mask of points in the height band
    """
    signed_distances = np.dot(points, plane[:3]) + plane[3]
    return (signed_distances >= min_h_m) & (signed_distances <= max_h_m)

def dbscan(points: np.ndarray, eps_m: float = 0.012, min_points: int = 12) -> np.ndarray:
    """DBSCAN clustering using scipy.spatial.cKDTree.
    
    Returns an int label array, -1 for noise.
    
    Standard definition: A core point has at least min_points neighbours within eps,
    itself included. Clusters are the connected components of core points, and a
    non-core point within eps of a core point joins that cluster as a border point.
    
    Args:
        points: (N,3) array of points in METRES
        eps_m: Neighborhood radius in metres
        min_points: Minimum number of points to form a cluster
        
    Returns:
        labels: Array of cluster labels, -1 for noise
    """
    if len(points) == 0:
        return np.array([], dtype=int)
    
    # Build KDTree
    tree = cKDTree(points)
    
    # Find neighbors within eps
    neighbors = tree.query_ball_point(points, eps_m, return_length=True)
    
    # Initialize labels
    labels = np.full(len(points), -1, dtype=int)
    cluster_id = 0
    
    # Find core points
    is_core = neighbors >= min_points
    
    # DBSCAN algorithm
    for i, is_core_point in enumerate(is_core):
        if labels[i] != -1:  # Already visited
            continue
        
        if not is_core_point:
            continue
        
        # Start new cluster
        seeds = [i]
        labels[i] = cluster_id
        
        seed_idx = 0
        while seed_idx < len(seeds):
            current = seeds[seed_idx]
            
            # Find all points within eps of current point
            neighbor_indices = tree.query_ball_point(points[current], eps_m)
            
            for neighbor in neighbor_indices:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    if is_core[neighbor]:
                        seeds.append(neighbor)
            
            seed_idx += 1
        
        cluster_id += 1
    
    return labels

def median_spacing(points: np.ndarray) -> float:
    """Median nearest-neighbour distance, in metres. scipy.spatial.cKDTree, k=2."""
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    # Query for nearest neighbor (k=2: first is self, second is nearest neighbor)
    distances, _ = tree.query(points, k=2)
    # Take median of the second column (nearest neighbor distances)
    return float(np.median(distances[:, 1]))


def cluster_objects(points: np.ndarray, eps_m: float = None, 
                    k_eps: float = 3.0, min_points: int = 12, min_cluster_points: int = 150) -> tuple:
    """Return a list of index arrays, one per cluster, largest first.

    Clusters smaller than min_cluster_points are DISCARDED.

    Args:
        points: (N,3) array of points in METRES
        eps_m: Neighborhood radius in metres (if None, computed as k_eps * median_spacing)
        k_eps: Multiplier for median spacing when eps_m is None
        min_points: Minimum points to form a cluster
        min_cluster_points: Minimum cluster size to keep

    Returns:
        tuple: (list of index arrays, list of warning strings)
    """
    # Compute eps_m if not provided
    if eps_m is None:
        eps_m = k_eps * median_spacing(points)
    
    labels = dbscan(points, eps_m, min_points)
    
    # Group indices by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:  # Skip noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    # Filter clusters by size and sort
    valid_clusters = []
    warnings = []
    
    for label, indices in clusters.items():
        if len(indices) >= min_cluster_points:
            valid_clusters.append(np.array(indices))
        else:
            warnings.append(f"Dropped cluster with {len(indices)} points (minimum {min_cluster_points})")
    
    # Sort by size (largest first)
    valid_clusters.sort(key=len, reverse=True)
    
    return valid_clusters, warnings

def segment(points: np.ndarray, **kwargs) -> Dict[str, Any]:
    """Full segmentation pipeline.
    
    Returns:
        {"plane": (4,), "above_mask": bool array, "clusters": list of index arrays,
         "n_clusters": int, "warnings": list[str]}
    
    A warning MUST be emitted when any cluster was discarded for being under
    min_cluster_points, naming how many points it had.
    """
    # Extract parameters with defaults
    threshold_m = kwargs.get('threshold_m', 0.001)
    iterations = kwargs.get('iterations', 2000)
    seed = kwargs.get('seed', 0)
    min_h_m = kwargs.get('min_h_m', 0.010)
    max_h_m = kwargs.get('max_h_m', 0.200)
    eps_m = kwargs.get('eps_m', None)
    k_eps = kwargs.get('k_eps', 3.0)
    min_points = kwargs.get('min_points', 12)
    min_cluster_points = kwargs.get('min_cluster_points', 150)
    
    warnings = []
    
    try:
        # Step 1: Fit plane
        plane, inlier_mask = fit_plane_ransac(points, threshold_m, iterations, seed)
        
        # Step 2: Get points above plane
        above_mask = points_above_plane(points, plane, min_h_m, max_h_m)
        
         # Step 3: Cluster objects above plane
        above_points = points[above_mask]
        clusters, cluster_warnings = cluster_objects(above_points, eps_m, k_eps, min_points, min_cluster_points)
        
        # Convert cluster indices back to original point indices
        cluster_indices = []
        for cluster in clusters:
            # Map from above_points indices to original points indices
            original_indices = np.where(above_mask)[0][cluster]
            cluster_indices.append(original_indices)
        
        # Add any cluster warnings
        if cluster_warnings:
            warnings.extend(cluster_warnings)

        return {
            "plane": plane,
            "above_mask": above_mask,
            "clusters": cluster_indices,
            "n_clusters": len(cluster_indices),
            "eps_m_used": eps_m if eps_m is not None else k_eps * median_spacing(above_points),
            "warnings": warnings
        }
        
    except Exception as e:
        warnings.append(f"Segmentation failed: {str(e)}")
        return {
            "plane": None,
            "above_mask": np.zeros(len(points), dtype=bool),
            "clusters": [],
            "n_clusters": 0,
            "warnings": warnings
        }
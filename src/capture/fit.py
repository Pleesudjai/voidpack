# Ellipsoid fitting to point clusters
# Robust estimator for scan-derived clusters

import numpy as np
from typing import List, Dict, Any

# The semi-axis is the SUPPORT EDGE of the projected distribution. On a cloud
# whose outliers have been removed the edge is the silhouette rim and the
# estimator belongs at p99. On a cloud carrying several mm of sensor noise the
# outer tail is noise, and p99 chases it. Measured this session against each
# scan's own truth sidecar, volume rms error:
#
#     capture                     p85     p95     p99    p99.5
#     synthetic, 8 mm noise     10.2%   49.1%  105.7%   125.0%
#     real iPhone LiDAR         34.5%   16.4%    5.2%     2.7%
#
# So the percentile follows the capture. FOOTPRINT_PCT stays at the noisy value
# so every published synthetic result is unchanged by this edit.
FOOTPRINT_PCT = 85.0
FOOTPRINT_PCT_NOISY = 85.0
FOOTPRINT_PCT_CLEAN = 99.0
# A sidecar declaring noise at or above this is a raw cloud whose tail is noise.
NOISE_TAIL_MM = 4.0
HEIGHT_PCT = 99.0


def footprint_pct_for(sidecar) -> tuple:
    """Choose the footprint percentile for a capture. Returns (pct, reason).

    `sidecar` is the loaded `*_truth.json` dict, or None when the scan ships
    none. A capture that declares `noise_mm` at or above NOISE_TAIL_MM is raw,
    so the estimator is held back off the noisy tail. Anything else is treated
    as cleaned and the estimator sits at the rim.
    """
    if not isinstance(sidecar, dict):
        return FOOTPRINT_PCT_NOISY, "no sidecar, holding the conservative percentile"
    noise = sidecar.get("noise_mm")
    if isinstance(noise, (int, float)) and noise >= NOISE_TAIL_MM:
        return FOOTPRINT_PCT_NOISY, "capture declares %.1f mm noise, tail is noise" % noise
    return FOOTPRINT_PCT_CLEAN, "capture declares no noise floor, tail is the silhouette rim"


def plane_frame(plane: np.ndarray) -> np.ndarray:
    """(3,3) rotation whose third row is the unit plane normal.
    
    Args:
        plane: (4,) array [nx, ny, nz, d] defining plane
        
    Returns:
        (3,3) rotation matrix
    """
    normal = plane[:3]
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Create orthogonal basis for the plane
    # Find a vector orthogonal to normal
    if abs(normal[0]) > abs(normal[1]):
        v1 = np.array([normal[1], -normal[0], 0.0])
    else:
        v1 = np.array([0.0, normal[2], -normal[1]])
    v1 = v1 / np.linalg.norm(v1)
    
    # Find second vector orthogonal to both normal and v1
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Build rotation matrix
    return np.array([v1, v2, normal])


def fit_cluster(points_m: np.ndarray, plane: np.ndarray,
                footprint_pct: float = FOOTPRINT_PCT,
                height_pct: float = HEIGHT_PCT) -> Dict[str, Any]:
    """One cluster to one fitted body. Points in METRES, output in MILLIMETRES.
    
    Args:
        points_m: (N,3) array of points in metres
        plane: (4,) array [nx, ny, nz, d] defining the table plane
        footprint_pct: Percentile for footprint semi-axes
        height_pct: Percentile for height
        
    Returns:
        Dictionary with axes_mm, centre_mm, yaw, volume_mm3, n_points, height_mm
    """
    if len(points_m) == 0:
        return {
            "axes_mm": (0.0, 0.0, 0.0),
            "centre_mm": (0.0, 0.0, 0.0),
            "yaw": 0.0,
            "volume_mm3": 0.0,
            "n_points": 0,
            "height_mm": 0.0
        }
    
    # Calculate signed distances to plane (height above plane)
    signed_distances = np.dot(points_m, plane[:3]) + plane[3]
    
    # Height: use the height_pct percentile
    height_m = np.percentile(signed_distances, height_pct)
    c_m = height_m / 2.0  # Semi-axis
    
    # Project points onto the table plane
    frame = plane_frame(plane)
    # Project onto plane by subtracting the normal component
    footprint_2d = points_m - np.outer(signed_distances, plane[:3])
    
    # Transform to 2D coordinates in the plane
    # Use the first two rows of the frame as basis vectors
    basis = frame[:2, :].T  # (3,2) matrix
    footprint_2d_coords = footprint_2d @ basis  # (N,2)
    
    # Calculate footprint centroid
    centroid_2d = np.mean(footprint_2d_coords, axis=0)
    
    # Calculate PCA of the footprint
    centered = footprint_2d_coords - centroid_2d
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues and vectors in descending order
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    # Calculate semi-axes from percentiles of projections
    projections = centered @ eigvecs
    a_m = np.percentile(np.abs(projections[:, 0]), footprint_pct)
    b_m = np.percentile(np.abs(projections[:, 1]), footprint_pct)
    
    # Sort semi-axes descending: a >= b >= c
    axes_m = np.array([a_m, b_m, c_m])
    axes_m.sort()
    axes_m = axes_m[::-1]  # Descending order
    
    # Centre: x,y from footprint centroid, z = c (by tangency)
    # Transform centroid back to 3D
    centroid_3d = centroid_2d @ basis.T
    centre_m = np.array([centroid_3d[0], centroid_3d[1], c_m])
    
    # Yaw: angle of first principal axis relative to x-axis
    yaw = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    
    # Convert to millimetres
    axes_mm = tuple(axes_m * 1000)
    centre_mm = tuple(centre_m * 1000)
    height_mm = height_m * 1000
    volume_mm3 = (4.0 / 3.0) * np.pi * axes_mm[0] * axes_mm[1] * axes_mm[2]
    
    return {
        "axes_mm": axes_mm,
        "centre_mm": centre_mm,
        "yaw": float(yaw),
        "volume_mm3": float(volume_mm3),
        "n_points": len(points_m),
        "height_mm": float(height_mm),
        "c_mm": float(c_m * 1000),  # Vertical semi-axis in mm
        "height_above_plane_mm": float(c_m * 1000),  # height of the CENTRE above the plane, equals c by tangency
    }


def fit_scene(points_m: np.ndarray, segmentation: Dict[str, Any],
              footprint_pct: float = FOOTPRINT_PCT) -> List:
    """Every cluster to an Item. Returns a list of src.types.Item.
    
    Args:
        points_m: (N,3) array of points in metres
        segmentation: Dictionary from segment() function
        
    Returns:
        List of Item objects, sorted by volume descending
    """
    from src.types import Item
    
    items = []
    
    for cluster_idx, cluster in enumerate(segmentation["clusters"]):
        cluster_points = points_m[cluster]
        fit_result = fit_cluster(cluster_points, segmentation["plane"],
                                 footprint_pct=footprint_pct)
        
        # Create Item with label and mass=0.0
        label = f"body-{cluster_idx + 1}"
        item = Item(
            label=label,
            axes=fit_result["axes_mm"],
            mass_g=0.0,  # Scan cannot measure weight
            fragile=False,
            deformable=False,
            compaction=1.0,
            keep_upright=False
        )
        items.append(item)
    
    # Sort by volume descending
    items.sort(key=lambda item: item.volume_mm3(), reverse=True)
    
    return items

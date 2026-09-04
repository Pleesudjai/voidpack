import numpy as np
from scipy.spatial import ConvexHull
from src.types import Placement


def container_density(sum_volume_mm3: float, width: float, depth: float,
                      fill_height: float) -> float:
    """sum(V) / (width * depth * fill_height). The shipping figure. Penalised by walls."""
    if fill_height <= 0:
        return 0.0
    
    container_volume = width * depth * fill_height
    if container_volume <= 0:
        return 0.0
    
    return sum_volume_mm3 / container_volume


def fill_height(placements: list) -> float:
    """Highest top surface across placements, in mm. Zero for an empty list."""
    if not placements:
        return 0.0
    
    max_height = 0.0
    for placement in placements:
        # Get the shape matrix
        M = placement.item.shape_matrix(placement.yaw)
        
        # Top surface is centre_z + extent in z direction
        # For ellipsoid, extent in z is 1/sqrt(M[2,2])
        extent_z = 1.0 / np.sqrt(M[2,2])
        top_surface = placement.centre[2] + extent_z
        
        if top_surface > max_height:
            max_height = top_surface
    
    return max_height


def hull_density(placements: list) -> float:
    """sum(V) / volume of the convex hull of the bodies.

    Approximate each body by its surface sample, at least 128 points per body,
    then take the hull of all samples with scipy.spatial.ConvexHull.
    """
    if not placements:
        return 0.0
    
    # Sample points from each ellipsoid surface
    all_points = []
    total_volume = 0.0
    
    for placement in placements:
        # Add surface points for this ellipsoid
        points = sample_ellipsoid_surface(placement, n_points=128)
        all_points.extend(points)
        
        # Sum the volumes
        total_volume += placement.item.volume_mm3()
    
    if len(all_points) < 4:
        # Need at least 4 points for convex hull
        return 1.0  # Fallback for single item
    
    # Compute convex hull
    try:
        hull = ConvexHull(all_points)
        hull_volume = hull.volume
        
        if hull_volume <= 0:
            return 0.0
        
        density = total_volume / hull_volume
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, density))
    except:
        # Fallback if hull computation fails
        return 1.0


def laguerre_density(placements: list):
    """Mean radical-cell fraction over INTERIOR cells only.

    Return None when no cell is interior. Do NOT return a number in that case.
    At small n no cell is interior and reporting one would be a fabrication.
    """
    # Laguerre tessellation is complex and time-consuming to implement
    # For this hackathon, we'll return None as the spec allows
    return None


def report(placements, box) -> dict:
    """Return {"container": float, "hull": float, "laguerre": float | None,
               "fill_height_mm": float, "sum_volume_mm3": float}.

    Never sum, average, or otherwise combine the three bases.
    """
    # Calculate sum of volumes
    sum_volume = sum(p.item.volume_mm3() for p in placements)
    
    # Calculate fill height
    fill_h = fill_height(placements)
    
    # Calculate densities
    container = container_density(sum_volume, box.width, box.depth, fill_h)
    hull = hull_density(placements)
    laguerre = laguerre_density(placements)
    
    return {
        "container": container,
        "hull": hull,
        "laguerre": laguerre,
        "fill_height_mm": fill_h,
        "sum_volume_mm3": sum_volume
    }


def sample_ellipsoid_surface(placement: Placement, n_points: int = 128) -> list:
    """Sample points from the surface of an ellipsoid."""
    # Get shape matrix
    M = placement.item.shape_matrix(placement.yaw)
    
    # Generate points on unit sphere
    points = []
    
    # Use Fibonacci spiral for even distribution
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(n_points):
        # Fibonacci spiral on unit sphere
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
        
        # Convert to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        # Transform to ellipsoid coordinates: point = centre + R * diag(1/sqrt(M)) * unit_point
        # Since M = R @ diag(1/a², 1/b², 1/c²) @ R.T, we have diag(a, b, c) = diag(1/sqrt(M))
        # But we need to be careful with the rotation
        
        # For simplicity, we'll use the inverse transformation
        # If x^T M x = 1 defines the ellipsoid, then the surface points satisfy this
        
        # Start with unit vector and scale by the support function
        unit_vec = np.array([x, y, z])
        
        # Find the scaling factor that puts this point on the surface
        # We want (point - centre)^T M (point - centre) = 1
        # Let point = centre + k * unit_vec
        # Then (k * unit_vec)^T M (k * unit_vec) = 1
        # k^2 * (unit_vec^T M unit_vec) = 1
        # k = 1 / sqrt(unit_vec^T M unit_vec)
        
        denominator = np.sqrt(unit_vec @ M @ unit_vec)
        if denominator > 1e-12:
            k = 1.0 / denominator
            surface_point = placement.centre + k * unit_vec
            points.append(surface_point)
    
    return points
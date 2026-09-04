import numpy as np
from src.types import Placement
from src.geometry.contact import contact_mu

TOLERANCE_MM = 1e-3


class OverlapRejected(Exception):
    """Carries .penetration_mm."""
    
    def __init__(self, penetration_mm: float):
        self.penetration_mm = penetration_mm
        super().__init__(f"Overlap detected: {penetration_mm:.6f} mm exceeds tolerance {TOLERANCE_MM:.6f} mm")


def pair_penetration_mm(pa: Placement, pb: Placement) -> float:
    """Linear interpenetration between two Placements, in mm. Zero when disjoint."""
    # Vector from pa centre to pb centre
    d = pb.centre - pa.centre
    d_norm = np.linalg.norm(d)
    
    if d_norm == 0.0:
        return 0.0  # Same centre, but contact_mu will handle this
    
    # Get shape matrices
    A = pa.item.shape_matrix(pa.yaw)
    B = pb.item.shape_matrix(pb.yaw)
    
    # Contact factor
    mu = contact_mu(A, B, d)
    
    # Penetration = |d| * (1 - mu) / mu, clamped to >= 0
    # For touching spheres: mu = 1.0, so penetration = 0.0
    penetration = d_norm * max(0.0, (1.0 - mu) / mu)
    
    # Clamp small numerical errors to zero for touching case
    if penetration < 1e-12:
        return 0.0
    
    return penetration


def max_penetration_mm(placements: list) -> float:
    """Largest pairwise penetration across all pairs, in mm. Zero for an empty
    or single-item list."""
    if len(placements) <= 1:
        return 0.0
    
    max_penetration = 0.0
    
    # Check all unique pairs
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            penetration = pair_penetration_mm(placements[i], placements[j])
            if penetration > max_penetration:
                max_penetration = penetration
    
    return max_penetration


def gate(placements: list) -> None:
    """Raise OverlapRejected when max_penetration_mm exceeds TOLERANCE_MM.

    Return None when the configuration is acceptable.
    """
    max_pen = max_penetration_mm(placements)
    
    if max_pen > TOLERANCE_MM:
        raise OverlapRejected(max_pen)
    
    # Configuration is acceptable
    return None
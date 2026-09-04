from dataclasses import dataclass
import numpy as np

MM_PER_M = 1000.0

@dataclass
class Item:
    """One packable body, modelled as a triaxial ellipsoid. All lengths in mm."""
    label: str
    axes: tuple[float, float, float]      # semi-axes a, b, c in mm
    mass_g: float = 0.0
    fragile: bool = False
    deformable: bool = False
    compaction: float = 1.0               # multiplies each semi-axis, 1.0 = rigid
    keep_upright: bool = False

    def volume_mm3(self) -> float:
        """Ellipsoid volume, 4/3 pi a b c, in mm^3."""
        a, b, c = self.axes
        return (4.0 / 3.0) * np.pi * a * b * c

    def shape_matrix(self, yaw: float = 0.0) -> np.ndarray:
        """(3,3) SPD matrix for the Perram-Wertheim test, rotated by yaw about z.

        Uses the EFFECTIVE semi-axes, that is axes scaled by compaction.
        """
        a, b, c = self.axes
        ae = a * self.compaction
        be = b * self.compaction
        ce = c * self.compaction
        
        # Rotation matrix about z-axis
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Shape matrix in canonical frame
        M_canon = np.diag([1.0/(ae**2), 1.0/(be**2), 1.0/(ce**2)])
        
        # Rotate to world frame
        return R @ M_canon @ R.T

@dataclass
class Box:
    """Container, inner dimensions in mm.

    `name` is a catalog label such as S, M, L or XL. It is display and lookup
    only, no solver code reads it, and it defaults to empty so a bare
    Box(w, d, h) stays valid.
    """
    width: float
    depth: float
    height: float
    name: str = ""

    def volume_mm3(self) -> float:
        """Box volume in mm^3."""
        return self.width * self.depth * self.height

@dataclass
class Placement:
    """One item placed. centre in mm from the box inner corner at the origin."""
    item: Item
    centre: np.ndarray                    # (3,) mm
    yaw: float = 0.0                      # radians about z
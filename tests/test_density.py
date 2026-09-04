import numpy as np
import pytest
from src.types import Item, Box, Placement
from src.solver.density import container_density, fill_height, hull_density, laguerre_density, report


def test_container_density_simple():
    """Simple container density calculation."""
    # 1000 mm³ volume in 100x100x10 mm container
    density = container_density(1000.0, 100.0, 100.0, 10.0)
    expected = 1000.0 / (100.0 * 100.0 * 10.0)  # = 0.1
    assert np.isclose(density, expected)


def test_container_density_zero_fill_height():
    """Zero fill height should return 0."""
    density = container_density(1000.0, 100.0, 100.0, 0.0)
    assert density == 0.0


def test_fill_height_empty():
    """Empty placements should return 0."""
    assert fill_height([]) == 0.0


def test_fill_height_single_sphere():
    """Single sphere fill height."""
    item = Item(label="test", axes=(10.0, 10.0, 10.0))
    placement = Placement(item, centre=np.array([0.0, 0.0, 5.0]))
    
    height = fill_height([placement])
    expected = 5.0 + 10.0  # centre + radius
    assert np.isclose(height, expected)


def test_fill_height_multiple():
    """Multiple items fill height."""
    item1 = Item(label="low", axes=(10.0, 10.0, 10.0))
    item2 = Item(label="high", axes=(8.0, 8.0, 8.0))
    
    p1 = Placement(item1, centre=np.array([0.0, 0.0, 10.0]))
    p2 = Placement(item2, centre=np.array([20.0, 20.0, 15.0]))
    
    height = fill_height([p1, p2])
    expected = 15.0 + 8.0  # high item centre + its radius
    assert np.isclose(height, expected)


def test_hull_density_single_item():
    """Single item hull density should be close to 1.0."""
    item = Item(label="test", axes=(10.0, 10.0, 10.0))
    placement = Placement(item, centre=np.array([0.0, 0.0, 0.0]))
    
    density = hull_density([placement])
    # Should be close to 1.0 for a single ellipsoid
    assert 0.9 <= density <= 1.0


def test_hull_density_empty():
    """Empty placements should return 0."""
    assert hull_density([]) == 0.0


def test_laguerre_density_returns_none():
    """Laguerre density should return None (not implemented)."""
    item = Item(label="test", axes=(10.0, 10.0, 10.0))
    placement = Placement(item, centre=np.array([0.0, 0.0, 0.0]))
    
    assert laguerre_density([placement]) is None


def test_report_structure():
    """Report should have correct structure."""
    item = Item(label="test", axes=(10.0, 10.0, 10.0))
    placement = Placement(item, centre=np.array([0.0, 0.0, 10.0]))
    box = Box(width=100.0, depth=100.0, height=50.0)
    
    result = report([placement], box)
    
    assert "container" in result
    assert "hull" in result
    assert "laguerre" in result
    assert "fill_height_mm" in result
    assert "sum_volume_mm3" in result
    
    # Check values
    expected_volume = (4.0 / 3.0) * np.pi * 10.0**3
    assert np.isclose(result["sum_volume_mm3"], expected_volume)
    assert np.isclose(result["fill_height_mm"], 20.0)  # 10 + 10
    
    # Container density
    container_vol = 100.0 * 100.0 * 20.0
    expected_container = expected_volume / container_vol
    assert np.isclose(result["container"], expected_container)
    
    # Hull density should be reasonable
    assert 0.0 <= result["hull"] <= 1.0
    
    # Laguerre should be None
    assert result["laguerre"] is None
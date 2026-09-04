# Concrete Packing Benchmark - Andreasen & Andersen
# Provides reference metrics for evaluating packing density

import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from src.types import Item, Box

# Best known packing density for 15 equal spheres in a cube (Gensane 2004)
GENSANE_N15_CUBE = 2500.0 * math.pi / 17576.0  # = 0.446858308715

def equivalent_diameter_mm(item: Item) -> float:
    """Calculate equivalent spherical diameter (mm) for an item.
    
    Equivalent diameter is 2 * (3V / 4pi)^(1/3), the diameter of 
    the sphere of the same volume.
    """
    volume_mm3 = item.volume_mm3()
    return 2.0 * (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0/3.0)

def aa_target_cpft(d_mm: float, d_min: float, d_max: float, q: float = 0.37) -> float:
    """Modified Andreasen and Andersen cumulative percent finer than.
    
    CPFT(D) = (D**q - d_min**q) / (d_max**q - d_min**q)
    
    Returns 0.0 at d_min and 1.0 at d_max exactly. Clamped outside [d_min, d_max].
    UNVERIFIED-EQUATION: the functional form has not been cross-checked against
    the Funk and Dinger primary source in this repository.
    """
    if d_mm <= d_min:
        return 0.0
    elif d_mm >= d_max:
        return 1.0
    else:
        numerator = d_mm**q - d_min**q
        denominator = d_max**q - d_min**q
        return numerator / denominator

def actual_cpft(items: List[Item], d_mm: float) -> float:
    """Volume fraction of item set with equivalent diameter <= d_mm.
    
    Weighted by volume, not by count.
    """
    total_volume = sum(item.volume_mm3() for item in items)
    if total_volume == 0:
        return 0.0
    
    volume_below = 0.0
    for item in items:
        eq_diam = equivalent_diameter_mm(item)
        if eq_diam <= d_mm:
            volume_below += item.volume_mm3()
    
    return volume_below / total_volume

def grading_deviation(items: List[Item], q: float = 0.37, samples: int = 64) -> float:
    """Root mean square difference between actual and target grading curves.
    
    Sampled uniformly in log10(d) across the item size range.
    Zero means the item set already sits on the ideal grading curve.
    """
    if not items:
        return 0.0
    
    # Get size range
    diameters = [equivalent_diameter_mm(item) for item in items]
    d_min = min(diameters)
    d_max = max(diameters)
    
    if d_min >= d_max:
        return 0.0
    
    # Sample points uniformly in log space
    log_min = math.log10(d_min)
    log_max = math.log10(d_max)
    log_samples = np.linspace(log_min, log_max, samples)
    d_samples = 10 ** log_samples
    
    # Calculate RMS deviation
    total_deviation = 0.0
    for d in d_samples:
        target = aa_target_cpft(d, d_min, d_max, q)
        actual = actual_cpft(items, d)
        deviation = target - actual
        total_deviation += deviation ** 2
    
    return math.sqrt(total_deviation / samples)

def best_q(items: List[Item], q_lo: float = 0.20, q_hi: float = 0.50, steps: int = 61) -> float:
    """Find the q in [q_lo, q_hi] that minimizes grading deviation.
    
    Grid search, deterministic.
    """
    if not items:
        return 0.37  # Default value
    
    q_values = np.linspace(q_lo, q_hi, steps)
    min_deviation = float('inf')
    best_q_val = 0.37
    
    for q in q_values:
        deviation = grading_deviation(items, q)
        if deviation < min_deviation:
            min_deviation = deviation
            best_q_val = q
    
    return best_q_val

def wall_ratio(box: Box, items: List[Item]) -> float:
    """Smallest box inner dimension divided by largest item equivalent diameter.
    
    A diagnostic for wall effect in confined packings.
    """
    if not items:
        return 0.0
    
    box_dims = [box.width, box.depth, box.height]
    min_box_dim = min(box_dims)
    
    diameters = [equivalent_diameter_mm(item) for item in items]
    max_item_diam = max(diameters)
    
    return min_box_dim / max_item_diam

def benchmark(items: List[Item], box: Box, achieved_container_density: float) -> Dict[str, Any]:
    """Generate benchmark report comparing achieved density to theoretical references.
    
    Returns comprehensive analysis including grading curve fit and wall effect.
    """
    n_items = len(items)
    notes = []
    
    # Calculate basic metrics
    q_used = 0.37  # Default value
    best_q_val = best_q(items, q_lo=0.20, q_hi=0.50, steps=61)
    grading_dev = grading_deviation(items, q=q_used)
    wall_ratio_val = wall_ratio(box, items)
    
    # Equal sphere ceiling (only for n=15)
    equal_sphere_ceiling = GENSANE_N15_CUBE if n_items == 15 else None
    fraction_of_ceiling = None
    
    if equal_sphere_ceiling is not None:
        fraction_of_ceiling = achieved_container_density / equal_sphere_ceiling
        notes.append(f"Equal-sphere ceiling (n=15): {equal_sphere_ceiling:.4f}")
        notes.append(f"Achieved {achieved_container_density:.4f} = {fraction_of_ceiling:.2%} of ceiling")
    else:
        notes.append(f"No published ceiling available for n={n_items} items")
    
    # Add interpretation notes
    if grading_dev < 0.05:
        notes.append(f"Item set closely follows ideal grading curve (deviation {grading_dev:.4f})")
    elif grading_dev < 0.15:
        notes.append(f"Item set moderately deviates from ideal grading (deviation {grading_dev:.4f})")
    else:
        notes.append(f"Item set significantly deviates from ideal grading (deviation {grading_dev:.4f})")
    
    if wall_ratio_val < 3.0:
        notes.append(f"Strong wall effect expected (wall ratio {wall_ratio_val:.2f})")
    elif wall_ratio_val < 5.0:
        notes.append(f"Moderate wall effect expected (wall ratio {wall_ratio_val:.2f})")
    else:
        notes.append(f"Minimal wall effect expected (wall ratio {wall_ratio_val:.2f})")
    
    return {
        "q_used": q_used,
        "best_q": best_q_val,
        "grading_deviation": grading_dev,
        "wall_ratio": wall_ratio_val,
        "n_items": n_items,
        "equal_sphere_ceiling": equal_sphere_ceiling,
        "fraction_of_ceiling": fraction_of_ceiling,
        "notes": notes
    }
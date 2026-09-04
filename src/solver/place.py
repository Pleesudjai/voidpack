import numpy as np
from src.types import Item, Box, Placement
from src.geometry.gate import gate, pair_penetration_mm


def pack(items: list, box: Box, ruleset: dict, seed: int = 0,
         grid: int = 12, yaws: int = 4) -> tuple[list, list]:
    """Place items into box under the compiled ruleset.

    Returns (placements, unplaced) where placements is a list of Placement and
    unplaced is a list of Item that did not fit.

    Deterministic for a given seed.
    """
    # Set random seed for determinism
    np.random.seed(seed)
    
    # Extract rules for quick lookup
    fragile_items = set()
    upright_items = set()
    max_weight_g = None
    
    for rule in ruleset.get("rules", []):
        if rule["kind"] == "fragile":
            _bind(rule["item"], items, fragile_items)
        elif rule["kind"] == "keep_upright":
            _bind(rule["item"], items, upright_items)
        elif rule["kind"] == "max_weight":
            max_weight_g = rule["limit_g"]
    
    # Sort items: largest first, deformable last
    def sort_key(item):
        # Deformable items go last (highest sort value)
        deformable_penalty = 1000000 if item.deformable else 0
        return deformable_penalty - item.volume_mm3()
    
    sorted_items = sorted(items, key=sort_key)
    
    placements = []
    unplaced = []
    current_weight_g = 0.0
    
    # Generate candidate grid positions
    # Candidate x, y positions are built PER BODY below, inset by the body's own
    # horizontal extent, so wall-snug placements are reachable and every candidate
    # is inside the box in x and y by construction. A fixed cell-centre grid could
    # never place a body whose footprint equals the box footprint.
    
    # Generate candidate yaw angles
    yaw_angles = np.linspace(0, np.pi, yaws, endpoint=False)
    
    for item in sorted_items:
        # Check weight limit before attempting placement
        if max_weight_g is not None:
            if current_weight_g + item.mass_g > max_weight_g:
                unplaced.append(item)
                continue
        
        best_placement = None
        best_z = float('inf')
        
        # Try all orientations
        for up_axis in range(3):  # Try each semi-axis as "up"
            # Skip if item must stay upright and this isn't the original up axis
            if item.label in upright_items and up_axis != 2:
                continue
            
            # Create permuted axes
            axes = list(item.axes)
            if up_axis == 0:
                # Rotate so first axis becomes up (z)
                permuted_axes = (axes[2], axes[1], axes[0])
            elif up_axis == 1:
                # Rotate so second axis becomes up (z)  
                permuted_axes = (axes[0], axes[2], axes[1])
            else:
                # Original orientation
                permuted_axes = tuple(axes)
            
            # Create temporary item with permuted axes, but keep reference to original
            temp_item = Item(
                label=item.label,
                axes=permuted_axes,
                mass_g=item.mass_g,
                fragile=item.fragile,
                deformable=item.deformable,
                compaction=item.compaction,
                keep_upright=item.keep_upright
            )
            # Store reference to original item for labelling and for identity.
            temp_item._original = item
            # When the permutation is the identity the original object IS the
            # correct geometry, so use it and preserve object identity.
            if permuted_axes == tuple(axes):
                temp_item = item
            
            # Inset grid for THIS body and orientation family. The horizontal
            # extent is bounded above by the larger horizontal semi-axis for any
            # yaw, so containment in x and y holds for every candidate.
            emax = float(max(permuted_axes[0], permuted_axes[1])) * float(item.compaction)
            if box.width - 2.0 * emax < -1e-9 or box.depth - 2.0 * emax < -1e-9:
                continue                                  # cannot fit in this orientation
            x_positions = np.unique(np.linspace(emax, box.width - emax, grid))
            y_positions = np.unique(np.linspace(emax, box.depth - emax, grid))

            # Try all grid positions
            for x_idx, x in enumerate(x_positions):
                for y_idx, y in enumerate(y_positions):
                    # Try all yaw angles
                    for yaw in yaw_angles:
                        # Find resting height using bisection
                        candidate = find_resting_position(temp_item, x, y, yaw, box, placements, fragile_items)
                        
                        if candidate is not None:
                            # Check if this is the best candidate so far
                            if candidate.centre[2] < best_z or (np.isclose(candidate.centre[2], best_z) and 
                                 (x_idx * grid + y_idx) < (best_x_idx * grid + best_y_idx if best_placement else float('inf'))):
                                best_placement = candidate
                                best_z = candidate.centre[2]
                                best_x_idx = x_idx
                                best_y_idx = y_idx
        
        if best_placement is not None:
            # Store the PERMUTED item, not the original.
            #
            # The search validated this position using the permuted axes, which are
            # the body's actual orientation once placed. Storing the original axes
            # instead put different geometry at the same centre, which produced a
            # 0.069 mm overlap the gate correctly refused at eight items. The
            # original is kept on `_original` for labelling only.
            placements.append(best_placement)
            current_weight_g += item.mass_g
        else:
            unplaced.append(item)
    
    # Final validation
    gate(placements)
    
    return placements, unplaced


def find_resting_position(item: Item, x: float, y: float, yaw: float, box: Box, 
                         existing: list, fragile_items: set) -> Placement:
    """Find the lowest z where item can rest without overlapping existing items."""
    
    # Get shape matrix for this orientation
    M = item.shape_matrix(yaw)
    
    # Calculate support radius in z direction (lowest possible z)
    # For ellipsoid, support in z direction is 1/sqrt(M[2,2])
    support_z = 1.0 / np.sqrt(M[2,2])
    
    # Bisection bracket
    z_low = support_z  # At least support radius above floor
    z_high = box.height - support_z  # At least support radius below ceiling
    
    # Check if item fits in box at all
    if z_low > z_high:
        return None
    
    # Bisection to find resting height
    for _ in range(50):  # 50 iterations gives ~1e-15 precision
        if z_high - z_low < 1e-6:
            break
        
        z_mid = (z_low + z_high) / 2
        centre = np.array([x, y, z_mid])
        
        # Check if this position is valid
        if is_position_valid(item, centre, yaw, box, existing, fragile_items):
            z_high = z_mid  # Try lower
        else:
            z_low = z_mid  # Need higher
    
    # Final check at the found position.
    #
    # The bisection assumes validity is monotonic in z, which fails when a body
    # can rest in a pocket between two others. That left residual overlaps of
    # about 0.07 mm at eight items, above the 1e-3 mm gate. Rather than trust the
    # bracket, lift the body until it is genuinely valid, then verify.
    centre = np.array([x, y, z_high])
    if is_position_valid(item, centre, yaw, box, existing, fragile_items):
        return Placement(item=item, centre=centre, yaw=yaw)

    step = 1e-4                      # 0.1 mm, 100x the gate tolerance
    for _ in range(60):
        z_high += step
        if z_high > box.height:
            return None
        centre = np.array([x, y, z_high])
        if is_position_valid(item, centre, yaw, box, existing, fragile_items):
            return Placement(item=item, centre=centre, yaw=yaw)
    return None


def _bind(name, items, target: set) -> None:
    """Resolve a rule's item name to the labels it governs.

    An EXACT label binds that body alone, so `egg-3 is fragile` leaves egg-4
    unconstrained. A name that is not a label but IS a kind present in the
    scene binds every copy of that kind, so `the eggs are fragile` covers
    egg-1 through egg-10. Reconciled 2026-09-03 with the scene-label check in
    src/air/tools.py, which refuses a name that is neither.
    """
    name = str(name)
    labels = {it.label for it in items}
    if name in labels:
        target.add(name)
        return
    matched = {it.label for it in items if base_kind(it.label) == name}
    target.update(matched or {name})


def base_kind(label: str) -> str:
    """Strip a copy suffix, so `egg-3` and `egg-7` are both `egg`.

    The library expands one fitted body into copies labelled `<kind>-<n>`, and
    the fragile rule has to tell "another egg" apart from "a rock".
    """
    head, sep, tail = str(label).rpartition("-")
    return head if (sep and tail.isdigit() and head) else str(label)


def same_kind(a: str, b: str) -> bool:
    """True when two labels are copies of the same library body."""
    return base_kind(a) == base_kind(b)


def is_position_valid(item: Item, centre: np.ndarray, yaw: float, box: Box, 
                     existing: list, fragile_items: set) -> bool:
    """Check if position is valid (inside box, no overlaps, no fragile violations)."""
    
    # Check box containment
    if not is_inside_box(item, centre, yaw, box):
        return False
    
    # Check overlaps with existing items.
    #
    # Conservative outer-bound early-out (T06b change 1). If the two bounding
    # spheres are separated the bodies are provably disjoint, so the exact test
    # is skipped. It is NEVER used to conclude that two bodies overlap, so it can
    # never hide an interpenetration. The largest semi-axis is rotation-invariant,
    # so no eigen-solve is needed.
    r_me = float(max(item.axes)) * float(item.compaction)
    me = None
    for existing_placement in existing:
        other = existing_placement.item
        r_other = float(max(other.axes)) * float(other.compaction)
        d = centre - existing_placement.centre
        if float(d @ d) > (r_me + r_other) ** 2:
            continue                                   # provably disjoint
        if me is None:
            me = Placement(item=item, centre=centre, yaw=yaw)
        penetration = pair_penetration_mm(me, existing_placement)
        if penetration > 1e-6:  # Any overlap
            return False
    
    # Fragile rule, BOTH directions. "Nothing above a fragile item" must hold
    # whichever body is placed second. The original check only ran when the
    # fragile body itself was being placed, so a heavier body placed later could
    # land on top of it unchallenged. Found by the theorem test on 2026-09-03.
    for existing_placement in existing:
        ex = existing_placement.item
        if (ex.label in fragile_items or ex.fragile) and            not same_kind(item.label, ex.label) and            footprint_overlaps(item, centre, yaw, ex, existing_placement.centre, existing_placement.yaw) and            centre[2] > existing_placement.centre[2] + 1e-6:
            return False

    # Check fragile items
    if item.label in fragile_items:
        # Check if any existing item is above this fragile item. A body of the
        # SAME kind is allowed above, because a tray of eggs stacks on itself.
        # Anything else is refused. Author's rule, 2026-09-03.
        for existing_placement in existing:
            if same_kind(item.label, existing_placement.item.label):
                continue
            # Check x,y footprint overlap
            if footprint_overlaps(item, centre, yaw, existing_placement.item, existing_placement.centre, existing_placement.yaw):
                # Check if existing item is above
                if existing_placement.centre[2] > centre[2] + 1e-6:
                    return False
    
    return True


def is_inside_box(item: Item, centre: np.ndarray, yaw: float, box: Box) -> bool:
    """Check if item is fully contained within box."""
    
    M = item.shape_matrix(yaw)
    
    # For each axis, check containment
    # x-axis (width)
    extent_x = 1.0 / np.sqrt(M[0,0])
    if centre[0] - extent_x < 0 or centre[0] + extent_x > box.width:
        return False
    
    # y-axis (depth)
    extent_y = 1.0 / np.sqrt(M[1,1])
    if centre[1] - extent_y < 0 or centre[1] + extent_y > box.depth:
        return False
    
    # z-axis (height)
    extent_z = 1.0 / np.sqrt(M[2,2])
    if centre[2] - extent_z < 0 or centre[2] + extent_z > box.height:
        return False
    
    return True


def footprint_overlaps(item1: Item, centre1: np.ndarray, yaw1: float,
                      item2: Item, centre2: np.ndarray, yaw2: float) -> bool:
    """Check if x,y footprints overlap (ignoring z)."""
    
    M1 = item1.shape_matrix(yaw1)
    M2 = item2.shape_matrix(yaw2)
    
    # x-extent for both items
    extent1_x = 1.0 / np.sqrt(M1[0,0])
    extent2_x = 1.0 / np.sqrt(M2[0,0])
    
    # y-extent for both items
    extent1_y = 1.0 / np.sqrt(M1[1,1])
    extent2_y = 1.0 / np.sqrt(M2[1,1])
    
    # Check x overlap
    x_overlap = not (centre1[0] + extent1_x < centre2[0] - extent2_x or 
                    centre1[0] - extent1_x > centre2[0] + extent2_x)
    
    # Check y overlap  
    y_overlap = not (centre1[1] + extent1_y < centre2[1] - extent2_y or
                    centre1[1] - extent1_y > centre2[1] + extent2_y)
    
    return x_overlap and y_overlap
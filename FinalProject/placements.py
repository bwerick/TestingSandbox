# placements.py


def compute_relative_xy(block, relation: str, offset: float = 30.0):
    """
    Given a reference block and a relation string, compute (x_mm, y_mm)
    where we should place the source block.

    relation: "left_of", "right_of", "above", "below", "on_top", or None.
    For "on_top", we just reuse (x, y); Z will be handled in skills.
    """
    x, y = block.x_mm, block.y_mm

    if relation == "left_of":
        return x - offset, y

    if relation == "right_of":
        return x + offset, y

    if relation == "above":
        return x, y + offset

    if relation == "below":
        return x, y - offset

    # For "on_top" or unknown, just return same XY;
    # the Z will be different in the executor/skills layer.
    return x, y

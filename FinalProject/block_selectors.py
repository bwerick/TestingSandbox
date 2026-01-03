# block_selectors.py

from scene_state import SceneState, Block


def resolve_block_selector(selector: dict, scene: SceneState) -> Block:
    """
    Convert a selector dict (from LLM) into a specific Block.

    Example selector:
    {
      "color": "red",
      "extremum": "leftmost",   # or rightmost, center, or null
      "relation": "furthest_from",  # or closest_to, or null
      "reference": { ... another selector ... }  # or null
    }
    """

    color = selector.get("color")
    extremum = selector.get("extremum")  # leftmost, rightmost, center, or None
    relation = selector.get("relation")  # furthest_from, closest_to, or None
    reference = selector.get("reference")

    # Filter by color if provided
    blocks = scene.get_blocks_by_color(color) if color else scene.blocks
    if not blocks:
        raise ValueError(f"No blocks found for selector {selector}")

    # -----------------
    # Extremum selectors
    # -----------------
    if extremum == "leftmost":
        return scene.leftmost(blocks)

    if extremum == "rightmost":
        return scene.rightmost(blocks)

    if extremum == "center":
        return scene.center_block(color=color)

    # -----------------
    # Relational selectors
    # -----------------
    if relation:
        if not reference:
            raise ValueError(f"relation={relation} but no reference given: {selector}")

        ref_block = resolve_block_selector(reference, scene)

        if relation == "furthest_from":
            return scene.furthest_from(ref_block, blocks)

        if relation == "closest_to":
            return scene.closest_to(ref_block, blocks)

    # -----------------
    # Fallback: just the first matching block
    # -----------------
    return blocks[0]

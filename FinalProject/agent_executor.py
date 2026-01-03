# agent_executor.py

from scene_state import SceneState
from block_selectors import resolve_block_selector
from placements import compute_relative_xy
from skills import pick_block_at, place_block_at, place_block_on_top_of


def execute_task(task: dict, scene: SceneState, device):
    """
    Execute a high-level task JSON on the robot.
    Supports:
      - task_type == "move_single"
      - task_type == "group_color"  (simple line layout)
    """
    task_type = task.get("task_type")

    if task_type == "move_single":
        return _execute_move_single(task, scene, device)

    if task_type == "group_color":
        return _execute_group_color(task, scene, device)

    raise ValueError(f"Unsupported task_type: {task_type}")


def _execute_move_single(task: dict, scene: SceneState, device):
    source_sel = task["source_selector"]
    target_sel = task.get("target_selector")
    placement = task.get("placement") or "right_of"
    offset = float(task.get("offset_mm", 30.0))

    # Resolve source block
    source_block = resolve_block_selector(source_sel, scene)

    # If we have a target, resolve it
    target_block = resolve_block_selector(target_sel, scene) if target_sel else None

    print(
        f"[move_single] source={source_block.color}@({source_block.x_mm:.1f},{source_block.y_mm:.1f})"
    )
    if target_block:
        print(
            f"[move_single] target={target_block.color}@({target_block.x_mm:.1f},{target_block.y_mm:.1f})"
        )
    print(f"[move_single] placement={placement}, offset={offset}mm")

    # 1) Pick up the source block
    pick_block_at(device, source_block.x_mm, source_block.y_mm)

    # 2) Place depending on placement
    if placement == "on_top" and target_block is not None:
        place_block_on_top_of(device, target_block)
    else:
        if target_block is not None:
            x_place, y_place = compute_relative_xy(target_block, placement, offset)
        else:
            # Fallback: just drop near where it was
            x_place, y_place = source_block.x_mm + offset, source_block.y_mm

        # For now, all these relative placements go on the table
        place_block_at(device, x_place, y_place, on_table=True)


def _execute_group_color(task: dict, scene: SceneState, device):
    """
    Simple: group all blocks of a color into a horizontal line on the left/right/center.
    """
    selector = task["group_selector"]
    color = selector["color"]

    region = task.get("group_region", "left")
    spacing = float(task.get("spacing_mm", 30.0))

    blocks = scene.get_blocks_by_color(color)
    if not blocks:
        print(f"[group_color] No blocks of color {color}")
        return

    # Sort blocks by current y (just to have a consistent order)
    blocks_sorted = sorted(blocks, key=lambda b: b.y_mm)

    # Decide base X depending on region
    all_x = [b.x_mm for b in scene.blocks]
    min_x, max_x = min(all_x), max(all_x)
    mid_x = 0.5 * (min_x + max_x)

    if region == "left":
        base_x = min_x - 2 * spacing
    elif region == "right":
        base_x = max_x + 2 * spacing
    else:  # center
        base_x = mid_x

    # Use average Y of the group as baseline
    avg_y = sum(b.y_mm for b in blocks_sorted) / len(blocks_sorted)

    print(
        f"[group_color] Grouping {len(blocks_sorted)} {color} blocks in region={region}"
    )
    print(
        f"[group_color] base_x={base_x:.1f}, base_y={avg_y:.1f}, spacing={spacing:.1f}"
    )

    # Move each block one by one
    for i, b in enumerate(blocks_sorted):
        target_x = base_x + i * spacing
        target_y = avg_y

        print(
            f"[group_color] Moving block {i}: from ({b.x_mm:.1f},{b.y_mm:.1f}) to ({target_x:.1f},{target_y:.1f})"
        )

        pick_block_at(device, b.x_mm, b.y_mm)
        place_block_at(device, target_x, target_y, on_table=True)

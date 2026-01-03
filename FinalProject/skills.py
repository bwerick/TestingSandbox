# skills.py

import time
from robot_control import move_to, set_suction, move_to_safe_pose, connect_robot

# --- Z levels (tweak if needed) ---

Z_TABLE = -55.6  # measured when tool just touches the table
Z_BLOCK_TOP = -49.6  # measured when tool touches top of a block
BLOCK_HEIGHT = Z_BLOCK_TOP - Z_TABLE  # ~6 mm

# Approach heights
Z_APPROACH_ABOVE_BLOCK = Z_BLOCK_TOP + 30.0  # 30 mm above block top
Z_APPROACH_ABOVE_TABLE = Z_TABLE + 30.0  # 30 mm above table

# Pick/place depths
PICK_OFFSET = -1.5  # mm into block
Z_PICK_ON_BLOCK = Z_BLOCK_TOP - PICK_OFFSET
Z_PLACE_ON_TABLE = Z_TABLE + 1.0  # ~1mm above table
Z_PLACE_ON_BLOCK = Z_BLOCK_TOP + 1.0  # ~1mm above top of a block


def pick_block_at(device, x_mm: float, y_mm: float, r_deg: float = 0.0):
    print(f"[pick_block_at] Target XY=({x_mm:.1f}, {y_mm:.1f})")

    print(f"  -> Moving to approach above block at Z={Z_APPROACH_ABOVE_BLOCK:.1f}")
    move_to(device, x_mm, y_mm, Z_APPROACH_ABOVE_BLOCK, r_deg, wait=True)
    time.sleep(0.15)  # prevents command pile-up

    print(f"  -> Moving down to pick at Z={Z_PICK_ON_BLOCK:.1f}")
    move_to(device, x_mm, y_mm, Z_PICK_ON_BLOCK, r_deg, wait=True)
    time.sleep(0.15)  # prevents command pile-up

    print("  -> Suction ON")
    set_suction(device, True)
    time.sleep(0.3)

    print(f"  -> Lifting back to Z={Z_APPROACH_ABOVE_BLOCK:.1f}")
    move_to(device, x_mm, y_mm, Z_APPROACH_ABOVE_BLOCK, r_deg, wait=True)
    time.sleep(0.15)  # prevents command pile-up


def place_block_at(
    device, x_mm: float, y_mm: float, on_table: bool = True, r_deg: float = 0.0
):
    if on_table:
        z_approach = Z_APPROACH_ABOVE_TABLE
        z_place = Z_PLACE_ON_TABLE
        where = "table"
    else:
        z_approach = Z_APPROACH_ABOVE_BLOCK
        z_place = Z_PLACE_ON_BLOCK
        where = "block"

    print(f"[place_block_at] Target XY=({x_mm:.1f}, {y_mm:.1f}) on {where}")

    print(f"  -> Moving to approach at Z={z_approach:.1f}")
    move_to(device, x_mm, y_mm, z_approach, r_deg, wait=True)
    time.sleep(0.15)  # prevents command pile-up

    print(f"  -> Moving down to place at Z={z_place:.1f}")
    move_to(device, x_mm, y_mm, z_place, r_deg, wait=True)
    time.sleep(0.15)  # prevents command pile-up

    print("  -> Suction OFF")
    set_suction(device, False)
    time.sleep(0.3)

    print(f"  -> Lifting back to Z={z_approach:.1f}")
    move_to(device, x_mm, y_mm, z_approach, r_deg, wait=True)
    time.sleep(0.15)  # prevents command pile-up


def place_block_on_top_of(device, ref_block, r_deg: float = 0.0):
    """
    Place the currently held block on top of ref_block (same XY, higher Z).
    We assume each block has height BLOCK_HEIGHT.
    """
    x_mm, y_mm = ref_block.x_mm, ref_block.y_mm

    # Top of a stack of two blocks:
    z_top_two = Z_BLOCK_TOP + BLOCK_HEIGHT
    z_approach = z_top_two + 30.0
    z_place = z_top_two + 4  # a tiny clearance

    print(f"[place_block_on_top_of] Ref block at XY=({x_mm:.1f}, {y_mm:.1f})")
    print(f"  -> Moving to approach above stack at Z={z_approach:.1f}")
    move_to(device, x_mm, y_mm, z_approach, r_deg, wait=True)
    time.sleep(0.15)  # prevents command pile-up

    print(f"  -> Moving down to place on stack at Z={z_place:.1f}")
    move_to(device, x_mm, y_mm, z_place, r_deg, wait=True)
    time.sleep(0.15)  # prevents command pile-up

    print("  -> Suction OFF")
    set_suction(device, False)
    time.sleep(0.3)

    print(f"  -> Lifting back to Z={z_approach:.1f}")
    move_to(device, x_mm, y_mm, z_approach, r_deg, wait=True)
    time.sleep(0.15)  # prevents command pile-up

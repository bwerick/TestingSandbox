#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maze_bfs.py
-----------
General BFS maze solver for black-wall / white-corridor images.

Features:
- Robust binarization (Otsu or Adaptive)
- Prevents "outside skirt" by flood-filling outside with sealed entrances
- Auto-detects border openings and picks a CONNECTED pair
- Or accept manual --start/--end (x,y pixels)
- BFS on interior-only mask (4- or 8-connected)
- Returns turning points (optionally merges tiny zigzags)
- Optional overlay PNG + debug masks

Usage (auto start/end):
    python maze_bfs.py maze.png --out-json turns.json --out-png solved.png

Manual start/end:
    python maze_bfs.py maze.png --start 12,742 --end 731,18 --out-png solved.png

Handy toggles:
    --merge-kinks 2       # merge tiny zigzags (Manhattan length ≤ 2)
    --seal-radius 1       # seal openings by a small radius during outside flood
    --morph-close 1       # close tiny gaps in walls (0..2 recommended)
    --thresh adaptive     # use adaptive threshold (default: otsu)
    --conn8               # use 8-connected moves
    --debug-dir debug     # dump binary/outside/interior masks for sanity check
"""

import argparse
import json
from collections import deque

import cv2
import numpy as np


# ------------------------- Utils -------------------------


def parse_xy(s: str):
    """Parse 'x,y' into internal (y,x)."""
    x, y = map(int, s.split(","))
    return (y, x)


def save_mask_png(path: str, mask: np.ndarray):
    """Save boolean/0-1/uint8 mask as 0/255 PNG."""
    if mask.dtype != np.uint8:
        mask_u8 = mask.astype(np.uint8) * 255
    else:
        # assume already 0/1 or 0/255
        mask_u8 = mask.copy()
        if mask_u8.max() == 1:
            mask_u8 *= 255
    cv2.imwrite(path, mask_u8)


# --------------------- Binarization ----------------------


def binarize(
    gray: np.ndarray, method: str = "otsu", morph_close: int = 1
) -> np.ndarray:
    """
    Return walls mask (uint8): 1=wall, 0=free.

    method: 'otsu' or 'adaptive'
    morph_close: 0..2 iterations to seal micro gaps in walls
    """
    if method == "adaptive":
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
    else:  # otsu
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    walls = (th == 0).astype(np.uint8)  # black=wall
    if morph_close > 0:
        k = np.ones((3, 3), np.uint8)
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, k, iterations=morph_close)
    return walls


def border_openings(free: np.ndarray):
    """List of free pixels on image border (in (y,x))."""
    H, W = free.shape
    ops = []
    for x in range(W):
        if free[0, x]:
            ops.append((0, x))
        if free[H - 1, x]:
            ops.append((H - 1, x))
    for y in range(H):
        if free[y, 0]:
            ops.append((y, 0))
        if free[y, W - 1]:
            ops.append((y, W - 1))
    # dedup
    out, seen = [], set()
    for p in ops:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ------------- Outside Flood (seal openings) --------------


def flood_outside_with_sealed_openings(
    free: np.ndarray, openings, seal_radius: int = 0
) -> np.ndarray:
    """
    Mark all border-connected free pixels as 'outside', but FIRST seal the chosen openings
    so flood doesn't leak into the maze corridors.
    Returns boolean mask: True=outside, False=not
    """
    H, W = free.shape
    # OpenCV drawing needs uint8
    sealed = free.astype(np.uint8).copy()

    # Seal the openings
    for y, x in openings:
        sealed[y, x] = 0
        if seal_radius > 0:
            cv2.circle(sealed, (x, y), seal_radius, 0, -1)

    outside = np.zeros((H, W), dtype=bool)
    q = deque()

    # Seed from border (on sealed)
    for x in range(W):
        if sealed[0, x]:
            outside[0, x] = True
            q.append((0, x))
        if sealed[H - 1, x]:
            outside[H - 1, x] = True
            q.append((H - 1, x))
    for y in range(H):
        if sealed[y, 0]:
            outside[y, 0] = True
            q.append((y, 0))
        if sealed[y, W - 1]:
            outside[y, W - 1] = True
            q.append((y, W - 1))

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        y, x = q.popleft()
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and sealed[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))

    return outside


def nudge_inward_v2(
    free: np.ndarray, outside: np.ndarray, opening, max_radius: int = 8
):
    """
    From a border opening (y,x), BFS outward up to max_radius to find the nearest
    pixel that is free AND not outside (i.e., interior corridor).
    """
    H, W = free.shape
    oy, ox = opening
    if 0 <= oy < H and 0 <= ox < W and free[oy, ox] and not outside[oy, ox]:
        return (oy, ox)

    visited = np.zeros((H, W), dtype=bool)
    q = deque([(oy, ox, 0)])
    visited[oy, ox] = True
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        y, x, d = q.popleft()
        if d > max_radius:
            break
        if free[y, x] and not outside[y, x]:
            return (y, x)
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx, d + 1))

    raise ValueError("Could not nudge opening into interior within radius.")


def quick_connected(mask: np.ndarray, a, b, conn8: bool = False) -> bool:
    """Quick BFS that stops once b is reached. mask==1 is traversable."""
    H, W = mask.shape
    ay, ax = a
    by, bx = b
    if mask[ay, ax] == 0 or mask[by, bx] == 0:
        return False
    visited = np.zeros((H, W), dtype=bool)
    q = deque([(ay, ax)])
    visited[ay, ax] = True
    if conn8:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        y, x = q.popleft()
        if (y, x) == (by, bx):
            return True
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    return False


def choose_connected_opening_pair(
    free: np.ndarray,
    outside: np.ndarray,
    openings,
    interior: np.ndarray,
    max_nudge_radius: int = 8,
    conn8: bool = False,
):
    """
    Try all border-opening pairs; nudge both into interior; keep only CONNECTED pairs.
    Return the pair with the longest Euclidean distance.
    """
    candidates = []
    for i in range(len(openings)):
        for j in range(i + 1, len(openings)):
            a0, b0 = openings[i], openings[j]
            try:
                a = nudge_inward_v2(free, outside, a0, max_nudge_radius)
                b = nudge_inward_v2(free, outside, b0, max_nudge_radius)
            except ValueError:
                continue
            if quick_connected(interior, a, b, conn8=conn8):
                dy = a[0] - b[0]
                dx = a[1] - b[1]
                d2 = dy * dy + dx * dx
                candidates.append((d2, a, b))

    if not candidates:
        raise ValueError(
            "No connected border-opening pair found. "
            "Try adjusting --morph-close / --seal-radius or provide --start/--end."
        )
    candidates.sort(reverse=True, key=lambda t: t[0])
    _, a, b = candidates[0]
    return a, b


# ------------------------- BFS ---------------------------


def bfs(mask: np.ndarray, start, end, conn8: bool = False):
    """Shortest path on mask (1=free), returns list of (y,x)."""
    H, W = mask.shape
    sy, sx = start
    gy, gx = end
    if mask[sy, sx] == 0 or mask[gy, gx] == 0:
        raise ValueError("Start or goal lies outside traversable mask.")
    prev = np.full((H, W, 2), -1, dtype=np.int32)
    visited = np.zeros((H, W), dtype=bool)
    q = deque([(sy, sx)])
    visited[sy, sx] = True

    if conn8:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    found = False
    while q:
        y, x = q.popleft()
        if (y, x) == (gy, gx):
            found = True
            break
        for dy, dx in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                prev[ny, nx] = [y, x]
                q.append((ny, nx))

    if not found:
        raise ValueError("No path found.")

    path = []
    cy, cx = gy, gx
    while (cy, cx) != (sy, sx):
        path.append((cy, cx))
        py, px = prev[cy, cx]
        cy, cx = py, px
    path.append((sy, sx))
    path.reverse()
    return path


# ------------------- Turning Points ----------------------


def turning_points(path, merge_kinks: int = 0):
    """
    Keep only direction changes (start/end included).
    Optionally merge tiny zigzags (Manhattan length <= merge_kinks).
    """
    if len(path) <= 2:
        return path[:]
    t = [path[0]]
    dy_prev = path[1][0] - path[0][0]
    dx_prev = path[1][1] - path[0][1]
    for i in range(1, len(path) - 1):
        dy = path[i + 1][0] - path[i][0]
        dx = path[i + 1][1] - path[i][1]
        if (dy, dx) != (dy_prev, dx_prev):
            t.append(path[i])
        dy_prev, dx_prev = dy, dx
    t.append(path[-1])

    if merge_kinks <= 0:
        return t

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    merged = [t[0]]
    for p in t[1:]:
        if len(merged) >= 2 and manhattan(merged[-1], p) <= merge_kinks:
            merged[-1] = p
        else:
            merged.append(p)
    return merged


# --------------------- Visualization ---------------------


def overlay_path(gray: np.ndarray, path, turns, out_png: str):
    """Save overlay PNG with path polyline and turning points."""
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if len(path) >= 2:
        pts = np.array([[x, y] for (y, x) in path], dtype=np.int32)
        cv2.polylines(color, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    for y, x in turns:
        cv2.circle(color, (x, y), 3, (0, 255, 0), -1)
    cv2.imwrite(out_png, color)


# ------------------------ Solve --------------------------


def solve_maze(
    image_path: str,
    start=None,
    end=None,  # (y,x) internal; if from CLI use parse_xy for 'x,y'
    thresh_method: str = "otsu",
    morph_close: int = 1,
    seal_radius: int = 0,
    conn8: bool = False,
    merge_kinks: int = 0,
    out_png: str | None = None,
    debug_dir: str | None = None,
):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)

    walls = binarize(gray, method=thresh_method, morph_close=morph_close)
    free = walls == 0

    # Auto openings or manual
    openings = border_openings(free)
    if (start is None or end is None) and len(openings) < 2:
        raise ValueError(
            "No (or only one) border opening found. Provide --start/--end manually."
        )

    # If manual provided, treat those as openings to seal
    openings_to_seal = [start, end] if (start is not None and end is not None) else []
    if not openings_to_seal:
        # Pick two far-away openings for sealing (prevents outside flood entering)
        # We'll find the connected pair later.
        openings_to_seal = openings[:2]

    outside = flood_outside_with_sealed_openings(
        free, openings_to_seal, seal_radius=seal_radius
    )
    interior = (free & (~outside)).astype(np.uint8)

    if start is not None and end is not None:
        start_in = nudge_inward_v2(free, outside, start, max_radius=8)
        end_in = nudge_inward_v2(free, outside, end, max_radius=8)
    else:
        start_in, end_in = choose_connected_opening_pair(
            free, outside, openings, interior, max_nudge_radius=8, conn8=conn8
        )

    path = bfs(interior, start_in, end_in, conn8=conn8)
    tpts = turning_points(path, merge_kinks=merge_kinks)

    # Optional outputs
    if out_png:
        overlay_path(gray, path, tpts, out_png)
    if debug_dir:
        # 0/1 masks
        save_mask_png(f"{debug_dir}/binary_walls.png", walls)
        save_mask_png(f"{debug_dir}/free.png", free.astype(np.uint8))
        save_mask_png(f"{debug_dir}/outside.png", outside.astype(np.uint8))
        save_mask_png(f"{debug_dir}/interior.png", interior)

    # Return as (x,y) for convenience
    tpts_xy = [(int(x), int(y)) for (y, x) in tpts]
    return {
        "turning_points": tpts_xy,
        "path_len": int(len(path)),
        "start_in": (int(start_in[1]), int(start_in[0])),
        "end_in": (int(end_in[1]), int(end_in[0])),
    }


# -------------------------- CLI --------------------------


def main():
    ap = argparse.ArgumentParser(
        description="BFS maze solver returning turning points."
    )
    ap.add_argument("image", help="Path to maze image (black walls, white corridors).")
    ap.add_argument(
        "--start",
        type=parse_xy,
        default=None,
        help="Manual start 'x,y' (pixel coords).",
    )
    ap.add_argument(
        "--end", type=parse_xy, default=None, help="Manual end   'x,y' (pixel coords)."
    )
    ap.add_argument(
        "--thresh",
        choices=["otsu", "adaptive"],
        default="otsu",
        help="Threshold method.",
    )
    ap.add_argument(
        "--morph-close", type=int, default=1, help="Morph close iterations (0..2)."
    )
    ap.add_argument(
        "--seal-radius",
        type=int,
        default=0,
        help="Radius to seal openings during outside flood.",
    )
    ap.add_argument(
        "--merge-kinks",
        type=int,
        default=0,
        help="Merge tiny zigzags (Manhattan length).",
    )
    ap.add_argument(
        "--conn8",
        action="store_true",
        help="Use 8-connected moves instead of 4-connected.",
    )
    ap.add_argument(
        "--out-json", default="turning_points.json", help="Where to save JSON result."
    )
    ap.add_argument("--out-png", default=None, help="Optional overlay PNG path.")
    ap.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory to save masks (binary, outside, interior).",
    )
    args = ap.parse_args()

    res = solve_maze(
        args.image,
        start=args.start,
        end=args.end,
        thresh_method=args.thresh,
        morph_close=args.morph_close,
        seal_radius=args.seal_radius,
        conn8=args.conn8,
        merge_kinks=args.merge_kinks,
        out_png=args.out_png,
        debug_dir=args.debug_dir,
    )

    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

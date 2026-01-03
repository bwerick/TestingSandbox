# Erick Duarte
# RAS 545 Midterm
# Fall 2025
# Dobot Magician Lite
# Tic Tac Toe Game

# --------------------------------------------------------------------------------------------
# Requirements
import cv2
import time, sys
from ultralytics import YOLO
from pydobot.dobot import PTPMode as MODE_PTP
from pydobot import Dobot
import pydobot
import sympy as sp
import numpy as np
from pathlib import Path
import json
import math

# --------------------------------------------------------------------------------------------
# Set Up

# Dobot
device = pydobot.Dobot(port="/dev/tty.usbmodem21301")
device.speed(100, 100)

# Robot Coordinates
allr = 0.0  # rotation angle never changes
penHover = 17  # pen above the paper, doesn't write at this height
penDown = 3.2  # pen down, writes at this height

# home coordinates, starts and ends the program here
homex = 240
homey = 0
homez = 150

# camera view coordinates @ 33x33 views the whole board
cameraX = 197.2
cameraY = -12.2
cameraZ = 64.9

# top right corner of the board, relative to the player
cellHeight = 29
cellWidth = 29
starting_position = (cameraX + 74, cameraY - 1.5 * cellHeight)
print(starting_position)

# --------------------------------------------------------------------------------------------
# Drawing Functions


# Calculate circle points
def pointsCircle(pointQty, x, y, radius):
    angle = 360 / pointQty
    points = []
    for i in range(pointQty):
        x1 = x + radius * sp.cos(sp.rad(angle * i))
        y1 = y + radius * sp.sin(sp.rad(angle * i))
        points.append((x1, y1))
    return points


# Draw circle
def drawCircle(x, y, radius):
    points = pointsCircle(12, x, y, radius)
    device.move_to(x=points[0][0], y=points[0][1], z=penHover, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[0][0], y=points[0][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    for point in points:
        device.move_to(x=point[0], y=point[1], z=penDown, r=allr, wait=False)
        time.sleep(0.1)
    device.move_to(x=points[0][0], y=points[0][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[0][0], y=points[0][1], z=penHover, r=allr, wait=False)
    time.sleep(2)


# Calculate cross points
def pointsCross(x, y, cellHeight, cellWidth):
    xtopLeft = (x - 0.4 * cellWidth / 2, y + 0.4 * cellHeight / 2)
    xtopRight = (x + 0.4 * cellWidth / 2, y + 0.4 * cellHeight / 2)
    xbottomLeft = (x - 0.4 * cellWidth / 2, y - 0.4 * cellHeight / 2)
    xbottomRight = (x + 0.4 * cellWidth / 2, y - 0.4 * cellHeight / 2)
    return [xtopLeft, xtopRight, xbottomLeft, xbottomRight]


# Draw cross
def drawCross(x, y, cellHeight, cellWidth):
    points = pointsCross(x, y, cellHeight, cellWidth)
    device.move_to(x=points[0][0], y=points[0][1], z=penHover, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[0][0], y=points[0][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[3][0], y=points[3][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[3][0], y=points[3][1], z=penHover, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[2][0], y=points[2][1], z=penHover, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[2][0], y=points[2][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[1][0], y=points[1][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[1][0], y=points[1][1], z=penHover, r=allr, wait=False)
    time.sleep(2)


# Calculate board lines points
def pointsBoard(starting_position, cellHeight, cellWidth):
    x, y = starting_position
    # vertical lines
    topHorizontalstart = x + cellWidth, y + 3 * cellHeight
    topHorizontalend = x + cellWidth, y
    bottomHorizontalstart = x + 2 * cellWidth, y
    bottomHorizontalend = x + 2 * cellWidth, y + 3 * cellHeight
    # Vertical lines
    rightVerticalstart = x, y + 2 * cellHeight
    rightVerticalend = x + 3 * cellWidth, y + 2 * cellHeight
    leftVerticalstart = x + 3 * cellWidth, y + 1 * cellHeight
    leftVerticalend = x, y + 1 * cellHeight
    return [
        leftVerticalstart,
        leftVerticalend,
        rightVerticalstart,
        rightVerticalend,
        bottomHorizontalstart,
        bottomHorizontalend,
        topHorizontalstart,
        topHorizontalend,
    ]


# Draw board
def drawBoard(starting_position, cellHeight, cellWidth):
    points = pointsBoard(starting_position, cellHeight, cellWidth)
    # vertical left line
    device.move_to(x=points[0][0], y=points[0][1], z=penHover, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[0][0], y=points[0][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[1][0], y=points[1][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[1][0], y=points[1][1], z=penHover, r=allr, wait=False)
    time.sleep(2)

    # vertical right line
    device.move_to(x=points[2][0], y=points[2][1], z=penHover, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[2][0], y=points[2][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[3][0], y=points[3][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[3][0], y=points[3][1], z=penHover, r=allr, wait=False)
    time.sleep(2)

    # horizontal top line
    device.move_to(x=points[4][0], y=points[4][1], z=penHover, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[4][0], y=points[4][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[5][0], y=points[5][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[5][0], y=points[5][1], z=penHover, r=allr, wait=False)
    time.sleep(2)

    device.move_to(x=points[6][0], y=points[6][1], z=penHover, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[6][0], y=points[6][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[7][0], y=points[7][1], z=penDown, r=allr, wait=False)
    time.sleep(2)
    device.move_to(x=points[7][0], y=points[7][1], z=penHover, r=allr, wait=False)
    time.sleep(2)

    # return to camera position
    device.move_to(x=cameraX + 4, y=cameraY - 5, z=cameraZ - 13, r=allr, wait=False)
    time.sleep(2)


# calculate robot cell centers
def cellCenters(starting_position, cellHeight, cellWidth):
    x, y = starting_position
    cell1 = (x + 5 * cellWidth / 2, y + 5 * cellHeight / 2)
    cell2 = (x + 5 * cellWidth / 2, y + 3 * cellHeight / 2)
    cell3 = (x + 5 * cellWidth / 2, y + cellHeight / 2)
    cell4 = (x + 3 * cellWidth / 2, y + 5 * cellHeight / 2)
    cell5 = (x + 3 * cellWidth / 2, y + 3 * cellHeight / 2)
    cell6 = (x + 3 * cellWidth / 2, y + cellHeight / 2)
    cell7 = (x + cellWidth / 2, y + 5 * cellHeight / 2)
    cell8 = (x + cellWidth / 2, y + 3 * cellHeight / 2)
    cell9 = (x + cellWidth / 2, y + cellHeight / 2)
    return [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]


def drawAllXO(starting_position, cellHeight, cellWidth):
    centers = cellCenters(starting_position, cellHeight, cellWidth)
    for cell in centers:
        drawCircle(cell[0], cell[1], 0.4 * (cellHeight / 2))
        drawCross(cell[0], cell[1], cellHeight, cellWidth)


# --------------------------------------------------------------------------------------------
# Camera stuff
MODEL_PATH = "/Users/erickduarte/git/TestingSandbox/runs/ttt_xo_mixed/weights/best.pt"  # <-- your YOLO model path
CONF_THRESH = 0.45
ROI_FILE = "roi.json"
STATE_FILE = "board_state.json"

# camera functions


def open_cam(indexes=(0, 1, 2), backends=None, width=1280, height=720, fps=30):
    if backends is None:
        if sys.platform == "darwin":
            backends = [cv2.CAP_AVFOUNDATION]
        elif sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW]
        else:
            backends = [cv2.CAP_V4L2, 0]

    for be in backends:
        for idx in indexes:
            cap = cv2.VideoCapture(idx, be) if be != 0 else cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            # Warm up
            for _ in range(20):
                ok, _ = cap.read()
                if not ok:
                    break
                time.sleep(0.01)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"[OK] Camera {idx} via backend {be}, frame {frame.shape}")
                return cap
            cap.release()
    return None


def snapshot_with_preview():
    cap = open_cam()
    if cap is None:
        raise RuntimeError("Could not open camera (permissions/index/backend).")
    print("[i] Press 's' to snapshot, 'q' to quit.")
    frame = None
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        cv2.imshow("Live Preview", f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            frame = f.copy()
            break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyWindow("Live Preview")
    if frame is None:
        raise RuntimeError("No snapshot taken.")
    return frame


def select_board_roi_on(frame):
    disp = frame.copy()
    max_side = 1200
    h, w = disp.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        disp = cv2.resize(disp, (int(w * scale), int(h * scale)))
    r = cv2.selectROI("Select Board ROI", disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Board ROI")
    if r == (0, 0, 0, 0):
        return None
    x, y, rw, rh = r
    x = int(x / scale)
    y = int(y / scale)
    rw = int(rw / scale)
    rh = int(rh / scale)
    return (x, y, rw, rh)


def save_roi(roi):
    with open(ROI_FILE, "w") as f:
        json.dump({"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]}, f)
    print(f"[i] ROI saved to {ROI_FILE}")


def load_roi():
    path = Path(ROI_FILE)
    if not path.exists():
        return None
    try:
        data = json.load(open(path))
        return (data["x"], data["y"], data["w"], data["h"])
    except Exception:
        return None


def read_board_by_simple_grid(model_path, conf_thresh=0.45, interactive_roi=True):
    model = YOLO(model_path)
    frame = snapshot_with_preview()

    # Load saved ROI if exists, otherwise ask user once
    roi_xywh = load_roi()
    if roi_xywh is None:
        print("[i] No saved ROI found — please select the board region.")
        roi_xywh = select_board_roi_on(frame)
        if roi_xywh is not None:
            save_roi(roi_xywh)

    # Fallback if ROI still missing
    if roi_xywh is None:
        h, w = frame.shape[:2]
        roi_xywh = (0, 0, w, h)

    # Divide ROI into 9 cells
    x, y, w, h = roi_xywh
    cw, ch = w // 3, h // 3
    rects = [
        (x + c * cw, y + r * ch, x + (c + 1) * cw, y + (r + 1) * ch)
        for r in range(3)
        for c in range(3)
    ]

    # Run detection
    res = model.predict(frame, conf=conf_thresh, verbose=False)[0]

    labels = [""] * 9
    best = np.zeros(9, dtype=np.float32)
    for b in res.boxes:
        cls_id = int(b.cls)
        conf = float(b.conf)
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for i, (X1, Y1, X2, Y2) in enumerate(rects):
            if X1 <= u < X2 and Y1 <= v < Y2:
                if conf > best[i]:
                    labels[i] = "X" if cls_id == 0 else "O"
                    best[i] = conf
                break

    # Visualization (optional)
    vis = frame.copy()
    v1, v2 = rects[1][0], rects[2][0]
    h1, h2 = rects[3][1], rects[6][1]
    cv2.line(vis, (v1, y), (v1, y + h), (0, 255, 0), 3)
    cv2.line(vis, (v2, y), (v2, y + h), (0, 255, 0), 3)
    cv2.line(vis, (x, h1), (x + w, h1), (0, 255, 0), 3)
    cv2.line(vis, (x, h2), (x + w, h2), (0, 255, 0), 3)

    for b in res.boxes:
        cls_id = int(b.cls)
        conf = float(b.conf)
        X1, Y1, X2, Y2 = map(int, b.xyxy[0])
        color = (0, 255, 255) if cls_id == 0 else (255, 200, 0)
        cv2.rectangle(vis, (X1, Y1), (X2, Y2), color, 3)
        lab = "X" if cls_id == 0 else "O"
        cv2.putText(
            vis,
            f"{lab} {conf:.2f}",
            (X1, max(0, Y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    for i, (X1, Y1, X2, Y2) in enumerate(rects):
        t = labels[i] if labels[i] else "-"
        cv2.putText(
            vis,
            f"{i+1}:{t}",
            (X1 + 6, Y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    grid = [labels[0:3], labels[3:6], labels[6:9]]
    print("\n--- Board State ---")
    for row in grid:
        print("|".join(c if c else " " for c in row))
    print("--------------------")

    return grid, labels, rects, frame


# Non interactive


STATE_FILE = "board_state.json"
ROI_FILE = "roi.json"  # we already save this elsewhere


def save_board_state(grid, path=STATE_FILE):
    data = {"grid": grid, "ts": time.time()}
    with open(path, "w") as f:
        json.dump(data, f)


def load_board_state(path=STATE_FILE):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.load(open(p))["grid"]
    except Exception:
        return None


def grids_equal(g1, g2):
    if g1 is None or g2 is None:
        return False
    return [c for r in g1 for c in r] == [c for r in g2 for c in r]


def load_roi(path=ROI_FILE):
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = json.load(open(p, "r"))
        return (int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"]))
    except Exception:
        return None


def capture_one_frame():
    # quick capture without preview
    cap = open_cam()
    if cap is None:
        raise RuntimeError("Camera not available.")
    # small warm-up
    for _ in range(5):
        ok, _ = cap.read()
        if not ok:
            break
        time.sleep(0.01)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError("Failed to capture frame.")
    return frame


def rects_from_roi(roi_xywh):
    x, y, w, h = roi_xywh
    cw, ch = w // 3, h // 3
    return [
        (x + c * cw, y + r * ch, x + (c + 1) * cw, y + (r + 1) * ch)
        for r in range(3)
        for c in range(3)
    ]


def read_board_once_noninteractive(model_path, conf_thresh=0.45):
    """Uses saved ROI (roi.json). Captures one frame, predicts, assigns cells; no popups."""
    roi = load_roi()
    frame = capture_one_frame()

    if roi is None:
        # fallback: full frame if ROI wasn't saved yet
        H, W = frame.shape[:2]
        roi = (0, 0, W, H)
    rects = rects_from_roi(roi)

    model = YOLO(model_path)
    res = model.predict(frame, conf=conf_thresh, verbose=False)[0]

    labels = [""] * 9
    best = np.zeros(9, dtype=np.float32)
    for b in res.boxes:
        cls_id = int(b.cls)
        conf = float(b.conf)
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for i, (X1, Y1, X2, Y2) in enumerate(rects):
            if X1 <= u < X2 and Y1 <= v < Y2:
                if conf > best[i]:
                    labels[i] = "X" if cls_id == 0 else "O"
                    best[i] = conf
                break

    grid = [labels[0:3], labels[3:6], labels[6:9]]
    return grid, labels, rects, frame


def wait_for_state_change(
    prev_grid,
    model_path,
    conf_thresh=0.45,
    timeout_sec=120,
    poll_sec=1.0,
    stable_reads=2,
):
    """
    Polls the camera until the grid changes from prev_grid, confirmed by 'stable_reads' consecutive
    identical new grids. Returns the new stable grid (or prev_grid if timeout).
    """
    start = time.time()
    stable_count = 0
    last_seen = None

    while time.time() - start < timeout_sec:
        try:
            new_grid, _, _, _ = read_board_once_noninteractive(model_path, conf_thresh)
        except Exception:
            time.sleep(poll_sec)
            continue

        if not grids_equal(new_grid, prev_grid):
            if last_seen is None or not grids_equal(new_grid, last_seen):
                last_seen = new_grid
                stable_count = 1
            else:
                stable_count += 1

            if stable_count >= stable_reads:
                return new_grid  # confirmed new state

        time.sleep(poll_sec)

    # timeout: no change
    return prev_grid


# --- utils to compare grids ---
def flatten(grid):
    return [c for row in grid for c in row]


def grid_diff(prev_grid, new_grid):
    """Return list of indices (0..8) where cells changed, and new symbols there."""
    a = flatten(prev_grid)
    b = flatten(new_grid)
    changes = [(i, b[i]) for i in range(9) if a[i] != b[i]]
    return changes


def check_illegal_human_move(prev_grid, new_grid, human_symbol, robot_symbol):
    """
    Works with your original grid_diff() that returns (idx, new_value).
    """
    diffs = grid_diff(prev_grid, new_grid)

    if len(diffs) == 0:
        return (False, "no_change", diffs)

    if len(diffs) > 1:
        return (False, "multiple_changes", diffs)

    # Extract change info
    idx, new_val = diffs[0]
    prev_flat = flatten(prev_grid)
    prev_val = prev_flat[idx]

    if prev_val != "":
        return (False, "overwrite_existing", diffs)

    if new_val not in ("X", "O", ""):
        return (False, "invalid_symbol", diffs)

    if new_val == "":
        return (False, "erasure", diffs)

    if new_val != human_symbol:
        if new_val == robot_symbol:
            return (False, "wrong_symbol", diffs)
        else:
            return (False, "invalid_symbol", diffs)

    return (True, None, diffs)


def exactly_one_new_mark(prev_grid, new_grid):
    """True iff exactly one cell changed from '' -> {'X','O'} and no other changes."""
    diffs = grid_diff(prev_grid, new_grid)
    if len(diffs) != 1:
        return False, None, None
    idx, sym = diffs[0]
    if sym not in ("X", "O") or flatten(prev_grid)[idx] != "":
        return False, None, None
    return True, idx, sym


# --------------------------------------------------------------------------------------------
# Game logic
def draw_symbol_at_cell(symbol, cell, cellHeight):
    """
    Draws a symbol ('X' or 'O') at the given cell center.
    You can plug in your own functions for drawing on the robot.

    Args:
        symbol: 'X' or 'O'
        cell: (x, y) tuple for the cell center
        cellHeight: height of one cell (used for sizing)
    """
    print(f"[i] Drawing {symbol} at cell center {cell}")

    if symbol == "O":
        # TODO: replace this line with your actual circle drawing command
        # Example:
        # drawCircle(cell[0], cell[1], 0.4 * (cellHeight / 2))
        drawCircle(cell[0], cell[1], 0.4 * (cellHeight / 2))
        pass

    elif symbol == "X":
        # TODO: replace this line with your actual X drawing command
        # Example:
        # drawX(cell[0], cell[1], 0.8 * (cellHeight / 2))
        drawCross(cell[0], cell[1], cellHeight, cellHeight)
        pass

    else:
        print("[!] Unknown symbol, skipping draw.")


def wait_for_first_human_mark(
    empty_grid,
    model_path,
    conf_thresh=0.45,
    timeout_sec=300,
    poll_sec=1.0,
    stable_reads=2,
):
    """
    Poll until we see exactly one new mark ('' -> 'X' or 'O') vs. an empty_grid.
    Confirms stability across 'stable_reads' identical detections.
    Returns: (human_symbol, new_grid, move_index)
    """
    stable_count = 0
    last_grid = None
    t0 = time.time()

    while time.time() - t0 < timeout_sec:
        try:
            new_grid, _, _, _ = read_board_once_noninteractive(model_path, conf_thresh)
        except Exception:
            time.sleep(poll_sec)
            continue

        ok, idx, sym = exactly_one_new_mark(empty_grid, new_grid)
        if not ok:
            # not yet a clean single move; reset stability and keep polling
            stable_count = 0
            last_grid = None
            time.sleep(poll_sec)
            continue

        # stability check: same grid repeated
        if last_grid is None or flatten(last_grid) != flatten(new_grid):
            last_grid = new_grid
            stable_count = 1
        else:
            stable_count += 1

        if stable_count >= stable_reads:
            return sym, new_grid, idx

        time.sleep(poll_sec)

    # timeout fallback: no clean mark detected
    return None, empty_grid, None


def choose_best_move(grid, my_symbol="O", opponent_symbol="X"):
    # Flatten for easier indexing
    flat = [c for row in grid for c in row]

    # All winning triplets
    lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],  # rows
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],  # cols
        [0, 4, 8],
        [2, 4, 6],  # diagonals
    ]

    # 1. Win if possible
    for line in lines:
        vals = [flat[i] for i in line]
        if vals.count(my_symbol) == 2 and vals.count("") == 1:
            return line[vals.index("")]

    # 2. Block opponent
    for line in lines:
        vals = [flat[i] for i in line]
        if vals.count(opponent_symbol) == 2 and vals.count("") == 1:
            return line[vals.index("")]

    # 3. Otherwise take center if free
    if flat[4] == "":
        return 4

    # 4. Otherwise pick first empty cell
    for i, c in enumerate(flat):
        if c == "":
            return i

    return None  # no moves left


def check_winner(board):
    """
    Check tic-tac-toe win/draw state.

    Args:
        board: either a 3x3 list of 'X'/'O'/'' OR a flat list of length 9 (row-major).

    Returns:
        winner: 'X' | 'O' | None
        line: list of 3 indices (0..8) that form the winning line, or [] if none
        is_draw: bool, True if no empties and no winner
    """
    # normalize to flat row-major list
    if (
        isinstance(board, list)
        and len(board) == 3
        and all(isinstance(r, list) for r in board)
    ):
        flat = [c for row in board for c in row]
    else:
        flat = list(board)
        if len(flat) != 9:
            raise ValueError("board must be 3x3 or flat length 9")

    lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],  # rows
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],  # cols
        [0, 4, 8],
        [2, 4, 6],  # diagonals
    ]

    for line in lines:
        a, b, c = line
        if flat[a] and flat[a] == flat[b] == flat[c]:
            return flat[a], line, False  # winner, winning line, not draw

    is_draw = all(cell != "" for cell in flat)
    return None, [], is_draw


# --------------------------------------------------------------------------------------------
# Robot makes a move
def robot_move(centers, current_grid, robot_symbol, human_symbol):
    winner, line, is_draw = check_winner(current_grid)
    if winner is not None or is_draw:
        return True, current_grid

    move = choose_best_move(
        current_grid, my_symbol=robot_symbol, opponent_symbol=human_symbol
    )
    if move is None:
        return True, current_grid

    print(f"[i] Robot ({robot_symbol}) chooses cell {move+1}")
    cell = centers[move]
    # Draw the correct symbol
    draw_symbol_at_cell(robot_symbol, cell, cellHeight)

    # Back to camera, verify state
    device.move_to(x=cameraX + 4, y=cameraY - 5, z=cameraZ - 13, r=allr, wait=True)
    time.sleep(1.0)
    new_grid, _, _, _ = read_board_once_noninteractive(MODEL_PATH, CONF_THRESH)

    save_board_state(new_grid)
    winner, line, is_draw = check_winner(new_grid)
    if winner is not None or is_draw:
        return True, new_grid
    return False, new_grid


import numpy as np
import math


def drawLine(line):
    print(f"Drawing line from {line[0]} to {line[1]}")
    device.move_to(x=line[0][0], y=line[0][1], z=penHover, r=allr, wait=True)
    time.sleep(2)
    device.move_to(x=line[0][0], y=line[0][1], z=penDown, r=allr, wait=True)
    time.sleep(2)

    device.move_to(x=line[1][0], y=line[1][1], z=penDown, r=allr, wait=True)
    time.sleep(2)
    device.move_to(x=line[0][0], y=line[0][1], z=penHover, r=allr, wait=True)
    time.sleep(2)


def draw_winning_line_from_check(
    grid,
    centers,
    cellWidth,
    cellHeight,
    drawLine=None,
    device=None,
    pen_down=None,
    pen_up=None,
    extend=0.45,
):
    """
    Draws a line through the winning cells returned by check_winner().

    Args:
        grid: current 3x3 board (passed to check_winner)
        centers: list of 9 (x, y) tuples (robot-space centers)
        cellWidth, cellHeight: for scaling / line extension
        drawLine: optional function drawLine(x1, y1, x2, y2)
        device, pen_down, pen_up: optional robot drawing backend
        extend: how far past the outer cells to extend the line (0.0–1.0)
    """

    # Determine winner and which cells won
    winner, line, is_draw = check_winner(grid)
    if not winner or not line:
        print("[i] No winning line to draw.")
        return None

    pts = np.array([centers[i] for i in line], dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]

    # Orientation test
    tol = 0.15 * min(cellWidth, cellHeight)
    if np.max(np.abs(ys - np.mean(ys))) < tol:
        # Horizontal
        y = float(np.mean(ys))
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        pad = extend * cellWidth
        start = (x_min - pad, y)
        end = (x_max + pad, y)

    elif np.max(np.abs(xs - np.mean(xs))) < tol:
        # Vertical
        x = float(np.mean(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        pad = extend * cellHeight
        start = (x, y_min - pad)
        end = (x, y_max + pad)

    else:
        # Diagonal: connect the farthest two centers
        dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        pA, pB = pts[i], pts[j]
        v = pB - pA
        L = np.linalg.norm(v)
        u = v / L
        pad = extend * math.hypot(cellWidth, cellHeight)
        start = (float(pA[0] - u[0] * pad), float(pA[1] - u[1] * pad))
        end = (float(pB[0] + u[0] * pad), float(pB[1] + u[1] * pad))

    # --- Draw ---
    if drawLine is not None:
        drawLine(start[0], start[1], end[0], end[1])
    else:
        # device is not None and penDown is not None and penHover is not None:
        device.move_to(x=start[0], y=start[1], z=penDown, r=allr, wait=True)

        device.move_to(x=end[0], y=end[1], z=penDown, r=allr, wait=True)

        print("[!] No drawing method provided (need drawLine() or device+pen).")

    print(f"[i] Drew winning line for {winner}: cells {line}")
    return start, end


# --------------------------------------------------------------------------------------------
# --- Initial setup (your existing code) ---
device.move_to(x=homex, y=homey, z=homez, r=allr, wait=False)
time.sleep(2)

centers = cellCenters(starting_position, cellHeight, cellWidth)
drawBoard(starting_position, cellHeight, cellWidth)

device.move_to(x=cameraX + 4, y=cameraY - 5, z=cameraZ - 13, r=allr, wait=False)
time.sleep(2)

# First run: pick ROI once (interactive) then it will be saved to roi.json
_, _, _, first_frame = read_board_by_simple_grid(
    MODEL_PATH, CONF_THRESH, interactive_roi=True
)
print("Press ENTER to start the game (human goes first).")
input()

# Read initial state non-interactively (reuses saved ROI)
current_grid, _, _, _ = read_board_once_noninteractive(MODEL_PATH, CONF_THRESH)
save_board_state(current_grid)

# Before the loop, after ROI is saved and camera positioned:
empty_grid = [["", "", ""], ["", "", ""], ["", "", ""]]

print("[i] Waiting for human to make the first mark...")
human_sym, current_grid, first_idx = wait_for_first_human_mark(
    empty_grid, MODEL_PATH, CONF_THRESH, timeout_sec=300, poll_sec=0.8, stable_reads=2
)

if human_sym is None:
    print("[!] No clear first move detected. Exiting.")
    # handle as you wish (abort/ retry)
else:
    robot_sym = "O" if human_sym == "X" else "X"
    print(f"[i] Human chose '{human_sym}' visually. Robot will play '{robot_sym}'.")
    save_board_state(current_grid)
robot_move(centers, current_grid, robot_sym, human_sym)

game_over = False
while not game_over:
    # Wait until human really played (grid changed & stabilized)
    print("[i] Waiting for human move...")
    prev_grid = current_grid
    new_grid = wait_for_state_change(
        current_grid,
        MODEL_PATH,
        CONF_THRESH,
        timeout_sec=300,
        poll_sec=1.0,
        stable_reads=2,
    )

    if grids_equal(new_grid, current_grid):
        print("[!] No human move detected within timeout. Ending.")
        break

    current_grid = new_grid
    save_board_state(current_grid)
    # ok, reason, diffs = check_illegal_human_move(
    #     prev_grid, current_grid, human_symbol=human_sym, robot_symbol=robot_sym
    # )
    # if not ok:
    #     print(f"[!] Illegal human move detected: {reason}, diffs: {diffs}. Ending.")
    #     break
    # Check if human just won
    winner, line, is_draw = check_winner(current_grid)
    if winner is not None:
        print(f"[i] Game over! {winner} wins!")
        break
    if is_draw:
        print("[i] Game over! It's a draw!")
        break

    # Robot turn (returns updated grid)
    game_over, current_grid = robot_move(
        centers, current_grid, robot_symbol=robot_sym, human_symbol=human_sym
    )
winner, line, is_draw = check_winner(current_grid)
if winner is not None:
    print(f"[i] Game over! {winner} wins!")
if is_draw:
    print("[i] Game over! It's a draw!")

winner, line, is_draw = check_winner(current_grid)

if winner and line:
    device.move_to(
        x=centers[line[0]][0], y=centers[line[0]][1], z=penHover, r=allr, wait=True
    )
    time.sleep(2.0)
    device.move_to(
        x=centers[line[0]][0], y=centers[line[0]][1], z=penDown, r=allr, wait=True
    )
    time.sleep(2.0)
    device.move_to(
        x=centers[line[2]][0], y=centers[line[2]][1], z=penDown, r=allr, wait=True
    )
    time.sleep(2.0)
    device.move_to(
        x=centers[line[2]][0], y=centers[line[2]][1], z=penHover, r=allr, wait=True
    )
    time.sleep(2.0)

# End: go home
device.move_to(x=homex, y=homey, z=homez, r=allr, wait=False)
time.sleep(2)

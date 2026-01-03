#!/usr/bin/env python3
"""
midterm2_combined_move.py

Midterm 2 – FULL PIPELINE, using maze-solving code (no YOLO, no best.pt).

Pipeline:
  1) Move Dobot to camera-view "home" pose.
  2) Open camera, show live preview.
     - User presses 'c' or SPACE to CAPTURE a frame.
  3) Detect maze corners (largest quadrilateral on black background).
     - Save:
         part_2_maze_solution/camera_capture.png
         part_2_maze_solution/maze_corners_overlay.png
         part_2_maze_solution/camera_capture_corners.json
  4) Warp/crop maze using corners JSON.
     - Save: part_2_maze_solution/maze_warp.png
  5) Detect red/green circles + grid + per-cell 0/1.
     - Save:
         circles_overlay.png, grid_overlay.png,
         walls_mask.png, grid_overlay_annot.png,
         result.json
     - **Gemini** is used here to classify circle color ("red"/"green").
  6) Ask user: start at red or green? [r/g]
     - Solve maze with BFS on the warped maze.
     - Save:
         solution_overlay.png (warped image with path)
         solution_path_points.json (path in warped coords)
  7) Unwarp path back to original image.
     - Save:
         original_with_path.png
         solution_path_points_unwarped.json
  8) Map unwarped pixel path -> robot mm via homography
     and move the Dobot along the path at TRACE_Z.
"""

import os
import time
import json
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw

# --- imports for Gemini + dotenv ---
from dotenv import load_dotenv
from google import genai
from google.genai import types
# ---------------------------------------


# ----------------------- CONFIG -----------------------
OUT_DIR = "part_2_maze_solution"

CAM_INDEX = 1      # camera index
REQ_W, REQ_H = 1600, 1200

COM_PORT  = "COM16"  # Dobot port

DOBOT_VIEW_POSE = (221.6, 0.0, 149.7, 0.0)  # robot pose to see the maze

TRACE_Z         = -45   # Z while tracing
TRACE_R         = 0.0     # tool rotation
TRACE_PAUSE_S   = 0.08    # pause between waypoints

# image->robot(mm) homography (old)
# H_IMG2MM = np.array([
 #   [-0.015141,   -0.17209,   370.77 ],
 #   [-0.18697,     0.0037308, 154.9  ],
 #   [-4.5791e-05,  5.6625e-05, 1.    ]
# ], dtype=np.float64)

# image->robot(mm) homography
H_IMG2MM = np.array([
    [  -0.029421,    -0.15776,      368.88],
    [   -0.17147,  -0.0037351,      141.73],
    [-0.00011097,  6.1361e-05,           1.]
], dtype=np.float64)


# Maze-processing constants
PROC_WIDTH = 1200          # processing width for corner detection
CORNER_EXPAND_PX = 10.0    # outward expand of maze corners in original pixels
GRID_SIZE_PX = 75          # grid cell size in pixels
WALL_THRESHOLD_PCT = 5.0   # % wall pixels to call a cell blocked
# ------------------------------------------------------


# ========== GEMINI CONFIG ==========
load_dotenv()  # loads GEMINI_API_KEY from .env in this folder

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

# choose a model you have access to; adjust if needed
GEMINI_MODEL = "gemini-2.0-flash"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
# ===================================


# -------------------- Dobot helpers -------------------
class _NoDobot:
    def move_to(self, *args, **kwargs):
        print("[WARN] Dobot not connected; skipping move.")

def get_dobot(port: str):
    try:
        from pydobot import Dobot
        bot = Dobot(port=port)
        print(f"[OK] Connected to Dobot on {port}")
        return bot, True
    except Exception as e:
        print(f"[WARN] Could not connect Dobot on {port}: {e}")
        return _NoDobot(), False

def move_robot_along(bot, robot_pts_mm, z=0.0, r=0.0, pause=0.08):
    for (xmm, ymm) in robot_pts_mm:
        try:
            bot.move_to(float(xmm), float(ymm), float(z), float(r))
        except Exception as e:
            print(f"[WARN] move_to failed at ({xmm:.1f},{ymm:.1f}): {e}")
        time.sleep(pause)

# ------------------------------------------------------


# ---------------- Camera / UI helpers -----------------
def make_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def open_camera(index: int, req_w: int, req_h: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    ok, _ = cap.read()
    if not ok:
        raise RuntimeError(f"Could not open camera at index {index}")
    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera ~{cw}x{ch} (requested {req_w}x{req_h})")
    return cap

def camera_capture_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Show live preview, return a single captured frame when user presses
    SPACE or 'c'. Return None if user quits with 'q' or ESC.
    """
    win = "Camera View (SPACE/c = capture, q = quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERR] Camera read failed.")
            break
        last_frame = frame
        disp = frame.copy()
        h, w = disp.shape[:2]
        cv2.putText(
            disp,
            "SPACE/c: capture   q/ESC: quit",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            last_frame = None
            break
        if key in (32, ord('c')):  # SPACE or c
            break

    cv2.destroyWindow(win)
    return last_frame

# ------------------------------------------------------


# -----------  Step 3 logic: corner detection --------
def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def binarize_white_foreground(gray: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (th > 0).mean() > 0.9:  # if almost everything is white, invert
        th = 255 - th
    return th

def morph_cleanup(bin_img: np.ndarray) -> np.ndarray:
    opened = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    return closed

def approx_to_quads(cnt: np.ndarray, max_iter: int = 25) -> Optional[np.ndarray]:
    peri = cv2.arcLength(cnt, True)
    for frac in np.linspace(0.01, 0.06, max_iter):
        approx = cv2.approxPolyDP(cnt, frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None

def find_largest_quad(bin_img: np.ndarray) -> Tuple[np.ndarray, str]:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found for maze. Check lighting / background.")

    h, w = bin_img.shape[:2]
    min_area = max(1000.0, 0.001 * (h * w))
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        candidates = [max(contours, key=cv2.contourArea)]

    best_quad, best_area = None, -1.0
    for c in candidates:
        quad = approx_to_quads(c)
        if quad is not None:
            area = cv2.contourArea(quad.astype(np.int32))
            if area > best_area:
                best_area = area
                best_quad = quad

    if best_quad is not None:
        return order_corners(best_quad), "largest_quadrilateral"

    # Fallback: oriented bounding box of largest
    largest = max(candidates, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_corners(box), "minAreaRect_fallback"

def expand_corners(corners: np.ndarray, expand_px: float) -> np.ndarray:
    if abs(expand_px) < 1e-9:
        return corners.astype(np.float32)
    center = corners.mean(axis=0)
    expanded = []
    for p in corners:
        v = p - center
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            expanded.append(p)
        else:
            scale = (norm + expand_px) / norm
            expanded.append(center + v * scale)
    return order_corners(np.array(expanded, dtype=np.float32))

def detect_corners_blackbg(img_bgr: np.ndarray, resize_width: Optional[int]) -> Tuple[np.ndarray, str, float]:
    if resize_width is not None and resize_width > 0 and img_bgr.shape[1] > resize_width:
        scale = resize_width / img_bgr.shape[1]
        img_proc = cv2.resize(
            img_bgr,
            (int(img_bgr.shape[1] * scale), int(img_bgr.shape[0] * scale)),
            cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        img_proc = img_bgr

    gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_img = binarize_white_foreground(gray)
    bin_img = morph_cleanup(bin_img)

    corners, method = find_largest_quad(bin_img)
    return corners, method, scale

def draw_corners_overlay(base_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    overlay = base_bgr.copy()
    c = corners.astype(int)
    labels = ["TL", "TR", "BR", "BL"]
    cv2.polylines(overlay, [c.reshape(-1, 1, 2)], True, (255, 0, 0), 3)
    for i, p in enumerate(c):
        cv2.circle(overlay, tuple(p), 8, (0, 255, 0), -1)
        cv2.putText(
            overlay,
            labels[i],
            tuple(p + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay



def capture_and_save_corners(cap: cv2.VideoCapture) -> Tuple[str, str, str]:
    """
    Capture a frame from camera, detect maze corners, and save:
      - raw image
      - overlay with corners
      - corners JSON
    """
    frame = camera_capture_frame(cap)
    if frame is None:
        raise SystemExit("[INFO] No frame captured; exiting.")

    make_out_dir()

    raw_path = os.path.join(OUT_DIR, "camera_capture.png")
    overlay_path = os.path.join(OUT_DIR, "maze_corners_overlay.png")
    json_path = os.path.join(OUT_DIR, "camera_capture_corners.json")

    corners_proc, method, scale = detect_corners_blackbg(frame, PROC_WIDTH)
    corners = corners_proc / scale if scale != 1.0 else corners_proc.copy()
    corners = expand_corners(corners, CORNER_EXPAND_PX)

    overlay = draw_corners_overlay(frame, corners)

    cv2.imwrite(raw_path, frame)
    cv2.imwrite(overlay_path, overlay)

    corners_list = corners.tolist()
    data = {
        "input": os.path.basename(raw_path),
        "method": method,
        "width_param": PROC_WIDTH,
        "expand_param": CORNER_EXPAND_PX,
        "corners": {
            "TL": corners_list[0],
            "TR": corners_list[1],
            "BR": corners_list[2],
            "BL": corners_list[3],
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Saved raw capture to {raw_path}")
    print(f"[OK] Saved corner overlay to {overlay_path}")
    print(f"[OK] Saved corners JSON to {json_path}")
    return raw_path, overlay_path, json_path

# ------------------------------------------------------


# --------- Step 4: warp maze from corners JSON --------
def read_corners_from_json(json_path: str) -> Tuple[np.ndarray, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_from_json = ""
    if isinstance(data, dict):
        if "corners" in data and isinstance(data["corners"], dict):
            tl = data["corners"]["TL"]
            tr = data["corners"]["TR"]
            br = data["corners"]["BR"]
            bl = data["corners"]["BL"]
            pts = np.array([tl, tr, br, bl], dtype=np.float32)
        else:
            raise ValueError("Corners JSON must contain 'corners': {TL,TR,BR,BL}.")
        if "input" in data and isinstance(data["input"], str):
            img_from_json = data["input"]
    else:
        raise ValueError("Corners JSON root must be an object.")

    pts = order_corners(pts)
    return pts, img_from_json

def infer_warp_size(corners: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    W = int(round((width_top + width_bottom) / 2.0))
    H = int(round((height_left + height_right) / 2.0))
    return max(W, 1), max(H, 1)

def warp_maze_from_json(corners_json: str) -> str:
    corners, img_from_json = read_corners_from_json(corners_json)
    json_dir = os.path.dirname(os.path.abspath(corners_json))
    img_path = os.path.join(json_dir, img_from_json)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"[ERR] Could not read image: {img_path}")

    W, H = infer_warp_size(corners)
    src = corners.astype(np.float32)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)

    out_path = os.path.join(OUT_DIR, "maze_warp.png")
    cv2.imwrite(out_path, warped)
    print(f"[OK] Warp complete: {img_path} -> {out_path}  (size {W}x{H})")
    return out_path

# ------------------------------------------------------


# --------- Step 5: circles, grid, walls JSON ----------
def detect_color(frame, center, radius):
    """Old deterministic HSV-based color detection (fallback)."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(max(1.0, radius * 0.8)), 255, -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    H = H[mask == 255]
    S = S[mask == 255]
    V = V[mask == 255]

    valid = (S > 50) & (V > 50)
    H = H[valid]

    if len(H) == 0:
        return "green"  # default

    red_mask = ((H <= 10) | (H >= 170))
    green_mask = ((H >= 35) & (H <= 85))

    red_ratio = np.sum(red_mask) / len(H)
    green_ratio = np.sum(green_mask) / len(H)

    if red_ratio > green_ratio and red_ratio > 0.1:
        return "red"
    elif green_ratio > red_ratio and green_ratio > 0.1:
        return "green"
    else:
        return "unknown"


def classify_dot_color_with_gemini(frame_bgr, center, radius) -> Optional[str]:
    """
    Use Gemini to classify the dot as 'red' or 'green'.

    Returns:
        'red', 'green', or None if the model is unsure / an error occurs.
    """
    try:
        u, v = center
        r = max(float(radius), 5.0)
        margin = int(r * 1.3)

        h, w = frame_bgr.shape[:2]
        x0 = max(0, int(u - margin))
        y0 = max(0, int(v - margin))
        x1 = min(w, int(u + margin))
        y1 = min(h, int(v + margin))

        if x1 <= x0 or y1 <= y0:
            return None

        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        # encode to JPEG bytes in-memory
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/jpeg",
        )

        prompt = (
            "You see a single solid-colored circular dot on a maze. "
            "Respond with EXACTLY one word: 'red' or 'green'. "
            "If you are unsure, guess."
        )

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image_part, prompt],
        )

        text = (response.text or "").strip().lower()
        # Basic parsing rules
        if "red" in text and "green" not in text:
            return "red"
        if "green" in text and "red" not in text:
            return "green"
        if text.startswith("red"):
            return "red"
        if text.startswith("green"):
            return "green"
        print("Gemini said:", text)
        return None
        

    except Exception as e:
        print(f"[WARN] Gemini classification failed: {e}")
        return None


def detect_circles_and_overlay(img_bgr: np.ndarray, out_path: str) -> List[Dict[str, Any]]:
    frame = img_bgr.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100,
    )
    results = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = circles[0].astype(np.float32)
        circles = circles[np.argsort(circles[:, 0])]  # sort left to right

        for (u, v, r) in circles:
            # 1) deterministic HSV color
            color_cv = detect_color(frame, (u, v), r)

            # 2) Gemini AI color
            color_ai = classify_dot_color_with_gemini(frame, (u, v), r)

            if color_ai in ("red", "green"):
                color = color_ai
            else:
                color = color_cv

            results.append(
                {
                    "center": [float(u), float(v)],
                    "radius": float(r),
                    "color": color,
                    "color_cv": color_cv, # unknown
                    "color_ai": color_ai,
                }
            )

            # draw with selected color
            cv2.circle(frame, (int(u), int(v)), int(r), (0, 255, 0), 2)
            cv2.circle(frame, (int(u), int(v)), 2, (0, 255, 0), 3)
            cv2.putText(
                frame,
                color,
                (int(u) + 10, int(v) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(out_path, frame)
    return results


def binarize_walls(gray: np.ndarray, adaptive: bool) -> np.ndarray:
    if adaptive:
        block = 35 if min(gray.shape[:2]) > 400 else 21
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            5,
        )
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th

def morph(mask: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
    m = mask.copy()
    if k_open > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k_open + 1, 2 * k_open + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
    if k_close > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k_close + 1, 2 * k_close + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc, iterations=1)
    return m

def draw_grid_lines(img: np.ndarray, grid: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    color = (255, 255, 255)
    for y in range(0, h, grid):
        cv2.line(out, (0, y), (w - 1, y), color, 1)
    for x in range(0, w, grid):
        cv2.line(out, (x, 0), (x, h - 1), color, 1)
    return out

def draw_grid_with_values(
    img: np.ndarray, grid: int, values_mat: np.ndarray, font_scale: float, thickness: int
) -> np.ndarray:
    out = draw_grid_lines(img, grid)
    h, w = out.shape[:2]
    gh, gw = values_mat.shape
    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            text = str(int(values_mat[gy, gx]))
            cv2.putText(
                out,
                text,
                (cx - 6, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                text,
                (cx - 6, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
    return out

def build_grid_and_json(warp_path: str) -> str:
    img = cv2.imread(warp_path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"[ERR] Cannot read warped maze: {warp_path}")

    grid = GRID_SIZE_PX

    circles_overlay_out = os.path.join(OUT_DIR, "circles_overlay.png")
    grid_overlay_out = os.path.join(OUT_DIR, "grid_overlay.png")
    grid_overlay_annot_out = os.path.join(OUT_DIR, "grid_overlay_annot.png")
    walls_mask_out = os.path.join(OUT_DIR, "walls_mask.png")
    json_out = os.path.join(OUT_DIR, "result.json")

    circles_info = detect_circles_and_overlay(img, circles_overlay_out)

    grid_overlay = draw_grid_lines(img, grid)
    cv2.imwrite(grid_overlay_out, grid_overlay)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    walls_mask = binarize_walls(gray, adaptive=False)
    walls_mask = morph(walls_mask, k_open=0, k_close=0)
    cv2.imwrite(walls_mask_out, walls_mask)

    h, w = walls_mask.shape[:2]
    gh = (h + grid - 1) // grid
    gw = (w + grid - 1) // grid
    values_mat = np.zeros((gh, gw), dtype=np.uint8)
    cells = []
    thresh_pct = float(WALL_THRESHOLD_PCT)

    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            blk = walls_mask[y0:y1, x0:x1]
            wall_pct = (blk > 0).mean() * 100.0 if blk.size > 0 else 0.0
            value = 0 if wall_pct >= thresh_pct else 1
            values_mat[gy, gx] = value
            cells.append(
                {
                    "row": int(gy),
                    "col": int(gx),
                    "value": int(value),
                    "center_px": [int(cx), int(cy)],
                }
            )

    grid_overlay_annot = draw_grid_with_values(
        img, grid, values_mat, font_scale=0.4, thickness=1
    )
    cv2.imwrite(grid_overlay_annot_out, grid_overlay_annot)

    meta = {
        "input": warp_path,
        "circles_overlay_path": circles_overlay_out,
        "grid_size_px": grid,
        "grid_rows": int(gh),
        "grid_cols": int(gw),
        "threshold_percent": thresh_pct,
        "grid_overlay_path": grid_overlay_out,
        "grid_overlay_annot_path": grid_overlay_annot_out,
        "walls_mask_path": walls_mask_out,
        "circles": circles_info,
        "cells": cells,
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Circles/grid JSON saved to {json_out}")
    return json_out

# ------------------------------------------------------


# --------- Step 6: BFS maze solver (warped) -----------
@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    value: int
    center_px: Tuple[int, int]

def load_maze(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def cells_to_grid(cells_json: List[Dict[str, Any]], rows: int, cols: int) -> List[List[Cell]]:
    grid: List[List[Cell]] = [[None for _ in range(cols)] for _ in range(rows)]
    for c in cells_json:
        cell = Cell(
            row=c["row"],
            col=c["col"],
            value=int(c["value"]),
            center_px=(int(c["center_px"][0]), int(c["center_px"][1])),
        )
        grid[cell.row][cell.col] = cell

    for r in range(rows):
        for q in range(cols):
            if grid[r][q] is None:
                raise ValueError(f"Missing cell at ({r},{q}) in JSON.")
    return grid

def nearest_cell_by_pixel(grid: List[List[Cell]], x: float, y: float) -> Tuple[int, int]:
    best = None
    best_d2 = 1e18
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            cx, cy = grid[r][c].center_px
            d2 = (cx - x) ** 2 + (cy - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (r, c)
    return best

def parse_start_end(
    grid, data, start_arg: Optional[str], end_arg: Optional[str]
):
    circles = data.get("circles", [])
    green = next((c for c in circles if c.get("color") == "green"), None)
    red   = next((c for c in circles if c.get("color") == "red"),   None)
    if green is None or red is None:
        raise ValueError("JSON must include green and red circles.")

    green_px = (float(green["center"][0]), float(green["center"][1]))
    red_px   = (float(red["center"][0]),   float(red["center"][1]))

    default_start = nearest_cell_by_pixel(grid, *green_px)
    default_end   = nearest_cell_by_pixel(grid, *red_px)

    def parse_point(arg, default_rc):
        if not arg:
            return default_rc
        a = arg.strip().lower()
        if a == "green": return default_start
        if a == "red":   return default_end
        if "," in a:
            r_str, c_str = a.split(",", 1)
            return (int(r_str), int(c_str))
        raise ValueError(f"Invalid start/end value: {arg}. Use 'green', 'red', or 'r,c'.")

    start_rc = parse_point(start_arg, default_start)
    end_rc   = parse_point(end_arg,   default_end)

    def choose_anchor(which, rc):
        if which and which.strip().lower() in ("green", "red"):
            return green_px if which.strip().lower() == "green" else red_px
        cx, cy = grid[rc[0]][rc[1]].center_px
        dg = (cx - green_px[0])**2 + (cy - green_px[1])**2
        dr = (cx - red_px[0])**2   + (cy - red_px[1])**2
        return green_px if dg <= dr else red_px

    start_dot = choose_anchor(start_arg, start_rc)
    end_dot   = choose_anchor(end_arg,   end_rc)

    return start_rc, end_rc, (int(start_dot[0]), int(start_dot[1])), (int(end_dot[0]), int(end_dot[1]))

def bfs_path(
    grid: List[List[Cell]],
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    rows, cols = len(grid), len(grid[0])
    sr, sc = start_rc
    tr, tc = end_rc

    def neighbors(r: int, c: int):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                yield rr, cc

    q = deque()
    q.append((sr, sc))
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    seen = set([(sr, sc)])

    while q:
        r, c = q.popleft()
        if (r, c) == (tr, tc):
            path = [(r, c)]
            while (r, c) != (sr, sc):
                r, c = prev[(r, c)]
                path.append((r, c))
            path.reverse()
            return path

        for rr, cc in neighbors(r, c):
            if (rr, cc) in seen:
                continue
            if (rr, cc) == (tr, tc):
                prev[(rr, cc)] = (r, c)
                seen.add((rr, cc))
                q.append((rr, cc))
                continue
            if grid[rr][cc].value != 1:
                continue
            prev[(rr, cc)] = (r, c)
            seen.add((rr, cc))
            q.append((rr, cc))
    return None

def draw_path_on_image(
    image_path: str,
    out_image_path: str,
    cell_path: List[Tuple[int, int]],
    grid: List[List[Cell]],
    start_circle_px: Tuple[int, int],
    end_circle_px: Tuple[int, int],
    line_width: int = 5,
) -> None:
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    poly: List[Tuple[int, int]] = []
    poly.append(start_circle_px)
    for (r, c) in cell_path:
        poly.append(grid[r][c].center_px)
    poly.append(end_circle_px)

    draw.line(poly, width=line_width, fill=(255, 0, 0, 255))
    r_rad = max(6, line_width * 2)
    g_rad = max(6, line_width * 2)
    gx, gy = start_circle_px
    rx, ry = end_circle_px
    draw.ellipse((gx - g_rad, gy - g_rad, gx + g_rad, gy + g_rad), outline=(0, 255, 0, 255), width=3)
    draw.ellipse((rx - r_rad, ry - r_rad, rx + r_rad, ry + r_rad), outline=(255, 0, 0, 255), width=3)

    img.save(out_image_path)

def write_path_json(
    out_json_path: str,
    cell_path: List[Tuple[int, int]],
    grid: List[List[Cell]],
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    start_circle_px: Tuple[int, int],
    end_circle_px: Tuple[int, int],
) -> None:
    path_cells_expanded = [
        {
            "row": r,
            "col": c,
            "value": grid[r][c].value,
            "center_px": [grid[r][c].center_px[0], grid[r][c].center_px[1]],
        }
        for (r, c) in cell_path
    ]
    pixel_polyline: List[List[int]] = []
    pixel_polyline.append([start_circle_px[0], start_circle_px[1]])
    for (r, c) in cell_path:
        x, y = grid[r][c].center_px
        pixel_polyline.append([x, y])
    pixel_polyline.append([end_circle_px[0], end_circle_px[1]])

    out = {
        "start_cell": {"row": start_rc[0], "col": start_rc[1]},
        "end_cell": {"row": end_rc[0], "col": end_rc[1]},
        "start_circle_px": [start_circle_px[0], start_circle_px[1]],
        "end_circle_px": [end_circle_px[0], end_circle_px[1]],
        "path_cells": path_cells_expanded,
        "path_pixels": pixel_polyline,
        "moves": max(0, len(cell_path) - 1),
        "notes": "Path respects rule: 1-only traversal; start/end cells may be 0.",
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def solve_maze_bfs(result_json_path: str, start_color: str, end_color: str) -> Tuple[str, str]:
    data = load_maze(result_json_path)
    rows = int(data["grid_rows"])
    cols = int(data["grid_cols"])
    grid = cells_to_grid(data["cells"], rows, cols)

    start_label = start_color.lower()
    end_label = end_color.lower()

    start_rc, end_rc, start_circle_px, end_circle_px = parse_start_end(
        grid, data, start_label, end_label
    )

    path = bfs_path(grid, start_rc, end_rc)
    if path is None:
        raise SystemExit("[ERR] No feasible path found under the given rules.")

    image_path = data["input"]
    if not os.path.exists(image_path):
        raise SystemExit(f"[ERR] Input image not found at: {image_path}")

    out_image = os.path.join(OUT_DIR, "solution_overlay.png")
    out_json = os.path.join(OUT_DIR, "solution_path_points.json")

    draw_path_on_image(
        image_path=image_path,
        out_image_path=out_image,
        cell_path=path,
        grid=grid,
        start_circle_px=start_circle_px,
        end_circle_px=end_circle_px,
        line_width=5,
    )
    write_path_json(
        out_json_path=out_json,
        cell_path=path,
        grid=grid,
        start_rc=start_rc,
        end_rc=end_rc,
        start_circle_px=start_circle_px,
        end_circle_px=end_circle_px,
    )

    print(f"[OK] BFS solution image saved to {out_image}")
    print(f"[OK] BFS path JSON saved to {out_json}")
    return out_image, out_json

# ------------------------------------------------------


# ----- Step 7: unwarp path to original image ----------
def read_corners_json(path: str) -> Tuple[np.ndarray, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "corners" in data:
            tl = data["corners"]["TL"]
            tr = data["corners"]["TR"]
            br = data["corners"]["BR"]
            bl = data["corners"]["BL"]
            corners = np.array([tl, tr, br, bl], dtype=np.float32)
        else:
            raise ValueError("Corners JSON must contain 'corners':{TL,TR,BR,BL}.")
        img_path = data.get("input", "")
        if not isinstance(img_path, str) or not img_path:
            raise ValueError("Corners JSON must include original image path in 'input'.")
    else:
        raise ValueError("Invalid corners JSON.")

    return order_corners(corners), img_path

def read_path_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_polyline_pixels(path_data: Dict[str, Any]) -> List[List[float]]:
    if isinstance(path_data.get("path_pixels"), list) and len(path_data["path_pixels"]) > 0:
        return [[float(x), float(y)] for x, y in path_data["path_pixels"]]

    if isinstance(path_data.get("path_cells"), list) and len(path_data["path_cells"]) > 0:
        poly: List[List[float]] = []
        if "start_circle_px" in path_data:
            sx, sy = path_data["start_circle_px"]
            poly.append([float(sx), float(sy)])
        for c in path_data["path_cells"]:
            cx, cy = c["center_px"]
            poly.append([float(cx), float(cy)])
        if "end_circle_px" in path_data:
            ex, ey = path_data["end_circle_px"]
            poly.append([float(ex), float(ey)])
        return poly

    raise ValueError("Path JSON must contain 'path_pixels' or 'path_cells'.")

def unwarp_path_and_overlay(
    corners_json: str,
    path_json: str,
    warped_image_path: str,
) -> Tuple[str, str]:
    corners, orig_img_rel = read_corners_json(corners_json)

    if not os.path.isabs(orig_img_rel):
        cj_dir = os.path.dirname(os.path.abspath(corners_json))
        orig_img_path = os.path.join(cj_dir, orig_img_rel)
    else:
        orig_img_path = orig_img_rel

    orig_img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
    if orig_img is None:
        raise SystemExit(f"[ERR] Could not read original image: {orig_img_path}")

    warped_img = cv2.imread(warped_image_path, cv2.IMREAD_COLOR)
    if warped_img is None:
        raise SystemExit(f"[ERR] Could not read warped image: {warped_image_path}")
    H, W = warped_img.shape[:2]

    src_rect = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    dst_quad = corners.astype(np.float32)
    M_inv = cv2.getPerspectiveTransform(src_rect, dst_quad)

    path_data = read_path_json(path_json)
    poly_warped = np.array(collect_polyline_pixels(path_data), dtype=np.float32).reshape(-1, 1, 2)
    poly_orig = cv2.perspectiveTransform(poly_warped, M_inv).reshape(-1, 2)

    img_overlay = orig_img.copy()
    poly_orig_int = [(int(round(x)), int(round(y))) for x, y in poly_orig]

    for i in range(len(poly_orig_int) - 1):
        cv2.line(
            img_overlay,
            poly_orig_int[i],
            poly_orig_int[i + 1],
            color=(0, 0, 255),
            thickness=5,
        )
    if len(poly_orig_int) >= 1:
        cv2.circle(
            img_overlay,
            poly_orig_int[0],
            10,
            (0, 255, 0),
            thickness=3,
        )
    if len(poly_orig_int) >= 2:
        cv2.circle(
            img_overlay,
            poly_orig_int[-1],
            10,
            (0, 0, 255),
            thickness=3,
        )

    out_image = os.path.join(OUT_DIR, "original_with_path.png")
    cv2.imwrite(out_image, img_overlay)

    out_json = os.path.join(OUT_DIR, "solution_path_points_unwarped.json")
    out = {
        "source_corners_json": os.path.abspath(corners_json),
        "source_path_json": os.path.abspath(path_json),
        "original_image": os.path.abspath(orig_img_path),
        "warped_image": os.path.abspath(warped_image_path),
        "unwarped_path_pixels": [[int(x), int(y)] for (x, y) in poly_orig_int],
        "notes": "Polyline points in ORIGINAL image pixel coordinates.",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] Unwarped overlay saved to {out_image}")
    print(f"[OK] Unwarped path JSON saved to {out_json}")
    return out_image, out_json

# ------------------------------------------------------


# ------- Map unwarped pixels -> robot mm via H_IMG2MM -
def homog_apply(H: np.ndarray, pt_xy: Tuple[float, float]) -> Tuple[float, float]:
    x, y = float(pt_xy[0]), float(pt_xy[1])
    v = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(v[2]) < 1e-9:
        return (v[0], v[1])
    return (v[0] / v[2], v[1] / v[2])

def build_robot_polyline_from_unwarped(unwarped_json_path: str) -> List[Tuple[float, float]]:
    with open(unwarped_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts_px = data.get("unwarped_path_pixels", [])
    if not pts_px:
        raise ValueError("No 'unwarped_path_pixels' found in JSON.")

    robot_pts = []
    for (x_px, y_px) in pts_px:
        xmm, ymm = homog_apply(H_IMG2MM, (x_px, y_px))
        robot_pts.append((xmm, ymm))
    print(f"[OK] Built robot polyline with {len(robot_pts)} points.")
    return robot_pts

# ------------------------------------------------------


# ============================ MAIN ====================
def main():
    make_out_dir()

    # 1) Connect Dobot and move to view pose
    bot, _ = get_dobot(COM_PORT)
    try:
        bot.move_to(*DOBOT_VIEW_POSE)
        print(f"[OK] Dobot moved to view pose {DOBOT_VIEW_POSE}")
    except Exception as e:
        print(f"[WARN] Could not move Dobot to view pose: {e}")

    # 2) Open camera and capture one frame
    cap = open_camera(CAM_INDEX, REQ_W, REQ_H)
    raw_path, overlay_path, corners_json = capture_and_save_corners(cap)
    cap.release()
    cv2.destroyAllWindows()

    # 3) Warp maze using corners
    warp_path = warp_maze_from_json(corners_json)

    # 4) Build grid, detect circles (with Gemini color), write result.json
    result_json = build_grid_and_json(warp_path)

    # 5) Ask user start color, solve maze BFS on warped image
    start_sel = input("Start at red or green? [r/g]: ").strip().lower()
    if start_sel == 'r':
        start_color, end_color = "red", "green"
    else:
        start_color, end_color = "green", "red"
    _, path_json = solve_maze_bfs(result_json, start_color, end_color)

    # 6) Unwarp path back to original image
    _, unwarped_json = unwarp_path_and_overlay(corners_json, path_json, warp_path)

    # 7) Build robot polyline from unwarped path and move
    robot_poly = build_robot_polyline_from_unwarped(unwarped_json)
    print(f"[INFO] Moving along {len(robot_poly)} points at Z={TRACE_Z} mm...")
    move_robot_along(bot, robot_poly, z=TRACE_Z, r=TRACE_R, pause=TRACE_PAUSE_S)

    print("[DONE] Maze solved and robot motion completed.")

if __name__ == "__main__":
    main()

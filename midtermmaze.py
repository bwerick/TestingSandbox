#!/usr/bin/env python3
"""
Maze → Dobot Pipeline (single-file)

Features:
- Camera capture (press 's' to snap) or load an image with --image
- Automatic corner detection (largest rectangular contour) with fallback manual clicking ('c')
- Rectify to 1000x1000
- Robust HSV dot detection for green/red (choose start color with --start-color)
- Binarize (adaptive), optional skeletonize (--no-skel to disable)
- BFS shortest path (8-neighbor), optional RDP simplification
- Homography mapping (pixels→robot mm) using YOUR 7-point calibration (already embedded)
- Simulate in matplotlib by default, or send to Dobot if --serial is given

Usage examples:
  # Live camera, simulate:
  python maze_dobot_pipeline.py --camera 0 --start-color green --simulate

  # Load image, export GIF:
  python maze_dobot_pipeline.py --image maze.jpg --simulate --export run.gif

  # Send to Dobot (example serial), densify to 5 mm, dwell 2 s:
  python maze_dobot_pipeline.py --camera 0 --serial /dev/ttyUSB0 --z 10 --r 0 --dwell 2

Controls (camera window):
  s  -> snap the current frame
  c  -> manual corner pick mode (click 4 corners in TL,TR,BR,BL order)
  q  -> quit
"""

import argparse
import math
import sys
import time
from typing import List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================
# 1) Homography (your 7-pt fit)
# =========================
H_IMG2ROB = np.array(
    [
        [-0.0153836124, -0.169789518, 398.57098947],
        [-0.177116480, 0.000843622, 146.66430761],
        [-0.0000541178, 0.0000156956, 1.0],
    ],
    dtype=float,
)

H_ROB2IMG = np.array(
    [
        [4.89509640e-02, -5.90909760e00, 8.47143276e02],
        [-5.67863527e00, -2.07643596e-01, 2.29379318e03],
        [9.17786173e-05, -3.16528239e-04, 1.00984310e00],
    ],
    dtype=float,
)


# =========================
# Common helpers
# =========================
def _homog(pts):
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    return np.c_[pts, np.ones((pts.shape[0], 1))]


def transform_points(pts, H):
    hp = _homog(pts) @ H.T
    return (hp[:, :2] / hp[:, 2:3]).astype(float)


def pix_to_robot(uv):
    return transform_points(uv, H_IMG2ROB)


def robot_to_pix(xy):
    return transform_points(xy, H_ROB2IMG)


# =========================
# Rectification / corners
# =========================
def order_corners_tl_tr_br_bl(pts4):
    pts = np.array(pts4, dtype=float)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(angles)
    pts = pts[idx]
    # rotate so TL (min sum) first
    s = pts.sum(axis=1)
    start = np.argmin(s)
    return np.roll(pts, -start, axis=0)


def detect_largest_rect_corners(img_bgr) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(-1, 2).astype(float)
            return order_corners_tl_tr_br_bl(pts)
    return None


def rectify_maze(frame_bgr, corners_px, size=(1000, 1000)):
    W, H = size
    tl, tr, br, bl = order_corners_tl_tr_br_bl(corners_px)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=float)
    src = np.array([tl, tr, br, bl], dtype=float)
    H_img2rect = cv2.getPerspectiveTransform(
        src.astype(np.float32), dst.astype(np.float32)
    )
    rectified = cv2.warpPerspective(frame_bgr, H_img2rect, (W, H))
    return rectified, H_img2rect


# =========================
# Dot detection (HSV)
# =========================
def detect_color_centroid_hsv(img_bgr, color: str) -> Optional[Tuple[float, float]]:
    """
    Robust centroid for 'green' or 'red'.
    Red uses two hue bands (wrap-around). Works on rectified image.
    Returns (x,y) in pixels or None if not found.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if color.lower() == "green":
        # broad green range; tweak if needed
        lower = np.array([35, 40, 40])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color.lower() == "red":
        lower1 = np.array([0, 40, 40])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 40, 40])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    else:
        raise ValueError("color must be 'green' or 'red'")

    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # centroid via moments
    M = cv2.moments(mask)
    if M["m00"] < 50:  # too small
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (float(cx), float(cy))


def detect_start_goal_dots(rectified_img, start_color: str):
    start = detect_color_centroid_hsv(rectified_img, start_color)
    other = "red" if start_color.lower() == "green" else "green"
    goal = detect_color_centroid_hsv(rectified_img, other)
    if start is None or goal is None:
        raise RuntimeError(
            "Could not detect both start and goal dots (HSV thresholds may need tuning)."
        )
    return start, goal


# =========================
# Binarize / skeletonize
# =========================
def binarize_maze(rectified_img, remove_outer_border=True):
    gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    # close small gaps in walls
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # outside border removal: fill outer background and invert back to free-space=255
    if remove_outer_border:
        h, w = walls.shape
        flood = walls.copy()
        cv2.floodFill(flood, None, (0, 0), 0)  # ensure outer is black
        bw = walls
    # free space = white
    free = cv2.bitwise_not(walls)
    return free


def skeletonize_path(binary_free: np.ndarray) -> np.ndarray:
    """Zhang-Suen–style morphological skeletonization (OpenCV variant)."""
    img = (binary_free > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


# =========================
# BFS path
# =========================
def nearest_white(img, pt):
    """Return nearest white pixel to pt in binary image."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = img.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    if img[y, x] > 0:
        return (x, y)
    # small expanding search
    for r in range(1, 15):
        x0, x1 = max(0, x - r), min(w - 1, x + r)
        y0, y1 = max(0, y - r), min(h - 1, y + r)
        roi = img[y0 : y1 + 1, x0 : x1 + 1]
        ys, xs = np.where(roi > 0)
        if len(xs):
            # pick nearest
            xs = xs + x0
            ys = ys + y0
            d2 = (xs - x) ** 2 + (ys - y) ** 2
            i = int(np.argmin(d2))
            return (int(xs[i]), int(ys[i]))
    return (x, y)


def bfs_shortest_path(binary, start_px, goal_px, use_8=True) -> List[Tuple[int, int]]:
    h, w = binary.shape
    start = nearest_white(binary, start_px)
    goal = nearest_white(binary, goal_px)
    from collections import deque

    q = deque([start])
    visited = np.full((h, w), False, dtype=bool)
    prev = np.full((h, w, 2), -1, dtype=int)
    visited[start[1], start[0]] = True
    if use_8:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            break
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if not visited[ny, nx] and binary[ny, nx] > 0:
                    visited[ny, nx] = True
                    prev[ny, nx] = (x, y)
                    q.append((nx, ny))

    # reconstruct
    path = []
    cur = goal
    if prev[cur[1], cur[0], 0] == -1 and cur != start:
        raise RuntimeError("BFS failed to find a path.")
    while cur != (-1, -1):
        path.append(cur)
        px, py = prev[cur[1], cur[0]]
        if px == -1 and py == -1:
            break
        cur = (int(px), int(py))
    path.reverse()
    return path


def rdp(points, eps=1.5):
    """Ramer–Douglas–Peucker simplification for pixel path."""
    if len(points) < 3:
        return points[:]
    import numpy as np

    def perp_dist(p, a, b):
        if a == b:
            return math.hypot(p[0] - a[0], p[1] - a[1])
        ax, ay = a
        bx, by = b
        px, py = p
        num = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
        den = math.hypot(bx - ax, by - ay)
        return num / den

    # find farthest
    dmax = 0.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], points[0], points[-1])
        if d > dmax:
            idx = i
            dmax = d
    if dmax > eps:
        left = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


# =========================
# Mapping rectified px → robot mm
# =========================
def map_rectified_pixels_to_robot(path_rect_px, H_img2rect):
    H_rect2img = np.linalg.inv(H_img2rect)

    def map_pts(pts, H):
        hp = _homog(pts) @ H.T
        return hp[:, :2] / hp[:, 2:3]

    img_px = map_pts(path_rect_px, H_rect2img)  # rect → image px
    rob_xy = map_pts(img_px, H_IMG2ROB)  # image px → robot mm
    return [(float(x), float(y)) for x, y in rob_xy]


# =========================
# Visualization / simulation
# =========================
def plot_robot_path(path_mm, start_xy=None, goal_xy=None, title="Robot Path (mm)"):
    xs = [p[0] for p in path_mm]
    ys = [p[1] for p in path_mm]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys)
    ax.plot(xs, ys, marker=".", linestyle="")
    if start_xy is not None:
        ax.plot([start_xy[0]], [start_xy[1]], marker="o")
        ax.text(start_xy[0], start_xy[1], "start")
    if goal_xy is not None:
        ax.plot([goal_xy[0]], [goal_xy[1]], marker="o")
        ax.text(goal_xy[0], goal_xy[1], "goal")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    plt.show()


def densify_path_mm(path_mm, max_step=5.0):
    if len(path_mm) < 2:
        return path_mm
    out = [path_mm[0]]
    for (x0, y0), (x1, y1) in zip(path_mm, path_mm[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(0, int(d // max_step))
        for k in range(1, n + 1):
            t = k / (n + 1)
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        out.append((x1, y1))
    return out


def animate_robot_path(
    path_mm,
    interval_ms=120,
    save_path: Optional[str] = None,
    title="Robot Path Simulation (mm)",
):
    xs = [p[0] for p in path_mm]
    ys = [p[1] for p in path_mm]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.plot(xs, ys)
    (point,) = ax.plot([], [], marker="o", linestyle="")
    pad = max(5.0, 0.05 * max(max(xs) - min(xs), max(ys) - min(ys)))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    def init():
        point.set_data([], [])
        return (point,)

    def update(i):
        point.set_data(xs[i], ys[i])
        return (point,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(path_mm),
        init_func=init,
        interval=interval_ms,
        blit=True,
    )
    if save_path:
        ext = save_path.lower().split(".")[-1]
        if ext == "mp4":
            ani.save(save_path, writer="ffmpeg")
        elif ext == "gif":
            ani.save(save_path, writer="pillow")
        else:
            raise ValueError("export must be .mp4 or .gif")
    else:
        plt.show()


# =========================
# Dobot execution (optional)
# =========================
def send_moves_to_dobot(device, path_mm, z=10.0, r=0.0, dwell_s=2.0, max_step=5.0):
    path_mm = densify_path_mm(path_mm, max_step=max_step)
    for x, y in path_mm:
        device.move_to(x, y, z, r, wait=False)
        time.sleep(dwell_s)


# =========================
# Camera helpers
# =========================
class ClickCollector:
    def __init__(self, win):
        self.win = win
        self.pts = []
        cv2.setMouseCallback(win, self._cb)

    def _cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))

    def reset(self):
        self.pts = []


def capture_frame_from_camera(cam_index):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")
    print("Press 's' to snap, 'c' for manual corner clicking, 'q' to quit.")
    win = "camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    clicker = ClickCollector(win)
    snapped = None
    manual_corners = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        disp = frame.copy()
        cv2.putText(
            disp,
            "s=save, c=manual corners, q=quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        if len(clicker.pts) > 0:
            for p in clicker.pts:
                cv2.circle(disp, p, 5, (0, 0, 255), -1)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            snapped = frame.copy()
            print("Snapped.")
            break
        elif key == ord("c"):
            clicker.reset()
            print("Click 4 corners in TL,TR,BR,BL order, then press 's' to snap.")
        elif key == ord("q"):
            break
    if snapped is not None and len(clicker.pts) == 4:
        manual_corners = np.array(clicker.pts, dtype=float)
    cap.release()
    cv2.destroyAllWindows()
    return snapped, manual_corners


# =========================
# Main pipeline
# =========================
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="Camera index (e.g., 0)")
    src.add_argument("--image", type=str, help="Path to maze image")
    ap.add_argument(
        "--start-color", type=str, default="green", choices=["green", "red"]
    )
    ap.add_argument(
        "--no-skel",
        action="store_true",
        help="Disable skeletonize (solve on raw free space)",
    )
    ap.add_argument(
        "--rdp",
        type=float,
        default=2.0,
        help="RDP epsilon (px) for path simplification",
    )
    ap.add_argument(
        "--simulate", action="store_true", help="Simulate instead of sending to Dobot"
    )
    ap.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export animation to .gif/.mp4 in simulate mode",
    )
    ap.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Dobot serial port (if omitted, simulate)",
    )
    ap.add_argument("--z", type=float, default=10.0)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--dwell", type=float, default=2.0)
    ap.add_argument(
        "--max-step", type=float, default=5.0, help="Max step between waypoints (mm)"
    )
    args = ap.parse_args()

    # 1) Get frame + corners
    if args.camera is not None:
        frame, manual_corners = capture_frame_from_camera(args.camera)
        if frame is None:
            print("No frame captured. Exiting.")
            sys.exit(1)
        corners = detect_largest_rect_corners(frame)
        if corners is None and manual_corners is not None:
            corners = manual_corners
        if corners is None:
            print("Could not detect corners (and none clicked). Exiting.")
            sys.exit(1)
    else:
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Failed to read {args.image}")
            sys.exit(1)
        corners = detect_largest_rect_corners(frame)
        if corners is None:
            print("WARNING: auto-corner detection failed on image.")
            print("You can run with --camera and use 'c' to click corners.")
            sys.exit(1)

    # 2) Rectify to 1000x1000
    rectified, H_img2rect = rectify_maze(frame, corners, size=(1000, 1000))

    # 3) Detect start/goal dots
    start_px, goal_px = detect_start_goal_dots(rectified, args.start_color)
    # 4) Binarize (+ optional skeleton)
    free = binarize_maze(rectified, remove_outer_border=True)
    work = skeletonize_path(free) if not args.no_skel else free

    # 5) BFS
    path_px = bfs_shortest_path(work, start_px, goal_px, use_8=True)
    if args.rdp > 0:
        path_px = rdp(path_px, eps=args.rdp)

    # 6) Map to robot mm
    # Note: BFS path is (x,y) pixel order; ensure it's floats:
    path_rect_xy = [(float(x), float(y)) for (x, y) in path_px]
    path_mm = map_rectified_pixels_to_robot(path_rect_xy, H_img2rect)

    # 7) Simulate OR send to Dobot
    plot_robot_path(path_mm, title="Planned Robot Path (mm)")
    if args.serial is None or args.simulate:
        animate_robot_path(path_mm, interval_ms=120, save_path=args.export)
    else:
        try:
            import pydobot
        except Exception as e:
            print("pydobot not available; falling back to sim.")
            animate_robot_path(path_mm, interval_ms=120, save_path=args.export)
            sys.exit(0)
        # Connect and send
        device = pydobot.Dobot(port=args.serial)
        try:
            send_moves_to_dobot(
                device,
                path_mm,
                z=args.z,
                r=args.r,
                dwell_s=args.dwell,
                max_step=args.max_step,
            )
        finally:
            device.close()


if __name__ == "__main__":
    main()

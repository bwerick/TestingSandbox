#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hands-free Maze → Dobot pipeline with AUTO corner detection + stability lock.

Usage (simulate only):
  python maze_dobot_autorun.py --camera 0 --start-color green --simulate --preview

Export animation:
  python maze_dobot_autorun.py --camera 0 --start-color red --simulate --export run.gif

Send to Dobot (if connected):
  python maze_dobot_autorun.py --camera 0 --serial /dev/ttyUSB0 --start-color green

Notes:
- No clicks/keypresses during run. The only control is --start-color.
- Corner detector: white-page HSV (with auto-sweep of S_max, V_min, inset%)
  + generic rectangle fallback. Stability lock ensures corners are steady.
"""

import argparse
import math
import sys
import time
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# 0) Homography (from your 7-pt fit)
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
# Helpers (homogeneous transforms)
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


def order_corners_tl_tr_br_bl(pts4):
    """Order 4 points TL, TR, BR, BL."""
    pts = np.array(pts4, dtype=float)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    s = pts.sum(axis=1)
    return np.roll(pts, -np.argmin(s), axis=0)


def _rect_quality(pts):
    """Higher is better; negative for bad quads."""
    if pts is None or np.any(np.isnan(pts)) or pts.shape != (4, 2):
        return -1e9
    p = order_corners_tl_tr_br_bl(pts)
    if not cv2.isContourConvex(p.astype(np.int32)):
        return -1e9

    # angles near 90
    def angle(a, b, c):
        ba = a - b
        bc = c - b
        nb = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
        cosang = np.dot(ba, bc) / nb
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    angs = [angle(p[(i - 1) % 4], p[i], p[(i + 1) % 4]) for i in range(4)]
    ang_err = sum(abs(a - 90) for a in angs)
    # roughly square + reasonable area
    tl, tr, br, bl = p
    w = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    h = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    area = w * h
    if area < 2e4:
        return -1e9
    aspect = max(w, h) / (min(w, h) + 1e-9)
    ar_err = abs(aspect - 1.0)
    return -(ang_err + 40 * ar_err)


# =========================
# 1) White-page HSV corner detector (+ tuner-style params)
# =========================
def _box_from_contour(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(float)
    return order_corners_tl_tr_br_bl(box)


def _intersections_from_lines(mask, linesP):
    if linesP is None or len(linesP) < 4:
        return None
    Ls = linesP.reshape(-1, 4)
    left = min(Ls, key=lambda L: min(L[0], L[2]))
    right = max(Ls, key=lambda L: max(L[0], L[2]))
    top = min(Ls, key=lambda L: min(L[1], L[3]))
    bot = max(Ls, key=lambda L: max(L[1], L[3]))

    def line(p1, p2):
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = A * p1[0] + B * p1[1]
        return A, B, C

    def inter(L1p1, L1p2, L2p1, L2p2):
        A1, B1, C1 = line(L1p1, L1p2)
        A2, B2, C2 = line(L2p1, L2p2)
        det = A1 * B2 - A2 * B1
        if abs(det) < 1e-6:
            return None
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return (x, y)

    P = lambda L: ((int(L[0]), int(L[1])), (int(L[2]), int(L[3])))
    tl = inter(*P(top), *P(left))
    tr = inter(*P(top), *P(right))
    br = inter(*P(bot), *P(right))
    bl = inter(*P(bot), *P(left))
    if None in (tl, tr, br, bl):
        return None
    return order_corners_tl_tr_br_bl(np.array([tl, tr, br, bl], dtype=float))


def detect_maze_corners_whiteHSV(
    frame_bgr, s_max=30, v_min=150, inset_frac=0.02, show_overlay=False
):
    """Detect outer square by isolating white page in HSV (S<=s_max, V>=v_min)."""
    H, W = frame_bgr.shape[:2]
    x0 = int(W * inset_frac)
    y0 = int(H * inset_frac)
    x1 = int(W * (1 - inset_frac))
    y1 = int(H * (1 - inset_frac))
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([0, 0, v_min], np.uint8), np.array([180, s_max, 255], np.uint8)
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1e9
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        # try polygonal approx
        for eps_scale in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(cnt, eps_scale * peri, True)
            if len(approx) == 4:
                cand = order_corners_tl_tr_br_bl(approx.reshape(-1, 2).astype(float))
                score = _rect_quality(cand)
                if score > best_score:
                    best_score, best = score, cand
        # try minAreaRect
        if best is None:
            box = _box_from_contour(cnt)
            score = _rect_quality(box)
            if score > best_score:
                best_score, best = score, box

    # Hough fallback inside ROI mask
    if best is None:
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=120,
            minLineLength=min(W, H) // 3,
            maxLineGap=25,
        )
        cand = _intersections_from_lines(mask, lines)
        if cand is not None:
            score = _rect_quality(cand)
            if score > best_score:
                best_score, best = score, cand

    if best is None:
        return None
    best[:, 0] += x0
    best[:, 1] += y0

    if show_overlay:
        dbg = frame_bgr.copy()
        tl, tr, br, bl = best.astype(int)
        cv2.line(dbg, tuple(tl), tuple(tr), (0, 255, 0), 2)
        cv2.line(dbg, tuple(tr), tuple(br), (0, 255, 0), 2)
        cv2.line(dbg, tuple(br), tuple(bl), (0, 255, 0), 2)
        cv2.line(dbg, tuple(bl), tuple(tl), (0, 255, 0), 2)
        cv2.line(dbg, tuple(tl), tuple(br), (0, 255, 0), 1)
        cv2.line(dbg, tuple(tr), tuple(bl), (0, 255, 0), 1)
        for name, p in zip(["TL", "TR", "BR", "BL"], [tl, tr, br, bl]):
            cv2.circle(dbg, tuple(p), 6, (0, 255, 255), -1)
            cv2.putText(
                dbg,
                name,
                tuple(p + np.array([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        cv2.putText(
            dbg, "maze: FOUND", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
        cv2.imshow("maze-corners", dbg)
        cv2.waitKey(1)

    return best


# =========================
# 2) Auto-sweep of HSV/inset + generic fallback
# =========================
def detect_rect_generic(img_bgr):
    """Generic approx-quad via adaptive threshold + contours (fallback)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = None
    best_score = -1e9
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:12]:
        peri = cv2.arcLength(cnt, True)
        for eps_scale in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(cnt, eps_scale * peri, True)
            if len(approx) == 4:
                cand = order_corners_tl_tr_br_bl(approx.reshape(-1, 2).astype(float))
                score = _rect_quality(cand)
                if score > best_score:
                    best_score, best = score, cand
    return best


def autosweep_white_params(frame_bgr):
    """Try a grid of (s_max, v_min, inset) and return best corners+params."""
    s_list = [20, 25, 30, 35, 40, 50, 60]
    v_list = [130, 140, 150, 160, 180, 200]
    inset_list = [0.00, 0.01, 0.02, 0.03, 0.05, 0.06]
    best = None
    best_score = -1e9
    best_params = (30, 150, 0.02)
    for s_max in s_list:
        for v_min in v_list:
            for inset in inset_list:
                cand = detect_maze_corners_whiteHSV(
                    frame_bgr,
                    s_max=s_max,
                    v_min=v_min,
                    inset_frac=inset,
                    show_overlay=False,
                )
                if cand is not None:
                    score = _rect_quality(cand)
                    if score > best_score:
                        best_score, best = score, cand
                        best_params = (s_max, v_min, inset)
    if best is None:
        # last resort generic
        cand = detect_rect_generic(frame_bgr)
        if cand is not None:
            return cand, best_params
        return None, best_params
    return best, best_params


def clear_dot_obstacles(rectified_img_bgr, free_mask):
    """
    Ensure the red/green dots don't block the maze:
    sets the dot regions to free (white) in the free-mask.
    """
    hsv = cv2.cvtColor(rectified_img_bgr, cv2.COLOR_BGR2HSV)

    # green
    g1 = np.array([35, 40, 40])
    g2 = np.array([85, 255, 255])
    mg = cv2.inRange(hsv, g1, g2)

    # red (two hue bands)
    r1a = np.array([0, 40, 40])
    r1b = np.array([10, 255, 255])
    r2a = np.array([170, 40, 40])
    r2b = np.array([180, 255, 255])
    mr = cv2.inRange(hsv, r1a, r1b) | cv2.inRange(hsv, r2a, r2b)

    dots = cv2.bitwise_or(mg, mr)

    # grow to cover the full paint spot
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    dots = cv2.morphologyEx(dots, cv2.MORPH_CLOSE, k, iterations=1)
    dots = cv2.dilate(dots, k, iterations=1)

    free = free_mask.copy()
    free[dots > 0] = 255
    return free


def ensure_free_disks(free_mask, centers, radius=10):
    """
    Punch small free disks at given centers (start/goal) to guarantee
    BFS can start/end on free pixels even if dots overlap walls.
    """
    out = free_mask.copy()
    for cx, cy in centers:
        if cx is None or cy is None:
            continue
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(radius), 255, -1)
    return out


def repair_corridors(free_mask, iterations=1):
    """
    Light morphology to bridge hairline breaks in corridors.
    free=255, walls=0. 'Closing' expands free slightly then shrinks back.
    """
    if iterations <= 0:
        return free_mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fm = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, k, iterations=iterations)
    return fm


# =========================
# 3) Corner stability lock (hands-free)
# =========================
def lock_corners_from_camera(
    cam_index: int,
    lock_seconds: float = 3.0,
    lock_eps_px: float = 2.0,
    max_wait_seconds: float = 45.0,
    show_preview: bool = False,
    s_max: int = 30,
    v_min: int = 150,
    inset_frac: float = 0.02,
    autosweep_every_n: int = 12,
):
    """
    Continuously detect corners; when stable (<= lock_eps_px) for >= lock_seconds,
    return (frame, corners). Tries white-HSV first; periodically auto-sweeps
    parameters; falls back to generic detector if needed. No keyboard/clicks.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")

    last_corners = None
    locked_start_t = None
    t0 = time.time()
    last_frame = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        last_frame = frame
        frame_idx += 1

        # primary detector with current params
        corners = detect_maze_corners_whiteHSV(
            frame,
            s_max=s_max,
            v_min=v_min,
            inset_frac=inset_frac,
            show_overlay=show_preview,
        )

        # every N frames attempt autosweep if not found
        if corners is None and (frame_idx % autosweep_every_n == 0):
            cand, (s_max, v_min, inset_frac) = autosweep_white_params(frame)
            corners = cand

        # final fallback
        if corners is None:
            corners = detect_rect_generic(frame)

        if corners is not None:
            corners = order_corners_tl_tr_br_bl(corners)
            if last_corners is None:
                last_corners = corners
                locked_start_t = None
            else:
                diffs = np.linalg.norm(corners - last_corners, axis=1)
                stable = np.all(diffs <= lock_eps_px)
                # overlay (non-interactive)
                if show_preview:
                    dbg = frame.copy()
                    for i, p in enumerate(corners.astype(int)):
                        cv2.circle(dbg, tuple(p), 6, (0, 255, 255), -1)
                        cv2.putText(
                            dbg,
                            ["TL", "TR", "BR", "BL"][i],
                            tuple(p + np.array([8, -8])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    if stable and locked_start_t is not None:
                        elapsed = time.time() - locked_start_t
                        cv2.putText(
                            dbg,
                            f"Stable: {elapsed:.1f}/{lock_seconds}s",
                            (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.putText(
                            dbg,
                            f"Stabilizing...",
                            (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                        )
                    cv2.imshow("preview", dbg)
                    cv2.waitKey(1)

                if stable:
                    if locked_start_t is None:
                        locked_start_t = time.time()
                    if (time.time() - locked_start_t) >= lock_seconds:
                        cap.release()
                        if show_preview:
                            cv2.destroyAllWindows()
                        return last_frame.copy(), corners.copy()
                else:
                    locked_start_t = None
                last_corners = corners

        if time.time() - t0 > max_wait_seconds:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            raise RuntimeError(
                "Auto-corner lock timed out. Try --preview and ensure the square is fully in frame with good contrast."
            )


# =========================
# 4) Rectify
# =========================
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
# 5) Dot detection (HSV)
# =========================
def detect_color_centroid_hsv(img_bgr, color: str) -> Optional[Tuple[float, float]]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if color.lower() == "green":
        lower = np.array([35, 40, 40])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif color.lower() == "red":
        l1 = np.array([0, 40, 40])
        u1 = np.array([10, 255, 255])
        l2 = np.array([170, 40, 40])
        u2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, l1, u1) | cv2.inRange(hsv, l2, u2)
    else:
        raise ValueError("color must be 'green' or 'red'")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    M = cv2.moments(mask)
    if M["m00"] < 50:
        return None
    return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))


def detect_start_goal_dots(rectified_img, start_color: str):
    start = detect_color_centroid_hsv(rectified_img, start_color)
    other = "red" if start_color.lower() == "green" else "green"
    goal = detect_color_centroid_hsv(rectified_img, other)
    if start is None or goal is None:
        raise RuntimeError(
            "Could not detect both start and goal dots. Adjust lighting/size or HSV bounds."
        )
    return start, goal


# =========================
# 6) Binarize / skeletonize / BFS
# =========================
def binarize_maze(rectified_img, remove_outer_border=True):
    gray = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    if remove_outer_border:
        flood = walls.copy()
        cv2.floodFill(flood, None, (0, 0), 0)
    free = cv2.bitwise_not(walls)
    return free


def skeletonize_path(binary_free: np.ndarray) -> np.ndarray:
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


def dump_debug(dirpath, rectified, free, work, start_px, goal_px):
    import os

    os.makedirs(dirpath, exist_ok=True)
    cv2.imwrite(os.path.join(dirpath, "rectified.png"), rectified)
    cv2.imwrite(os.path.join(dirpath, "free.png"), free)
    color = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR) if work.ndim == 2 else work.copy()
    sp = (int(round(start_px[0])), int(round(start_px[1])))
    gp = (int(round(goal_px[0])), int(round(goal_px[1])))
    cv2.circle(color, sp, 9, (0, 255, 0), -1)
    cv2.circle(color, gp, 9, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(dirpath, "work_with_dots.png"), color)


def nearest_white(img, pt):
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = img.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    if img[y, x] > 0:
        return (x, y)
    for r in range(1, 50):
        x0, x1 = max(0, x - r), min(w - 1, x + r)
        y0, y1 = max(0, y - r), min(h - 1, y + r)
        roi = img[y0 : y1 + 1, x0 : x1 + 1]
        ys, xs = np.where(roi > 0)
        if len(xs):
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
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)] + (
        [(-1, -1), (1, -1), (-1, 1), (1, 1)] if use_8 else []
    )
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
    if len(points) < 3:
        return points[:]

    def perp_dist(p, a, b):
        if a == b:
            return math.hypot(p[0] - a[0], p[1] - a[1])
        ax, ay = a
        bx, by = b
        px, py = p
        num = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
        den = math.hypot(bx - ax, by - ay)
        return num / den

    dmax = 0.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], points[0], points[-1])
        if d > dmax:
            idx, dmax = i, d
    if dmax > eps:
        left = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


# =========================
# 7) Map rectified px → robot mm
# =========================
def map_rectified_pixels_to_robot(path_rect_px, H_img2rect):
    H_rect2img = np.linalg.inv(H_img2rect)

    def map_pts(pts, H):
        hp = _homog(pts) @ H.T
        return hp[:, :2] / hp[:, 2:3]

    img_px = map_pts(path_rect_px, H_rect2img)
    rob_xy = map_pts(img_px, H_IMG2ROB)
    return [(float(x), float(y)) for x, y in rob_xy]


# =========================
# 8) Visualization / simulation
# =========================
def plot_robot_path(path_mm, title="Planned Robot Path (mm)"):
    xs = [p[0] for p in path_mm]
    ys = [p[1] for p in path_mm]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys)
    ax.plot(xs, ys, marker=".", linestyle="")
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
# 9) Dobot (optional)
# =========================
def send_moves_to_dobot(device, path_mm, z=10.0, r=0.0, dwell_s=2.0, max_step=5.0):
    path_mm = densify_path_mm(path_mm, max_step=max_step)
    for x, y in path_mm:
        device.move_to(x, y, z, r, wait=False)
        time.sleep(dwell_s)


def tune_corner_detector(cam_index=0):
    """Live knob tuner for white-maze HSV thresholds."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")

    s_max = 30
    v_min = 150
    inset = 0.02

    def nothing(x):
        pass

    cv2.namedWindow("tuner")
    cv2.createTrackbar("white S max", "tuner", s_max, 255, nothing)
    cv2.createTrackbar("white V min", "tuner", v_min, 255, nothing)
    cv2.createTrackbar("inset x100", "tuner", int(inset * 100), 10, nothing)

    print("\n[ESC] to quit and print final values\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        s_max = cv2.getTrackbarPos("white S max", "tuner")
        v_min = cv2.getTrackbarPos("white V min", "tuner")
        inset = cv2.getTrackbarPos("inset x100", "tuner") / 100.0

        _ = detect_maze_corners_whiteHSV(
            frame, s_max=s_max, v_min=v_min, inset_frac=inset, show_overlay=True
        )
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final: s_max={s_max}, v_min={v_min}, inset={inset:.2f}")


# =========================
# 10) Main (hands-free)
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--camera", type=int, required=True, help="Camera index, e.g., 0")
    ap.add_argument(
        "--start-color", type=str, default="green", choices=["green", "red"]
    )
    ap.add_argument(
        "--no-skel",
        action="store_true",
        help="Disable skeleton (solve on raw free-space)",
    )
    ap.add_argument("--rdp", type=float, default=2.0, help="RDP epsilon (px)")
    ap.add_argument(
        "--simulate", action="store_true", help="Simulate instead of sending to Dobot"
    )
    ap.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export animation .gif/.mp4 (simulate mode)",
    )
    ap.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Dobot serial port; omit for simulation",
    )
    ap.add_argument("--z", type=float, default=10.0)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--dwell", type=float, default=2.0)
    ap.add_argument("--max-step", type=float, default=5.0)
    # Corner lock controls
    ap.add_argument(
        "--lock-seconds", type=float, default=3.0, help="Stable time required (s)"
    )
    ap.add_argument(
        "--lock-eps", type=float, default=2.0, help="Per-corner jitter tolerance (px)"
    )
    ap.add_argument(
        "--lock-timeout", type=float, default=45.0, help="Max wait for lock (s)"
    )
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Show preview overlay during lock (no keypresses)",
    )
    # Initial HSV/inset guesses (autosweep runs if these fail)
    ap.add_argument("--s-max", type=int, default=30, help="HSV white S max initial")
    ap.add_argument("--v-min", type=int, default=150, help="HSV white V min initial")
    ap.add_argument(
        "--inset", type=float, default=0.02, help="Inset fraction initial [0..0.08]"
    )
    # Maze cleaning & debug options
    ap.add_argument(
        "--open-radius",
        type=int,
        default=12,
        help="Free disk radius at start/goal (px)",
    )
    ap.add_argument(
        "--repair", type=int, default=1, help="Corridor repair (closing) iterations"
    )
    ap.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Folder to save rectified/free/work images",
    )
    ap.add_argument(
        "--tune-corners",
        action="store_true",
        help="Open interactive HSV/inset tuner with trackbars",
    )

    args = ap.parse_args()

    if args.tune_corners:
        tune_corner_detector(args.camera)
        return

    # 1) Auto-lock corners (hands-free)
    frame, corners = lock_corners_from_camera(
        cam_index=args.camera,
        lock_seconds=args.lock_seconds,
        lock_eps_px=args.lock_eps,
        max_wait_seconds=args.lock_timeout,
        show_preview=args.preview,
        s_max=args.s_max,
        v_min=args.v_min,
        inset_frac=args.inset,
        autosweep_every_n=12,
    )

    # 2) Rectify
    rectified, H_img2rect = rectify_maze(frame, corners, size=(1000, 1000))

    # 3) Dots
    start_px, goal_px = detect_start_goal_dots(rectified, args.start_color)

    # 4) Binarize (+ optional skeleton)
    free = binarize_maze(rectified, remove_outer_border=True)

    # 1) make sure colored dots are traversable
    free = clear_dot_obstacles(rectified, free)

    # 2) ensure BFS can start/end on free
    free = ensure_free_disks(free, [start_px, goal_px], radius=args.open_radius)

    # 3) optionally repair tiny gaps
    free = repair_corridors(free, iterations=args.repair)

    # 4) choose solver image (keep skeleton off until connectivity is proven)
    work = free if args.no_skel else skeletonize_path(free)

    # (optional) dump masks for quick inspection
    if args.dump_dir:
        dump_debug(args.dump_dir, rectified, free, work, start_px, goal_px)

    # 5) BFS
    path_px = bfs_shortest_path(work, start_px, goal_px, use_8=True)
    if args.rdp > 0:
        path_px = rdp(path_px, eps=args.rdp)

    # 6) Rectified px → robot mm
    path_rect_xy = [(float(x), float(y)) for (x, y) in path_px]
    path_mm = map_rectified_pixels_to_robot(path_rect_xy, H_img2rect)

    # 7) Simulate OR send
    if args.serial is None or args.simulate:
        plot_robot_path(path_mm, title="Planned Robot Path (mm)")
        animate_robot_path(path_mm, interval_ms=120, save_path=args.export)
    else:
        try:
            import pydobot
        except Exception:
            print("pydobot not available; simulating instead.")
            plot_robot_path(path_mm, title="Planned Robot Path (mm)")
            animate_robot_path(path_mm, interval_ms=120, save_path=args.export)
            sys.exit(0)
        device = pydobot.Dobot(port=args.serial)
        device.speed(80, 80)
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
            device.move_to(240, 0, 145, 0, wait=True)
            device.close()


if __name__ == "__main__":
    main()

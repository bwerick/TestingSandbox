import cv2
import numpy as np
from collections import deque
from time import time

CAM_INDEX = 0
FRAME_SIZE = (1280, 1024)
RECT_SIZE = 480  # pixels for rectified square
AUTO_GRID = True  # press 'g' in UI to toggle to forced 4x4

robotMiddleLeft = 280.1, 75.7
robotBottomLeft = 213.2, 145.2
robotMiddleRight = 284.6, 3.5
robotBottomRight = 213.2, -84.4
cameraMiddleLeft = 406, 659
cameraBottomLeft = 9, 1019
cameraMiddleRight = 805, 626
cameraBottomRight = 1272, 1019


# ---------------------- small utils ----------------------
def order_tl_tr_bl_br(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, bl, br], dtype=np.float32)


def draw_text(img, txt, org, color=(255, 255, 255), scale=0.6):
    cv2.putText(
        img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA
    )
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------- page detect + rectify ----------------------
def find_maze_quad(frame, white_S_max=60, white_V_min=180, inset_frac=0.06):
    """
    1) HSV 'white' segmentation -> mask
    2) largest contour -> minAreaRect (robust if clipped)
    3) inset the rectangle (ignore margins)
    4) warp to RECT_SIZE x RECT_SIZE square
    """
    H0, W0 = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, white_V_min], dtype=np.uint8)
    upper = np.array([179, white_S_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None, None, mask
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.05 * W0 * H0:  # too small
        return False, None, None, None, mask

    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect).astype(np.float32)
    quad = order_tl_tr_bl_br(box)

    # inset along rectangle axes
    tl, tr, bl, br = quad
    top_dir = tr - tl
    top_len = np.linalg.norm(top_dir) + 1e-6
    top_dir /= top_len
    left_dir = bl - tl
    left_len = np.linalg.norm(left_dir) + 1e-6
    left_dir /= left_len
    inset_t, inset_l = inset_frac * top_len, inset_frac * left_len
    tl_i = tl + top_dir * inset_t + left_dir * inset_l
    tr_i = tr - top_dir * inset_t + left_dir * inset_l
    bl_i = bl + top_dir * inset_t - left_dir * inset_l
    br_i = br - top_dir * inset_t - left_dir * inset_l
    quad_inset = np.stack([tl_i, tr_i, bl_i, br_i], axis=0).astype(np.float32)

    dst = np.array(
        [
            [0, 0],
            [RECT_SIZE - 1, 0],
            [0, RECT_SIZE - 1],
            [RECT_SIZE - 1, RECT_SIZE - 1],
        ],
        dtype=np.float32,
    )
    H_img2rect = cv2.getPerspectiveTransform(quad_inset, dst)
    rectified = cv2.warpPerspective(
        frame,
        H_img2rect,
        (RECT_SIZE, RECT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return True, quad_inset, H_img2rect, rectified, mask


# ---------------------- grid size detection ----------------------
def smooth1d(x, k=11):
    k = max(3, int(k) | 1)
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def fundamental_period_from_projection(proj, min_period=60, max_period=220):
    """
    Find dominant spacing via autocorrelation of the projection.
    Returns estimated period in pixels (or None).
    """
    s = proj.astype(np.float32)
    s = s - s.mean()
    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac) // 2 :]  # non-negative lags
    # Ignore very small lags; search in [min_period,max_period]
    lo = int(min_period)
    hi = min(len(ac) - 1, int(max_period))
    if hi <= lo + 3:
        return None
    # Smooth to reduce spurious peaks
    ac_s = smooth1d(ac, k=21)
    lag = lo + np.argmax(ac_s[lo:hi])
    return lag


def detect_grid_size(rectified, N_min=3, N_max=8):
    """
    Hybrid grid-size detector using direct boundary scoring.
    Works reliably up to 8x8.
    """
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    # strengthen walls
    kx = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    ky = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kx, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ky, iterations=1)

    def boundary_score(N):
        step = RECT_SIZE / N
        pad = 0.18
        score = 0
        cnt = 0
        # vertical
        for r in range(N):
            for c in range(N - 1):
                x = (c + 1) * step
                y0 = r * step + pad * step
                y1 = (r + 1) * step - pad * step
                hit = sample_line_has_wall(
                    bw, (x, y0), (x, y1), thickness=4, samples=40
                )
                score += hit
                cnt += 1
        # horizontal
        for r in range(N - 1):
            y = (r + 1) * step
            for c in range(N):
                x0 = c * step + pad * step
                x1 = (c + 1) * step - pad * step
                hit = sample_line_has_wall(
                    bw, (x0, y), (x1, y), thickness=4, samples=40
                )
                score += hit
                cnt += 1
        return score / cnt

    bestN, bestS = None, -1
    scores = []
    for N in range(N_min, N_max + 1):
        s = boundary_score(N)
        scores.append((N, s))
        if s > bestS:
            bestS, bestN = s, N

    # optional: print or overlay scores to see confidence
    print("Grid scores:", scores)
    return bestN


# ---------------------- red / green detection ----------------------
def find_dot_rectified(rectified, color="red"):
    """
    Return (cx,cy) in rectified coords or None.
    Assumes saturated marker dots.
    """
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
    if color == "red":
        # red wraps hue (two ranges)
        m1 = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 120, 80), (179, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    else:  # green
        mask = cv2.inRange(hsv, (40, 80, 60), (85, 255, 255))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 20:
        return None
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


# ---------------------- grid & graph build ----------------------
def cell_centers(N):
    step = RECT_SIZE / N
    return [((c + 0.5) * step, (r + 0.5) * step) for r in range(N) for c in range(N)]


def sample_line_has_wall(bw, p0, p1, thickness=5, samples=60, frac=0.18):
    """
    Returns True if enough black pixels along a line segment (with thickness).
    """
    h, w = bw.shape
    hit = 0
    for t in np.linspace(0, 1, samples):
        x = int(round(p0[0] * (1 - t) + p1[0] * t))
        y = int(round(p0[1] * (1 - t) + p1[1] * t))
        x0, x1 = max(0, x - thickness), min(w - 1, x + thickness)
        y0, y1 = max(0, y - thickness), min(h - 1, y + thickness)
        if (bw[y0 : y1 + 1, x0 : x1 + 1] > 0).any():
            hit += 1
    return hit >= frac * samples


def build_graph_from_rectified(rectified, N):
    """
    Determine walls between neighboring cells by sampling the mid-boundaries.
    Returns adjacency dict and debug arrays walls_v, walls_h.
    """
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )

    step = RECT_SIZE / N
    pad = 0.18  # keep away from cell corners
    walls_v = np.zeros((N, N - 1), dtype=bool)
    walls_h = np.zeros((N - 1, N), dtype=bool)

    for r in range(N):
        for c in range(N - 1):
            x = (c + 1) * step
            y0 = r * step + pad * step
            y1 = (r + 1) * step - pad * step
            walls_v[r, c] = sample_line_has_wall(bw, (x, y0), (x, y1), thickness=4)

    for r in range(N - 1):
        y = (r + 1) * step
        for c in range(N):
            x0 = c * step + pad * step
            x1 = (c + 1) * step - pad * step
            walls_h[r, c] = sample_line_has_wall(bw, (x0, y), (x1, y), thickness=4)

    adj = {(r, c): [] for r in range(N) for c in range(N)}
    for r in range(N):
        for c in range(N):
            if c < N - 1 and not walls_v[r, c]:
                adj[(r, c)].append((r, c + 1))
                adj[(r, c + 1)].append((r, c))
            if r < N - 1 and not walls_h[r, c]:
                adj[(r, c)].append((r + 1, c))
                adj[(r + 1, c)].append((r, c))
    return adj, walls_v, walls_h


def pixel_to_cell(pt, N):
    step = RECT_SIZE / N
    c = int(np.clip(pt[0] / step, 0, N - 1))
    r = int(np.clip(pt[1] / step, 0, N - 1))
    return (r, c)


# ---------------------- BFS ----------------------
def bfs(adj, start, goal):
    q = deque([start])
    prev = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    if goal not in prev:
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


# ---------------------- draw helpers ----------------------
def draw_solution(rectified, path, N, start=None, goal=None):
    vis = rectified.copy()
    step = RECT_SIZE / N
    pts = [(int((c + 0.5) * step), int((r + 0.5) * step)) for (r, c) in path]
    if len(pts) >= 2:
        cv2.polylines(
            vis,
            [np.array(pts, np.int32).reshape(-1, 1, 2)],
            False,
            (0, 0, 255),
            4,
            cv2.LINE_AA,
        )
    if start:
        s = (int((start[1] + 0.5) * step), int((start[0] + 0.5) * step))
        cv2.circle(vis, s, 10, (0, 255, 0), 2, cv2.LINE_AA)
    if goal:
        g = (int((goal[1] + 0.5) * step), int((goal[0] + 0.5) * step))
        cv2.circle(vis, g, 10, (0, 140, 255), 2, cv2.LINE_AA)
    return vis


def draw_main(
    frame, status, quad=None, rect_thumb=None, mask_thumb=None, fps=None, info_lines=()
):
    vis = frame.copy()
    draw_text(
        vis, status, (12, 28), (40, 220, 40) if "FOUND" in status else (0, 0, 255), 0.8
    )
    if fps is not None:
        draw_text(vis, f"{fps:.1f} FPS", (vis.shape[1] - 120, 28))
    if quad is not None:
        q = quad.astype(int)
        cv2.polylines(vis, [q], True, (0, 255, 0), 2, cv2.LINE_AA)
        labels = ["TL", "TR", "BL", "BR"]
        for i, (x, y) in enumerate(q):
            cv2.circle(vis, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
            draw_text(vis, labels[i], (x + 6, y - 6), (0, 255, 255), 0.5)
    if rect_thumb is not None:
        small = cv2.resize(rect_thumb, (260, 260))
        x0 = vis.shape[1] - 10 - 260
        vis[10 : 10 + 260, x0 : x0 + 260] = small
        cv2.rectangle(vis, (x0, 10), (x0 + 260, 10 + 260), (255, 255, 255), 1)
    if mask_thumb is not None:
        mcol = cv2.cvtColor(mask_thumb, cv2.COLOR_GRAY2BGR)
        mcol = cv2.resize(mcol, (170, 120))
        vis[-120 - 10 : -10, 10 : 10 + 170] = mcol
        cv2.rectangle(
            vis,
            (10, vis.shape[0] - 120 - 10),
            (10 + 170, vis.shape[0] - 10),
            (255, 255, 255),
            1,
        )
    y = vis.shape[0] - 40
    for line in info_lines:
        draw_text(vis, line, (12, y), (220, 220, 220), 0.6)
        y -= 22
    draw_text(
        vis,
        "[F]reeze  [R]esume  [G]rid 4x4 toggle  [ESC] quit",
        (12, vis.shape[0] - 12),
        (240, 240, 240),
        0.6,
    )
    return vis


# ---------------------- main ----------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    if not cap.isOpened():
        raise SystemExit("❌ Cannot open camera")

    cv2.namedWindow("Maze Live", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("white S max", "Maze Live", 60, 120, lambda v: None)
    cv2.createTrackbar("white V min", "Maze Live", 180, 255, lambda v: None)
    cv2.createTrackbar("inset% x100", "Maze Live", int(0.06 * 100), 20, lambda v: None)

    freeze = False
    frozen = dict(found=False, quad=None, H=None, rect=None, mask=None)

    fps = None
    t0 = time()
    frames = 0
    force_4x4 = not AUTO_GRID  # start based on AUTO_GRID flag

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, FRAME_SIZE)

        if freeze and frozen["rect"] is not None:
            found, quad, rectified, mask = True, frozen["quad"], frozen["rect"], None
        else:
            Smax = cv2.getTrackbarPos("white S max", "Maze Live")
            Vmin = cv2.getTrackbarPos("white V min", "Maze Live")
            inset = cv2.getTrackbarPos("inset% x100", "Maze Live") / 100.0
            found, quad, H, rectified, mask = find_maze_quad(frame, Smax, Vmin, inset)

        status = "maze: FOUND" if found else "maze: NOT FOUND"
        info = []

        # If frozen (or currently found), run the solver stack
        rect_show = None
        if found and rectified is not None:
            # grid size
            if force_4x4:
                N = 4
                info.append("Grid: forced 4x4")
            else:
                N = detect_grid_size(rectified)
                info.append(f"Grid: auto {N}x{N}")

            # build graph
            adj, walls_v, walls_h = build_graph_from_rectified(rectified, N)

            # detect red/green
            red_pt = find_dot_rectified(rectified, "red")
            green_pt = find_dot_rectified(rectified, "green")
            start_cell = end_cell = None
            if red_pt is not None:
                rc = pixel_to_cell(red_pt, N)
                end_cell = rc
                cv2.circle(rectified, red_pt, 10, (0, 0, 255), 2, cv2.LINE_AA)
                info.append(f"Red @ cell {rc}")
            if green_pt is not None:
                rc = pixel_to_cell(green_pt, N)
                start_cell = rc
                cv2.circle(rectified, green_pt, 10, (0, 255, 0), 2, cv2.LINE_AA)
                info.append(f"Green @ cell {rc}")

            # two paths (both directions) so you can pick at demo time
            path_g2r = (
                bfs(adj, start_cell, end_cell) if (start_cell and end_cell) else []
            )
            path_r2g = (
                bfs(adj, end_cell, start_cell) if (start_cell and end_cell) else []
            )

            if start_cell and end_cell:
                info.append(f"Path len g→r: {len(path_g2r)}  r→g: {len(path_r2g)}")
                # default show green→red if start is green; else red→green
                show_path = path_g2r if start_cell else path_r2g
                rect_show = draw_solution(rectified, show_path, N, start_cell, end_cell)
            else:
                rect_show = rectified.copy()
        else:
            rect_show = None

        vis = draw_main(
            frame, status, quad if found else None, rect_show, mask, fps, info
        )
        cv2.imshow("Maze Live", vis)

        frames += 1
        if frames % 15 == 0:
            now = time()
            fps = 15.0 / max(1e-6, now - t0)
            t0 = now

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord("f"), ord("F")) and found:
            freeze = True
            frozen.update(
                found=True, quad=quad.copy(), H=H, rect=rectified.copy(), mask=mask
            )
        elif key in (ord("r"), ord("R")):
            freeze = False
            frozen = dict(found=False, quad=None, H=None, rect=None, mask=None)
        elif key in (ord("g"), ord("G")):
            force_4x4 = not force_4x4

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

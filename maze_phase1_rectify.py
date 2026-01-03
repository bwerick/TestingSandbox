import cv2
import numpy as np
from pydobot.dobot import PTPMode as MODE_PTP
from pydobot import Dobot
import pydobot
import time

device = pydobot.Dobot(port="/dev/tty.usbmodem21301")
device.speed(120, 120)

CAM_INDEX = 0  # change if your camera is not 0
RECT_SIZE = 600  # size of the rectified square (pixels)


# ---------- path direction toggle ----------
# 0 = Green -> Red, 1 = Red -> Green
DIRECTION_MODE = 0

# will be updated every frame once we know the window size
TOGGLE_RECT = (0, 0, 0, 0)  # (x0, y0, x1, y1)


def on_mouse(event, x, y, flags, param):
    global DIRECTION_MODE, TOGGLE_RECT

    # Debug: log EVERY mouse event
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[MOUSE] Left click at ({x}, {y}), TOGGLE_RECT={TOGGLE_RECT}")

        x0, y0, x1, y1 = TOGGLE_RECT
        if x0 <= x <= x1 and y0 <= y <= y1:
            DIRECTION_MODE = 1 - DIRECTION_MODE
            print(
                "Toggled path direction:", "G → R" if DIRECTION_MODE == 0 else "R → G"
            )


# ------------ camera <-> robot calibration points ------------
CAMERA_PTS = np.array(
    [
        [184.0, 368.0],
        [214.0, 696.0],
        [440.0, 526.0],
        [665.0, 255.0],
        [843.0, 546.0],
        [1045.0, 272.0],
        [1053.0, 761.0],
    ],
    dtype=np.float32,
)

ROBOT_PTS = np.array(
    [
        [335.5, 113.8],
        [277.3, 109.9],
        [305.8, 70.7],
        [356.7, 31.6],
        [304.5, -3.9],
        [354.5, -40.6],
        [265.5, -40.5],
    ],
    dtype=np.float32,
)

H_cam2robot, _ = cv2.findHomography(CAMERA_PTS, ROBOT_PTS)


# ---------- small utilities ----------


def draw_text(img, txt, org, color=(255, 255, 255), scale=0.6):
    cv2.putText(
        img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 3, cv2.LINE_AA
    )
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def order_tl_tr_bl_br(pts):
    """
    Given 4 unordered points, return them as [TL, TR, BL, BR].
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)  # x + y
    d = np.diff(pts, axis=1).ravel()  # x - y

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, bl, br], np.float32)


def find_maze_quad(frame, white_S_max=103, white_V_min=158, inset_frac=0.01):
    """
    1) Threshold for 'white' in HSV to isolate the maze paper.
    2) Take largest contour -> minAreaRect -> 4-corner box.
    3) Optionally inset a bit so we avoid the paper margins.
    4) Return:
       - success flag
       - 4 corner points (TL, TR, BL, BR) in image coords
       - homography H (image -> rectified square)
       - rectified color image
       - mask (for debugging)
    """
    H0, W0 = frame.shape[:2]

    # HSV white-ish detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, white_V_min], dtype=np.uint8)  # low S, high V
    upper = np.array([179, white_S_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # clean mask: close gaps, remove specks
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None, None, mask

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.05 * W0 * H0:
        # too small to be the maze paper
        return False, None, None, None, mask

    # min-area rectangle around the contour
    rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect).astype(np.float32)
    quad = order_tl_tr_bl_br(box)

    # optional: inset the quad a little along its axes to avoid paper border
    tl, tr, bl, br = quad
    top_dir = tr - tl
    top_len = np.linalg.norm(top_dir) + 1e-6
    top_dir /= top_len

    left_dir = bl - tl
    left_len = np.linalg.norm(left_dir) + 1e-6
    left_dir /= left_len

    inset_t = inset_frac * top_len
    inset_l = inset_frac * left_len

    tl_i = tl + top_dir * inset_t + left_dir * inset_l
    tr_i = tr - top_dir * inset_t + left_dir * inset_l
    bl_i = bl + top_dir * inset_t - left_dir * inset_l
    br_i = br - top_dir * inset_t - left_dir * inset_l

    quad_inset = np.stack([tl_i, tr_i, bl_i, br_i]).astype(np.float32)

    # destination square for rectified maze
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


import numpy as np
import cv2

# ---------- 1D smoothing + period finder ----------


def smooth1d(x, k=11):
    """
    Simple 1D moving average smoother for projections.
    k is forced to be odd and at least 3.
    """
    k = max(3, int(k) | 1)
    ker = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), ker, mode="same")


def fundamental_period_from_projection(proj, min_period, max_period):
    """
    Given a 1D projection (e.g. sum of wall pixels per column),
    estimate the dominant spacing via autocorrelation.

    Returns an integer lag in [min_period, max_period], or None if invalid.
    """
    s = proj.astype(np.float32)
    s = s - s.mean()
    if np.allclose(s, 0):
        return None

    ac = np.correlate(s, s, mode="full")
    ac = ac[len(ac) // 2 :]  # keep non-negative lags

    lo = int(min_period)
    hi = min(len(ac) - 1, int(max_period))
    if hi <= lo + 3:
        return None

    ac_s = smooth1d(ac, k=21)
    lag_rel = np.argmax(ac_s[lo:hi])
    lag = lo + int(lag_rel)
    return lag


def detect_grid_size(rectified, N_min=3, N_max=10):
    """
    AUTO-DETECT N (N_min..N_max) from the rectified maze using
    periodicity of vertical/horizontal wall projections.

    Returns N (int) or None.
    """
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    # Binary image: walls dark, background light
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # walls = 1, background = 0
    walls = (bw < 128).astype(np.float32)

    H, W = walls.shape
    assert H == W, "Rectified maze should be square"
    size = float(H)  # e.g. 600

    # Projections: how much wall per column / row
    proj_x = walls.sum(axis=0)  # vertical grid lines → variation in x
    proj_y = walls.sum(axis=1)  # horizontal grid lines → variation in y

    # Expected line spacing range (in pixels)
    min_period = size / (N_max * 1.3)
    max_period = size / (N_min * 0.7)

    if max_period <= min_period + 3:
        print("[grid detect] degenerate period range")
        return None

    per_x = fundamental_period_from_projection(proj_x, min_period, max_period)
    per_y = fundamental_period_from_projection(proj_y, min_period, max_period)

    periods = [p for p in (per_x, per_y) if p is not None]
    if not periods:
        print("[grid detect] could not estimate period from projections")
        return None

    candidates = []
    for p in periods:
        if p <= 0:
            continue
        # Raw estimate
        N_est = size / p
        N_round = int(round(N_est))
        # Keep only candidates in the valid range
        if N_round < N_min or N_round > N_max:
            continue
        # Ideal period if the grid really were N_round x N_round
        ideal_p = size / N_round
        error = abs(p - ideal_p)
        candidates.append((error, N_round, p, ideal_p))

    if not candidates:
        print("[grid detect] no valid N candidates from periods:", periods)
        return None

    # Choose the N whose measured period is closest to its ideal period
    candidates.sort()  # sort by error ascending
    best_error, best_N, best_p, best_ideal = candidates[0]

    print(
        f"[grid detect] size={size:.1f}, periods={periods}, "
        f"candidates={[(c[1], c[0]) for c in candidates]}, "
        f"chosen N={best_N}, error={best_error:.1f}"
    )

    return best_N


def find_dot_rectified(rectified, color="red"):
    """
    Find the colored start/goal dot in the rectified maze.

    Returns (cx, cy) in rectified pixel coordinates, or None.
    """
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)

    if color == "red":
        # red wraps the hue wheel: two ranges
        m1 = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 120, 80), (179, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    else:  # "green"
        # tune if needed based on lighting / marker
        mask = cv2.inRange(hsv, (40, 80, 60), (85, 255, 255))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 30:  # ignore tiny specks
        return None

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


from collections import deque

# ---------- wall sampling + graph ----------


def sample_line_has_wall(bw, p0, p1, thickness=3, samples=40, frac=0.25):
    """
    Check if a line segment between p0 and p1 crosses a wall.
    bw: binary image with walls dark (0) and background light (255).
    We sample along the line, look in a small square around each point,
    and count how many samples are mostly dark.
    """
    h, w = bw.shape
    hits = 0
    for t in np.linspace(0, 1, samples):
        x = int(round(p0[0] * (1 - t) + p1[0] * t))
        y = int(round(p0[1] * (1 - t) + p1[1] * t))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        x0 = max(0, x - thickness)
        x1 = min(w - 1, x + thickness)
        y0 = max(0, y - thickness)
        y1 = min(h - 1, y + thickness)
        patch = bw[y0 : y1 + 1, x0 : x1 + 1]
        # walls are dark → mean < 128
        if np.mean(patch) < 128:
            hits += 1
    return hits >= frac * samples


def build_graph_from_rectified(rectified, N):
    """
    For an N×N maze, check boundaries between adjacent cells and
    determine which cells are connected.
    Returns:
      adj: adjacency dict
      walls_v: (N, N-1) bool array, True if wall between (r,c) and (r,c+1)
      walls_h: (N-1, N) bool array, True if wall between (r,c) and (r+1,c)
    """
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # IMPORTANT: do NOT invert; walls are dark (0), background light (255)

    H, W = bw.shape
    size = float(H)
    step = size / N
    pad = 0.20 * step  # stay away from corners

    adj = {(r, c): [] for r in range(N) for c in range(N)}
    walls_v = np.zeros((N, N - 1), dtype=bool)
    walls_h = np.zeros((N - 1, N), dtype=bool)

    # we'll sample 3 slightly shifted lines for each boundary
    # offsets are a fraction of cell size
    offset_frac = 0.08  # 8% of cell size
    offset_pix = offset_frac * step

    # vertical boundaries (between (r,c) and (r,c+1))
    for r in range(N):
        for c in range(N - 1):
            x_base = (c + 1) * step
            y0 = r * step + pad
            y1 = (r + 1) * step - pad
            if y1 <= y0:
                continue

            has_wall = False
            for dx in (-offset_pix, 0.0, +offset_pix):
                x = x_base + dx
                if sample_line_has_wall(bw, (x, y0), (x, y1)):
                    has_wall = True
                    break

            walls_v[r, c] = has_wall
            if not has_wall:
                adj[(r, c)].append((r, c + 1))
                adj[(r, c + 1)].append((r, c))

    # horizontal boundaries (between (r,c) and (r+1,c))
    for r in range(N - 1):
        y_base = (r + 1) * step
        for c in range(N):
            x0 = c * step + pad
            x1 = (c + 1) * step - pad
            if x1 <= x0:
                continue

            has_wall = False
            for dy in (-offset_pix, 0.0, +offset_pix):
                y = y_base + dy
                if sample_line_has_wall(bw, (x0, y), (x1, y)):
                    has_wall = True
                    break

            walls_h[r, c] = has_wall
            if not has_wall:
                adj[(r, c)].append((r + 1, c))
                adj[(r + 1, c)].append((r, c))

    return adj, walls_v, walls_h


# ---------- BFS + drawing ----------


def bfs(adj, start, goal):
    """
    Standard BFS shortest path on adjacency dict.
    Returns list of cells [start,...,goal] or [] if unreachable.
    """
    if start not in adj or goal not in adj:
        return []

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
    path.reverse()
    return path


def draw_solution(rect_vis, path, N, color=(0, 0, 255)):
    """
    Draw the BFS path as a polyline through cell centers on rect_vis.
    """
    if not path:
        return rect_vis

    H, W = rect_vis.shape[:2]
    step = float(H) / N

    pts = []
    for r, c in path:
        x = int((c + 0.5) * step)
        y = int((r + 0.5) * step)
        pts.append((x, y))

    pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(rect_vis, [pts_arr], False, color, 3, cv2.LINE_AA)
    return rect_vis


def cells_to_robot_path(path_cells, N, H_img2rect, H_cam2robot):
    """
    Convert BFS cell path into robot (X, Y) coordinates.

    path_cells   : list[(row, col)] from BFS
    N            : grid size (NxN)
    H_img2rect   : 3x3 homography, camera -> rectified
    H_cam2robot  : 3x3 homography, camera -> robot (from your calibration)

    returns: list[(X, Y)] in robot coordinates
    """
    if not path_cells or H_img2rect is None or H_cam2robot is None:
        return []

    # 1) cell centers in rectified coordinates
    step = RECT_SIZE / float(N)
    pts_rect = []
    for r, c in path_cells:
        x = (c + 0.5) * step  # center of cell in rectified image
        y = (r + 0.5) * step
        pts_rect.append([x, y])
    pts_rect = np.array(pts_rect, dtype=np.float32).reshape(-1, 1, 2)

    # 2) rectified -> camera (inverse of H_img2rect)
    H_rect2img = np.linalg.inv(H_img2rect)
    pts_cam = cv2.perspectiveTransform(pts_rect, H_rect2img)

    # 3) camera -> robot
    pts_robot = cv2.perspectiveTransform(pts_cam, H_cam2robot)

    # 4) flatten
    robot_path = []
    for p in pts_robot.reshape(-1, 2):
        X, Y = float(p[0]), float(p[1])
        robot_path.append((X, Y))

    return robot_path


def execute_robot_path(device, robot_path, z_height=-85, r=0.0, wait_each=True):
    """
    Send the robot along the given (X, Y) path at a fixed Z and orientation.
    """
    if not robot_path:
        print("No robot path to execute.")
        return

    print("Executing robot path with", len(robot_path), "points")

    # Optional: go to the first point
    for X, Y in robot_path:
        device.move_to(x=X - 29, y=Y + 5, z=z_height, r=r, wait=False)
        time.sleep(1)  # small delay between points
    device.move_to(x=240, y=0, z=145, r=r, wait=wait_each)


# ---------- main loop ----------
# --- DRAW THE DIRECTION TOGGLE BUTTON ON 'display' ---
def draw_direction_toggle(display):
    global TOGGLE_RECT, DIRECTION_MODE

    h, w = display.shape[:2]
    btn_w, btn_h = 190, 36

    x1 = w - 10
    x0 = x1 - btn_w
    y0 = 10
    y1 = y0 + btn_h

    # IMPORTANT: update the GLOBAL rect so the mouse callback sees it
    TOGGLE_RECT = (x0, y0, x1, y1)

    # button background
    cv2.rectangle(display, (x0, y0), (x1, y1), (40, 40, 40), -1)
    cv2.rectangle(display, (x0, y0), (x1, y1), (255, 255, 255), 1)

    # label
    label = "Mode: G → R" if DIRECTION_MODE == 0 else "Mode: R → G"
    cv2.putText(
        display,
        label,
        (x0 + 8, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    return display


def main():
    # ---------- stability tracking for auto-execute ----------
    global H_cam2robot  # if you defined it globally
    stable_sig = None  # last stable "maze state" signature
    stable_since = None  # time() when this signature started
    executed_for_sig = False  # whether we've already run the robot for this sig
    last_robot_path = None  # optional: keep for debugging
    global DIRECTION_MODE
    cap = cv2.VideoCapture(CAM_INDEX)
    device.move_to(x=240, y=0, z=145, r=0, wait=False)
    time.sleep(1)
    if not cap.isOpened():
        raise SystemExit("❌ Cannot open camera")

    # Optional: ask the camera what resolution it is giving us
    ok, test_frame = cap.read()
    if not ok:
        raise SystemExit("❌ Could not read from camera")
    print("Camera frame shape:", test_frame.shape)  # (h, w, 3)

    cv2.namedWindow("Maze Camera", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Maze Camera", on_mouse)
    # cv2.createTrackbar("white S max", "Maze Camera", 60, 120, lambda v: None)
    # cv2.createTrackbar("white V min", "Maze Camera", 180, 255, lambda v: None)
    # cv2.createTrackbar(
    #     "inset% x100", "Maze Camera", int(0.06 * 100), 20, lambda v: None
    # )

    cv2.namedWindow("Rectified Maze", cv2.WINDOW_NORMAL)
    cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)

    # Trackbars to tune white-paper segmentation
    cv2.createTrackbar("white S max", "Maze Camera", 103, 120, lambda v: None)
    cv2.createTrackbar("white V min", "Maze Camera", 150, 255, lambda v: None)
    cv2.createTrackbar(
        "inset % x100", "Maze Camera", int(0.01 * 100), 25, lambda v: None
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # DO NOT resize here – keep native aspect ratio
        # (If you later want to resize for display, keep aspect ratio.)

        # get parameters from trackbars
        Smax = cv2.getTrackbarPos("white S max", "Maze Camera")
        Vmin = cv2.getTrackbarPos("white V min", "Maze Camera")
        inset_frac = cv2.getTrackbarPos("inset % x100", "Maze Camera") / 100.0

        found, quad, H, rectified, mask = find_maze_quad(
            frame, white_S_max=Smax, white_V_min=Vmin, inset_frac=inset_frac
        )
        # -------- RECTIFIED VIEW: grid, dots, path --------
        if found and rectified is not None:

            N = detect_grid_size(rectified)

            # start from a copy so we can draw on it
            rect_vis = rectified.copy()
            Hrect, Wrect = rect_vis.shape[:2]
            size = float(Hrect)

            start_cell = None
            goal_cell = None

            if N is not None:
                draw_text(rect_vis, f"Grid: {N}x{N}", (10, 30), (0, 255, 0), 0.8)

                # ---- DOT DETECTION ----
                red_pt = find_dot_rectified(rectified, "red")
                green_pt = find_dot_rectified(rectified, "green")

                step = size / N

                def pt_to_cell(pt):
                    x, y = pt
                    c = int(np.clip(x / step, 0, N - 1))
                    r = int(np.clip(y / step, 0, N - 1))
                    return (r, c)

                green_cell = None
                red_cell = None

                # detect green dot
                if green_pt is not None:
                    cv2.circle(rect_vis, green_pt, 10, (0, 255, 0), 2, cv2.LINE_AA)
                    green_cell = pt_to_cell(green_pt)
                    draw_text(
                        rect_vis,
                        f"G {green_cell}",
                        (green_pt[0] + 10, green_pt[1]),
                        (0, 180, 0),
                        0.7,
                    )
                    print("Green dot cell:", green_cell)

                # detect red dot
                if red_pt is not None:
                    cv2.circle(rect_vis, red_pt, 10, (0, 0, 255), 2, cv2.LINE_AA)
                    red_cell = pt_to_cell(red_pt)
                    draw_text(
                        rect_vis,
                        f"R {red_cell}",
                        (red_pt[0] + 10, red_pt[1]),
                        (0, 0, 180),
                        0.7,
                    )
                    print("Red dot cell:", red_cell)

                # ---- PICK START/GOAL BASED ON TOGGLE ----
                if green_cell is not None and red_cell is not None:
                    if DIRECTION_MODE == 0:
                        # Mode: G → R
                        start_cell = green_cell
                        goal_cell = red_cell
                    else:
                        # Mode: R → G
                        start_cell = red_cell
                        goal_cell = green_cell

                    print(
                        "DIRECTION_MODE:",
                        DIRECTION_MODE,
                        "start_cell:",
                        start_cell,
                        "goal_cell:",
                        goal_cell,
                    )

                # ---- GRAPH + BFS ----
                if start_cell is not None and goal_cell is not None:
                    adj, walls_v, walls_h = build_graph_from_rectified(rectified, N)
                    path = bfs(adj, start_cell, goal_cell)

                    if path:
                        rect_vis = draw_solution(rect_vis, path, N)
                        draw_text(
                            rect_vis,
                            f"path len: {len(path)}",
                            (10, 60),
                            (255, 0, 0),
                            0.7,
                        )
                        robot_path = cells_to_robot_path(path, N, H, H_cam2robot)
                        # in your code, after you build robot_path
                        test_X, test_Y = robot_path[0]  # first cell in path
                        print("Test robot point:", test_X, test_Y)

                        # ----- STABILITY CHECK & AUTO-EXECUTE AFTER 5s -----
                        now = time.time()

                        # Build a compact signature of the current maze state
                        # (grid size, direction, start/goal cells, path length)
                        sig = (N, DIRECTION_MODE, start_cell, goal_cell, len(path))

                        if sig == stable_sig:
                            # Same maze state as before
                            if stable_since is None:
                                stable_since = now
                            # If we haven't executed yet and it's been stable long enough:
                            elif not executed_for_sig and (now - stable_since) >= 5.0:
                                print("Maze stable for 5s — executing robot path.")
                                execute_robot_path(
                                    device, robot_path, z_height=-40, r=0.0
                                )
                                executed_for_sig = True
                        else:
                            # Maze state changed (new dots, new N, or mode toggled)
                            stable_sig = sig
                            stable_since = now
                            executed_for_sig = False

                        last_robot_path = robot_path  # optional: for debugging

                    else:
                        draw_text(rect_vis, "no path found", (10, 60), (0, 0, 255), 0.7)
                        stable_sig = None
                        stable_since = None
                        executed_for_sig = False

            else:
                draw_text(rect_vis, "Grid: ?", (10, 30), (0, 0, 255), 0.8)

            cv2.imshow("Rectified Maze", rect_vis)

        else:
            # if no maze found, show a blank rectified view
            blank = np.zeros((RECT_SIZE, RECT_SIZE, 3), np.uint8)
            cv2.imshow("Rectified Maze", blank)

        display = frame.copy()
        if found and quad is not None:
            q = quad.astype(int)
            cv2.polylines(display, [q], True, (0, 255, 0), 2, cv2.LINE_AA)
            for i, (x, y) in enumerate(q):
                cv2.circle(display, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
                draw_text(display, f"P{i}", (x + 6, y - 6), (0, 255, 255), 0.5)
            draw_text(display, "maze: FOUND", (12, 30), (40, 220, 40), 0.8)
        else:
            draw_text(display, "maze: NOT FOUND", (12, 30), (0, 0, 255), 0.8)

        display = draw_direction_toggle(display)
        cv2.imshow("Maze Camera", display)

        # Show the white mask for debugging the segmentation
        if mask is not None:
            cv2.imshow("White Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # ESC or q to quit
            break
    device.move_to(x=240, y=0, z=145, r=0, wait=False)
    time.sleep(1)
    device.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

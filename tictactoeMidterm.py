import cv2
import time
from ultralytics import YOLO
from pydobot.dobot import PTPMode as MODE_PTP
from pydobot import Dobot
import pydobot
import sympy as sp
import numpy as np

MODEL_PATH = "runs/detect/ttt_xo_synth/weights/best.pt"  # your trained YOLO
CONF_THRESH = 0.45
H_FILE = "h.npy"  # saved homography file

# device = pydobot.Dobot(port="/dev/tty.usbmodem21201")
# device.speed(80, 80)
allr = 0.0
penHover = 10
penDown = -11

# home coordinates
homex = 240
homey = 0
homez = 150

# camera view coordinates
cameraX = 250
cameraY = 0
cameraZ = 34

# top right corner of the board
cellHeight = 33
cellWidth = 33
starting_position = (cameraX + 20, cameraY - 1.5 * cellHeight)
print(starting_position)


def pointsCircle(pointQty, x, y, radius):
    angle = 360 / pointQty
    points = []
    for i in range(pointQty):
        x1 = x + radius * sp.cos(sp.rad(angle * i))
        y1 = y + radius * sp.sin(sp.rad(angle * i))
        points.append((x1, y1))
    return points


def drawCircle(x, y, radius):
    points = pointsCircle(8, x, y, radius)
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


def pointsCross(x, y, cellHeight, cellWidth):
    xtopLeft = (x - 0.8 * cellWidth / 2, y + 0.8 * cellHeight / 2)
    xtopRight = (x + 0.8 * cellWidth / 2, y + 0.8 * cellHeight / 2)
    xbottomLeft = (x - 0.8 * cellWidth / 2, y - 0.8 * cellHeight / 2)
    xbottomRight = (x + 0.8 * cellWidth / 2, y - 0.8 * cellHeight / 2)
    return [xtopLeft, xtopRight, xbottomLeft, xbottomRight]


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


def cellCenters(starting_position, cellHeight, cellWidth):
    x, y = starting_position
    cell1 = (x + cellWidth / 2, y + cellHeight / 2)
    cell2 = (x + cellWidth / 2, y + 3 * cellHeight / 2)
    cell3 = (x + cellWidth / 2, y + 5 * cellHeight / 2)
    cell4 = (x + 3 * cellWidth / 2, y + cellHeight / 2)
    cell5 = (x + 3 * cellWidth / 2, y + 3 * cellHeight / 2)
    cell6 = (x + 3 * cellWidth / 2, y + 5 * cellHeight / 2)
    cell7 = (x + 5 * cellWidth / 2, y + cellHeight / 2)
    cell8 = (x + 5 * cellWidth / 2, y + 3 * cellHeight / 2)
    cell9 = (x + 5 * cellWidth / 2, y + 5 * cellHeight / 2)
    return [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]


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
    device.move_to(
        x=starting_position[0], y=starting_position[1], z=penHover, r=allr, wait=False
    )
    time.sleep(2)


# ------------------------------------------------------------


def pix_to_robot(u, v):
    p = np.array([u, v, 1.0], dtype=np.float32)
    q = H @ p
    return float(q[0] / q[2]), float(q[1] / q[2])  # (Xr, Yr)


def nearest_cell(xr, yr):

    d = np.linalg.norm(cells_robot - np.array([xr, yr]), axis=1)
    return int(np.argmin(d))  # 0..8


def pretty(grid):
    rows = [grid[0:3], grid[3:6], grid[6:9]]
    return "\n".join(" | ".join(c or " ") for c in rows)


def choose_bot_move(grid):
    # simplest strategy: center -> first empty
    order = [4, 0, 2, 6, 8, 1, 3, 5, 7]
    for i in order:
        if grid[i] == "":
            return i
    return None


# # move home
# device.move_to(x=homex, y=homey, z=homez, r=allr, wait=False)
# time.sleep(2)
# # Test draw board
# drawBoard(starting_position, cellHeight, cellWidth)
# # Test draw circle and cross
# drawCircle(
#     starting_position[0] + cellWidth,
#     starting_position[1] - cellHeight,
#     0.8 * (cellHeight / 2),
# )
# drawCross(
#     starting_position[0] + 2 * cellWidth,
#     starting_position[1] - cellHeight,
#     cellHeight,
#     cellWidth,
# )

# if you can win, place where you can win
# if opponent can win, block opponent
# if center is free, take center
# if center is not free, take corner
# if corner is not free, take side

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # cols
    (0, 4, 8),
    (2, 4, 6),  # diags
]
CORNERS = [0, 2, 6, 8]
SIDES = [1, 3, 5, 7]
CENTER = 4

IMAGE_PATH = "calib_board.jpg"  # your snapshot
MODE = "cells9"  # "corners" or "cells9"

############################################
# START
############################################
# drawBoard(starting_position, cellHeight, cellWidth)

# device.move_to(x=cameraX, y=cameraY, z=cameraZ, r=allr, wait=False)
time.sleep(2)
print("calibrating!")
input()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press S to Save", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):  # press S to save the image
        cv2.imwrite("calib_board.jpg", frame)
        print("Saved image: calib_board.jpg")
        break
    elif key == ord("q"):  # press Q to quit without saving
        break

cap.release()
cv2.destroyAllWindows()
# calibrte
IMAGE_PATH = "calib_board.jpg"  # your snapshot
MODE = "cells9"  # "corners" or "cells9"


robot_pts_corners = np.array(
    [
        [starting_position[0], starting_position[1]],  # bottom right in camera view
        [
            starting_position[0] + 3 * cellWidth,
            starting_position[1],
        ],  # top-right in camera view
        [
            starting_position[0],
            starting_position[1] + 3 * cellHeight,
        ],  # bottom-left in camera view
        [
            starting_position[0] + 3 * cellWidth,
            starting_position[1] + 3 * cellHeight,
        ],  # top-left in camera view
    ],
    dtype=np.float32,
)
centers = cellCenters(starting_position, cellHeight, cellWidth)

robot_pts_cells9 = np.array(
    [
        [centers[0][0], centers[0][1]],
        [centers[1][0], centers[1][1]],
        [centers[2][0], centers[2][1]],
        [centers[3][0], centers[3][1]],
        [centers[4][0], centers[4][1]],
        [centers[5][0], centers[5][1]],
        [centers[6][0], centers[6][1]],
        [centers[7][0], centers[7][1]],
        [centers[8][0], centers[8][1]],
    ],
    dtype=np.float32,
)
# mouse click calibration
clicked = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append([x, y])
        print(f"clicked: {(x, y)}")


img = cv2.imread(IMAGE_PATH)
if img is None:
    raise SystemExit(f"Could not load {IMAGE_PATH}")

cv2.namedWindow("Click in order")
cv2.setMouseCallback("Click in order", on_mouse)

if MODE == "corners":
    print("Click 4 points on the IMAGE in this order:")
    print("  1) top-left corner, 2) top-right, 3) bottom-right, 4) bottom-left")
    need = 4
    robot_pts = robot_pts_corners
elif MODE == "cells9":
    print("Click the 9 CELL CENTERS on the IMAGE in row-major order:")
    print("  r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2")
    need = 9
    robot_pts = robot_pts_cells9
else:
    raise SystemExit("MODE must be 'corners' or 'cells9'.")


while True:
    vis = img.copy()
    for i, (x, y) in enumerate(clicked):
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(
            vis, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    cv2.imshow("Click in order", vis)
    key = cv2.waitKey(20)
    if key == 27:  # ESC
        cv2.destroyAllWindows()
        raise SystemExit("Canceled.")
    if len(clicked) >= need:
        break

cv2.destroyAllWindows()

image_pts = np.array(clicked[:need], dtype=np.float32)

H, mask = cv2.findHomography(
    image_pts, robot_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0
)
if H is None:
    raise SystemExit("findHomography failed. Check point order and values.")
print("Homography H =\n", H)
np.save("H.npy", H)
print("Saved H.npy")

ones = np.ones((image_pts.shape[0], 1), np.float32)
pix_h = np.hstack([image_pts, ones])  # N x 3
rob_proj = (H @ pix_h.T).T  # N x 3
rob_proj = rob_proj[:, :2] / rob_proj[:, 2:3]

# err = np.linalg.norm(rob_proj - robot_pts, axis=1)
# print("Per-point error (robot units):", np.round(err, 3))
# print("Mean error:", float(err.mean()), "Max error:", float(err.max()))
H = np.load("H.npy")  # from the calibration step

# known cell centers in ROBOT coordinates (row-major TL->BR)


# if you already calibrated, use H to convert image (u,v) -> robot (Xr,Yr)
H = np.load("H.npy")


def pix_to_robot(u, v):
    p = np.array([u, v, 1.0], dtype=np.float32)
    q = H @ p
    return float(q[0] / q[2]), float(q[1] / q[2])


def nearest_cell(xr, yr):
    d = np.linalg.norm(robot_pts_cells9 - np.array([xr, yr]), axis=1)
    return int(np.argmin(d)), float(d.min())


# current board state we maintain (what you called the “mental” board)
# "", "X", or "O" per cell
state = [""] * 9


def update_state_from_detections(yolo_results, conf_thr=0.5, snap_radius=30.0):
    """
    - yolo_results: Ultralytics result for one frame
    - conf_thr: ignore weak detections
    - snap_radius: (robot units) max distance allowed to snap a detection to a cell
    """
    best_conf = [-1.0] * 9
    labels = [""] * 9  # temp view from this frame

    for b in yolo_results.boxes:
        c = float(b.conf[0])
        if c < conf_thr:
            continue
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        xr, yr = pix_to_robot(u, v)

        cid, dist = nearest_cell(xr, yr)
        if dist > snap_radius:
            # too far from any known center → ignore (likely false positive)
            continue

        cls = int(b.cls[0])  # 0=X, 1=O
        lab = "X" if cls == 0 else "O"

        # keep the highest-confidence label per cell
        if c > best_conf[cid]:
            best_conf[cid] = c
            labels[cid] = lab

    # now fold this frame’s labels into our persistent state
    # (only promote definitive, simple rule: if we see X or O, mark occupied)
    for i in range(9):
        if labels[i] in ("X", "O"):
            state[i] = labels[i]


def choose_robot_move(state):
    """Return index of an empty cell (simple policy)."""
    # favorite order: center → corners → edges
    order = [4, 0, 2, 6, 8, 1, 3, 5, 7]
    for i in order:
        if state[i] == "":
            return i
    return None


# # loop
# model = YOLO("/Users/erickduarte/git/TestingSandbox/runs/ttt_xo_synth/weights/best.pt")
# cap = cv2.VideoCapture(0)

# turn = "player"
# K = 3  # debounce frames for accepting a new X
# candidate = None
# stable = 0

# while True:
#     ok, frame = cap.read()
#     if not ok:
#         break

#     res = model.predict(frame, conf=0.5, verbose=False)[0]
#     # build a momentary copy before committing
#     snapshot = state.copy()
#     update_state_from_detections(res, conf_thr=0.5, snap_radius=30.0)
#     # state now reflects what the model currently sees

#     if turn == "player":
#         # accept exactly one new X (empty -> X), stable for K frames
#         diffs = [i for i, (a, b) in enumerate(zip(snapshot, state)) if a != b]
#         x_adds = [i for i in diffs if snapshot[i] == "" and state[i] == "X"]
#         if len(x_adds) == 1:
#             if candidate != x_adds[0]:
#                 candidate = x_adds[0]
#                 stable = 1
#             else:
#                 stable += 1
#             if stable >= K:
#                 # lock in player's move (already in state)
#                 turn = "robot"
#                 candidate = None
#                 stable = 0
#         else:
#             candidate = None
#             stable = 0

#     elif turn == "robot":
#         move = choose_robot_move(state)
#         if move is None:
#             print("No moves left.")
#             break
#         # command robot here (you already have cells_robot[move] for (Xr,Yr))
#         xr, yr = robot_pts_cells9[move]
#         device.move_to(x=xr, y=yr, z=penHover, r=allr, wait=False)
#         time.sleep(2)
#         drawCircle(xr, yr, 0.8 * (cellHeight / 2))
#         print(f"Robot: draw O at cell {move} ({xr:.1f},{yr:.1f})")
#         # after drawing, immediately mark occupied so we don't choose it again
#         state[move] = "O"
#         turn = "player"

#     # (optional) show image
#     cv2.imshow("XO", res.plot())
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

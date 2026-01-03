import cv2
import numpy as np
import glob

# ========= BOARD PARAMETERS ==========
GRID_COLS = 8  # markers along X
GRID_ROWS = 5  # markers along Y
MARKER_LENGTH = 0.024  # meters (24 mm)
MARKER_SEPARATION = 0.006  # meters (30 - 24 = 6 mm gap)

DICT = cv2.aruco.DICT_4X4_50
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

board = cv2.aruco.GridBoard(
    (GRID_COLS, GRID_ROWS), MARKER_LENGTH, MARKER_SEPARATION, aruco_dict
)

# ========= LOAD IMAGES ==========
images = sorted(glob.glob("calib_imgs/*.jpg"))
print("Found images:", len(images))

all_corners = []  # flat list of all marker corners
all_ids = []  # flat list of all marker IDs
counter = []  # markers-per-image
img_size = None

for fname in images:
    print(f"\nProcessing: {fname}")
    img = cv2.imread(fname)
    if img is None:
        print("  Could not read image, skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]  # (w, h)

    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)

    if ids is None or len(ids) == 0:
        print("  No markers detected, skipping.")
        continue

    n_markers = len(ids)
    print("  Markers detected:", n_markers)

    # corners is already a list of np arrays for each marker: (4,1,2)
    all_corners.extend(corners)
    # flatten ids into plain Python ints
    all_ids.extend(ids.flatten().tolist())
    counter.append(n_markers)

print("\nTotal markers overall:", len(all_corners))
print("Images with markers:", len(counter))

if len(all_corners) == 0:
    print("ERROR: No markers detected in any frame.")
    raise SystemExit

# Convert ids and counter to the formats OpenCV expects
ids_array = np.array(all_ids, dtype=np.int32).reshape(-1, 1)
counter_array = np.array(counter, dtype=np.int32)

# ========= CALIBRATION ==========
ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
    all_corners, ids_array, counter_array, board, img_size, None, None
)

print("\n==== ARUCO GRID 8x5 CALIBRATION RESULTS ====")
print("Reprojection error:", ret)
print("\nCamera matrix (K):\n", K)
print("\nDistortion coefficients:\n", dist.ravel())

np.savez("aruco_grid_8x5_calibration.npz", K=K, dist=dist, reproj_error=ret)

print("\nSaved calibration to aruco_grid_8x5_calibration.npz")

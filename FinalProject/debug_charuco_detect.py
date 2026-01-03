import cv2
import numpy as np
import glob

# ---------- Your board settings ----------
CHARUCO_COLS = 8
CHARUCO_ROWS = 5
SQUARE_SIZE = 0.03
MARKER_SIZE = 0.024

# Try this first (matches what we used for the board script)
DICT = cv2.aruco.DICT_4X4_50

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard(
    (CHARUCO_COLS, CHARUCO_ROWS), SQUARE_SIZE, MARKER_SIZE, aruco_dict
)

images = sorted(glob.glob("calib_imgs/*.jpg"))
print("Found images:", len(images))

if not images:
    print("No images found in calib_imgs/")
    exit(1)

for fname in images:
    print("\n=== File:", fname, "===")
    img = cv2.imread(fname)
    if img is None:
        print("  Could not read image.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)

    num_markers = 0 if ids is None else len(ids)
    print("  Markers detected:", num_markers)

    # Draw markers (even if few)
    debug_img = img.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)

    # Try Charuco interpolation
    if ids is not None and len(ids) >= 1:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=board
        )

        if retval and charuco_corners is not None and len(charuco_corners) > 0:
            print("  Charuco corners:", len(charuco_corners))
            cv2.aruco.drawDetectedCornersCharuco(
                debug_img, charuco_corners, charuco_ids
            )
        else:
            print("  No Charuco corners found")
    else:
        print("  Not enough markers to attempt Charuco")

    # HUD text
    text = f"{fname.split('/')[-1]} | markers: {num_markers}"
    cv2.putText(
        debug_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    cv2.imshow("Charuco Debug", debug_img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()

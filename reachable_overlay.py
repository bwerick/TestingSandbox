import cv2
import numpy as np

# ======================================================
# === 1.  FILL IN YOUR OWN MEASURED VALUES BELOW ===
# ======================================================

# Four pixel coordinates from your camera frame (in 640x480)
# Replace these with the pixel locations of the paper corners you drew.
pix = np.array(
    [
        [0, 0],  # top-left  pixel coordinate
        [640, 0],  # top-right pixel coordinate
        [0, 480],  # bottom-left  pixel coordinate
        [640, 480],  # bottom-right pixel coordinate
    ],
    np.float32,
)

# Corresponding robot XY coordinates in millimeters
rob = np.array(
    [
        [330.4, 111.7],  # top-left
        [336.3, -104.8],  # top-right
        [162.1, 111.6],  # bottom-left
        [156.8, -105.2],  # bottom-right
    ],
    np.float32,
)

# Robot reach limits (tune these)
r_min = 188.0  # mm, closest distance to base
r_max = 330.0  # mm, measure actual farthest comfortable reach
x_max = 340.0  # mm, measure actual +X limit once known

# Choose image source
use_live_camera = True
camera_index = 0
frame_path = "camera_frame.jpg"  # used if not live
# ======================================================


# === 2. Compute homography (pixels → robot XY) ===
H, _ = cv2.findHomography(pix, rob)
Hinv = np.linalg.inv(H)


def pix_to_robot_xy(uv):
    """Convert pixel coords (N×2) → robot XY (mm) using H."""
    pts = np.hstack([uv.astype(np.float32), np.ones((len(uv), 1), np.float32)])
    w = (H @ pts.T).T
    return w[:, :2] / w[:, 2:3]


def robot_to_pix(xy):
    """Convert robot XY (mm) → pixel coords (N×2) using H⁻¹."""
    pts = np.hstack([xy.astype(np.float32), np.ones((len(xy), 1), np.float32)])
    w = (Hinv @ pts.T).T
    return w[:, :2] / w[:, 2:3]


# === 3. Generate reachability mask ===
def make_unreachable_mask(w=640, h=480, step=4):
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    uv = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    pts = np.hstack([uv, np.ones((len(uv), 1), np.float32)])
    out = (H @ pts.T).T
    XY = out[:, :2] / out[:, 2:3]

    r = np.linalg.norm(XY, axis=1)
    unreachable = (r < r_min) | (r > r_max) | (XY[:, 0] > x_max)
    mask = np.zeros((h, w), np.uint8)
    mask[ys.ravel(), xs.ravel()] = unreachable.astype(np.uint8) * 255
    mask = cv2.blur(mask, (15, 15))
    return mask


# === 4. Draw overlay ===
def overlay_unreachable(frame, mask):
    overlay = frame.copy()
    # tint unreachable areas red
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(
        np.uint8
    )
    blended = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    return blended


# === 5. Display once or in live loop ===
def main():
    mask = make_unreachable_mask()

    if use_live_camera:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            blended = overlay_unreachable(frame, mask)
            cv2.imshow("Reachability Overlay", blended)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
        cap.release()
    else:
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (640, 480))
        blended = overlay_unreachable(frame, mask)
        cv2.imshow("Reachability Overlay", blended)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

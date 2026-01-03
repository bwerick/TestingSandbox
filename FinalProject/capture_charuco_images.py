import cv2
import os
import time

# ---------- Settings ----------
CAMERA_INDEX = 0  # change if needed
OUTPUT_DIR = "calib_imgs"  # folder to save snapshots
BASE_NAME = "charuco"  # filename prefix
# ------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera index {CAMERA_INDEX}")
    exit(1)

# Optional: set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

img_count = 0

print("Press 's' to save a frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # ---- Make a copy for display only ----
    display = frame.copy()
    text = f"[q] quit  |  [s] save  |  saved: {img_count}"
    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Live Camera - Charuco Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        # Save the *clean* frame (no overlay)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{BASE_NAME}_{timestamp}_{img_count:02d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved (clean): {filepath}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()

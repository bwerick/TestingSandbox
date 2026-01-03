import cv2
import time
import os

# Choose your preferred resolution
WIDTH, HEIGHT = 1280, 1024  # change to 1280,1024 if that's your native
SAVE_DIR = "captures"

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

# Try to set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Confirm what the camera actually accepted
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera initialized at {int(actual_w)}x{int(actual_h)}")

print("\nPress 'S' to save a frame, 'Q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Camera View (press S to save, Q to quit)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"capture_{ts}.png")
        cv2.imwrite(filename, frame)
        print(f"✅ Saved {filename}")

    elif key == ord("q"):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

# block_detection.py

import cv2
import numpy as np

# ------------ Color detection helpers ------------


def get_color_masks(hsv_frame):
    """
    Given an HSV frame, return a dict of color_name -> binary mask.
    Adjust the ranges to match your actual blocks if needed.
    """

    masks = {}

    # RED is special: it wraps around the hue range, so we combine two intervals.
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 80])
    upper_red2 = np.array([179, 255, 255])

    mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    masks["red"] = cv2.bitwise_or(mask_red1, mask_red2)

    # BLUE
    lower_blue = np.array([100, 120, 80])
    upper_blue = np.array([135, 255, 255])
    masks["blue"] = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # GREEN
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])
    masks["green"] = cv2.inRange(hsv_frame, lower_green, upper_green)

    # YELLOW
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    masks["yellow"] = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    return masks


def find_colored_blocks(frame_bgr):
    """
    Detect colored 'blocks' in the frame.
    Returns a list of detections:
        [
          {
            "color": "red",
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "contour": contour
          },
          ...
        ]
    """

    detections = []

    # Convert to HSV
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Get masks for each color
    masks = get_color_masks(hsv)

    for color, mask in masks.items():
        # Clean up the mask a bit
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # ignore tiny noise blobs; adjust threshold
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            detections.append(
                {
                    "color": color,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "contour": cnt,
                }
            )

    return detections


# ------------ Main test loop ------------


def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("❌ Failed to open camera. Try another camera_index.")
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened at {width} x {height}")

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        detections = find_colored_blocks(frame)

        from scene_state import Block, SceneState

        # Build scene state
        blocks = []
        for i, det in enumerate(detections):
            x, y, w, h = det["bbox"]
            cx, cy = det["center"]
            color_name = det["color"]

            blocks.append(
                Block(id=i, color=color_name, x_px=cx, y_px=cy, width_px=w, height_px=h)
            )

        scene = SceneState(blocks)

        # Draw detections
        for det in detections:
            x, y, w, h = det["bbox"]
            cx, cy = det["center"]
            color_name = det["color"]

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

            # Put label text
            label = f"{color_name}"
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # (Optional) show how many of each were found
        counts = {}
        for det in detections:
            counts[det["color"]] = counts.get(det["color"], 0) + 1

        if counts:
            text = " ".join([f"{c}:{n}" for c, n in counts.items()])
            cv2.putText(
                frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        # Debug print
        if scene.blocks:
            print([f"{b.color}@({b.x_px},{b.y_px})" for b in scene.blocks])

        cv2.imshow("Colored Block Detection (press q to exit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()

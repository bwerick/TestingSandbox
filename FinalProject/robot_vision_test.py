# robot_vision_test.py

import time
import cv2

from robot_control import connect_robot, move_to, move_to_safe_pose
from block_detection import find_colored_blocks
from calibration import image_to_robot


# Adjust this if needed: a safe hover height above the table
HOVER_Z = 0.0  # same as your home Z for now, just to hover


def choose_leftmost_block(detections):
    """
    Pick the leftmost block in image space (smallest x of center).
    detections: list of dicts from find_colored_blocks(...)
    """
    if not detections:
        return None

    return min(detections, key=lambda d: d["center"][0])


def main(camera_index=0):
    # 1) Connect to robot and move to vision/home pose
    print("Connecting to robot...")
    device = connect_robot()
    print("Connected.")

    print("Moving to vision/home pose...")
    move_to_safe_pose(device)
    time.sleep(1.0)

    # 2) Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    print("Camera opened. Make sure the robot is in the HOME/VISION pose.")
    print("Press 'm' to select a block and move above it, 'q' to quit.")

    chosen_block = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        detections = find_colored_blocks(frame)

        # Draw detections so you can see what it will pick
        for det in detections:
            x, y, w, h = det["bbox"]
            cx, cy = det["center"]
            color_name = det["color"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
            cv2.putText(
                frame,
                color_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Robot Vision Test (m=move, q=quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("m"):
            if not detections:
                print("No blocks detected; cannot move.")
                continue

            chosen_block = choose_leftmost_block(detections)
            cx, cy = chosen_block["center"]
            color_name = chosen_block["color"]
            print(f"Chosen block: {color_name} at pixel ({cx},{cy})")

            # Convert to robot coords
            try:
                x_mm, y_mm = image_to_robot(cx, cy)
            except Exception as e:
                print(f"❌ Homography failed for ({cx},{cy}): {e}")
                continue

            print(f"Mapped to robot XY: ({x_mm:.1f}, {y_mm:.1f}), Z={HOVER_Z:.1f}")

            # 3) Move robot above that block
            print("Moving robot above block...")
            move_to(device, x_mm, y_mm, HOVER_Z, r=0.0, wait=True)
            print("Arrived above block (hover).")

            # (Optional) pause so you can observe
            time.sleep(1.0)

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

    try:
        device.close()
    except AttributeError:
        pass

    print("Robot connection closed.")
    print("Done.")


if __name__ == "__main__":
    main()

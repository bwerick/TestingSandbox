# pick_place_test.py

import time
import cv2

from robot_control import connect_robot, move_to_safe_pose
from block_detection import find_colored_blocks
from calibration import image_to_robot
from skills import pick_block_at, place_block_at


# Choose a safe drop location on the table (adjust if needed)
DROP_X = 280.0
DROP_Y = -60.0


def choose_leftmost_block(detections):
    if not detections:
        return None
    return min(detections, key=lambda d: d["center"][0])


def main(camera_index=0):
    # 1) Connect and go to vision pose
    print("Connecting to robot...")
    device = connect_robot()
    print("Connected.")

    print("Moving to home/vision pose...")
    move_to_safe_pose(device)
    time.sleep(1.0)

    # 2) Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    print("Camera opened. Make sure blocks are in view.")
    print("Press 'p' to pick & place leftmost block, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        detections = find_colored_blocks(frame)

        # Draw detections
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

        cv2.imshow("Pick & Place Test (p=pick/place, q=quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("p"):
            if not detections:
                print("No blocks detected; cannot pick.")
                continue

            chosen = choose_leftmost_block(detections)
            cx, cy = chosen["center"]
            color_name = chosen["color"]
            print(f"Chosen block: {color_name} at pixel ({cx},{cy})")

            # Convert to robot XY
            x_mm, y_mm = image_to_robot(cx, cy)
            print(f"  -> Mapped to robot XY=({x_mm:.1f}, {y_mm:.1f})")

            # Move robot to home/vision pose first
            print("Returning to home/vision pose before action...")
            move_to_safe_pose(device)
            time.sleep(0.5)

            # 3) Pick the block
            pick_block_at(device, x_mm, y_mm, r_deg=0.0)

            # 4) Place at fixed DROP location on table
            print(f"Placing at DROP location: ({DROP_X:.1f}, {DROP_Y:.1f})")
            place_block_at(device, DROP_X, DROP_Y, on_table=True, r_deg=0.0)

            # 5) Return to home pose
            print("Returning to home/vision pose...")
            move_to_safe_pose(device)
            time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

    try:
        device.close()
    except AttributeError:
        pass

    print("Done.")


if __name__ == "__main__":
    main()

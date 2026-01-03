# language_main.py

import cv2
import os
from datetime import datetime

from robot_control import connect_robot, move_to_safe_pose
from block_detection import find_colored_blocks
from calibration import image_to_robot
from scene_state import Block, SceneState
from llm_interpreter import interpret_prompt
from agent_executor import execute_task


def build_scene_from_frame(frame) -> SceneState:
    detections = find_colored_blocks(frame)
    blocks = []

    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        cx, cy = det["center"]
        color = det["color"]

        x_mm, y_mm = image_to_robot(cx, cy)

        blocks.append(
            Block(
                id=i,
                color=color,
                x_px=cx,
                y_px=cy,
                width_px=w,
                height_px=h,
                x_mm=x_mm,
                y_mm=y_mm,
            )
        )

    return SceneState(blocks), detections


def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on the frame for visualization.
    """
    vis = frame.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        cx, cy = det["center"]
        color_name = det["color"]

        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)
        cv2.putText(
            vis,
            color_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return vis


def save_scene_image(frame, out_dir="scenes"):
    """
    Save the current frame as a timestamped PNG in out_dir.
    Returns the filepath.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scene_{ts}.png"
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, frame)
    print(f"📸 Scene screenshot saved to: {path}")
    return path


def main(camera_index=0):
    device = connect_robot()
    print("Connected to robot.")

    cap = cv2.VideoCapture(camera_index)
    # Force resolution (example: 1280x1024)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Optional: verify
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened at {w} x {h}")
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    try:
        while True:
            print("\n=== New command cycle ===")
            move_to_safe_pose(device)

            # --- Live preview loop ---
            print("Showing live camera. Press 'c' to capture scene, 'q' to quit.")
            captured_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame.")
                    break

                # Run detection just for visualization
                _, detections = build_scene_from_frame(frame)
                vis = draw_detections(frame, detections)

                cv2.imshow("Robot View (press 'c' to capture, 'q' to quit)", vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    # Quit entire program
                    return
                elif key == ord("c"):
                    captured_frame = frame.copy()
                    print("✅ Scene captured.")
                    # Optional: also save what you see (with detections)
                    save_scene_image(draw_detections(captured_frame, detections))
                    break

            if captured_frame is None:
                print("No frame captured; exiting.")
                break

            # --- Build scene from the captured frame ---
            scene, detections = build_scene_from_frame(captured_frame)
            scene_summary = {"blocks": scene.summary()}
            print("Scene summary:", scene_summary)

            # --- Ask user for instruction ---
            prompt = input(
                "\nEnter a natural language instruction (or 'q' to quit):\n> "
            )
            if prompt.strip().lower() in ("q", "quit", "exit"):
                break

            # --- Call LLM to get JSON task ---
            task = interpret_prompt(prompt, scene_summary)
            print("\nTask JSON:", task)

            # --- Execute task on robot ---
            execute_task(task, scene, device)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")
        try:
            device.close()
        except Exception:
            pass
        print("Robot connection closed.")


if __name__ == "__main__":
    main()

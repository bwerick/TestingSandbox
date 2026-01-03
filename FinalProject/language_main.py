# language_main.py

import cv2

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

    return SceneState(blocks)


def main(camera_index=0):
    device = connect_robot()
    print("Connected to robot.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    try:
        while True:
            # 1) Go to vision pose
            print("\n=== New command cycle ===")
            move_to_safe_pose(device)

            # 2) Grab a frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame.")
                break

            # 3) Build scene
            scene = build_scene_from_frame(frame)
            scene_summary = {"blocks": scene.summary()}
            print("Scene summary:", scene_summary)

            # 4) Ask user for instruction
            prompt = input(
                "\nEnter a natural language instruction (or 'q' to quit):\n> "
            )
            if prompt.strip().lower() in ("q", "quit", "exit"):
                break

            # 5) Get task JSON from LLM
            task = interpret_prompt(prompt, scene_summary)
            print("\nTask JSON:", task)

            # 6) Execute task on robot
            execute_task(task, scene, device)

    finally:
        cap.release()
        print("Camera closed.")
        try:
            device.close()
        except Exception:
            pass
        print("Robot connection closed.")


if __name__ == "__main__":
    main()

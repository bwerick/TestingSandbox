# llm_agent_test.py

import cv2

from robot_control import connect_robot, move_to_safe_pose
from block_detection import find_colored_blocks
from calibration import image_to_robot
from scene_state import Block, SceneState
from llm_interpreter import interpret_prompt


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
    # 1) Connect & go to vision pose
    device = connect_robot()
    move_to_safe_pose(device)

    # 2) Grab one frame
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to grab frame.")
        return

    # 3) Build scene
    scene = build_scene_from_frame(frame)
    scene_summary = {"blocks": scene.summary()}
    print("Scene summary:", scene_summary)

    # 4) Ask user for an instruction
    prompt = input("\nEnter a natural language instruction:\n> ")

    # 5) Get task from LLM
    task = interpret_prompt(prompt, scene_summary)

    print("\nLLM returned task JSON:")
    print(task)


if __name__ == "__main__":
    main()

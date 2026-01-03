import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0 faster, 2 more accurate
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Pose Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# vision_test.py

import cv2


def main(camera_index=0):
    """
    Opens the camera and displays frames in real time.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("❌ Failed to open camera. Try camera_index=1 or another number.")
        return

    # Try to print the default resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened at {width} x {height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        cv2.imshow("Camera View (press q to exit)", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == "__main__":
    main()

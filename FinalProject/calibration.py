# calibration.py

import numpy as np
import cv2

# ------------ camera <-> robot calibration points ------------

CAMERA_PTS = np.array(
    [
        [184.0, 368.0],
        [214.0, 696.0],
        [440.0, 526.0],
        [665.0, 255.0],
        [843.0, 546.0],
        [1045.0, 272.0],
        [1053.0, 761.0],
    ],
    dtype=np.float32,
)

ROBOT_PTS = np.array(
    [
        [335.5, 113.8],
        [277.3, 109.9],
        [305.8, 70.7],
        [356.7, 31.6],
        [304.5, -3.9],
        [354.5, -40.6],
        [265.5, -40.5],
    ],
    dtype=np.float32,
)

# Compute homography from camera pixels to robot XY plane
H_cam2robot, _ = cv2.findHomography(CAMERA_PTS, ROBOT_PTS)
print(H_cam2robot.shape)

#  3x3 affine matrix for pixel -> robot (X,Y)
H_cam2robot = np.array(
    [
        [6.00650232e-03, -4.84214952e-01, 3.85653329e02],
        [-4.69079919e-01, 3.74996755e-03, 1.605349575e02],
    ],
    dtype=np.float64,
)
print(H_cam2robot.shape)


def image_to_robot(u: float, v: float) -> tuple[float, float]:
    """
    Map image pixel coordinates (u, v) in the HOME/VISION camera pose
    to robot XY coordinates in mm on the table plane.

    Uses the homography H_cam2robot.
    """
    pt_cam = np.array([[u], [v], [1.0]], dtype=np.float64)
    print(pt_cam.shape)
    pt_robot = H_cam2robot @ pt_cam

    # w = pt_robot[2]
    w = 1.0
    if abs(w) < 1e-6:
        raise ValueError(
            "Homogeneous scale is too small, homography may be degenerate."
        )

    x = pt_robot[0] / w
    y = pt_robot[1] / w
    return float(x), float(y)


def _test_homography():
    """
    Simple reprojection test: apply image_to_robot to the CAMERA_PTS
    and compare to ROBOT_PTS. Prints errors in mm.
    """
    print("Testing homography reprojection error:")
    total_err = 0.0

    for i, (uv, xy_true) in enumerate(zip(CAMERA_PTS, ROBOT_PTS)):
        u, v = uv
        x_true, y_true = xy_true
        x_est, y_est = image_to_robot(u, v)

        err = np.sqrt((x_est - x_true) ** 2 + (y_est - y_true) ** 2)
        total_err += err

        print(
            f"Point {i}: pixel=({u:.1f},{v:.1f}) "
            f"robot_true=({x_true:.2f},{y_true:.2f}) "
            f"robot_est=({x_est:.2f},{y_est:.2f}) "
            f"error={err:.2f} mm"
        )

    avg_err = total_err / len(CAMERA_PTS)
    print(f"\nAverage reprojection error: {avg_err:.2f} mm")


if __name__ == "__main__":
    _test_homography()

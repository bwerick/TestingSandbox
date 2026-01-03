# robot_control.py

import time
from pydobot import Dobot
from pydobot.dobot import PTPMode as MODE_PTP


# ---------- Connection & basic control ----------


def connect_robot(
    port: str = "/dev/tty.usbmodem21301", velocity: int = 120, acceleration: int = 120
) -> Dobot:
    """
    Connect to the Dobot and set a default speed.
    Adjust the port if your device shows up differently.
    """
    device = Dobot(port=port)
    # Set joint/linear speeds (units are Dobot-specific)
    device.speed(velocity, acceleration)
    return device


def move_to(
    device: Dobot, x: float, y: float, z: float, r: float = 0.0, wait: bool = True
):
    """
    Wrapper around Dobot's move_to for readability.
    """
    device.move_to(x=x, y=y, z=z, r=r, wait=wait)


def set_suction(device: Dobot, on: bool):
    """
    Turn the suction gripper on or off.
    """
    device.suck(on)


# ---------- High-level poses ----------


def move_to_safe_pose(device: Dobot):
    """
    Move to a known safe 'home' position above the workspace.
    This uses the pose you showed in your example code.
    """
    # You can tweak these if you find a safer or more central pose
    HOME_X = 240
    HOME_Y = 0
    HOME_Z = 150
    HOME_R = 0.0

    print(f"Moving to safe pose: x={HOME_X}, y={HOME_Y}, z={HOME_Z}, r={HOME_R}")
    move_to(device, HOME_X, HOME_Y, HOME_Z, HOME_R, wait=True)


# ---------- Test routine ----------


def test_move():
    """
    Simple test:
    1. Connects to the robot.
    2. Moves to a safe pose.
    3. Jogs a bit in X and comes back.
    4. Toggles suction on and off.
    """

    print("Connecting to Dobot...")
    device = connect_robot()
    print("Connected to Dobot.")

    # 1) Move to safe pose
    move_to_safe_pose(device)
    time.sleep(1)

    # 2) Small jog in +X direction and back
    print("Jogging a bit in +X...")
    JOG_DISTANCE = 20  # mm

    # Get current pose if your API supports it, otherwise just offset X
    # Here we'll just offset from the known safe pose.
    x_jog = 240 + JOG_DISTANCE
    y_jog = 0
    z_jog = 145
    r_jog = 0.0

    move_to(device, x_jog, y_jog, z_jog, r_jog, wait=True)
    time.sleep(1)

    print("Moving back to safe pose...")
    move_to_safe_pose(device)
    time.sleep(1)

    # 3) Test suction on/off
    print("Turning suction ON...")
    set_suction(device, True)
    time.sleep(1)

    print("Turning suction OFF...")
    set_suction(device, False)
    time.sleep(1)

    # 4) (Optional) close connection
    try:
        device.close()
    except AttributeError:
        # Some versions may not have close(); it's okay to ignore
        pass

    print("Test sequence complete.")


if __name__ == "__main__":
    test_move()

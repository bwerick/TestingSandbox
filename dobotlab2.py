from pydobot.dobot import PTPMode as MODE_PTP
from pydobot import Dobot
import pydobot


device = pydobot.Dobot(port="/dev/tty.usbmodem2101")  # For Windows, replace COMXX with the actual port, e.g., COM3
# device = pydobot.Robot(port="/dev/ttyUSB0") # For Linux
# device = pydobot.Robot(port="/dev/tty.usbserial-XXXX")


#device.home()  # Home the robot to the origin position


(x, y, z, r, j1, j2, j3, j4) = device.pose()  # Get the current position and joint angles



pose = [x, y, z, r]
joint = [j1, j2, j3, j4] 
print(f"pose: {pose}, j: {joint}")
# position, joint = pose.position, pose.joints
print(pose)
device.speed(80, 80)


# Preferred way to move
#green block
device.move_to(x=285.0, y=11.0, z=7, r=0.0, wait=False)  # Move to position (x=250, y=0, z=50) with r=0
# device.move_to(x=255.0, y=-80.0, z=-43, r=0.0, wait=False)
# device.suck(True)
# device.move_to(x=285.0, y=11.0, z=7, r=0.0, wait=False)
# device.move_to(x=256.2, y=43.2, z=-40, r=0.0, wait=False)
# device.suck(False)
# device.move_to(x=285.0, y=11.0, z=7, r=0.0, wait=False)

#yellow block
# device.move_to(x=255.0, y=-22.5, z=-44, r=0.0, wait=False)
# device.suck(True)
# device.move_to(x=285.0, y=11.0, z=7, r=0.0, wait=False)
# device.move_to(x=255.5, y=101.2, z=-44, r=0.0, wait=False)
# device.suck(False)
# device.move_to(x=255.5, y=101.2, z=-39, r=0.0, wait=False)
# device.move_to(x=285.0, y=11.0, z=7, r=0.0, wait=False)

#blue block
# device.move_to(x=314.0, y=-22.5, z=-44, r=0.0, wait=False)
# device.suck(True)
# device.move_to(x=285.0, y=11.0, z=7, r=0.0, wait=False)
# device.move_to(x=315.0, y=102.2, z=-44, r=0.0, wait=False)
# device.suck(False)
# device.move_to(x=315.0, y=103.2, z=-39, r=0.0, wait=False)
# device.move_to(x=285.0, y=11.0, z=7, r=0.0, wait=False)

#red block
device.move_to(x=315.5, y=-85.5, z=-44, r=0.0, wait=False)
device.suck(True)
device.move_to(x=285.0, y=11.0, z=7, r=0.0, wait=False)
device.move_to(x=319.5, y=44.2, z=-40, r=0.0, wait=False)
device.suck(False)
device.move_to(x=319.5, y=44.2, z=-39, r=0.0, wait=False)
device.move_to(x=285.0, y=11.0, z=7, r=0.0, wait=False)

  # Move to joint angles (j1=0, j2=0, j3=0, j4=0)
device.close()
# # Control gripper and suction cup
# device.grip(True)  # Close the gripper
# device.grip(False)  # Open the gripper
# device.suck(True)  # Turn on the suction cup
# device.suck(False)  # Turn off the suction cup
# device.close()  # Very Important

# Advanced way to move
# device.move_to(mode=int(MODE_PTP.JUMP_XYZ), x=230, y=30.0, z=20, r=0.0)  # Lift and move to position (x=250, y=0, z=50) with r=0
# device.move_to(mode=int(MODE_PTP.JUMP_ANGLE), x=200, y=30.0, z=20, r=0.0)  # Lift and move to joint angles (j1=0, j2=0, j3=0, j4=0)


# # Move by relative amounts
# device.move_to(mode=int(MODE_PTP.MOVJ_INC), x=-10, y=-10.0, z=-10, r=0.0)  # Move relative to current angles by (dj1=-10, dj2=-10, dj3=-10, dj4=0)
# device.move_to(mode=int(MODE_PTP.MOVJ_XYZ_INC), x=10, y=10.0, z=30, r=0.0)  # Move relative to current position by (dx=10, dy=10, dz=10) with dr=0

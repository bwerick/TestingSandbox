from pydobot.dobot import PTPMode as MODE_PTP
from pydobot import Dobot
import pydobot
import time

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


allr =0

homez = 156
homey = 5.4
homex = 250.7

middlex = 214
middley = 1
middlez = 25
#Block positions
#red block start
redxs = 250.2
redys = -78.8
redzs = -17.9
redza = 10
#red block end
redxe = 247.0
redye = 36.8
redze = -15.3
#blue block start
bluexs = 249.0
blueys = -19.7
bluezs = -16.4
blueza = 10
#blue block end
bluexe = 244.6
blueye = 95.9
blueze = -13.9
#yellow block start
yellowxs = 187.0
yellowys = -21.9
yellowzs = -14
yellowza = 10
#yellow block end
yellowxe = 183.2
yellowye = 93.9
yellowze = -15.4
#green block start
greenxs = 189.8
greenys = -78.7
greenzs = -18.3
greenza = -10
#green block end
greenxe = 185.0
greenye = 34.8
greenze = -13.8

# Preferred way to move
#start position
device.move_to(x=homex, y=homey, z=homez, r=allr, wait=False)
cycletimestart = time.time()
#move to middle
device.move_to(x=middlex, y=middley, z=middlez, r=allr, wait=False)

#green block 
device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#move to above block
device.move_to(x=greenxs, y=greenys, z=greenzs, r=0.0, wait=False) #move to block
device.grip(True)#grip block
device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#move to above drop position
device.move_to(x=greenxe, y=greenye, z=greenze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.grip(False)#drop block

device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#yellow block
device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#move to above block
device.move_to(x=yellowxs, y=yellowys, z=yellowzs, r=0.0, wait=False)#move down to block
device.grip(True)#grip block
device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#move to above drop position
device.move_to(x=yellowxe, y=yellowye, z=yellowze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.grip(False)#drop block

device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#red block
device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#move to above block
device.move_to(x=redxs, y=redys, z=redzs, r=0.0, wait=False)#move down to block
device.grip(True)#grip block
device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#move to above drop position
device.move_to(x=redxe, y=redye, z=redze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.grip(False)#drop block

device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#blue block
device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#move to above block
device.move_to(x=bluexs, y=blueys, z=bluezs, r=0.0, wait=False)#move down to block
device.grip(True)#grip block
device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#move to above drop position
device.move_to(x=bluexe, y=blueye, z=blueze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.grip(False)#drop block

device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#return green block
device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#move to above block
device.move_to(x=greenxe, y=greenye, z=greenze, r=0.0, wait=False) #move to block
device.grip(True)#grip block
device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#move to above drop position
device.move_to(x=greenxs, y=greenys, z=greenzs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.grip(False)#drop block

device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#return yellow block
device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#move to above block
device.move_to(x=yellowxe, y=yellowye, z=yellowze, r=0.0, wait=False) #move to block
device.grip(True)#grip block
device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#move to above drop position
device.move_to(x=yellowxs, y=yellowys, z=yellowzs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.grip(False)#drop block

device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#return red block
device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#move to above block
device.move_to(x=redxe, y=redye, z=redze, r=0.0, wait=False) #move to block
device.grip(True)#grip block
device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#move to above drop position
device.move_to(x=redxs, y=redys, z=redzs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.grip(False)#drop block

device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#return blue block
device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#move to above block
device.move_to(x=bluexe, y=blueye, z=blueze, r=0.0, wait=False) #move to block
device.grip(True)#grip block
device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#move to above drop position
device.move_to(x=bluexs, y=blueys, z=bluezs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.grip(False)#drop block

device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#move back to home
device.move_to(x=homex, y=homey, z=homez, r=allr, wait=False) #move to home position
# Calculate cycle time
cycletimeend = time.time()
cycletime = cycletimeend - cycletimestart
print(f"Cycle time: {cycletime:.2f} seconds")
device.close()  # Close the connection to the robot

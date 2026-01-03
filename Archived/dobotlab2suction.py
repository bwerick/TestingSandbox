from pydobot.dobot import PTPMode as MODE_PTP
from pydobot import Dobot
import pydobot
import time

cycletimestart = time.time()

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


allr =0.0

homez = 7
homey = 11
homex = 285

middlex = 285
middley = 11
middlez = 7
#Block positions
#red block start
redxs = 315.5
redys = -85.5
redzs = -44

redza = -40
#red block end
redxe = 315.5
redye = -85.5
redze = -44

#blue block start
bluexs = 314.0
blueys = -22.5
bluezs = -44
blueza = -40
#blue block end
bluexe = 315.0
blueye = 102.2
blueze = -44
#yellow block start
yellowxs = 255.0
yellowys = -22.5
yellowzs = -44
yellowza = -40
#yellow block end
yellowxe = 255.5
yellowye = 101.2
yellowze = -44
#green block start
greenxs = 256.2
greenys = 43.2
greenzs = -44
greenza = -40
#green block end
greenxe = 256.2
greenye = 43.2
greenze = -40

# Preferred way to move
#start position
device.move_to(x=homex, y=homey, z=homez, r=allr, wait=False)
cycletimestart = time.time()
#move to middle
device.move_to(x=middlex, y=middley, z=middlez, r=allr, wait=False)

#green block 
device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#move to above block
device.move_to(x=greenxs, y=greenys, z=greenzs, r=0.0, wait=False) #move to block
device.suck(True)#suck block
device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#move to above drop position
device.move_to(x=greenxe, y=greenye, z=greenze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle

#yellow block
device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#move to above block
device.move_to(x=yellowxs, y=yellowys, z=yellowzs, r=0.0, wait=False)#move down to block
device.suck(True)#suck block
device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#move to above drop position
device.move_to(x=yellowxe, y=yellowye, z=yellowze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle

#red block
device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#move to above block
device.move_to(x=redxs, y=redys, z=redzs, r=0.0, wait=False)#move down to block
device.suck(True)#suck block
device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#move to above drop position
device.move_to(x=redxe, y=redye, z=redze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle

#blue block
device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#move to above block
device.move_to(x=bluexs, y=blueys, z=bluezs, r=0.0, wait=False)#move down to block
device.suck(True)#suck block
device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#move to above drop position
device.move_to(x=bluexe, y=blueye, z=blueze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle

#return green block
device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#move to above block
device.move_to(x=greenxe, y=greenye, z=greenze, r=0.0, wait=False) #move to block
device.suck(True)#suck block
device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#move to above drop position
device.move_to(x=greenxs, y=greenys, z=greenzs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle

#return yellow block
device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#move to above block
device.move_to(x=yellowxe, y=yellowye, z=yellowze, r=0.0, wait=False) #move to block
device.suck(True)#suck block
device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#move to above drop position
device.move_to(x=yellowxs, y=yellowys, z=yellowzs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle

#return red block
device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#move to above block
device.move_to(x=redxe, y=redye, z=redze, r=0.0, wait=False) #move to block
device.suck(True)#suck block
device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#move to above drop position
device.move_to(x=redxs, y=redys, z=redzs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle

#return blue block
device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#move to above block
device.move_to(x=bluexe, y=blueye, z=blueze, r=0.0, wait=False) #move to block
device.suck(True)#suck block
device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#move to above drop position
device.move_to(x=bluexs, y=blueys, z=bluezs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle

#move back to home
device.move_to(x=homex, y=homey, z=homez, r=allr, wait=False) #move to home position

# Calculate cycle time
cycletimeend = time.time()
cycletime = cycletimeend - cycletimestart
print(f"Cycle time: {cycletime:.2f} seconds")
device.close()  # Close the connection to the robot

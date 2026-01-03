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
redxs = 297.1
redys = -93.7
redzs = -43.1
redza = -39
#red block end
redxe = 297.8
redye = 21.5
redze = -43.0
#blue block start
bluexs = 296.6
blueys = -34.9
bluezs = -44.0
blueza = -39
#blue block end
bluexe = 299.9
blueye = 80.0
blueze = -44
#yellow block start
yellowxs = 237.7
yellowys = -34.1
yellowzs = -44
yellowza = -39
#yellow block end
yellowxe = 239.5
yellowye = 83.5
yellowze = -44
#green block start
greenxs = 239.8
greenys = -92.6
greenzs = -43.4
greenza = -39
#green block end
greenxe = 239.0
greenye = 23.5
greenze = -44.0

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
time.sleep(2)
device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#move to above drop position
device.move_to(x=greenxe, y=greenye, z=greenze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#yellow block
device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#move to above block
device.move_to(x=yellowxs, y=yellowys, z=yellowzs, r=0.0, wait=False)#move down to block
device.suck(True)#suck block
device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#move to above drop position
device.move_to(x=yellowxe, y=yellowye, z=yellowze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#red block
device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#move to above block
device.move_to(x=redxs, y=redys, z=redzs, r=0.0, wait=False)#move down to block
device.suck(True)#suck block
device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#move to above drop position
device.move_to(x=redxe, y=redye, z=redze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#blue block
device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#move to above block
device.move_to(x=bluexs, y=blueys, z=bluezs, r=0.0, wait=False)#move down to block
device.suck(True)#suck block
device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#move to above drop position
device.move_to(x=bluexe, y=blueye, z=blueze, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#return green block
device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#move to above block
device.move_to(x=greenxe, y=greenye, z=greenze, r=0.0, wait=False) #move to block
device.suck(True)#suck block
device.move_to(x=greenxe, y=greenye, z=greenza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#move to above drop position
device.move_to(x=greenxs, y=greenys, z=greenzs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=greenxs, y=greenys, z=greenza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#return yellow block
device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#move to above block
device.move_to(x=yellowxe, y=yellowye, z=yellowze, r=0.0, wait=False) #move to block
device.suck(True)#suck block
device.move_to(x=yellowxe, y=yellowye, z=yellowza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#move to above drop position
device.move_to(x=yellowxs, y=yellowys, z=yellowzs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=yellowxs, y=yellowys, z=yellowza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#return red block
device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#move to above block
time.sleep(.5)
device.move_to(x=redxe, y=redye, z=redze, r=0.0, wait=False) #move to block
device.suck(True)#suck block
device.move_to(x=redxe, y=redye, z=redza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#move to above drop position
time.sleep(.5)
device.move_to(x=redxs, y=redys, z=redzs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

device.move_to(x=redxs, y=redys, z=redza, r=0.0, wait=False)#lift up
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)

#return blue block
device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#move to above block
time.sleep(.5)
device.move_to(x=bluexe, y=blueye, z=blueze, r=0.0, wait=False) #move to block
device.suck(True)#suck block
device.move_to(x=bluexe, y=blueye, z=blueza, r=0.0, wait=False)#lift block
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)#move to middle
time.sleep(2)
device.move_to(x=bluexs, y=blueys, z=blueza, r=0.0, wait=False)#move to above drop position
time.sleep(.5)
device.move_to(x=bluexs, y=blueys, z=bluezs, r=0.0, wait=False)#move to drop position
time.sleep(2)  # Pause for 2 seconds
device.suck(False)#drop block

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

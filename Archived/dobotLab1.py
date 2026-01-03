from pydobot import Dobot
import time
from serial.tools import list_ports
import pydobot

available_ports = list_ports.comports()
print(f"available ports: {[x.device for x in available_ports]}")
port = available_ports[0].device
device = pydobot.Dobot(port="/dev/tty.usbmodem21201", verbose=True)
(x, y, z, r, j1, j2, j3, j4) = device.pose()
print(f"x:{x} y:{y} z:{z} j1:{j1} j2:{j2} j3:{j3} j4:{j4}")


# set home
homex = 285
homey = 11
homez = 7

# set middle
middlex = 285
middley = 11
middlez = 7
z = 52.5

####Letter C
# point A
pointAx = 315.5
pointAy = -7
pointAz = -z
# point B
pointBx = 315.5
pointBy = -27
pointBz = -z
# point C
pointCx = 335.5
pointCy = -27
pointCz = -z
# point D
pointDx = 335.45
pointDy = -7
pointDz = -z

# Letter I
# point E
pointEx = 315.5
pointEy = 3
pointEz = -z
# point F
pointFx = 335.5
pointFy = 3
pointFz = -z

# Letter M
# point G
pointGx = 335.5
pointGy = 13
pointGz = -z
# point H
pointHx = 315.5
pointHy = 13
pointHz = -z
# point I
pointIx = 335.5
pointIy = 23
pointIz = -z
# point J
pointJx = 315.5
pointJy = 33
pointJz = -z
# point K
pointKx = 335.5
pointKy = 33
pointKz = -z

# move home
device.move_to(x=homex, y=homey, z=homez, r=0.0, wait=False)
time.sleep(2)
# move to middle
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)
time.sleep(2)


# move to point A
device.move_to(
    x=pointAx, y=pointAy, z=pointAz, r=0.0, wait=False
)  # move to above point A
time.sleep(2)
# LineAB
device.move_to(x=pointBx, y=pointBy, z=pointBz, r=0.0, wait=False)  # move to point B
time.sleep(2)
# LineBC
device.move_to(x=pointCx, y=pointCy, z=pointCz, r=0.0, wait=False)  # move to point C
time.sleep(2)
# LineCD
device.move_to(x=pointDx, y=pointDy, z=pointDz, r=0.0, wait=False)  # move to point D
time.sleep(2)
# move to middle
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)
time.sleep(2)
# move to point E
device.move_to(x=pointEx, y=pointEy, z=pointEz, r=0.0, wait=False)
time.sleep(2)  # move to above point E
# LineEF
device.move_to(x=pointFx, y=pointFy, z=pointFz, r=0.0, wait=False)
time.sleep(2)  # move to point F
# move to middle
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)
time.sleep(2)
# move to point G
device.move_to(x=pointGx, y=pointGy, z=pointGz, r=0.0, wait=False)
time.sleep(2)  # move to above point G
# LineGH
device.move_to(x=pointHx, y=pointHy, z=pointHz, r=0.0, wait=False)
time.sleep(2)  # move to point H
# LineHI
device.move_to(x=pointIx, y=pointIy, z=pointIz, r=0.0, wait=False)
time.sleep(2)  # move to point I
# LineIJ
device.move_to(x=pointJx, y=pointJy, z=pointJz, r=0.0, wait=False)
time.sleep(2)  # move to point J
# LineJK
device.move_to(x=pointKx, y=pointKy, z=pointKz, r=0.0, wait=False)  # move to point K
time.sleep(2)
# move to middle
device.move_to(x=middlex, y=middley, z=middlez, r=0.0, wait=False)
time.sleep(2)
# move home
device.move_to(x=homex, y=homey, z=homez, r=0.0, wait=False)
time.sleep(2)
# close device
device.close()

from pydobot import Dobot

from serial.tools import list_ports
import pydobot

available_ports = list_ports.comports()
print(f"available ports: {[x.device for x in available_ports]}")

port = available_ports[0].device
device = pydobot.Dobot(port="/dev/tty.usbmodem21301", verbose=True)
device.move_to(x=240, y=0, z=150, r=0, wait=False)
(x, y, z, r, j1, j2, j3, j4) = device.pose()
print(f"x:{x} y:{y} z:{z} j1:{j1} j2:{j2} j3:{j3} j4:{j4}")
# device.suck(False)

# device.move_to(x, y, z, r, wait=True)  # we wait until this movement is done before continuing
# device.close()

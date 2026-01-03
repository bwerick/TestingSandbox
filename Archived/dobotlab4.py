from pydobot import Dobot
import time
from serial.tools import list_ports
import pydobot
import numpy as np
import math

available_ports = list_ports.comports()
print(f"available ports: {[x.device for x in available_ports]}")
port = available_ports[0].device
device = pydobot.Dobot(port="/dev/tty.usbmodem21201", verbose=True)
(x, y, z, r, j1, j2, j3, j4) = device.pose()
print(f"x:{x} y:{y} z:{z} j1:{j1} j2:{j2} j3:{j3} j4:{j4}")

theta1 = 0 * 180 / math.pi
theta2 = 0 * 180 / math.pi
theta3 = 0 * 180 / math.pi
theta4 = 0 * 180 / math.pi
theta5 = 0 * 180 / math.pi
a1 = 51
a2 = 150
a2y = 25
a3 = 150
a3x = 50
a4x = 40
a5y = 76

H01 = np.array(
    [math.cos(theta1), 0, -math.sin(theta1), 0],
    [math.sin(theta1), 0, math.cos(theta1), 0],
    [0, -1, 0, a1],
    [0, 0, 0, 1],
)
H12 = np.array(
    [math.sin(theta2), math.cos(theta2), 0, a2 * math.sin(theta2)],
    [-math.cos(theta2), math.sin(theta2), 0, -a2 * math.cos(theta2)],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
)
H12a = np.array([1, 0, 0, a2y], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
H12b = np.array([0, -1, 0, 0], [1, 0, 0, a3x], [0, 0, 1, 0], [0, 0, 0, 1])
H23 = np.array(
    [math.cos(theta3 - theta2), math.sin(theta3 - theta2), 0, a3 * math.cos(theta3)],
    [-math.sin(theta3 - theta2), math.cos(theta3 - theta2), 0, a3 * math.sin(theta3)],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
)
H34 = np.array(
    [math.cos(theta4), 0, -math.sin(theta4), 0],
    [math.sin(theta4), 0, math.cos(theta4), 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
)
H45 = np.array(
    [math.cos(theta5), -math.sin(theta5), 0, 0],
    [-math.cos(theta2), math.sin(theta2), 0, 0],
    [0, 0, 1, -a5y],
    [0, 0, 0, 1],
)


H05 = H01 @ H12 @ H12a @ H12b @ H23 @ H34 @ H45
print(H05)

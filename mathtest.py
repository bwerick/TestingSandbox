import math
import numpy as np

np.set_printoptions(precision=2, suppress=True)

theta1 = math.radians(90)
theta2 = math.radians(20)
theta3 = math.radians(30)
theta4 = math.radians(0) - theta3 - theta2
theta5 = math.radians(0)
a1 = 51
a2 = 150
a2y = 25
a3 = 150
a3x = 50
a4x = 40
a5y = 76

H01 = np.array(
    [
        [math.cos(theta1), 0, -math.sin(theta1), 0],
        [math.sin(theta1), 0, math.cos(theta1), 0],
        [0, -1, 0, a1],
        [0, 0, 0, 1],
    ]
)
print(H01)
print()
H12 = np.array(
    [
        [math.sin(theta2), math.cos(theta2), 0, a2 * math.sin(theta2)],
        [-math.cos(theta2), math.sin(theta2), 0, -a2 * math.cos(theta2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
print(H12)
print()
H12a = np.array([[1, 0, 0, a2y], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
print(H12a)
print()
H12b = np.array([[0, -1, 0, 0], [1, 0, 0, a3x], [0, 0, 1, 0], [0, 0, 0, 1]])
print(H12b)
print()
H23 = np.array(
    [
        [
            math.cos(theta3 - theta2),
            math.sin(theta3 - theta2),
            0,
            a3 * math.cos(theta3),
        ],
        [
            -math.sin(theta3 - theta2),
            math.cos(theta3 - theta2),
            0,
            a3 * math.sin(theta3),
        ],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
print(H23)
print()
H34 = np.array(
    [
        [math.cos(theta4), 0, math.sin(theta4), a4x],
        [math.sin(theta4), 0, -math.cos(theta4), 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)
print(H34)
print()
H45 = np.array(
    [
        [math.cos(theta5), -math.sin(theta5), 0, 0],
        [math.sin(theta5), math.cos(theta5), 0, 0],
        [0, 0, 1, -a5y],
        [0, 0, 0, 1],
    ]
)
print(H45)
print()
H02 = H01 @ H12 @ H12a @ H12b
print(H02)
print()
H03 = H02 @ H23
print(H03)
print()
H04 = H03 @ H34
print(H04)
print()
H05 = H01 @ H12 @ H12a @ H12b @ H23 @ H34 @ H45
print(H05)

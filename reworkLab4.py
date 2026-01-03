import math
import numpy as np
import sympy as sp
from sympy.plotting import plot3d
from sympy.geometry import Point3D, Line3D
from sympy.plotting import plot3d_parametric_line
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.set_printoptions(precision=2, suppress=True)
theta = sp.symbols("theta")
rotationMatrix = sp.Matrix(
    [
        [sp.cos(theta), -sp.sin(theta), 0],
        [sp.sin(theta), sp.cos(theta), 0],
        [0, 0, 1],
    ]
)

proj01 = sp.Matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
proj12 = sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
proj23 = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
proj34 = sp.Matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
proj45 = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

rproj01 = rotationMatrix @ proj01
sp.pprint(rproj01)
rproj12 = rotationMatrix @ proj12
sp.pprint(rproj12)
rproj23 = rotationMatrix @ proj23
sp.pprint(rproj23)
rproj34 = rotationMatrix @ proj34
sp.pprint(rproj34)
rproj45 = rotationMatrix @ proj45
sp.pprint(rproj45)


a1 = 51
a2 = 175
a3 = 200
a4 = 40
a5 = 76
theta1 = 90 * sp.pi / 180
theta2 = 20 * sp.pi / 180
theta3 = (30 * sp.pi / 180) - theta2
theta4 = 0 * sp.pi / 180 - theta3
theta5 = 0 * sp.pi / 180
H01 = sp.Matrix(
    [
        [sp.cos(theta1), 0, -sp.sin(theta1), 0],
        [sp.sin(theta1), 0, sp.cos(theta1), 0],
        [0, -1, 0, a1],
        [0, 0, 0, 1],
    ]
)
sp.pprint(H01.evalf())
x1, y1, z1 = H01[0, 3], H01[1, 3], H01[2, 3]


H12 = sp.Matrix(
    [
        [sp.sin(theta2), sp.cos(theta2), 0, a2 * sp.sin(theta2)],
        [-sp.cos(theta2), sp.sin(theta2), 0, -a2 * sp.cos(theta2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
sp.pprint(H12.evalf())
x2, y2, z2 = H12[0, 3], H12[1, 3], H12[2, 3]
H23 = sp.Matrix(
    [
        [
            -sp.sin(theta3),
            -sp.cos(theta3),
            0,
            -a3 * sp.sin(theta3),
        ],
        [
            sp.cos(theta3),
            -sp.sin(theta3),
            0,
            a3 * sp.cos(theta3),
        ],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
sp.pprint(H23.evalf())
x3, y3, z3 = H23[0, 3], H23[1, 3], H23[2, 3]
H34 = sp.Matrix(
    [
        [
            sp.cos(theta4),
            0,
            sp.sin(theta4),
            a4 * sp.cos(theta4 - theta2),
        ],
        [sp.sin(theta4), 0, -sp.cos(theta4), a4 * sp.sin(theta4 - theta2)],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)
sp.pprint(H34.evalf())
x4, y4, z4 = H34[0, 3], H34[1, 3], H34[2, 3]
H45 = sp.Matrix(
    [
        [sp.cos(theta5), -sp.sin(theta5), 0, a5 * sp.sin(0 + theta2)],
        [sp.sin(theta5), sp.cos(theta5), 0, 0],
        [0, 0, 1, -a5 * sp.cos(0 + theta2)],
        [0, 0, 0, 1],
    ]
)
sp.pprint(H45.evalf())
x5, y5, z5 = H45[0, 3], H45[1, 3], H45[2, 3]


xjoint0, yjoint0, zjoint0 = 0, 0, 0
xjoint1, yjoint1, zjoint1 = H01[0, 3], H01[1, 3], H01[2, 3]
H02 = H01 * H12
xjoint2, yjoint2, zjoint2 = H02[0, 3], H02[1, 3], H02[2, 3]
H03 = H02 * H23
xjoint3, yjoint3, zjoint3 = H03[0, 3], H03[1, 3], H03[2, 3]
H04 = H03 * H34
xjoint4, yjoint4, zjoint4 = H04[0, 3], H04[1, 3], H04[2, 3]
H05 = H04 * H45
xjoint5, yjoint5, zjoint5 = H05[0, 3], H05[1, 3], H05[2, 3]
xend, yend, zend = H05[0, 3], H05[1, 3], H05[2, 3]


points = [
    (xjoint0, yjoint0, zjoint0),
    (xjoint1, yjoint1, zjoint1),
    (xjoint2, yjoint2, zjoint2),
    (xjoint3, yjoint3, zjoint3),
    (xjoint4, yjoint4, zjoint4),
    (xjoint5, yjoint5, zjoint5),
    (xend, yend, zend),
]
# --- SAFE 3D PLOT BLOCK (drop-in replacement) ---

# If you're on macOS and still see crashes, uncomment the next two lines BEFORE importing pyplot:
# import matplotlib
# matplotlib.use("TkAgg")  # or "Qt5Agg"


import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale for accurate geometry."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# === your plotting code ===
colors = ["red", "orange", "yellow", "green", "blue", "purple"]

P = np.array([[float(sp.N(v)) for v in p] for p in points], dtype=float)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(P[:, 0], P[:, 1], P[:, 2], color="gray", linewidth=2, alpha=0.6)

for i, ((x, y, z), c) in enumerate(zip(P, colors), start=1):
    ax.scatter([x], [y], [z], c=[c], s=80, edgecolors="black", depthshade=True)
    ax.text(
        x,
        y,
        z + 5,
        f"J{i}\n({x:.2f}, {y:.2f}, {z:.2f})",
        color="black",
        fontsize=9,
        ha="center",
        va="bottom",
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Robot Arm with Equal Axes")
ax.view_init(elev=25, azim=45)

set_axes_equal(ax)
plt.show()


H05 = H01 * H12 * H23 * H34 * H45
H05 = H05.evalf()
sp.pprint(H05)

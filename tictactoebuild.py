import sympy as sp
import matplotlib.pyplot as plt


# top right corner of the board
starting_position = (10, 20)
cellHeight = 20
cellWidth = 20


def pointsCircle(pointQty, x, y, radius):
    angle = 360 / pointQty
    points = []
    for i in range(pointQty):
        x1 = x + radius * sp.cos(sp.rad(angle * i))
        y1 = y + radius * sp.sin(sp.rad(angle * i))
        points.append((x1, y1))
    return points


def pointsCross(x, y, cellHeight, cellWidth):
    xtopLeft = (x - 0.8 * cellWidth / 2, y + 0.8 * cellHeight / 2)
    xtopRight = (x + 0.8 * cellWidth / 2, y + 0.8 * cellHeight / 2)
    xbottomLeft = (x - 0.8 * cellWidth / 2, y - 0.8 * cellHeight / 2)
    xbottomRight = (x + 0.8 * cellWidth / 2, y - 0.8 * cellHeight / 2)
    return [xtopLeft, xtopRight, xbottomLeft, xbottomRight]


def pointsBoard(starting_position, cellHeight, cellWidth):
    x, y = starting_position
    # vertical lines
    leftVerticalstart = x + cellWidth, y + 3 * cellHeight
    leftVerticalend = x + cellWidth, y
    rightVerticalstart = x + 2 * cellWidth, y
    rightVerticalend = x + 2 * cellWidth, y + 3 * cellHeight
    # horizontal lines
    topHorizontalstart = x, y + 2 * cellHeight
    topHorizontalend = x + 3 * cellWidth, y + 2 * cellHeight
    bottomHorizontalstart = x + 3 * cellWidth, y + cellHeight
    bottomHorizontalend = x, y + cellHeight
    return [
        leftVerticalstart,
        leftVerticalend,
        rightVerticalstart,
        rightVerticalend,
        topHorizontalstart,
        topHorizontalend,
        bottomHorizontalstart,
        bottomHorizontalend,
    ]


def cellCenters(starting_position, cellHeight, cellWidth):
    x, y = starting_position
    cell1 = (x + cellWidth / 2, y + 5 * cellHeight / 2)
    cell2 = (x + 3 * cellWidth / 2, y + 5 * cellHeight / 2)
    cell3 = (x + 5 * cellWidth / 2, y + 5 * cellHeight / 2)
    cell4 = (x + cellWidth / 2, y + 3 * cellHeight / 2)
    cell5 = (x + 3 * cellWidth / 2, y + 3 * cellHeight / 2)
    cell6 = (x + 5 * cellWidth / 2, y + 3 * cellHeight / 2)
    cell7 = (x + cellWidth / 2, y + cellHeight / 2)
    cell8 = (x + 3 * cellWidth / 2, y + cellHeight / 2)
    cell9 = (x + 5 * cellWidth / 2, y + cellHeight / 2)
    return [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]


boardNew = pointsBoard(starting_position, cellHeight, cellWidth)
centers = cellCenters(starting_position, cellHeight, cellWidth)
boardxlist = [x[0] for x in boardNew]
boardylist = [y[1] for y in boardNew]
plt.plot(boardxlist[0:2], boardylist[0:2], color="black")  # left vertical
plt.plot(boardxlist[2:4], boardylist[2:4], color="green")  # right vertical
plt.plot(boardxlist[4:6], boardylist[4:6], color="red")  # top horizontal
plt.plot(boardxlist[6:8], boardylist[6:8], color="blue")  # bottom horizontal
centersxlist = [x[0] for x in centers]
centersylist = [y[1] for y in centers]
plt.scatter(centersxlist, centersylist, color="orange")

plt.axis("equal")
plt.show()

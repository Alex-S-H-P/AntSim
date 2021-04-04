from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states as createRNGState, xoroshiro128p_normal_float64 as normal
from typing import Tuple, Dict
import numpy as np
import math
from random import randint

foodColor = np.array((0, 0, 0, 0, 0, 0, 1)).astype(float)
seekingAntColor = np.array((1, 0, 0, 0, 0, 0, 0)).astype(float)
homingAntColor = np.array((0, 1, 0, 0, 0, 0, 0)).astype(float)
colonyColor = np.array((0, 0, 0, 0, 0, 1, 0)).astype(float)
HomeTrailColor = np.array((0, 0, 0, 0, 1, 0, 0)).astype(float)
FoodTrailColor = np.array((0., 0., 0., 1., 0., 0., 0.))


def randomPos(dimensions: Tuple[int, int]):
    return randint(0, dimensions[0] - 1), randint(0, dimensions[1] - 1)


# Each and every pixel on the drawn screen will actually contain multiple values
# in order :
# One dimension for seekingAnts, one for homing ants (two ants cannot coexist on the same pixel)
# One dimension for an angle that the ant wants to move in (dim2)
# One for each trail (dim 3 and 4)
# one for the colony (dim 5)
# one for the food (dim 6)

class UniverseScreen:

    def __init__(self, screenDim: Tuple[int, int] = None, imgArray: np.ndarray = None,
                 antColonyCoord: Tuple[int, int] = None, diffusionFactor: float = 0.01,
                 evaporationFactor: float = 0.99, antNb: int = 20, foodNb=40):
        if imgArray is not None:
            self.mainScreen = imgArray
            self.screenDim = imgArray.shape[:2]
            self.antNb = self.countAnts()
        else:
            assert screenDim is not None, "must specify screenDimensions if no image is given"
            self.screenDim = screenDim
            self.antNb = antNb
            self.mainScreen = np.zeros(screenDim + (7,), float)
            if antColonyCoord is not None:
                self.mainScreen[antColonyCoord] = colonyColor
                x, y = antColonyCoord
            else:
                x = screenDim[0] // 2
                y = screenDim[1] // 2
                self.mainScreen[x, y] += colonyColor
            for _ in range(antNb):
                x, y = randomPos(screenDim)
                self.mainScreen[x, y] += (1, 0, np.random.random() * 2 * math.pi, 0, 0, 0, 0)
            x, y = screenDim
            for _ in range(foodNb):
                dx, dy = np.random.randint(0, screenDim) // 5
                self.mainScreen[x - dx - 1, y - dy - 1] += np.array(foodColor)
        self.diffusionFactor = diffusionFactor
        self.evaporationFactor = evaporationFactor

    def countAnts(self) -> int:
        count = 0
        for pixel in self.mainScreen.reshape((self.screenDim[0] * self.screenDim[1], 7)):
            if pixel[1] != 0. or pixel[0] != 0.:
                count += 1
        return count

    def update(self):
        n, p = self.screenDim
        threadsPerBlock = (16, 16)
        blocksPerGrid_x = math.ceil(n / threadsPerBlock[0])
        blocksPerGrid_y = math.ceil(p / threadsPerBlock[1])
        BlocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
        rng = createRNGState(256 * blocksPerGrid_x * blocksPerGrid_y, seed=1)

        motionRequests = np.ones(self.screenDim + (3,), float) * (-1)

        # self.debugUpdate(self.mainScreen, self.diffusionFactor, self.evaporationFactor, motionRequests, rng)

        self._update[BlocksPerGrid, threadsPerBlock](self.mainScreen, self.diffusionFactor, self.evaporationFactor,
                                                     motionRequests, rng)
        # all pixels are updated
        destinations: Dict[Tuple[int, int], Tuple[int, int]] = {}
        n, p = motionRequests.shape[:2]
        for i in range(n):
            for j in range(p):
                if motionRequests[i, j, 0] < 0:
                    continue
                else:
                    # position adjustments
                    x, y = motionRequests[i, j, :2]
                    if motionRequests[i, j, 2] > .5:
                        # SWITCH MODE
                        self.mainScreen[i, j, :2] = np.flip(self.mainScreen[i, j, :2], 0)
                    x, y = int(x + .5), int(y + .5)
                    if x < 0 or x >= n:
                        x = max(0, min(x, n-1))
                        self.mainScreen[i, j, 2] = (math.pi  - self.mainScreen[i, j, 2])
                    if y < 0 or y >= n:
                        y = max(0, min( y, n-1))
                        self.mainScreen[i, j, 2] *= -1
                    while (x, y) in destinations or 0 > x or x > n or y < 0 or p < y:
                        x += [-1, 0, 0, 1][randint(0, 3)]
                        y += [-1, 0, 0, 1][randint(0, 3)]
                    destinations[(x, y)] = (i, j)
        # all ants can move
        overWrittenAnts: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for xf, yf in destinations:
            xO, yO = destinations[xf, yf]  # ants origin
            antType = 1
            if self.mainScreen[xO, yO, 1] > .5:
                antType = 2
            # protecting the destination cell in case it contains a cell
            if self.mainScreen[xf, yf, 1] > .5 or self.mainScreen[xf, yf, 0] > .5:
                destType = 1
                if self.mainScreen[xf, yf, 1] > .5:
                    destType = 2
                destAngle = self.mainScreen[xf, yf, 2]
                overWrittenAnts[xf, yf] = destType, destAngle
            # write into that space
            if (xO, yO) in overWrittenAnts:
                antType, angle = overWrittenAnts[xO, yO]
                self.mainScreen[xf, yf, 2] = angle
                self.mainScreen[xf, yf, antType - 1] = 1.
            else:
                self.mainScreen[xf, yf, :3] = self.mainScreen[xO, yO, :3]
                self.mainScreen[xO, yO, :3] = (0., 0., 0.)
        print(len(overWrittenAnts), len(destinations))

    def draw(self):
        n, p = self.screenDim
        screen = np.ones((n, p, 3), float)

        threadsPerBlock = (16, 16)
        blocksPerGrid_x = math.ceil(n / threadsPerBlock[0])
        blocksPerGrid_y = math.ceil(p / threadsPerBlock[1])
        BlocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)

        # self._debugDraw(self.mainScreen, screen)
        self._draw[BlocksPerGrid, threadsPerBlock](self.mainScreen, screen)
        return screen

    @staticmethod
    @cuda.jit
    def _draw(mainScreen, realScreen):

        i, j = cuda.grid(2)
        n, p, t = realScreen.shape

        def thereIsAHoleNearby(i, j):
            for k in range(-2, 2):
                for l in range(-2, 2):
                    if n > i + k >= 0 and p > j + l >= 0:
                        if mainScreen[i + k, j + l, 5] > .5:
                            return True

        realScreen[i, j, 1] -= mainScreen[i, j, 3]
        realScreen[i, j, 2] -= mainScreen[i, j, 4]
        if i < n and j < p:
            if mainScreen[i, j, 6] > 0.5:
                realScreen[i, j] = (0., 0., 0.)
            elif mainScreen[i, j, 5] > 0.5:
                realScreen[i, j] = 165 / 255, 42 / 255, 42 / 255
            elif mainScreen[i, j, 0] > 0.5:
                realScreen[i, j, 0] = 1.
                realScreen[i, j, 1] = 0.
                realScreen[i, j, 2] = 0.
            elif mainScreen[i, j, 1] > 0.5:
                realScreen[i, j] = (1. + 0. + .7)
            elif thereIsAHoleNearby(i, j):
                realScreen[i, j] = 165 / 257, 42 / 257, 42 / 257

    @staticmethod
    @cuda.jit
    def _update(screenArray, diff, evaporationFactor, motionRequests, rngState):
        ## setting values
        speed = 3
        grabRange = 3
        environmentRadius = 5
        visionAngle = math.pi * 0.45

        ## cuda handling
        i, j = cuda.grid(2)
        n, p, t = screenArray.shape
        cudaIndex = n * i + p
        ## algo
        if i < n and j < p:
            thisCell = screenArray[i, j]
            antType = 0
            if thisCell[0] > .5:
                antType = 1
            elif thisCell[1] > .5:
                antType = 2
            neighbours = screenArray[max(i - environmentRadius, 0):min(i + environmentRadius, n),
                                     max(j - environmentRadius, 0):min(j + environmentRadius, p)]
            xCur, yCur = (min(environmentRadius, i), min(environmentRadius, j))
            trailCenter_0 = 0.
            trailCenter_1 = 0.
            trailCenter_2 = 0.
            trailCenter_3 = 0.
            foodCoord_0 = 0.
            foodCoord_1 = 0.
            foodFound = False
            foodCounter = 0
            switching = 0.
            nbCases = neighbours.shape[0] * neighbours.shape[1]
            if antType != 0:
                thisCell[antType + 2] += 1.
            for row in range(neighbours.shape[0]):
                for col in range(neighbours.shape[1]):
                    if xCur == row and yCur == col:
                        continue
                    inverseDistanceToCenter = 1 / math.sqrt((row - xCur) ** 2 + (col - yCur) ** 2)
                    # Handling diffusion and evaporation
                    factor = inverseDistanceToCenter * inverseDistanceToCenter * diff
                    thisCell[3] += (neighbours[row, col, 3]-thisCell[3]) * factor
                    thisCell[4] += (neighbours[row, col, 4]-thisCell[4]) * factor
                    trailCenter_0 += neighbours[row, col, 3] * (row - xCur) / nbCases
                    trailCenter_2 += neighbours[row, col, 4] * (row - xCur) / nbCases
                    trailCenter_1 += neighbours[row, col, 3] * (col - yCur) / nbCases
                    trailCenter_3 += neighbours[row, col, 4] * (col - yCur) / nbCases
                    if neighbours[row, col, 6] == 1. and antType == 1:
                        foodCoord_0 += row
                        foodCoord_1 += col
                        foodCounter += 1
                        if inverseDistanceToCenter > 1 / grabRange:
                            switching = 1.
                    elif neighbours[row, col, 5] == 1. and not foodFound and antType == 2:
                        foodCoord_0 = row
                        foodCoord_1 = col
                        if inverseDistanceToCenter > 1 / grabRange:
                            switching = 1.
            thisCell[3] = max(0., thisCell[3] * evaporationFactor)
            thisCell[4] = max(0., thisCell[4] * evaporationFactor)
            angle = thisCell[2]
            if foodCounter > 0:
                foodCoord_0 /= foodCounter
                foodCoord_1 /= foodCounter
            deviation = 0.
            if antType != 0:
                if not foodFound:
                    if antType == 2:
                        xO, yO = trailCenter_0, trailCenter_1
                    else:
                        xO, yO = trailCenter_2, trailCenter_3
                else:
                    xO, yO = foodCoord_0, foodCoord_1
                if xO != 0 and yO != 0:
                    angleOfInterest = math.atan(xO / yO)
                    if abs(angleOfInterest - angle) > math.pi:
                        deviation = math.pi - (angleOfInterest - angle)
                    else:
                        deviation = angleOfInterest - angle
                deviation += normal(rngState, cudaIndex) * math.pi * 0.08
                thisCell[2] += deviation / 30
                motionRequests[i, j][0] = i + speed * math.cos(thisCell[2])
                motionRequests[i, j][1] = j - speed * math.sin(thisCell[2])
                motionRequests[i, j][2] = switching

    @staticmethod
    def debugUpdate(screenArray, diff, evaporationFactor, motionRequests, rngState):
        ## setting values
        speed = 2
        grabRange = 3
        environmentRadius = 5

        ## cuda handling
        n, p, t = screenArray.shape

        ## algo
        for i in range(n):
            for j in range(p):
                cudaIndex = n * i + p
                thisCell = screenArray[i, j]
                antType = 0
                if thisCell[0] > .5:
                    antType = 1

                elif thisCell[1] > .5:
                    antType = 2

                neighbours = screenArray[max(i - environmentRadius, 0):min(i + environmentRadius, n),
                                         max(j - environmentRadius, 0):min(j + environmentRadius, p)]
                xCur, yCur = (min(environmentRadius, i), min(environmentRadius, j))
                trailCenter_0 = 0.
                trailCenter_1 = 0.
                trailCenter_2 = 0.
                trailCenter_3 = 0.
                foodCoord_0 = 0.
                foodCoord_1 = 0.
                foodFound = False
                switching = 0.
                nbCases = neighbours.shape[0] * neighbours.shape[1]
                if antType != 0:
                    thisCell[antType + 2] += 1
                for row in range(neighbours.shape[0]):
                    for col in range(neighbours.shape[1]):
                        if xCur == row and yCur == col:
                            continue
                        inverseDistanceToCenter = 1 / math.sqrt((row - xCur) ** 2 + (col - yCur) ** 2)
                        # Handling diffusion and evaporation
                        thisCell[3] += neighbours[
                                           row, col, 3] * inverseDistanceToCenter * inverseDistanceToCenter * diff
                        thisCell[4] += neighbours[
                                           row, col, 4] * inverseDistanceToCenter * inverseDistanceToCenter * diff
                        trailCenter_0 += neighbours[row, col, 3] * (row - xCur) / nbCases
                        trailCenter_2 += neighbours[row, col, 4] * (row - xCur) / nbCases
                        trailCenter_1 += neighbours[row, col, 3] * (col - yCur) / nbCases
                        trailCenter_3 += neighbours[row, col, 4] * (col - yCur) / nbCases
                        if neighbours[row, col, 6] == 1. and not foodFound and antType == 1:
                            foodCoord_0 = row
                            foodCoord_1 = col
                            if inverseDistanceToCenter > 1 / grabRange:
                                switching = 1.
                        elif neighbours[row, col, 5] == 1. and not foodFound and antType == 2:
                            foodCoord_0 = row
                            foodCoord_1 = col
                            if inverseDistanceToCenter > 1 / grabRange:
                                switching = 1.
                thisCell[3] /= nbCases
                thisCell[3] = max(0., thisCell[3] * evaporationFactor)
                thisCell[4] /= nbCases
                thisCell[4] = max(0., thisCell[4] * evaporationFactor)
                angle = thisCell[2]
                deviation = 0.
                if antType != 0:
                    if not foodFound:
                        if antType == 1:
                            xO, yO = trailCenter_0, trailCenter_1
                        else:
                            xO, yO = trailCenter_2, trailCenter_3
                    else:
                        xO, yO = foodCoord_0, foodCoord_1
                    if xO != 0 and yO != 0:
                        angleOfInterest = math.atan(yO / xO)
                        if abs(angleOfInterest - angle) > math.pi:
                            deviation = math.pi - (angleOfInterest - angle)
                        else:
                            deviation = angleOfInterest - angle
                    deviation += np.random.normal(0, 1) * math.pi * 0.10
                    print(deviation)
                    thisCell[2] += deviation / 30
                    motionRequests[i, j][0] = float(i) + speed * math.cos(thisCell[2])
                    motionRequests[i, j][1] = float(j) - speed * math.sin(thisCell[2])
                    motionRequests[i, j][2] = switching

    @staticmethod
    def _debugDraw(mainScreen, realScreen):

        n, p, t = realScreen.shape

        def thereIsAHoleNearby(i, j):
            for k in range(-2, 2):
                for l in range(-2, 2):
                    if n > i + k >= 0 and p > j + l >= 0:
                        if mainScreen[i + k, j + l, 5] > .5:
                            return True

        for i in range(n):
            for j in range(p):
                if mainScreen[i, j, 6] > 0.5:
                    realScreen[i, j] = (0., 0., 0.)
                elif mainScreen[i, j, 5] > 0.5:
                    realScreen[i, j] = 165 / 255, 42 / 255, 42 / 255
                elif mainScreen[i, j, 0] > 0.5:
                    realScreen[i, j, 0] = 1.
                    realScreen[i, j, 1] = 0.
                    realScreen[i, j, 2] = 0.
                elif mainScreen[i, j, 1] > 0.5:
                    realScreen[i, j, 0] = 1.
                    realScreen[i, j, 1] = 0.
                    realScreen[i, j, 2] = .7
                elif thereIsAHoleNearby(i, j):
                    realScreen[i, j] = 165 / 257, 42 / 257, 42 / 257
                else:
                    realScreen[i, j, 0] = 1
                    realScreen[i, j, 1] = 1-mainScreen[i, j, 3]
                    realScreen[i, j, 2] = 1-mainScreen[i, j, 4]

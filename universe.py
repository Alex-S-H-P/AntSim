from numba import cuda
import numpy as np
from typing import Tuple, Dict, List, Union
from numba.cuda.random import create_xoroshiro128p_states as createStates, xoroshiro128p_normal_float32 as normal
from math import ceil, isqrt, asin, sqrt, fabs as abs
import pygame


def randomPosition(maxCoord: Tuple[float, float]):
    mult = np.array(maxCoord)
    return tuple(np.random.random((2,)) * mult)


class Universe:

    def __init__(self, antRange=5., antVisionAngle=110., antNb=5, foodNb=10, antSpeed=3., screenDim=(100, 100),
                 evaporationMultiplier=0.99, antGrabbingRange=1.):
        self.inverseGrabbingRange = 1 / antGrabbingRange
        self.evaporationMultiplier = evaporationMultiplier
        self.frame = 0
        self.antSpeed = antSpeed
        self.antVisionAngle = antVisionAngle
        self.screenDim = screenDim
        self.antRange = antRange
        self.chunksDim = ceil(screenDim[0] / antRange), ceil(screenDim[1] / antRange)
        self.chunks = [[[] for _ in range(self.chunksDim[1])]
                       for _ in range(self.chunksDim[0])]
        n = len(self.chunks)
        p = len(self.chunks[0])
        for ant in range(antNb):
            self.chunks[n // 2][p // 2] = [0,
                                           0,
                                           self.screenDim[0] / 2,
                                           self.screenDim[1] / 2,
                                           np.random.random() * 360,
                                           1,
                                           0]
        for ant in range(foodNb):
            self.chunks[n // 2][p // 2] = [7,
                                           0,
                                           self.screenDim[0] / 2,
                                           self.screenDim[1] / 2,
                                           0,
                                           1,
                                           0]
        # ants have id 0-4
        # home trails have id 4
        # food trails have id 5
        # go back trails have id 6
        # food has id 7

    def update(self):
        threadsPerBlock = 16, 16
        x = ceil(self.chunksDim[0] / threadsPerBlock[0])
        y = ceil(self.chunksDim[1] / threadsPerBlock[1])
        blocksPerGrid = x, y
        rngState = createStates(self.frame, 1)
        numThreads = self.chunksDim[0] * self.chunksDim[1]
        chunkless = [[] for i in range(numThreads)]
        nestCoords = (self.screenDim[0] / 2,
                      self.screenDim[1] / 2)
        self._update[threadsPerBlock, blocksPerGrid](
            self.chunks, self.antSpeed, self.antRange, self.antVisionAngle, rngState, self.antVisionAngle / 100,
            self.evaporationMultiplier, chunkless, self.inverseGrabbingRange, nestCoords)
        for item in chunkless:
            x, y = item[2:4]
            if x < 0 or x > self.screenDim[0]:
                x = abs(self.screenDim[0] - x)
            if y < 0 or y > self.screenDim[1]:
                y = abs(self.screenDim[1] - y)
            i, j = int(x / self.screenDim[0]), int(y / self.screenDim[1])
            self.chunks[j][i].append(item)

    def draw(self, screen):
        nestCoords = (self.screenDim[0] / 2,
                      self.screenDim[1] / 2)
        colors = [(255, 0, 0),
                  (255, 255, 0),
                  (255, 0, 255),
                  (255, 120, 120),
                  (0, 120, 255),
                  (0, 255, 255),
                  (120, 255, 255),
                  (0, 255, 255)]
        for _ in self.chunks:
            for chunk in _:
                for item in chunk:
                    x = item[2]
                    y = item[3]
                    color = colors[item[0]]
                    rect = pygame.Rect(x - 2, y - 2, 4, 4)
                    pygame.draw.rect(screen, color, rect, 1)
            colony = pygame.Rect(nestCoords[0] - 5, nestCoords[1] - 5, 10, 10)
            color = (165, 42, 42)
            pygame.draw.rect(screen, color, colony)

    @staticmethod
    @cuda.jit
    def _update(chunks, speed, visionRange, angle, rngState, AntSpeedDeviationFactor, evaporationMultiplier, chunkless,
                inverseGrabbingRange, nestCoords):

        def isInView(pos, alpha, item):
            dx = item[2] - pos[0]
            dy = item[3] - pos[1]
            theta = asin(dx)
            if abs(theta - alpha) < angle:
                return sqrt(dx ** 2 + dy ** 2) < visionRange
            else:
                return False

        def antMotion(neighbors, ant):
            direction = [normal(rngState, ThreadId) * AntSpeedDeviationFactor,
                         normal(rngState, ThreadIdPlus) * AntSpeedDeviationFactor]  # random deviation for exploration
            for item in neighbors:
                if not isInView(ant[2:4], ant[4], item):
                    continue
                if item[0] == ant[0] + 4.:
                    # found the right trail
                    inverseDistance = isqrt((item[2] - pos[0]) ** 2 + (item[3] - pos[1]) ** 2)
                    signalAmplitude = item[1] * inverseDistance ** 2
                    direction[0] += (item[2] - pos[0]) * signalAmplitude
                    direction[1] += (item[3] - pos[1]) * signalAmplitude
                elif item[0] == 7. and item[1] == 0. and ant[0] == 0.:
                    # Food takes precedences and therefore returns it's own direction
                    inverseDistance = isqrt((item[2] - pos[0]) ** 2 + (item[3] - pos[1]) ** 2)
                    factor = inverseDistance * speed
                    direction = [(item[2] - pos[0]) * factor,
                                 (item[3] - pos[1]) * factor]
                    if inverseDistance > inverseGrabbingRange:
                        ant[0] = 1.
                        ant[1] = direction[0]
                        item[1] = direction[0]
                    return direction
                elif item[0] == 7. and ant[0] == 1. and ant[1] == item[1] and (
                        distance := sqrt((nestCoords[0] - ant[2]) ** 2 + (nestCoords[1] - ant[3]) ** 2)) < visionRange:
                    factor = speed / distance
                    direction = [(item[2] - pos[0]) * factor,
                                 (item[3] - pos[1]) * factor]
                    if distance > 1 / inverseGrabbingRange:
                        ant[1] = 1.
                    return direction
            normalisationFactor = isqrt(direction[0] ** 2 + direction[1] ** 2)
            direction[0] *= normalisationFactor * speed
            direction[1] *= normalisationFactor * speed
            return direction

        i, j = cuda.grid(2)
        n, p = len(chunks), len(chunks[0])
        ThreadId = n * i + j
        ThreadIdPlus = n * p + ThreadId
        if i < n and j < p:
            additional_data = chunkless[ThreadId]
            cur_chunk = chunks[i][j][:]  # we copy the chunk as a way of securing it

            if len(cur_chunk) != 0:
                neighbours = [[0., 0., 0., 0., 0., 0., 0.]]
                for k in range(-1, 2):
                    if 0 < i + k < n:
                        for l in range(-1, 2):
                            if 0 < j + l < p:
                                neighbours += chunks[k][l]
                if len(neighbours) > 1:
                    neighbours = neighbours[1:]
                else:
                    # todo handle there being no neighbors
                for index in range(len(cur_chunk)):
                    item = cur_chunk[index]
                    EntityType = item[0]
                    timer = item[1]
                    pos = item[2:4]
                    alpha = item[4]
                    vel = item[5:7]
                    if EntityType < 4.:
                        # Ant
                        vel = antMotion(neighbours, item)
                        timer += 0.03
                        if timer > 0.5:
                            timer = 0.
                            additional_data.append([4 + EntityType, 1] + pos + [alpha] + vel)
                        pos[0] += vel[0]
                        pos[1] += vel[1]
                        # food acquisition and relay
                    elif 4. <= EntityType < 7.:
                        # Trail
                        timer = timer * evaporationMultiplier
                        if timer < 0.001:
                            cur_chunk = cur_chunk[:index] + cur_chunk[index + 1:]
                            continue
                    else:
                        # Food
                        pass
            chunks[i][j] = cur_chunk

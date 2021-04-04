from matplotlib import pyplot as plt, image
from universe import UniverseScreen
from numba import cuda, njit, jit, types

size = height, width = 200, 200

leaveMainLoop = False

universe = UniverseScreen(size, evaporationFactor=.951, diffusionFactor=0.001)
print(universe.countAnts())

# main loop
while not leaveMainLoop:
    try:
        img = universe.draw()
        plt.clf()
        plt.imshow(img)
        plt.draw()
        plt.pause(1/30)
        universe.update()
    except KeyboardInterrupt:
        leaveMainLoop = True
        print(img[:10, :10])

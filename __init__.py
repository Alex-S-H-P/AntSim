from matplotlib import pyplot as plt, image
from universe import UniverseScreen
from numba import cuda, njit, jit, types

size = height, width = 200, 200

leaveMainLoop = False

universe = UniverseScreen(size, evaporationFactor=0.995, diffusionFactor=0.005)
print(universe.countAnts())

# main loop
while not leaveMainLoop:
    try:
        universe.update()
        img = universe.draw()
        plt.clf()
        plt.imshow(img)
        plt.draw()
        plt.pause(1)
    except KeyboardInterrupt:
        leaveMainLoop = True
        print(universe.mainScreen[:10, :10])

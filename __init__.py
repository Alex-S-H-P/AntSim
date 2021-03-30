import pygame
from universe import Universe
from numba import cuda, njit, jit, types

pygame.init()

size = height, width = 1920, 760

screen = pygame.display.set_mode(size)
leaveMainLoop = False

universe = Universe(5, 110., 5, 10, 1, size, antGrabbingRange=3.)

# main loop
while not leaveMainLoop:
    screen.fill((255, 255, 255))
    pygame.display.flip()
    universe.update()
    universe.draw(screen)

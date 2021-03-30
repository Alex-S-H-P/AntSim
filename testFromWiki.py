import sys, pygame
pygame.init()

size = width, height = 1600, 760
speed = [2, 2]
black = 0, 0, 0
white = 255, 255, 255
screen = pygame.display.set_mode(size)

x, y = 0, 0
rect = pygame.Rect(x, y, 50, 50)
while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    screen.fill(white)
    pygame.draw.rect(screen, color=(0, 0, 255), rect=rect)
    pygame.display.flip()